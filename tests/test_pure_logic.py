"""
純邏輯的單元測試：不碰 Discord、不打真的 LLM API。
涵蓋 schedule 邊界、模型輸出解析、時間差格式化。
"""

import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from response_parser import parse_response
from schedule import format_elapsed, get_schedule_status
from session import UserSession
from proactive import should_attempt_proactive, attempt_proactive_message
from memory import MemoryManager

TZ = ZoneInfo("Asia/Taipei")


def dt(hour, minute=0, weekday_date="2026-08-24"):  # 2026-08-24 是週一
    d = datetime.fromisoformat(f"{weekday_date}T{hour:02d}:{minute:02d}:00")
    return d.replace(tzinfo=TZ)


class TestScheduleStatus:
    def test_sleep_start_boundary_is_blocking(self):
        status = get_schedule_status(dt(0, 0))
        assert status.blocking is True
        assert status.reason == "sleep"

    def test_wake_hour_boundary_is_not_sleep(self):
        status = get_schedule_status(dt(10, 0))
        assert status.blocking is False

    def test_just_before_wake_hour_is_still_sleep(self):
        status = get_schedule_status(dt(9, 59))
        assert status.blocking is True
        assert status.reason == "sleep"

    def test_gym_start_boundary_is_blocking(self):
        status = get_schedule_status(dt(17, 0))
        assert status.blocking is True
        assert status.reason == "gym"

    def test_gym_end_boundary_is_not_blocking(self):
        status = get_schedule_status(dt(19, 0))
        assert status.blocking is False

    def test_weekday_daytime_label(self):
        status = get_schedule_status(dt(11, 0, "2026-08-24"))  # 週一
        assert status.blocking is False
        assert status.label == "白天上課中"

    def test_weekend_label(self):
        status = get_schedule_status(dt(11, 0, "2026-08-29"))  # 週六
        assert status.blocking is False
        assert status.label == "假日，沒有固定行程"

    def test_weekday_evening_label(self):
        status = get_schedule_status(dt(20, 0, "2026-08-24"))
        assert status.blocking is False
        assert status.label == "晚上自由時間"

    def test_available_at_is_end_of_blocking_period(self):
        status = get_schedule_status(dt(2, 0))
        assert status.available_at.hour == 10
        assert status.available_at.minute == 0


class TestFormatElapsed:
    def test_under_an_hour(self):
        start = dt(10, 0)
        end = dt(10, 45)
        assert format_elapsed(start, end) == "45分鐘"

    def test_exact_hour(self):
        start = dt(10, 0)
        end = dt(12, 0)
        assert format_elapsed(start, end) == "2小時"

    def test_hour_with_remainder(self):
        start = dt(10, 0)
        end = dt(12, 30)
        assert format_elapsed(start, end) == "2小時30分鐘"


class TestUserSessionPendingBuffer:
    def test_take_pending_buffer_returns_current_content(self):
        session = UserSession()
        session.pending_buffer.append({"text": "hi", "images": []})

        taken = session.take_pending_buffer()

        assert taken == [{"text": "hi", "images": []}]

    def test_take_pending_buffer_leaves_fresh_empty_list(self):
        session = UserSession()
        session.pending_buffer.append({"text": "hi", "images": []})

        taken = session.take_pending_buffer()
        session.pending_buffer.append({"text": "new message during processing", "images": []})

        # 舊 snapshot 不受之後寫入影響
        assert taken == [{"text": "hi", "images": []}]
        # 新訊息進到全新的 buffer，不會跟舊 snapshot 混在一起
        assert session.pending_buffer == [{"text": "new message during processing", "images": []}]

    def test_clear_pending_does_not_wipe_buffer_written_during_processing(self):
        session = UserSession()
        session.pending_task = object()
        session.pending_mode = "debounce"

        session.take_pending_buffer()
        session.pending_buffer.append({"text": "arrived while awaiting LLM", "images": []})
        session.clear_pending()

        assert session.pending_task is None
        assert session.pending_mode is None
        # clear_pending 只清旗標，不應清掉處理期間新寫入的 buffer
        assert session.pending_buffer == [{"text": "arrived while awaiting LLM", "images": []}]


class TestParseResponse:
    def test_normal_response(self):
        raw = json.dumps({
            "reply": "今天過得不錯",
            "state": "心情平靜，剛洗完澡",
            "image_prompt": None,
        })
        result = parse_response(raw)
        assert result.reply == "今天過得不錯"
        assert result.state == "心情平靜，剛洗完澡"
        assert result.image_prompt is None

    def test_response_with_image_prompt(self):
        raw = json.dumps({
            "reply": "傳張照片給你看",
            "state": "剛拍完照，心情不錯",
            "image_prompt": "casual outfit, smiling, indoor, soft lighting, close-up",
        })
        result = parse_response(raw)
        assert result.reply == "傳張照片給你看"
        assert result.state == "剛拍完照，心情不錯"
        assert result.image_prompt == "casual outfit, smiling, indoor, soft lighting, close-up"

    def test_empty_state_string_becomes_none(self):
        raw = json.dumps({"reply": "回覆內容", "state": "", "image_prompt": None})
        result = parse_response(raw)
        assert result.state is None

    def test_strips_whitespace(self):
        raw = json.dumps({
            "reply": "  回覆內容  ",
            "state": "  狀態內容  ",
            "image_prompt": None,
        })
        result = parse_response(raw)
        assert result.reply == "回覆內容"
        assert result.state == "狀態內容"

    def test_malformed_json_falls_back_to_raw_text_as_reply(self):
        raw = "不是合法 JSON 的原始文字"
        result = parse_response(raw)
        assert result.reply == "不是合法 JSON 的原始文字"
        assert result.state is None
        assert result.image_prompt is None

    def test_missing_fields_do_not_crash(self):
        raw = json.dumps({"reply": "只有 reply 欄位"})
        result = parse_response(raw)
        assert result.reply == "只有 reply 欄位"
        assert result.state is None
        assert result.image_prompt is None


class TestShouldAttemptProactive:
    def test_no_last_interaction_returns_false(self):
        session = UserSession()
        assert should_attempt_proactive(session, dt(12, 0)) is False

    def test_before_min_quiet_minutes_returns_false(self):
        session = UserSession(last_interaction_time=dt(12, 0))
        now = dt(12, config.PROACTIVE_MIN_QUIET_MINUTES - 1)
        assert should_attempt_proactive(session, now) is False

    def test_at_min_quiet_minutes_rolls_probability(self, monkeypatch):
        session = UserSession(last_interaction_time=dt(12, 0))
        now = dt(12, config.PROACTIVE_MIN_QUIET_MINUTES)

        monkeypatch.setattr(random, "random", lambda: 0.0)
        assert should_attempt_proactive(session, now) is True

        monkeypatch.setattr(random, "random", lambda: 0.999)
        assert should_attempt_proactive(session, now) is False

    def test_probability_is_capped_at_max(self, monkeypatch):
        session = UserSession(last_interaction_time=dt(0, 0))
        now = dt(23, 59)  # 閒置快一整天，機率應該被封頂在 GATE_MAX_PROBABILITY

        monkeypatch.setattr(random, "random", lambda: config.PROACTIVE_GATE_MAX_PROBABILITY - 0.01)
        assert should_attempt_proactive(session, now) is True

        monkeypatch.setattr(random, "random", lambda: config.PROACTIVE_GATE_MAX_PROBABILITY + 0.01)
        assert should_attempt_proactive(session, now) is False


class _FakeChannel:
    """最小可用的假 Discord channel：可設定 send() 是否要拋錯，並記錄呼叫次數。"""

    def __init__(self, raise_on_send: bool = False):
        self.raise_on_send = raise_on_send
        self.send_calls = 0

    async def send(self, *args, **kwargs):
        self.send_calls += 1
        if self.raise_on_send:
            raise RuntimeError("discord api boom")


class _FakeMemoryManager:
    def __init__(self):
        self.bot_memory = ""
        self.append_turn_calls = 0

    def append_turn(self, session, user_content, reply_content):
        self.append_turn_calls += 1


class _FakeStateHolder:
    def __init__(self):
        self.value = "原本的狀態"
        self.update_calls = 0

    def update(self, new_state):
        self.update_calls += 1
        self.value = new_state


class TestAttemptProactiveMessageSendFailure:
    """#004：channel.send() 失敗不應讓例外往外冒，且不該誤記成功送出的狀態/記憶。"""

    def test_send_failure_does_not_raise_and_skips_state_update(self, monkeypatch):
        session = UserSession()
        session.last_channel = _FakeChannel(raise_on_send=True)
        memory_manager = _FakeMemoryManager()
        state_holder = _FakeStateHolder()

        async def fake_chat_completion(messages, response_format=None):
            return json.dumps({"reply": "在嗎", "state": "有點想你", "image_prompt": None})

        monkeypatch.setattr("proactive.chat_completion", fake_chat_completion)

        # 不應拋出例外
        asyncio.run(attempt_proactive_message(1, session, memory_manager, state_holder))

        assert session.last_channel.send_calls == 1
        # 發送失敗，狀態與記憶都不該被更新成「好像有送出去」的樣子
        assert state_holder.update_calls == 0
        assert memory_manager.append_turn_calls == 0

    def test_send_success_updates_state_and_memory(self, monkeypatch):
        session = UserSession()
        session.last_channel = _FakeChannel(raise_on_send=False)
        memory_manager = _FakeMemoryManager()
        state_holder = _FakeStateHolder()

        async def fake_chat_completion(messages, response_format=None):
            return json.dumps({"reply": "在嗎", "state": "有點想你", "image_prompt": None})

        monkeypatch.setattr("proactive.chat_completion", fake_chat_completion)

        asyncio.run(attempt_proactive_message(1, session, memory_manager, state_holder))

        assert session.last_channel.send_calls == 1
        assert state_holder.update_calls == 1
        assert memory_manager.append_turn_calls == 1


class TestMemoryManagerQueueSerialization:
    """#002：溢出的長期記憶更新要依序處理，不能因為同時觸發而互相覆蓋。"""

    def _make_manager(self, tmp_path, monkeypatch):
        bot_mem_path = tmp_path / "bot_mem.md"
        bot_mem_path.write_text("", encoding="utf-8")
        monkeypatch.setattr(config, "BOT_MEMORY_PATH", str(bot_mem_path))
        return MemoryManager(), bot_mem_path

    def test_overflow_updates_are_processed_in_order(self, tmp_path, monkeypatch):
        manager, bot_mem_path = self._make_manager(tmp_path, monkeypatch)

        calls = []

        async def fake_chat_completion(messages, response_format=None):
            # 每次回應都疊加目前收到的既有長期記憶，藉此驗證後一筆
            # 是不是真的讀到前一筆「已經寫回」的最新值，而不是同時讀到舊值。
            existing = manager.bot_memory
            calls.append(existing)
            return f"{existing}+new"

        monkeypatch.setattr("memory.chat_completion", fake_chat_completion)

        # 直接透過 queue 排入三筆溢出，比湊 append_turn 的觸發條件更直接、更聚焦在序列化本身
        async def run_three_overflows():
            manager.start_worker()
            for i in range(3):
                await manager.summarize_into_long_term([{"role": "assistant", "content": f"msg{i}"}])
            await manager.stop_worker()

        asyncio.run(run_three_overflows())

        # 三次呼叫應該依序發生，且每次看到的都是前一次已經寫回的結果
        assert calls == ["", "+new", "+new+new"]
        assert manager.bot_memory == "+new+new+new"
        assert bot_mem_path.read_text(encoding="utf-8") == "+new+new+new"

    def test_stop_worker_waits_for_queue_to_drain(self, tmp_path, monkeypatch):
        manager, _ = self._make_manager(tmp_path, monkeypatch)

        processed = []

        async def slow_chat_completion(messages, response_format=None):
            await asyncio.sleep(0.05)
            processed.append(messages)
            return "done"

        monkeypatch.setattr("memory.chat_completion", slow_chat_completion)

        async def run():
            manager.start_worker()
            await manager.summarize_into_long_term([{"role": "assistant", "content": "a"}])
            await manager.summarize_into_long_term([{"role": "assistant", "content": "b"}])
            # stop_worker 應該等兩筆都處理完才返回，不會提早砍斷還沒處理的項目
            await manager.stop_worker()

        asyncio.run(run())

        assert len(processed) == 2

    def test_append_turn_does_not_block_on_overflow(self, tmp_path, monkeypatch):
        """append_turn 只需要把溢出排入 queue（sync），不應該自己去 await LLM。"""
        manager, _ = self._make_manager(tmp_path, monkeypatch)

        called = False

        async def fake_chat_completion(messages, response_format=None):
            nonlocal called
            called = True
            return "updated"

        monkeypatch.setattr("memory.chat_completion", fake_chat_completion)

        session = UserSession()
        for i in range(config.MAX_HISTORY_MESSAGES + 2):
            manager.append_turn(session, f"user{i}", f"reply{i}")

        # append_turn 本身是 sync 呼叫，還沒有事件迴圈機會執行 worker，
        # 所以此刻 LLM 一定還沒被呼叫到——證明沒有卡在這裡等待。
        assert called is False
        assert len(session.history) <= config.MAX_HISTORY_MESSAGES


class TestMemoryManagerSizeLimit:
    """#005：長期記憶超過字數上限時要再壓縮一次，避免無限增長。"""

    def _make_manager(self, tmp_path, monkeypatch):
        bot_mem_path = tmp_path / "bot_mem.md"
        bot_mem_path.write_text("", encoding="utf-8")
        monkeypatch.setattr(config, "BOT_MEMORY_PATH", str(bot_mem_path))
        return MemoryManager(), bot_mem_path

    def test_under_limit_does_not_trigger_compression(self, tmp_path, monkeypatch):
        manager, bot_mem_path = self._make_manager(tmp_path, monkeypatch)
        monkeypatch.setattr(config, "BOT_MEMORY_MAX_CHARS", 100)

        call_count = 0

        async def fake_chat_completion(messages, response_format=None):
            nonlocal call_count
            call_count += 1
            return "短短的記憶"  # 遠小於 100 字上限

        monkeypatch.setattr("memory.chat_completion", fake_chat_completion)

        async def run():
            manager.start_worker()
            await manager.summarize_into_long_term([{"role": "assistant", "content": "hi"}])
            await manager.stop_worker()

        asyncio.run(run())

        # 只有第一次的 summarize 呼叫，沒有額外觸發壓縮呼叫
        assert call_count == 1
        assert manager.bot_memory == "短短的記憶"
        assert bot_mem_path.read_text(encoding="utf-8") == "短短的記憶"

    def test_over_limit_triggers_compression_and_writes_compressed_result(self, tmp_path, monkeypatch):
        manager, bot_mem_path = self._make_manager(tmp_path, monkeypatch)
        monkeypatch.setattr(config, "BOT_MEMORY_MAX_CHARS", 10)

        responses = ["這是一段超過十個字元長度的長期記憶內容", "濃縮後的短版本"]
        call_count = 0

        async def fake_chat_completion(messages, response_format=None):
            nonlocal call_count
            result = responses[call_count]
            call_count += 1
            return result

        monkeypatch.setattr("memory.chat_completion", fake_chat_completion)

        async def run():
            manager.start_worker()
            await manager.summarize_into_long_term([{"role": "assistant", "content": "hi"}])
            await manager.stop_worker()

        asyncio.run(run())

        # 第一次 summarize + 第二次壓縮，共兩次呼叫
        assert call_count == 2
        assert manager.bot_memory == "濃縮後的短版本"
        assert bot_mem_path.read_text(encoding="utf-8") == "濃縮後的短版本"

    def test_compression_failure_keeps_uncompressed_version(self, tmp_path, monkeypatch):
        manager, bot_mem_path = self._make_manager(tmp_path, monkeypatch)
        monkeypatch.setattr(config, "BOT_MEMORY_MAX_CHARS", 10)

        long_text = "這是一段超過十個字元長度的長期記憶內容"
        call_count = 0

        async def fake_chat_completion(messages, response_format=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return long_text
            raise RuntimeError("壓縮呼叫失敗")

        monkeypatch.setattr("memory.chat_completion", fake_chat_completion)

        async def run():
            manager.start_worker()
            await manager.summarize_into_long_term([{"role": "assistant", "content": "hi"}])
            await manager.stop_worker()

        asyncio.run(run())

        # 壓縮失敗時寧可維持超標的原版本，也不能整份記憶遺失
        assert manager.bot_memory == long_text
        assert bot_mem_path.read_text(encoding="utf-8") == long_text
