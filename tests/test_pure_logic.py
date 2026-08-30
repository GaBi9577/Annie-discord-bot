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
