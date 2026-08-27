"""
純邏輯的單元測試：不碰 Discord、不打真的 LLM API。
涵蓋 schedule 邊界、模型輸出解析、時間差格式化。
"""

import random
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from response_parser import split_model_output
from schedule import format_elapsed, get_schedule_status
from session import UserSession
from proactive import should_attempt_proactive

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


class TestSplitModelOutput:
    def test_no_delimiter_returns_raw_text_as_reply(self):
        result = split_model_output("單純的回覆內容")
        assert result.reply == "單純的回覆內容"
        assert result.state is None
        assert result.image_prompt is None

    def test_reply_and_state(self):
        raw = "今天過得不錯\n===STATE===\n心情平靜，剛洗完澡"
        result = split_model_output(raw)
        assert result.reply == "今天過得不錯"
        assert result.state == "心情平靜，剛洗完澡"
        assert result.image_prompt is None

    def test_reply_state_and_image_prompt(self):
        raw = (
            "傳張照片給你看\n===STATE===\n剛拍完照，心情不錯"
            "\n===IMAGE===\ncasual outfit, smiling, indoor, soft lighting, close-up"
        )
        result = split_model_output(raw)
        assert result.reply == "傳張照片給你看"
        assert result.state == "剛拍完照，心情不錯"
        assert result.image_prompt == "casual outfit, smiling, indoor, soft lighting, close-up"

    def test_strips_whitespace(self):
        raw = "  回覆內容  \n===STATE===\n  狀態內容  "
        result = split_model_output(raw)
        assert result.reply == "回覆內容"
        assert result.state == "狀態內容"


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
