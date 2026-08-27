"""
生活作息與時間感知模組：判斷目前是否為硬性阻擋回覆的時段（睡覺／健身），
以及背景時段標籤。

時間一律使用 Asia/Taipei（zoneinfo），不再依賴執行主機的系統時區——
原本用 datetime.now() 在本地筆電（台北時區）沒問題，但如果之後搬到雲端主機
且系統時區非 UTC+8，作息判斷會整組跑掉。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import config

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
WEEKDAY_NAMES = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]


@dataclass
class ScheduleStatus:
    blocking: bool          # 是否為硬性阻擋回覆的時段
    reason: str | None      # "sleep" | "gym" | None
    label: str               # 目前狀態的簡短描述，用於注入 prompt 當背景資訊
    available_at: datetime | None  # 若 blocking，時段結束的時間點；否則為 None


def now_taipei() -> datetime:
    """取得目前的台北時間（timezone-aware）。"""
    return datetime.now(TAIPEI_TZ)


def get_schedule_status(now: datetime | None = None) -> ScheduleStatus:
    """判斷目前的作息狀態。傳入的 now 若是 naive datetime，視為已經是台北時間。"""
    now = now or now_taipei()
    if now.tzinfo is None:
        now = now.replace(tzinfo=TAIPEI_TZ)

    hour = now.hour + now.minute / 60
    weekday = now.weekday()  # 0=週一 ... 6=週日

    if config.SLEEP_START_HOUR <= hour < config.WAKE_HOUR:
        available_at = now.replace(hour=config.WAKE_HOUR, minute=0, second=0, microsecond=0)
        return ScheduleStatus(True, "sleep", "睡覺中", available_at)

    if config.GYM_START_HOUR <= hour < config.GYM_END_HOUR:
        available_at = now.replace(hour=config.GYM_END_HOUR, minute=0, second=0, microsecond=0)
        return ScheduleStatus(True, "gym", "健身中", available_at)

    if weekday < 5 and config.WAKE_HOUR <= hour < config.GYM_START_HOUR:
        label = "白天上課中"
    elif weekday >= 5:
        label = "假日，沒有固定行程"
    else:
        label = "晚上自由時間"

    return ScheduleStatus(False, None, label, None)


def format_elapsed(start: datetime, end: datetime) -> str:
    """把時間差轉成中文簡短描述，用於「剛結束忙碌時段」的 catch-up 提示。"""
    minutes = int((end - start).total_seconds() // 60)
    if minutes < 60:
        return f"{minutes}分鐘"
    hours, remaining = divmod(minutes, 60)
    return f"{hours}小時{remaining}分鐘" if remaining else f"{hours}小時"
