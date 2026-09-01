"""
作息覆寫：讓 bot 能把「跟預設作息不同的安排」自己寫進一個純文字檔案，
之後每輪組 prompt 時讀出來，讓 LLM 自己判斷要不要套用、怎麼套用。

不在這裡寫時間解析邏輯（例如判斷「晚點睡到幾點」要怎麼影響 get_schedule_status()
的硬性阻擋時間）——那樣要處理的自然語言情境太多、容易寫出一堆特例判斷，
違反 YAGNI/KISS。改成把覆寫內容整段塞進 prompt，交給 LLM 自己讀文字判斷
「現在算不算在忙碌時段」，回覆時自然反映即可。

檔案格式固定兩段：
    【常態調整】
    ...持續套用，直到被下一次覆寫...

    【今日突發】YYYY-MM-DD
    ...只在寫入當天有效，隔天視為過期，不會再注入...
"""

from __future__ import annotations

import logging

import config
from schedule import now_taipei

logger = logging.getLogger(__name__)

_RECURRING_HEADER = "【常態調整】"
_TODAY_HEADER = "【今日突發】"


def _read_raw() -> str:
    try:
        with open(config.SCHEDULE_OVERRIDE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _write_raw(content: str) -> None:
    with open(config.SCHEDULE_OVERRIDE_PATH, "w", encoding="utf-8") as f:
        f.write(content)


def _parse(raw: str) -> tuple[str, str, str | None]:
    """回傳 (recurring_text, today_text, today_date_str)，三者都可能是空字串/None。"""
    recurring = ""
    today_text = ""
    today_date = None

    if _RECURRING_HEADER in raw:
        after = raw.split(_RECURRING_HEADER, 1)[1]
        recurring = after.split(_TODAY_HEADER, 1)[0].strip()

    if _TODAY_HEADER in raw:
        after = raw.split(_TODAY_HEADER, 1)[1]
        first_line, _, rest = after.partition("\n")
        today_date = first_line.strip() or None
        today_text = rest.strip()

    return recurring, today_text, today_date


def get_active_override_text() -> str | None:
    """讀取目前有效的覆寫內容，組成可直接塞進 prompt 的一段文字。

    今日突發只在寫入當天有效；過了就當作不存在，不需要額外清檔案，
    下次被覆寫時自然會被新內容取代。
    """
    raw = _read_raw()
    if not raw.strip():
        return None

    recurring, today_text, today_date = _parse(raw)
    today_str = now_taipei().strftime("%Y-%m-%d")

    parts = []
    if recurring:
        parts.append(f"常態調整：{recurring}")
    if today_text and today_date == today_str:
        parts.append(f"今日突發：{today_text}")

    if not parts:
        return None
    return "\n".join(parts)


def write_recurring_override(text: str) -> None:
    """寫入／更新常態調整。保留現有的今日突發區塊。"""
    raw = _read_raw()
    _, today_text, today_date = _parse(raw)
    _write_override(recurring=text, today_text=today_text, today_date=today_date)


def write_today_override(text: str) -> None:
    """寫入今日突發，標記為今天的日期，明天自動視為過期。保留現有的常態調整。"""
    raw = _read_raw()
    recurring, _, _ = _parse(raw)
    _write_override(recurring=recurring, today_text=text, today_date=now_taipei().strftime("%Y-%m-%d"))


def _write_override(recurring: str, today_text: str, today_date: str | None) -> None:
    lines = [_RECURRING_HEADER, recurring.strip(), ""]
    if today_text:
        lines.append(f"{_TODAY_HEADER}{today_date}")
        lines.append(today_text.strip())
    _write_raw("\n".join(lines) + "\n")
    logger.info("作息覆寫已更新")
