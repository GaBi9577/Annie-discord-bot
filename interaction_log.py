"""
互動 log：把每一輪回覆（一般回覆／主動發訊）用固定格式 append 進一份
人類可讀的 markdown 檔案，供 /狀態 類指令與人工查閱使用。

選用 .md 而非 .jsonl：這份 log 主要是「翔想看」，可讀性優先於程式易讀性。
用固定分隔符號（"---"）＋固定欄位順序（標籤: 內容）維持結構穩定，
所以即使選了 .md，程式仍能穩定用簡單規則解析出最近 N 筆（見 read_recent_entries）。

只負責寫入與讀取，不管呼叫時機——呼叫時機在 main.py / proactive.py 裡決定。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import config

_SEPARATOR = "---"


@dataclass
class LogEntry:
    timestamp: str
    kind: str  # "一般回覆" | "主動發訊"
    user_message: str | None
    reply: str
    image_prompt: str | None


def _format_entry(entry: LogEntry) -> str:
    lines = [
        _SEPARATOR,
        f"時間: {entry.timestamp}",
        f"類型: {entry.kind}",
    ]
    if entry.user_message:
        lines.append(f"使用者訊息: {entry.user_message}")
    lines.append(f"回覆: {entry.reply}")
    if entry.image_prompt:
        lines.append(f"圖片 Prompt: {entry.image_prompt}")
    return "\n".join(lines) + "\n"


def append_entry(
    kind: str,
    reply: str,
    user_message: str | None = None,
    image_prompt: str | None = None,
    now: datetime | None = None,
) -> None:
    """寫入一筆互動紀錄。now 未指定則用目前時間（呼叫端通常已經有 now_taipei()）。"""
    from schedule import now_taipei

    now = now or now_taipei()
    entry = LogEntry(
        timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
        kind=kind,
        user_message=user_message,
        reply=reply,
        image_prompt=image_prompt,
    )
    with open(config.INTERACTION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(_format_entry(entry))


def read_recent_entries(limit: int = 10) -> list[LogEntry]:
    """讀取最近 N 筆紀錄，供斜線指令顯示用。檔案不存在時回傳空列表。"""
    try:
        with open(config.INTERACTION_LOG_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return []

    blocks = [b.strip() for b in content.split(_SEPARATOR) if b.strip()]
    entries: list[LogEntry] = []
    for block in blocks[-limit:]:
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ": " in line:
                key, _, value = line.partition(": ")
                fields[key] = value
        entries.append(LogEntry(
            timestamp=fields.get("時間", ""),
            kind=fields.get("類型", ""),
            user_message=fields.get("使用者訊息"),
            reply=fields.get("回覆", ""),
            image_prompt=fields.get("圖片 Prompt"),
        ))
    return entries
