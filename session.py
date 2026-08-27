"""
Session 狀態管理。

原本 bot.py 用四個平行的 global dict（pending / pending_mode / pending_buffer /
conversation_history）各自用 user_id 當 key，但這四個字典其實描述的是同一個
「使用者當下互動狀態」，拆散容易漏改其中一個、也難以一次看懂某個使用者現在的狀態。
這裡合併成單一 UserSession，SessionManager 負責 per-user 存取。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class UserSession:
    """單一使用者的所有暫存狀態（debounce/等待狀態 + 短期對話歷史 + 主動發訊判斷用資訊）。"""

    pending_task: asyncio.Task | None = None
    pending_mode: str | None = None  # "debounce" | "waiting"
    pending_buffer: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    last_interaction_time: datetime | None = None  # 使用者發訊或她主動發訊都算，主動發訊判斷用
    last_channel: Any = None  # discord 頻道物件，記錄最後互動頻道，主動發訊要送去哪裡

    def clear_pending(self) -> None:
        """清掉這輪的 debounce/waiting 狀態，history 不受影響。"""
        self.pending_task = None
        self.pending_mode = None
        self.pending_buffer = []


class SessionManager:
    """管理所有使用者的 UserSession。"""

    def __init__(self) -> None:
        self._sessions: dict[int, UserSession] = {}

    def get(self, user_id: int) -> UserSession:
        return self._sessions.setdefault(user_id, UserSession())

    def all_sessions(self):
        return self._sessions.items()
