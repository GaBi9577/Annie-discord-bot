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
    response_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # 正常回覆與主動發訊共用同一把鎖，確保同一個使用者的「組 prompt →
    # 呼叫 LLM → 送出」關鍵段落不會同時被兩邊搶著跑（P0-1）。

    def take_pending_buffer(self) -> list[dict]:
        """取出目前的 pending_buffer 並立刻換上一個新的空 list（snapshot/consume）。

        呼叫後，這輪 task 手上的回傳值是「當下那個時間點」的訊息快照，之後
        新訊息（包含 task 正在 await LLM 期間收到的）會寫進新的空 list，
        不會混進正在被這輪 task 處理的資料，也不會在 clear_pending() 時被誤刪。
        """
        buffer = self.pending_buffer
        self.pending_buffer = []
        return buffer

    def clear_pending(self) -> None:
        """清掉這輪的 debounce/waiting 狀態旗標，不動 pending_buffer（用 take_pending_buffer 處理）。"""
        self.pending_task = None
        self.pending_mode = None

    def requeue_pending_buffer(self, buffer: list[dict]) -> None:
        """LLM 最終失敗時，把已取出的訊息塞回 pending_buffer 最前面（P1-1）。

        用「取出時的內容 + 之後新進來的內容」重組，而不是直接覆蓋，
        因為 take_pending_buffer() 之後、這次呼叫之前，可能已經有新訊息
        寫進新的 pending_buffer；直接覆蓋會遺失那些新訊息。
        """
        self.pending_buffer = buffer + self.pending_buffer


class SessionManager:
    """管理所有使用者的 UserSession。"""

    def __init__(self) -> None:
        self._sessions: dict[int, UserSession] = {}

    def get(self, user_id: int) -> UserSession:
        return self._sessions.setdefault(user_id, UserSession())

    def all_sessions(self):
        return self._sessions.items()