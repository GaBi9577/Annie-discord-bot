"""
current_state（現況小抄）的存取封裝。

正常回覆流程與主動發訊流程都會讀取、更新這個值，如果各自用 global 處理，
模組間會需要互相 import 對方的 global 變數，容易出錯又難追蹤。
用一個小物件封裝讀寫，兩邊都拿同一個實例操作即可。
"""

from __future__ import annotations

from memory import load_current_state, save_current_state


class CurrentStateHolder:
    def __init__(self) -> None:
        self.value: str = load_current_state()

    def update(self, new_value: str) -> None:
        self.value = new_value
        save_current_state(new_value)
