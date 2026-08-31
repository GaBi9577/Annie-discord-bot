"""
記憶模組：短期記憶（sliding window，存在 UserSession.history）+
長期記憶（bot_mem.md 摘要）+ 現況小抄。

現階段仍是「單一 md 檔案整份塞進 prompt」，先不拆 short_term/long_term/storage
多層架構、也不導入 SQLite/Vector DB——那些等真的碰到 bot_mem.md 太大、或需要
使用者模型/情緒快照時再評估（YAGNI）。
"""

from __future__ import annotations

import asyncio
import logging

import config
from llm_client import chat_completion
from schedule import now_taipei

logger = logging.getLogger(__name__)

MEMORY_QUEUE_WARN_SIZE = 3
# queue 積壓超過這個筆數就額外警告——正常情況下 worker 會很快消化掉，
# 累積到這個量通常代表 summarize 持續失敗或變慢（P2-1 可觀測性）。


def load_bot_memory() -> str:
    """讀取長期記憶檔案，不存在則建立空白檔案。"""
    try:
        with open(config.BOT_MEMORY_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        with open(config.BOT_MEMORY_PATH, "w", encoding="utf-8") as f:
            f.write("")
        return ""


def load_current_state() -> str:
    """讀取現況小抄檔案，不存在則建立預設狀態。"""
    try:
        with open(config.CURRENT_STATE_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        default_state = "剛開始互動，沒有特別的持續性狀態。"
        with open(config.CURRENT_STATE_PATH, "w", encoding="utf-8") as f:
            f.write(default_state)
        return default_state


def save_current_state(state: str) -> None:
    with open(config.CURRENT_STATE_PATH, "w", encoding="utf-8") as f:
        f.write(state)


def strip_images_for_history(content):
    """存進短期歷史前，把圖片部分換成文字佔位符——圖片只在當輪分析，不重複佔用後續 token。"""
    if isinstance(content, str):
        return content

    text_parts = [item["text"] for item in content if item["type"] == "text"]
    image_count = sum(1 for item in content if item["type"] == "image_url")
    text = " ".join(text_parts)
    if image_count:
        placeholder = f"[傳送了{image_count}張圖片]"
        text = f"{text}\n{placeholder}" if text else placeholder
    return text


def stringify_content(content) -> str:
    """把 content（可能是純文字或多模態 array）轉成摘要用的純文字描述。"""
    if isinstance(content, str):
        return content

    parts = []
    for item in content:
        if item["type"] == "text":
            parts.append(item["text"])
        elif item["type"] == "image_url":
            parts.append("[附上一張圖片]")
    return " ".join(parts)


class MemoryManager:
    """管理長期記憶讀寫，並提供把對話寫入短期歷史／溢出摘要的方法。

    溢出的對話透過 asyncio.Queue 交給單一背景 worker 依序處理，避免短時間內
    多個 summarize 同時讀到同一份舊 bot_memory、最後互相覆蓋彼此的更新結果。
    """

    def __init__(self) -> None:
        self.bot_memory = load_bot_memory()
        self._queue: asyncio.Queue[list[dict]] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self.last_memory_update_success: object = None  # 最近一次成功更新的時間(datetime | None)
        self.last_memory_update_failure: object = None  # 最近一次失敗的時間(datetime | None)

    def start_worker(self) -> None:
        """啟動背景 worker。重複呼叫是安全的：worker 還在跑就不會重建。"""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop_worker(self) -> None:
        """等 queue 清空後停止 worker，確保關閉前所有已排入的更新都處理完，不留下無限制的背景 task。"""
        if self._worker_task is None:
            return
        await self._queue.join()
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    @property
    def queue_size(self) -> int:
        """目前待處理的溢出批次數，供日後需要時查詢佇列健康狀態用。"""
        return self._queue.qsize()

    async def _worker_loop(self) -> None:
        """依序消化 queue 中的溢出對話，一次只處理一筆，天生序列化。"""
        while True:
            overflow_messages = await self._queue.get()

            queue_size = self._queue.qsize()
            if queue_size >= MEMORY_QUEUE_WARN_SIZE:
                logger.warning("長期記憶佇列積壓中，待處理筆數：%d", queue_size)

            try:
                await self._summarize_into_long_term(overflow_messages)
            except Exception:
                self.last_memory_update_failure = now_taipei()
                logger.exception("長期記憶背景更新發生未預期錯誤")
            finally:
                self._queue.task_done()

    def append_turn(self, session, user_content, reply_content) -> None:
        """把這一輪對話存入短期歷史，超過上限的部分交給長期記憶 worker 依序整理。

        user_content 為 None 時代表這輪是主動發訊（沒有對應的使用者訊息），
        只存 assistant 這一半，不要塞一句假的 user 發言進去污染歷史紀錄。
        """
        if user_content is not None:
            session.history.append({"role": "user", "content": strip_images_for_history(user_content)})
        session.history.append({"role": "assistant", "content": reply_content})

        if len(session.history) > config.MAX_HISTORY_MESSAGES:
            overflow_count = len(session.history) - config.MAX_HISTORY_MESSAGES
            overflow = session.history[:overflow_count]
            del session.history[:overflow_count]
            self._queue.put_nowait(overflow)

    async def summarize_into_long_term(self, overflow_messages: list[dict]) -> None:
        """供外部（例如 shutdown 時的 flush）直接排入 queue，走跟一般 overflow 相同的序列化路徑。"""
        self._queue.put_nowait(overflow_messages)

    async def _summarize_into_long_term(self, overflow_messages: list[dict]) -> None:
        """呼叫 LLM 把即將被短期記憶砍掉的內容整理進長期記憶檔案。

        每次都讀取 self.bot_memory「當下最新」的值，搭配 worker 序列化執行，
        確保後一筆更新一定是疊加在前一筆已經寫回的結果之上。
        """
        if not overflow_messages:
            return

        overflow_text = "\n".join(
            f"{'使用者' if m['role'] == 'user' else '亞妮'}：{stringify_content(m['content'])}"
            for m in overflow_messages
        )
        summarize_prompt = (
            "你是記憶整理工具，不是在扮演角色。"
            "請把「新對話片段」中值得長期記住的資訊（使用者的習慣、喜好、重要事件、"
            "關係狀態變化等），整合進「既有長期記憶」，輸出更新後的完整長期記憶。"
            "用精簡條列呈現，不要重複贅述、不要流水帳、不要加入不重要的閒聊內容。"
            "只需要輸出更新後的長期記憶內容本身，不要有其他說明文字。\n\n"
            f"【既有長期記憶】\n{self.bot_memory if self.bot_memory else '（目前無）'}\n\n"
            f"【新對話片段】\n{overflow_text}"
        )

        try:
            updated_memory = await chat_completion([{"role": "user", "content": summarize_prompt}])
            updated_memory = updated_memory.strip()
        except Exception:
            self.last_memory_update_failure = now_taipei()
            logger.exception("長期記憶整理失敗")
            return

        updated_memory = await self._compress_if_too_long(updated_memory)

        self.bot_memory = updated_memory
        with open(config.BOT_MEMORY_PATH, "w", encoding="utf-8") as f:
            f.write(updated_memory)
        self.last_memory_update_success = now_taipei()
        logger.info("長期記憶已更新")

    async def _compress_if_too_long(self, memory_text: str) -> str:
        """長期記憶超過字數上限時，再請 LLM 濃縮一次；避免 prompt token 成本無限增加。

        壓縮失敗（例如 LLM 呼叫出錯）就照原樣回傳未壓縮的版本——寧可長度超標，
        也不要因為壓縮這步失敗而遺失整份長期記憶。
        """
        if len(memory_text) <= config.BOT_MEMORY_MAX_CHARS:
            return memory_text

        compress_prompt = (
            "以下是一份長期記憶摘要，目前太長了，請在不遺失重要資訊的前提下"
            "進一步濃縮，去除重複、次要或已經不重要的細節，用更精簡的條列呈現。"
            "只需要輸出濃縮後的長期記憶內容本身，不要有其他說明文字。\n\n"
            f"【目前的長期記憶】\n{memory_text}"
        )
        try:
            compressed = await chat_completion([{"role": "user", "content": compress_prompt}])
            compressed = compressed.strip()
        except Exception:
            logger.exception("長期記憶壓縮失敗，暫時維持未壓縮版本")
            return memory_text

        if not compressed:
            logger.warning("長期記憶壓縮結果為空，維持未壓縮版本")
            return memory_text

        logger.info("長期記憶已壓縮：%d 字 → %d 字", len(memory_text), len(compressed))
        return compressed
