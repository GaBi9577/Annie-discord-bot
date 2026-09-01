"""
進入點：Discord client、訊息收發、debounce/schedule 阻擋，串起
prompt_builder / llm_client / response_parser / image_gen / memory。

Discord 層只負責：收訊息、顯示 typing、取得附件、發送文字與圖片。
角色的 context / memory / schedule / decision / prompt / LLM 都在其他模組。
"""

from __future__ import annotations

import asyncio
import io
import logging

import discord

import config
from schedule import get_schedule_status, format_elapsed, now_taipei
from session import SessionManager
from prompt_builder import build_user_content, build_messages
from llm_client import chat_completion
from response_parser import parse_response, RESPONSE_JSON_SCHEMA
from image_gen import generate_image
from memory import MemoryManager
from state import CurrentStateHolder
from proactive import proactive_loop
import interaction_log
import schedule_override
import status_commands

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

sessions = SessionManager()
memory_manager = MemoryManager()
state_holder = CurrentStateHolder()

status_commands.register_commands(tree, sessions, memory_manager, state_holder)

_proactive_task: asyncio.Task | None = None
_commands_synced = False


def extract_image_urls(message: discord.Message) -> list[str]:
    """從 Discord 訊息附件中取出圖片 URL。"""
    return [
        att.url for att in message.attachments
        if att.content_type and att.content_type.startswith("image/")
    ]


@client.event
async def on_ready():
    logger.info("Bot 已上線：%s", client.user)

    global _proactive_task, _commands_synced
    # Discord reconnect 會再次觸發 on_ready；只在 loop 還沒啟動、或前一個已經
    # 結束（例如例外導致退出）時才建立新的，避免同時存在多個 proactive loop。
    if _proactive_task is None or _proactive_task.done():
        _proactive_task = client.loop.create_task(
            proactive_loop(client, sessions, memory_manager, state_holder)
        )

    if not _commands_synced:
        # 斜線指令 sync 有 rate limit，只在真正第一次啟動時做一次；
        # reconnect 不會重新 sync（指令定義沒變，不需要）。
        await tree.sync()
        _commands_synced = True
        logger.info("斜線指令已同步")

    memory_manager.start_worker()


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    user_id = message.author.id
    session = sessions.get(user_id)
    session.last_channel = message.channel
    session.pending_buffer.append({
        "text": message.content,
        "images": extract_image_urls(message),
    })

    status = get_schedule_status()

    # 已經在「等待忙碌時段結束」模式時，新訊息只需要進緩衝區，不用動計時器——
    # 因為等待時間是算到固定的時段結束點，不是每則訊息重新倒數
    if session.pending_mode == "waiting" and status.blocking:
        return

    if session.pending_task:
        session.pending_task.cancel()

    if status.blocking:
        session.pending_mode = "waiting"
        wait_seconds = max((status.available_at - now_taipei()).total_seconds(), 0)
        busy_started_at = now_taipei()
        task = asyncio.create_task(
            wait_then_reply(
                user_id, message, wait_seconds,
                busy_reason=status.reason, busy_started_at=busy_started_at,
            )
        )
    else:
        session.pending_mode = "debounce"
        task = asyncio.create_task(wait_then_reply(user_id, message, config.REPLY_DELAY))

    session.pending_task = task


async def wait_then_reply(
    user_id: int,
    message: discord.Message,
    wait_seconds: float,
    busy_reason: str | None = None,
    busy_started_at=None,
):
    await asyncio.sleep(wait_seconds)

    session = sessions.get(user_id)
    buffered = session.take_pending_buffer()
    if not buffered:
        session.clear_pending()
        return

    merged_text = "\n".join(item["text"] for item in buffered if item["text"])
    merged_images = [url for item in buffered for url in item["images"]]
    user_content = build_user_content(merged_text, merged_images)

    catch_up_note = None
    if busy_reason:
        elapsed = format_elapsed(busy_started_at, now_taipei())
        busy_label = "睡覺" if busy_reason == "sleep" else "健身"
        catch_up_note = (
            f"【剛結束忙碌時段】你剛結束{busy_label}，這段期間（約{elapsed}）沒看訊息，"
            "現在才剛看到累積的訊息，語氣自然反映剛回神／剛醒來／剛練完的狀態即可，"
            "不用逐句回應每一則，抓重點自然回應。"
        )

    messages = build_messages(
        history=session.history,
        bot_memory=memory_manager.bot_memory,
        current_state=state_holder.value,
        user_content=user_content,
        catch_up_note=catch_up_note,
        last_interaction_time=session.last_interaction_time,
    )

    # 關鍵段落（呼叫 LLM → 送出）要拿到 session 的鎖才能跑，避免跟同一個
    # 使用者的主動發訊同時執行、互相搶著送訊息（P0-1）。
    async with session.response_lock:
        async with message.channel.typing():
            try:
                raw_text = await chat_completion(messages, response_format=RESPONSE_JSON_SCHEMA)
            except Exception:
                logger.exception("LLM 呼叫最終失敗，訊息塞回 pending buffer 避免遺失")
                session.requeue_pending_buffer(buffered)
                await message.channel.send("（訊號有點不穩，等一下再試一次）")
                session.clear_pending()
                return

        parsed = parse_response(raw_text)

        image_bytes = None
        if parsed.image_prompt:
            image_bytes = await generate_image(parsed.image_prompt)

        logger.info("回傳內容：%s", parsed.reply)
        interaction_log.append_entry(
            kind="一般回覆",
            reply=parsed.reply,
            user_message=merged_text or None,
            image_prompt=parsed.image_prompt,
        )

        if image_bytes:
            discord_file = discord.File(io.BytesIO(image_bytes), filename="annie.png")
            await message.channel.send(content=parsed.reply, file=discord_file)
        else:
            await message.channel.send(parsed.reply)

        if parsed.state:
            state_holder.update(parsed.state)
        else:
            logger.warning("模型沒有依格式輸出現況小抄，本輪狀態維持不變")

        if parsed.schedule_override_type == "recurring":
            schedule_override.write_recurring_override(parsed.schedule_override_text)
        elif parsed.schedule_override_type == "today":
            schedule_override.write_today_override(parsed.schedule_override_text)

        memory_manager.append_turn(session, user_content, parsed.reply)
        session.last_interaction_time = now_taipei()
        session.clear_pending()


async def flush_remaining_history():
    """程式關閉前，把還沒被截斷整理過的短期記憶排入 queue，並等 worker 處理完再停止。

    這裡不能直接 await 每個 summarize，因為 worker 才是唯一實際執行整理的地方
    （避免跟一般 overflow 用不同路徑寫入 bot_memory、破壞序列化保證）；
    所以是「排入 queue → 啟動/沿用 worker → 等 queue 清空 → 停止 worker」。
    """
    for _, session in sessions.all_sessions():
        if session.history:
            await memory_manager.summarize_into_long_term(session.history)
            session.history.clear()

    memory_manager.start_worker()  # 保險：萬一 on_ready 從未觸發過
    await memory_manager.stop_worker()
    logger.info("關閉前已將剩餘短期記憶存入長期記憶")


async def shutdown():
    """關閉前的清理，跟 client 共用同一個 event loop（P0-2）。

    _proactive_task 要先取消並 await，確保它不會在 flush 進行到一半時
    又醒來嘗試送訊息、或在 loop 收尾階段留下未處理的例外。
    """
    if _proactive_task is not None:
        _proactive_task.cancel()
        try:
            await _proactive_task
        except asyncio.CancelledError:
            pass

    await flush_remaining_history()


async def main():
    """單一 async 入口：啟動、執行、關閉都在同一個 event loop 裡完成，
    不再有『client.run() 結束後另開一個 asyncio.run()』的第二層生命週期。
    """
    async with client:
        try:
            await client.start(config.TOKEN)
        finally:
            await shutdown()


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
