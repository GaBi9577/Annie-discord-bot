"""
主動發訊模組：bot 主控性核心，不需要使用者輸入也能主動開口。

兩階段判斷：
1. should_attempt_proactive() — 便宜的機率過濾器，不呼叫 LLM，決定這次要不要
   花錢問她真正的意願。
2. attempt_proactive_message() — 真的呼叫 LLM，讓她根據狀態/長期記憶/時段
   判斷這個當下想不想主動開口、想說什麼。

proactive_loop() 是背景迴圈，定期喚醒檢查每個使用者是否符合條件。
"""

from __future__ import annotations

import asyncio
import io
import logging
import random
from datetime import datetime

import discord

import config
from schedule import get_schedule_status, now_taipei
from prompt_builder import build_proactive_check_messages
from llm_client import chat_completion
from response_parser import parse_response, RESPONSE_JSON_SCHEMA
from image_gen import generate_image

logger = logging.getLogger(__name__)


def should_attempt_proactive(session, now: datetime) -> bool:
    """第一關：便宜的過濾器，決定這次要不要花錢問她真正的意願。
    距離上次互動太近就不考慮；過了安全時間後，閒置越久機率越高，但有上限。
    """
    if session.last_interaction_time is None:
        return False  # 還沒有任何互動紀錄，不主動

    elapsed_minutes = (now - session.last_interaction_time).total_seconds() / 60
    if elapsed_minutes < config.PROACTIVE_MIN_QUIET_MINUTES:
        return False

    elapsed_hours = elapsed_minutes / 60
    probability = min(
        config.PROACTIVE_GATE_MAX_PROBABILITY,
        config.PROACTIVE_GATE_BASE_PROBABILITY + elapsed_hours * config.PROACTIVE_GATE_GROWTH_PER_HOUR,
    )
    return random.random() < probability


async def attempt_proactive_message(user_id, session, memory_manager, state_holder) -> None:
    """第二關：真的呼叫 LLM，讓她根據狀態/長期記憶/時段判斷這次要不要主動傳訊息、傳什麼。"""
    channel = session.last_channel
    if channel is None:
        return  # 還沒有已知的頻道可以送，跳過

    if session.pending_task is not None:
        return  # 使用者訊息正在處理中（debounce 或等待忙碌時段結束），這輪不搶著發

    messages = build_proactive_check_messages(
        history=session.history,
        bot_memory=memory_manager.bot_memory,
        current_state=state_holder.value,
    )

    try:
        raw_text = await chat_completion(messages, response_format=RESPONSE_JSON_SCHEMA)
    except Exception:
        logger.exception("主動發訊檢查呼叫 LLM 失敗")
        return

    parsed = parse_response(raw_text)

    if parsed.reply == config.NO_PROACTIVE_TOKEN or parsed.reply.startswith(config.NO_PROACTIVE_TOKEN):
        return  # 她這次判斷不想主動開口，不用做任何事

    if not parsed.reply:
        return

    image_bytes = None
    if parsed.image_prompt:
        image_bytes = await generate_image(parsed.image_prompt)

    logger.info("主動發訊：%s", parsed.reply)
    if image_bytes:
        discord_file = discord.File(io.BytesIO(image_bytes), filename="annie.png")
        await channel.send(content=parsed.reply, file=discord_file)
    else:
        await channel.send(parsed.reply)

    if parsed.state:
        state_holder.update(parsed.state)

    memory_manager.append_turn(session, None, parsed.reply)
    session.last_interaction_time = now_taipei()


async def proactive_loop(client, sessions, memory_manager, state_holder) -> None:
    """背景迴圈：每隔一段隨機時間醒來檢查一次，決定要不要主動傳訊息。"""
    await client.wait_until_ready()
    while not client.is_closed():
        wait_minutes = random.uniform(
            config.PROACTIVE_CHECK_INTERVAL_MIN_MINUTES,
            config.PROACTIVE_CHECK_INTERVAL_MAX_MINUTES,
        )
        await asyncio.sleep(wait_minutes * 60)

        now = now_taipei()
        status = get_schedule_status(now)
        if status.blocking:
            continue  # 睡覺／健身時段不主動；其他時段（含上課中）都可能觸發

        for user_id, session in list(sessions.all_sessions()):
            if should_attempt_proactive(session, now):
                await attempt_proactive_message(user_id, session, memory_manager, state_holder)
