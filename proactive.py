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
from dataclasses import dataclass
from datetime import datetime, timedelta

import discord

import config
from schedule import get_schedule_status, now_taipei
from prompt_builder import build_proactive_check_messages
from llm_client import chat_completion
from response_parser import parse_response, RESPONSE_JSON_SCHEMA
from image_gen import generate_image
import interaction_log
import schedule_override

logger = logging.getLogger(__name__)


@dataclass
class ProactiveObservability:
    """目前主動發訊迴圈的可觀測狀態，供 /狀態 類指令讀取，不影響迴圈本身邏輯。"""
    backoff_multiplier: int = 1
    last_check_at: datetime | None = None
    next_check_at: datetime | None = None
    paused: bool = False
    # /annie_pause 設 True 時，proactive_loop 仍照常醒來計時，但跳過實際判斷與發訊，
    # 這樣 /annie_resume 後不需要額外邏輯重啟迴圈，只是單純的旗標檢查（KISS）。


observability = ProactiveObservability()


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


async def attempt_proactive_message(user_id, session, memory_manager, state_holder) -> bool:
    """第二關：真的呼叫 LLM，讓她根據狀態/長期記憶/時段判斷這次要不要主動傳訊息、傳什麼。

    回傳值只代表「呼叫 LLM 是否失敗」，給 proactive_loop 用來決定要不要
    套用 backoff；沒頻道、鎖被占用、判斷不想開口、發送失敗等情況都不算
    LLM 失敗，回傳 False（不影響 backoff 狀態）。
    """
    channel = session.last_channel
    if channel is None:
        return False  # 還沒有已知的頻道可以送，跳過

    if session.pending_task is not None:
        return False  # 使用者訊息正在處理中（debounce 或等待忙碌時段結束），這輪不搶著發

    if session.response_lock.locked():
        return False  # 對方的訊息正在跑關鍵段落，這輪主動發訊直接放棄，不排隊等待

    # 進入關鍵段落前才正式拿鎖：跟 wait_then_reply 共用同一把鎖，確保同一個
    # session 不會同時有兩邊在組 prompt / 呼叫 LLM / 送出（P0-1）。
    async with session.response_lock:
        # 拿到鎖之後，pending_task 可能在等待期間被建立（使用者剛好在這時發訊），
        # 因此進鎖後要再檢查一次，才能保證是拿到鎖當下才成立的狀態。
        if session.pending_task is not None:
            return False

        messages = build_proactive_check_messages(
            history=session.history,
            bot_memory=memory_manager.bot_memory,
            current_state=state_holder.value,
            last_interaction_time=session.last_interaction_time,
        )

        try:
            raw_text = await chat_completion(messages, response_format=RESPONSE_JSON_SCHEMA)
        except Exception:
            logger.exception("主動發訊檢查呼叫 LLM 失敗")
            return True

        parsed = parse_response(raw_text)

        if parsed.reply == config.NO_PROACTIVE_TOKEN or parsed.reply.startswith(config.NO_PROACTIVE_TOKEN):
            return False  # 她這次判斷不想主動開口，不用做任何事

        if not parsed.reply:
            return False

        image_bytes = None
        if parsed.image_prompt:
            image_bytes = await generate_image(parsed.image_prompt)

        logger.info("主動發訊：%s", parsed.reply)
        interaction_log.append_entry(
            kind="主動發訊",
            reply=parsed.reply,
            image_prompt=parsed.image_prompt,
        )
        try:
            if image_bytes:
                discord_file = discord.File(io.BytesIO(image_bytes), filename="annie.png")
                await channel.send(content=parsed.reply, file=discord_file)
            else:
                await channel.send(parsed.reply)
        except Exception:
            # 單次發送失敗（例如 Discord API 錯誤）不該讓整個 proactive_loop 停擺，
            # 記錄下來、跳過這次即可；下次背景迴圈醒來會再重新判斷一次。
            logger.exception("主動發訊發送失敗，本次跳過")
            return False

        if parsed.state:
            state_holder.update(parsed.state)

        if parsed.schedule_override_type == "recurring":
            schedule_override.write_recurring_override(parsed.schedule_override_text)
        elif parsed.schedule_override_type == "today":
            schedule_override.write_today_override(parsed.schedule_override_text)

        memory_manager.append_turn(session, None, parsed.reply)
        session.last_interaction_time = now_taipei()
        return False


async def proactive_loop(client, sessions, memory_manager, state_holder) -> None:
    """背景迴圈：每隔一段隨機時間醒來檢查一次，決定要不要主動傳訊息。

    連續呼叫 LLM 失敗時，等待間隔會依 PROACTIVE_BACKOFF_MULTIPLIER 逐次放大
    （封頂於 PROACTIVE_BACKOFF_MAX_MULTIPLIER），避免在 API 出問題或設定錯誤
    期間持續浪費呼叫額度；只要有一次成功執行完整輪次（不論最終有沒有真的送
    訊息），就把倍率重置回 1。
    """
    await client.wait_until_ready()
    backoff_multiplier = 1
    while not client.is_closed():
        wait_minutes = random.uniform(
            config.PROACTIVE_CHECK_INTERVAL_MIN_MINUTES,
            config.PROACTIVE_CHECK_INTERVAL_MAX_MINUTES,
        ) * backoff_multiplier
        observability.backoff_multiplier = backoff_multiplier
        observability.next_check_at = now_taipei() + timedelta(minutes=wait_minutes)
        await asyncio.sleep(wait_minutes * 60)

        now = now_taipei()
        observability.last_check_at = now

        if observability.paused:
            continue  # 已暫停：照常計時醒來，但不做任何判斷或發訊

        status = get_schedule_status(now)
        if status.blocking:
            continue  # 睡覺／健身時段不主動；其他時段（含上課中）都可能觸發

        any_failure = False
        for user_id, session in list(sessions.all_sessions()):
            if should_attempt_proactive(session, now):
                failed = await attempt_proactive_message(user_id, session, memory_manager, state_holder)
                any_failure = any_failure or failed

        if any_failure:
            backoff_multiplier = min(
                backoff_multiplier * config.PROACTIVE_BACKOFF_MULTIPLIER,
                config.PROACTIVE_BACKOFF_MAX_MULTIPLIER,
            )
            logger.warning("主動發訊呼叫 LLM 失敗，下次等待間隔倍率調整為 %sx", backoff_multiplier)
        else:
            backoff_multiplier = 1
