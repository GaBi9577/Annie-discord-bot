"""
斜線指令面板：/annie_status 系列指令。

獨立成一個模組而不是塞進 main.py，因為 main.py 已經是訊息收發的核心流程，
指令面板是額外的觀測介面，職責不同（單一職責原則）。

這是單一使用者的個人專案，不需要權限檢查、不需要多伺服器 guild 隔離，
指令直接對 client 底下唯一的 SessionManager／proactive 狀態做唯讀查詢即可。
"""

from __future__ import annotations

from typing import Literal

import discord
from discord import app_commands

import config
import interaction_log
import nanogpt_balance
import proactive
import schedule_override
from schedule import format_elapsed, get_schedule_status, now_taipei


def register_commands(tree: app_commands.CommandTree, sessions, memory_manager, state_holder) -> None:
    """把所有斜線指令掛到傳入的 CommandTree 上。sessions／memory_manager／state_holder
    都是 main.py 裡本來就存在的單例，這裡只是拿來做唯讀查詢（或少數幾個直接寫入
    schedule_override 檔案／proactive 旗標的操作），不擁有它們的生命週期。
    """

    @tree.command(name="annie_status", description="看 Annie 目前狀態、主動發訊狀況")
    async def annie_status(interaction: discord.Interaction):
        now = now_taipei()
        status = get_schedule_status(now)

        lines = ["**Bot 狀態**：運作中"]
        lines.append(f"**目前時段**：{status.label}")

        override_text = schedule_override.get_active_override_text()
        if override_text:
            lines.append(f"**生效中的作息調整**：\n{override_text}")

        lines.append("")
        lines.append("**主動發訊狀態**")
        obs = proactive.observability
        if obs.paused:
            lines.append("⏸ 已暫停（/annie_resume 恢復）")

        session = sessions.get(interaction.user.id)
        if session.last_interaction_time:
            elapsed = format_elapsed(session.last_interaction_time, now)
            lines.append(f"距離上次互動：約 {elapsed}")
        else:
            lines.append("距離上次互動：尚無紀錄")

        lines.append(f"目前 backoff 倍率：{obs.backoff_multiplier}x")
        if obs.next_check_at:
            lines.append(f"下次檢查時間：{obs.next_check_at.strftime('%H:%M:%S')}")
        if obs.last_check_at:
            lines.append(f"上次檢查時間：{obs.last_check_at.strftime('%H:%M:%S')}")

        await interaction.response.send_message("\n".join(lines))

    @tree.command(name="annie_recent", description="看最近幾輪的回覆與圖片 prompt")
    @app_commands.describe(count="要看最近幾筆，預設 5 筆，最多 20 筆")
    async def annie_recent(interaction: discord.Interaction, count: int = 5):
        count = max(1, min(count, 20))
        entries = interaction_log.read_recent_entries(limit=count)

        if not entries:
            await interaction.response.send_message("目前還沒有任何紀錄。")
            return

        lines = [f"**最近 {len(entries)} 筆紀錄**"]
        for e in entries:
            lines.append(f"\n`{e.timestamp}` [{e.kind}]")
            lines.append(f"回覆：{e.reply}")
            if e.image_prompt:
                lines.append(f"圖片 Prompt：{e.image_prompt}")

        text = "\n".join(lines)
        # Discord 單則訊息上限 2000 字，超過就截斷並提示，避免直接發送失敗
        if len(text) > 1900:
            text = text[:1900] + "\n…（內容過長，已截斷）"

        await interaction.response.send_message(text)

    @tree.command(name="annie_balance", description="看 NanoGPT 帳戶餘額")
    async def annie_balance(interaction: discord.Interaction):
        await interaction.response.defer()  # 查詢是網路 call，先讓 Discord 知道還在處理
        balance = await nanogpt_balance.get_balance()
        if balance is None:
            await interaction.followup.send("查詢餘額失敗，稍後再試一次。")
            return
        await interaction.followup.send(
            f"**NanoGPT 帳戶餘額**\nUSD：{balance.usd_balance}\nNano：{balance.nano_balance}"
        )

    @tree.command(name="annie_memory", description="看目前的長期記憶內容")
    async def annie_memory(interaction: discord.Interaction):
        content = memory_manager.bot_memory
        if not content:
            await interaction.response.send_message("目前長期記憶還是空的。", ephemeral=True)
            return

        text = f"**長期記憶**\n{content}"
        if len(text) > 1900:
            text = text[:1900] + "\n…（內容過長，已截斷，完整內容請查看 bot_mem.md）"
        await interaction.response.send_message(text, ephemeral=True)

    @tree.command(name="annie_state", description="看目前的現況小抄")
    async def annie_state(interaction: discord.Interaction):
        await interaction.response.send_message(
            f"**目前現況小抄**\n{state_holder.value}", ephemeral=True
        )

    @tree.command(name="annie_schedule_set", description="手動設定作息調整，不用透過對話讓她自己判斷")
    @app_commands.describe(
        scope="today：只影響今天，明天自動失效；recurring：持續套用，直到下次被覆寫",
        text="調整內容的描述，例如「熬夜到凌晨兩點」",
    )
    async def annie_schedule_set(
        interaction: discord.Interaction,
        scope: Literal["today", "recurring"],
        text: str,
    ):
        if scope == "today":
            schedule_override.write_today_override(text)
        else:
            schedule_override.write_recurring_override(text)
        await interaction.response.send_message(
            f"作息調整已寫入（{'今日突發' if scope == 'today' else '常態調整'}）：{text}",
            ephemeral=True,
        )

    @tree.command(name="annie_pause", description="暫停主動發訊，直到 /annie_resume")
    async def annie_pause(interaction: discord.Interaction):
        proactive.observability.paused = True
        await interaction.response.send_message("主動發訊已暫停。", ephemeral=True)

    @tree.command(name="annie_resume", description="恢復主動發訊")
    async def annie_resume(interaction: discord.Interaction):
        proactive.observability.paused = False
        await interaction.response.send_message("主動發訊已恢復。", ephemeral=True)
