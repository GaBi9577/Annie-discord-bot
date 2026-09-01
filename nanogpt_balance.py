"""
NanoGPT 帳戶餘額查詢：獨立成一支模組而不是塞進 llm_client.py，因為這是
一次性查詢的 REST call（跟 image_provider_nanogpt.py 的 requests.post 模式
一致），跟 chat_completion 的職責不同（單一職責原則）。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import requests

import config

logger = logging.getLogger(__name__)

BALANCE_REQUEST_TIMEOUT_SECONDS = 15


@dataclass
class NanoGptBalance:
    usd_balance: str
    nano_balance: str


def _request() -> dict:
    resp = requests.post(
        config.NANOGPT_BALANCE_URL,
        headers={"x-api-key": config.NANOGPT_API_KEY},
        timeout=BALANCE_REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


async def get_balance() -> NanoGptBalance | None:
    """查詢目前 NanoGPT 帳戶餘額。失敗回傳 None，呼叫端（斜線指令）自行決定
    要顯示什麼樣的錯誤訊息——這裡不擅自決定使用者看到的文字。
    """
    try:
        data = await asyncio.to_thread(_request)
        return NanoGptBalance(
            usd_balance=str(data.get("usd_balance", "unknown")),
            nano_balance=str(data.get("nano_balance", "unknown")),
        )
    except Exception:
        logger.exception("查詢 NanoGPT 餘額失敗")
        return None
