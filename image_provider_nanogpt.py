"""
NanoGPT 圖片生成 provider：呼叫 NanoGPT 的圖片生成 API，回傳圖片 bytes。
"""

from __future__ import annotations

import asyncio
import base64
import logging

import requests

import config
from prompt_builder import load_pic_prompt_prefix

logger = logging.getLogger(__name__)

IMAGE_REQUEST_TIMEOUT_SECONDS = 60

_PIC_PROMPT_PREFIX = load_pic_prompt_prefix()


def _request(full_prompt: str) -> dict:
    resp = requests.post(
        config.NANOGPT_IMAGE_URL,
        headers={
            "Authorization": f"Bearer {config.NANOGPT_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "prompt": full_prompt,
            "model": config.IMAGE_MODEL,
            "resolution": config.IMAGE_RESOLUTION,
            "n": 1,
        },
        timeout=IMAGE_REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


async def generate_image(scene_prompt: str) -> bytes | None:
    """呼叫圖片生成 API。失敗回傳 None——圖片是加分項，不該讓整輪回覆失敗。"""
    full_prompt = f"{_PIC_PROMPT_PREFIX}, {scene_prompt}"
    try:
        data = await asyncio.to_thread(_request, full_prompt)
        b64_json = data["data"][0]["b64_json"]
        return base64.b64decode(b64_json)
    except Exception:
        logger.exception("圖片生成失敗")
        return None
