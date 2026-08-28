"""
圖片生成 facade：依照 config.IMAGE_PROVIDER 選擇要呼叫 NanoGPT 還是 PixAI。

上層（main.py / proactive.py）只依賴這裡的 generate_image()，不需要知道
現在用的是哪個 provider——要切換 provider 只需要改 config.IMAGE_PROVIDER，
呼叫端完全不用改。
"""

from __future__ import annotations

import logging

import config
import image_provider_nanogpt
import image_provider_pixai

logger = logging.getLogger(__name__)

_PROVIDERS = {
    "nanogpt": image_provider_nanogpt.generate_image,
    "pixai": image_provider_pixai.generate_image,
}


async def generate_image(scene_prompt: str) -> bytes | None:
    provider_fn = _PROVIDERS.get(config.IMAGE_PROVIDER)
    if provider_fn is None:
        logger.error("未知的 IMAGE_PROVIDER：%s（可用值：%s）", config.IMAGE_PROVIDER, list(_PROVIDERS))
        return None
    return await provider_fn(scene_prompt)
