"""
模型輸出解析：把原始文字切成三段——要送給使用者的回覆、更新後的現況小抄、
（可選）圖片生成 prompt。

先維持現有 delimiter 解析方式（第一階段），改成 structured output/Pydantic
是之後真的碰到「模型常常不遵守格式」的痛點時再評估，現階段沒有必要。
"""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass
class ParsedResponse:
    reply: str
    state: str | None
    image_prompt: str | None


def split_model_output(raw_text: str) -> ParsedResponse:
    reply_part = raw_text
    state_part = None
    image_prompt = None

    if config.STATE_DELIMITER in raw_text:
        reply_part, rest = raw_text.split(config.STATE_DELIMITER, 1)
        if config.IMAGE_DELIMITER in rest:
            state_part, image_prompt = rest.split(config.IMAGE_DELIMITER, 1)
            image_prompt = image_prompt.strip()
        else:
            state_part = rest
        state_part = state_part.strip()

    return ParsedResponse(reply_part.strip(), state_part, image_prompt)
