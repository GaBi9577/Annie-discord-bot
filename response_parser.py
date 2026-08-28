"""
模型輸出解析：定義結構化輸出的 JSON Schema，並把 API 回傳的 JSON 字串轉成
ParsedResponse。

原本用 ===STATE===／===IMAGE=== 文字 delimiter 手動切字串，缺點是只要模型
輸出格式跑掉（漏加分隔線、換行位置不對、大小寫或空白差異），整段解析就失敗，
current_state 因此吃不到更新。改用 OpenAI-compatible 的 response_format
json_schema（NanoGPT 的 chat/completions endpoint 有支援 constrained decoding），
由 API 端強制模型輸出符合結構的 JSON，不會再有格式跑掉的問題。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedResponse:
    reply: str
    state: str | None
    image_prompt: str | None


RESPONSE_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "annie_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reply": {
                    "type": "string",
                    "description": "要傳送給使用者看的回覆內容，符合角色人設與語氣。",
                },
                "state": {
                    "type": "string",
                    "description": (
                        "旁白角度描述這輪回覆後「會持續到下一輪對話仍然成立」的狀態"
                        "（例如正在做的事、身體狀態如頭髮還沒吹乾、情緒是否還沒平復及原因）。"
                        "只描述會延續下去的狀態，不要重複列出這輪講過的對話內容。"
                        "如果沒有需要更新的狀態，就照抄前一版的現況小抄內容，這個欄位不能空白。"
                    ),
                },
                "image_prompt": {
                    "type": ["string", "null"],
                    "description": (
                        "如果這一輪判斷想主動分享一張照片給對方看（不需要每輪都分享），"
                        "這裡填給圖片生成模型用的英文 prompt，不是講給使用者聽的話。"
                        "角色身分與畫風不用描述，程式會自動加上，只需要依場景自由決定並描述"
                        "Clothing、Action、Expression、Environment Background、"
                        "Lighting Atmosphere、Composition and Camera 六個面向，"
                        "合併成一段逗號分隔的描述，不要加類別標籤。不想分享照片就填 null。"
                    ),
                },
            },
            "required": ["reply", "state", "image_prompt"],
            "additionalProperties": False,
        },
    },
}


def parse_response(raw_json_text: str) -> ParsedResponse:
    """把 API 回傳的 JSON 字串轉成 ParsedResponse。

    理論上 response_format=json_schema 已經強制格式正確，這裡的 try/except
    只是防止極端情況（例如某個 provider 沒有真的套用 schema）讓整輪對話直接
    炸掉——退化成把整段原始文字當作 reply，至少使用者還看得到回覆。
    """
    try:
        data = json.loads(raw_json_text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("模型輸出不是合法 JSON，整段當作 reply 處理：%r", raw_json_text)
        return ParsedResponse(reply=(raw_json_text or "").strip(), state=None, image_prompt=None)

    reply = (data.get("reply") or "").strip()

    state = data.get("state")
    state = state.strip() if isinstance(state, str) and state.strip() else None

    image_prompt = data.get("image_prompt")
    image_prompt = image_prompt.strip() if isinstance(image_prompt, str) and image_prompt.strip() else None

    return ParsedResponse(reply=reply, state=state, image_prompt=image_prompt)
