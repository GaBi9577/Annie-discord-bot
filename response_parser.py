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
    schedule_override_type: str | None = None  # "recurring" | "today" | None
    schedule_override_text: str | None = None


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
                "schedule_override_type": {
                    "type": ["string", "null"],
                    "enum": ["recurring", "today", None],
                    "description": (
                        "只有在翔明確提到要調整作息安排時才填寫，其他情況一律填 null。"
                        "\"recurring\" 表示這是持續性的常態調整（例如以後週三都晚點睡）；"
                        "\"today\" 表示只影響今天的臨時變化（例如今晚熬夜到兩點）。"
                    ),
                },
                "schedule_override_text": {
                    "type": ["string", "null"],
                    "description": (
                        "搭配 schedule_override_type 使用，用一句話描述調整內容"
                        "（例如「熬夜到凌晨兩點才睡」），純文字即可，不用結構化格式。"
                        "schedule_override_type 為 null 時，這裡也填 null。"
                    ),
                },
            },
            "required": [
                "reply", "state", "image_prompt",
                "schedule_override_type", "schedule_override_text",
            ],
            "additionalProperties": False,
        },
    },
}


FALLBACK_REPLY = "（訊號有點不穩，等一下再試一次）"
# 結構化輸出解析失敗時的安全回覆文字，統一用跟 LLM 呼叫失敗一致的措辭，
# 使用者不會感覺到「這其實是另一種內部錯誤」。


def parse_response(raw_json_text: str) -> ParsedResponse:
    """把 API 回傳的 JSON 字串轉成 ParsedResponse。

    理論上 response_format=json_schema 已經強制格式正確，這裡的 try/except
    只是防止極端情況（例如某個 provider 沒有真的套用 schema）讓整輪對話直接
    炸掉。但解析失敗時的原始文字可能是不完整的 JSON 片段或協定層雜訊
    （例如截斷到一半的字串），不適合直接當成 reply 送給使用者看，所以改成
    回傳固定的安全 fallback 訊息，並把原始內容完整記錄到 log 供除錯。
    """
    try:
        data = json.loads(raw_json_text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("模型輸出不是合法 JSON，改用安全 fallback 回覆：%r", raw_json_text)
        return ParsedResponse(reply=FALLBACK_REPLY, state=None, image_prompt=None)

    if not isinstance(data, dict):
        logger.warning("模型輸出不是 JSON object，改用安全 fallback 回覆：%r", raw_json_text)
        return ParsedResponse(reply=FALLBACK_REPLY, state=None, image_prompt=None)

    reply = (data.get("reply") or "").strip()
    if not reply:
        logger.warning("模型輸出缺少 reply 內容，改用安全 fallback 回覆：%r", raw_json_text)
        reply = FALLBACK_REPLY

    state = data.get("state")
    state = state.strip() if isinstance(state, str) and state.strip() else None

    image_prompt = data.get("image_prompt")
    image_prompt = image_prompt.strip() if isinstance(image_prompt, str) and image_prompt.strip() else None

    override_type = data.get("schedule_override_type")
    override_type = override_type if override_type in ("recurring", "today") else None

    override_text = data.get("schedule_override_text")
    override_text = override_text.strip() if isinstance(override_text, str) and override_text.strip() else None

    if override_type and not override_text:
        logger.warning("模型標記了 schedule_override_type 但沒給內容，忽略這次覆寫")
        override_type = None

    return ParsedResponse(
        reply=reply,
        state=state,
        image_prompt=image_prompt,
        schedule_override_type=override_type,
        schedule_override_text=override_text if override_type else None,
    )
