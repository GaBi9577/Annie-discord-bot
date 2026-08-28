"""
LLM Client：集中管理 NanoGPT (OpenAI-compatible) client 的初始化與呼叫。

上層（prompt_builder / memory / main）只依賴 chat_completion()，不需要知道
provider 細節，未來若要換 provider，只需要改這個檔案。

加上 timeout + 簡單的指數退避 retry，只對 transient error（timeout / 連線錯誤 /
429 / 5xx）重試；4xx 驗證或授權錯誤不盲目重試。
"""

from __future__ import annotations

import asyncio
import logging

import openai
from openai import OpenAI

import config

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

_client = OpenAI(
    api_key=config.NANOGPT_API_KEY,
    base_url=config.NANOGPT_BASE_URL,
)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return False


async def chat_completion(
    messages: list[dict], model: str | None = None, response_format: dict | None = None
) -> str:
    """呼叫 chat completion。transient error 會自動重試，最終失敗才把例外往上拋。

    response_format 可傳入 OpenAI-compatible 的 json_schema 設定，強制模型輸出
    符合指定結構的 JSON（NanoGPT 的 chat/completions endpoint 有支援 constrained
    decoding），不需要另外處理則留 None，行為跟以前一樣。
    """
    model = model or config.MODEL
    last_exc: Exception | None = None
    extra_kwargs = {"response_format": response_format} if response_format else {}

    for attempt in range(MAX_RETRIES):
        try:
            response = await asyncio.to_thread(
                _client.chat.completions.create,
                model=model,
                messages=messages,
                timeout=REQUEST_TIMEOUT_SECONDS,
                **extra_kwargs,
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                raise
            wait = BACKOFF_BASE_SECONDS * (2 ** attempt)
            logger.warning(
                "LLM 呼叫失敗（第 %d/%d 次），%s 秒後重試：%s",
                attempt + 1, MAX_RETRIES, wait, exc,
            )
            await asyncio.sleep(wait)

    raise last_exc  # pragma: no cover — 迴圈內不是 return 就是 raise，理論上不會到這裡
