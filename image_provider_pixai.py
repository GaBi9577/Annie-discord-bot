"""
PixAI 圖片生成 provider（v2 image API，task-based）。

流程跟 NanoGPT 不一樣：PixAI 是非同步任務制，建立任務後要輪詢狀態，
狀態變成 completed 才有圖片 URL 可以下載。介面（generate_image）跟
NanoGPT provider 保持一致，讓 image_gen.py 可以無痛切換。
"""

from __future__ import annotations

import asyncio
import logging
import time

import requests

import config
from prompt_builder import load_pic_prompt_prefix

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_REQUEST_TIMEOUT_SECONDS = 30

_PIC_PROMPT_PREFIX = load_pic_prompt_prefix()


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {config.PIXAI_API_KEY}",
        "Content-Type": "application/json",
    }


def _create_task(full_prompt: str) -> str:
    resp = requests.post(
        f"{config.PIXAI_BASE_URL}/v2/image/create",
        headers=_headers(),
        json={
            "modelVersionId": config.PIXAI_MODEL_VERSION_ID,
            "prompt": full_prompt,
            "aspectRatio": config.PIXAI_ASPECT_RATIO,
            "mode": config.PIXAI_MODE,
            "batchSize": 1,
        },
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def _get_task(task_id: str) -> dict:
    resp = requests.get(
        f"{config.PIXAI_BASE_URL}/v1/task/{task_id}",
        headers=_headers(),
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def _poll_until_done(task_id: str) -> dict:
    """輪詢任務狀態直到終止狀態（completed/failed/cancelled）或逾時。"""
    deadline = time.time() + config.PIXAI_POLL_TIMEOUT_SECONDS
    while True:
        task = _get_task(task_id)
        if task.get("status") in _TERMINAL_STATUSES:
            return task
        if time.time() > deadline:
            raise TimeoutError(f"PixAI 任務 {task_id} 輪詢逾時（超過 {config.PIXAI_POLL_TIMEOUT_SECONDS} 秒）")
        time.sleep(config.PIXAI_POLL_INTERVAL_SECONDS)


def _generate_sync(scene_prompt: str) -> bytes | None:
    full_prompt = f"{_PIC_PROMPT_PREFIX}, {scene_prompt}"
    task_id = _create_task(full_prompt)
    task = _poll_until_done(task_id)

    if task.get("status") != "completed":
        logger.warning("PixAI 任務未成功完成：task_id=%s status=%s", task_id, task.get("status"))
        return None

    media_urls = task.get("outputs", {}).get("mediaUrls") or []
    if not media_urls:
        logger.warning("PixAI 任務完成但沒有回傳圖片：task_id=%s", task_id)
        return None

    image_resp = requests.get(media_urls[0], timeout=_REQUEST_TIMEOUT_SECONDS)
    image_resp.raise_for_status()
    return image_resp.content


async def generate_image(scene_prompt: str) -> bytes | None:
    """呼叫 PixAI 圖片生成 API。失敗（含逾時）回傳 None，不中斷整輪回覆。"""
    try:
        return await asyncio.to_thread(_generate_sync, scene_prompt)
    except Exception:
        logger.exception("PixAI 圖片生成失敗")
        return None
