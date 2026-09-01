"""
Prompt 組裝：載入人設檔案，組出每次呼叫 API 要用的完整 messages。

拆成 build_base_system_messages()（人設＋長期記憶＋現況小抄＋時間感知）與
output_format_system_message()（輸出格式技術規定），供正常回覆與未來的主動發訊
共用同一套組裝邏輯（DRY）。
"""

from __future__ import annotations

from datetime import datetime

import config
import schedule_override
from schedule import WEEKDAY_NAMES, format_elapsed, get_schedule_status, now_taipei


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_system_prompt() -> str:
    persona = _load_text(config.SYSTEM_PROMPT_PATH)
    few_shot = _load_text(config.FEW_SHOT_PATH)
    return persona + "\n\n" + few_shot


def load_pic_prompt_prefix() -> str:
    return _load_text(config.PIC_PROMPT_PATH).strip()


SYSTEM_PROMPT = load_system_prompt()

_OUTPUT_FORMAT_INSTRUCTIONS = (
    "【輸出格式技術規定，此段不屬於角色設定】\n"
    "你的輸出會被強制成三個欄位的 JSON（reply / state / image_prompt），"
    "以下說明每個欄位該填什麼：\n"
    "reply：正常要給使用者看的回覆內容，符合角色人設與語氣。\n"
    "state：用旁白角度簡短描述回覆後「會持續到下一輪對話仍然成立」的狀態"
    "（例如：正在做的事、身體狀態如頭髮還沒吹乾、情緒是否還沒平復及原因等）。"
    "只描述會延續下去的狀態，不要重複列出這輪講過的對話內容，"
    "如果沒有需要更新的狀態就照抄前一版的【目前持續中的狀態】內容，這個欄位不能空白。\n"
    "image_prompt：如果這一輪你判斷想主動分享一張照片給對方看（例如聊到你正在做的事、"
    "想讓對方看看你現在的樣子，不需要每輪都分享），才需要填這個欄位；"
    "內容是給圖片生成模型用的英文 prompt，這是給程式用的，不是講給使用者聽的話。"
    "角色身分與畫風不用你描述，程式會自動加上，"
    "你只需要依照這次場景自由決定並描述以下六個面向，"
    "全部合併成一段逗號分隔的描述，不要加類別標籤：\n"
    "1. Clothing（服裝：款式、顏色、材質）\n"
    "2. Action（動作：姿勢、肢體、手部動作、視線方向）\n"
    "3. Expression（表情：情緒、細微表情特徵）\n"
    "4. Environment Background（背景：場景、環境細節、景深）\n"
    "5. Lighting Atmosphere（光線：光源方向、光影對比、氛圍）\n"
    "6. Composition and Camera（構圖與運鏡：拍攝距離如全身／半身／特寫、"
    "拍攝角度如平視／俯角／仰角、視角如第一人稱自拍或第三人稱旁觀，"
    "依照這次想呈現的場景自由決定，不要每次都套用同一種構圖）\n"
    "每個面向都要有具體內容，不要籠統帶過。不想分享照片就把這個欄位填 null。\n"
    "schedule_override_type／schedule_override_text：只有在翔明確提到要調整作息時才填寫"
    "（例如「今晚熬夜到兩點」「以後週三都晚點睡」），其他情況兩個欄位都填 null。"
    "type 填 \"today\"（只影響今天）或 \"recurring\"（持續性調整），"
    "text 用一句話描述調整內容即可，不用結構化格式。\n"
    "以上這些技術段落使用者都看不到，不用顧慮角色語氣，直接客觀描述即可。"
)


def build_user_content(text: str, image_urls: list[str]):
    """組出這一輪 user 訊息的 content：純文字就用字串，有圖片就用 array 格式。"""
    if not image_urls:
        return text

    content = []
    if text:
        content.append({"type": "text", "text": text})
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def build_base_system_messages(
    bot_memory: str,
    current_state: str,
    now: datetime | None = None,
    last_interaction_time: datetime | None = None,
) -> list[dict]:
    """人設 + 長期記憶 + 現況小抄 + 時間感知，供正常回覆與主動發訊共用。

    last_interaction_time 有給的話，會額外注入「距離上次互動過了多久」，
    讓時間感知不只是「現在幾點」，也包含「這段對話中斷了多久」。
    """
    now = now or now_taipei()
    status = get_schedule_status(now)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if bot_memory:
        messages.append({
            "role": "system",
            "content": f"【關於使用者與過往互動的長期記憶】\n{bot_memory}",
        })

    messages.append({
        "role": "system",
        "content": f"【目前持續中的狀態】\n{current_state}",
    })

    time_awareness = (
        f"【目前時間感知】現在是{WEEKDAY_NAMES[now.weekday()]} {now.strftime('%H:%M')}，"
        f"依照平常作息，這個時段通常是：{status.label}。"
        "回覆的語氣長短可以自然反映這個時段的狀態，不用刻意提起。"
    )
    if last_interaction_time is not None:
        elapsed = format_elapsed(last_interaction_time, now)
        time_awareness += f"\n距離上次互動已經過了約{elapsed}。"
    messages.append({"role": "system", "content": time_awareness})

    override_text = schedule_override.get_active_override_text()
    if override_text:
        messages.append({
            "role": "system",
            "content": (
                f"【目前生效的作息調整，覆蓋預設作息】\n{override_text}\n"
                "回覆時依照這個調整過的作息判斷現在的狀態，而不是原本的預設時段。"
            ),
        })

    return messages


def output_format_system_message() -> dict:
    return {"role": "system", "content": _OUTPUT_FORMAT_INSTRUCTIONS}


def long_silence_note(elapsed_label: str) -> str:
    """距離上次互動過了很久之後的 catch-up 提示，跟「剛結束忙碌時段」是同一種

    「給提示、不強制」的注入模式：明確告訴 LLM 這是隔了很久之後重新開始的對話，
    不用執著在沉默前的舊話題，語氣也可以自然轉換，而不是被 context 慣性卡住。
    """
    return (
        f"【隔了一段時間才繼續對話】距離上次互動已經過了約{elapsed_label}，"
        "這不是接續剛才的話題，是隔了一段時間後重新開始的對話。"
        "不需要執著在沉默前的舊話題，語氣也可以自然轉換，依照現在的心情、狀態自然回應即可。"
    )


def build_messages(
    history: list[dict],
    bot_memory: str,
    current_state: str,
    user_content,
    catch_up_note: str | None = None,
    last_interaction_time: datetime | None = None,
    now: datetime | None = None,
) -> list[dict]:
    """組出這次要送給 API 的完整 messages：base system + catch-up + 輸出格式 + 歷史 + 這輪訊息。

    catch_up_note（忙碌時段結束）跟長時間沉默提示可能同時成立，兩者都給、
    不互斥；呼叫端決定 catch_up_note 內容，這裡只負責在滿足時數門檻時
    額外附加長時間沉默提示。
    """
    now = now or now_taipei()
    messages = build_base_system_messages(
        bot_memory, current_state, now, last_interaction_time
    )

    if catch_up_note:
        messages.append({"role": "system", "content": catch_up_note})

    if last_interaction_time is not None:
        elapsed_hours = (now - last_interaction_time).total_seconds() / 3600
        if elapsed_hours >= config.LONG_SILENCE_HOURS:
            elapsed_label = format_elapsed(last_interaction_time, now)
            messages.append({"role": "system", "content": long_silence_note(elapsed_label)})

    messages.append(output_format_system_message())
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def build_proactive_check_messages(
    history: list[dict],
    bot_memory: str,
    current_state: str,
    now: datetime | None = None,
    last_interaction_time: datetime | None = None,
) -> list[dict]:
    """組出「主動發訊檢查」用的 messages：跟正常回覆共用人設/記憶/狀態/時間感知，
    但沒有使用者這輪的訊息，改成一段系統說明，讓她自己判斷這個當下想不想主動開口。
    """
    messages = build_base_system_messages(bot_memory, current_state, now, last_interaction_time)

    messages.append({
        "role": "system",
        "content": (
            "【系統定期檢查，不是使用者傳來的訊息】\n"
            "現在沒有新訊息，這是系統定期檢查你想不想主動開口。"
            "根據你目前的狀態、心情、跟對方的關係，自己判斷這個當下想不想主動傳訊息給翔。\n"
            f"如果不想，reply 欄位只填「{config.NO_PROACTIVE_TOKEN}」這幾個字，"
            "state 欄位照抄前一版的現況小抄即可，image_prompt 填 null。\n"
            "如果想，才需要照平常的格式輸出：reply 填你想說的話，"
            "state／image_prompt 照樣照下一段技術規定填寫。"
        ),
    })
    messages.append(output_format_system_message())
    messages.extend(history)
    return messages
