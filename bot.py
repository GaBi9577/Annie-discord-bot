import discord
import asyncio
import base64
import io
import requests
from datetime import datetime
from openai import OpenAI
from config import *

with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    persona = f.read()

with open(FEW_SHOT_PATH, "r", encoding="utf-8") as f:
    few_shot = f.read()

SYSTEM_PROMPT = persona + "\n\n" + few_shot

with open(PIC_PROMPT_PATH, "r", encoding="utf-8") as f:
    PIC_PROMPT_PREFIX = f.read().strip()

llm_client = OpenAI(
    api_key=NANOGPT_API_KEY,
    base_url=NANOGPT_BASE_URL,
)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

pending = {}              # user_id -> asyncio.Task（計時器，debounce 或等待忙碌時段結束）
pending_mode = {}         # user_id -> "debounce" | "waiting"（目前計時器屬於哪種模式）
pending_buffer = {}       # user_id -> list[str]（等待期間收到的訊息，等等會合併）
conversation_history = {}  # user_id -> list[{"role": ..., "content": ...}]（短期記憶）


def load_bot_memory():
    """讀取長期記憶檔案，不存在則建立空白檔案"""
    try:
        with open(BOT_MEMORY_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        with open(BOT_MEMORY_PATH, "w", encoding="utf-8") as f:
            f.write("")
        return ""


bot_memory = load_bot_memory()


def load_current_state():
    """讀取現況小抄檔案，不存在則建立預設狀態"""
    try:
        with open(CURRENT_STATE_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        default_state = "剛開始互動，沒有特別的持續性狀態。"
        with open(CURRENT_STATE_PATH, "w", encoding="utf-8") as f:
            f.write(default_state)
        return default_state


current_state = load_current_state()

WEEKDAY_NAMES = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]


def get_schedule_status(now=None):
    """
    判斷目前的作息狀態。
    回傳 dict：
      blocking    - 是否為硬性阻擋回覆的時段（睡覺／健身）
      label       - 目前狀態的簡短描述，用於注入 prompt 當背景資訊
      available_at - 若 blocking，時段結束的時間點（datetime）；否則為 None
    """
    now = now or datetime.now()
    hour = now.hour + now.minute / 60
    weekday = now.weekday()  # 0=週一 ... 6=週日

    if SLEEP_START_HOUR <= hour < WAKE_HOUR:
        available_at = now.replace(hour=WAKE_HOUR, minute=0, second=0, microsecond=0)
        return {"blocking": True, "reason": "sleep", "label": "睡覺中", "available_at": available_at}

    if GYM_START_HOUR <= hour < GYM_END_HOUR:
        available_at = now.replace(hour=GYM_END_HOUR, minute=0, second=0, microsecond=0)
        return {"blocking": True, "reason": "gym", "label": "健身中", "available_at": available_at}

    if weekday < 5 and WAKE_HOUR <= hour < GYM_START_HOUR:
        label = "白天上課中"
    elif weekday >= 5:
        label = "假日，沒有固定行程"
    else:
        label = "晚上自由時間"

    return {"blocking": False, "reason": None, "label": label, "available_at": None}


def format_elapsed(start, end):
    """把時間差轉成中文簡短描述，用於「剛結束忙碌時段」的 catch-up 提示"""
    minutes = int((end - start).total_seconds() // 60)
    if minutes < 60:
        return f"{minutes}分鐘"
    hours = minutes // 60
    remaining = minutes % 60
    return f"{hours}小時{remaining}分鐘" if remaining else f"{hours}小時"


def extract_image_urls(message):
    """從 Discord 訊息附件中取出圖片 URL"""
    return [
        att.url for att in message.attachments
        if att.content_type and att.content_type.startswith("image/")
    ]


def build_user_content(text, image_urls):
    """組出這一輪 user 訊息的 content：純文字就用字串，有圖片就用 array 格式"""
    if not image_urls:
        return text
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def build_messages(user_id, user_content, catch_up_note=None):
    """組出這次要送給 API 的 messages：角色人設 + 長期記憶 + 現況小抄 + 時間感知 + 短期歷史 + 這次的訊息"""
    history = conversation_history.setdefault(user_id, [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if bot_memory:
        messages.append({
            "role": "system",
            "content": f"【關於使用者與過往互動的長期記憶】\n{bot_memory}"
        })
    messages.append({
        "role": "system",
        "content": f"【目前持續中的狀態】\n{current_state}"
    })

    now = datetime.now()
    status = get_schedule_status(now)
    messages.append({
        "role": "system",
        "content": (
            f"【目前時間感知】現在是{WEEKDAY_NAMES[now.weekday()]} {now.strftime('%H:%M')}，"
            f"依照平常作息，這個時段通常是：{status['label']}。"
            "回覆的語氣長短可以自然反映這個時段的狀態，不用刻意提起。"
        )
    })

    if catch_up_note:
        messages.append({"role": "system", "content": catch_up_note})

    messages.append({
        "role": "system",
        "content": (
            "【輸出格式技術規定，此段不屬於角色設定】\n"
            f"正常回覆完之後，換行加上分隔線「{STATE_DELIMITER.strip()}」，"
            "接著用旁白角度簡短描述回覆後「會持續到下一輪對話仍然成立」的狀態"
            "（例如：正在做的事、身體狀態如頭髮還沒吹乾、情緒是否還沒平復及原因等）。"
            "只描述會延續下去的狀態，不要重複列出這輪講過的對話內容，"
            "如果沒有需要更新的狀態就照抄前一版的【目前持續中的狀態】內容。\n"
            "如果這一輪你判斷想主動分享一張照片給對方看（例如聊到你正在做的事、"
            "想讓對方看看你現在的樣子，不需要每輪都分享），"
            f"在狀態段落後面再換行加上「{IMAGE_DELIMITER.strip()}」，"
            "接著用英文寫這張照片的 prompt，這是給圖片生成模型用的，不是講給使用者聽的話。"
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
            "每個面向都要有具體內容，不要籠統帶過。不想分享照片就不要加這一段。\n"
            "以上這些技術段落使用者都看不到，不用顧慮角色語氣，直接客觀描述即可。"
        )
    })
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def split_model_output(raw_text):
    """把模型輸出切成三段：要送給使用者的回覆、更新後的現況小抄、（可選）圖片生成 prompt"""
    reply_part = raw_text
    state_part = None
    image_prompt = None

    if STATE_DELIMITER in raw_text:
        reply_part, rest = raw_text.split(STATE_DELIMITER, 1)
        if IMAGE_DELIMITER in rest:
            state_part, image_prompt = rest.split(IMAGE_DELIMITER, 1)
            image_prompt = image_prompt.strip()
        else:
            state_part = rest
        state_part = state_part.strip()

    return reply_part.strip(), state_part, image_prompt


async def generate_image(scene_prompt):
    """呼叫 NanoGPT 圖片生成 API，回傳圖片的 bytes；失敗回傳 None"""
    full_prompt = f"{PIC_PROMPT_PREFIX}, {scene_prompt}"

    def _request():
        resp = requests.post(
            NANOGPT_IMAGE_URL,
            headers={
                "Authorization": f"Bearer {NANOGPT_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "prompt": full_prompt,
                "model": IMAGE_MODEL,
                "resolution": IMAGE_RESOLUTION,
                "n": 1,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    try:
        data = await asyncio.to_thread(_request)
        b64_json = data["data"][0]["b64_json"]
        return base64.b64decode(b64_json)
    except Exception as e:
        print("圖片生成失敗：", e)
        return None


def strip_images_for_history(content):
    """存進短期歷史前，把圖片部分換成文字佔位符——圖片只在當輪分析，不重複佔用後續 token"""
    if isinstance(content, str):
        return content
    text_parts = [item["text"] for item in content if item["type"] == "text"]
    image_count = sum(1 for item in content if item["type"] == "image_url")
    text = " ".join(text_parts)
    if image_count:
        placeholder = f"[傳送了{image_count}張圖片]"
        text = f"{text}\n{placeholder}" if text else placeholder
    return text


def update_history(user_id, user_content, reply_content):
    """把這一輪對話存入短期歷史，超過上限的部分交給長期記憶整理後砍掉"""
    history = conversation_history.setdefault(user_id, [])
    history.append({"role": "user", "content": strip_images_for_history(user_content)})
    history.append({"role": "assistant", "content": reply_content})
    if len(history) > MAX_HISTORY_MESSAGES:
        overflow_count = len(history) - MAX_HISTORY_MESSAGES
        overflow = history[:overflow_count]
        del history[:overflow_count]
        asyncio.create_task(update_bot_memory(overflow))


def stringify_content(content):
    """把 content（可能是純文字或多模態 array）轉成摘要用的純文字描述"""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if item["type"] == "text":
            parts.append(item["text"])
        elif item["type"] == "image_url":
            parts.append("[附上一張圖片]")
    return " ".join(parts)


async def update_bot_memory(overflow_messages):
    """呼叫 LLM 把即將被短期記憶砍掉的內容整理進長期記憶檔案"""
    global bot_memory

    overflow_text = "\n".join(
        f"{'使用者' if m['role'] == 'user' else '亞妮'}：{stringify_content(m['content'])}"
        for m in overflow_messages
    )

    summarize_prompt = (
        "你是記憶整理工具，不是在扮演角色。"
        "請把「新對話片段」中值得長期記住的資訊（使用者的習慣、喜好、重要事件、"
        "關係狀態變化等），整合進「既有長期記憶」，輸出更新後的完整長期記憶。"
        "用精簡條列呈現，不要重複贅述、不要流水帳、不要加入不重要的閒聊內容。"
        "只需要輸出更新後的長期記憶內容本身，不要有其他說明文字。\n\n"
        f"【既有長期記憶】\n{bot_memory if bot_memory else '（目前無）'}\n\n"
        f"【新對話片段】\n{overflow_text}"
    )

    try:
        response = await asyncio.to_thread(
            llm_client.chat.completions.create,
            model=MODEL,
            messages=[{"role": "user", "content": summarize_prompt}]
        )
        updated_memory = response.choices[0].message.content.strip()
    except Exception as e:
        print("長期記憶整理失敗：", e)
        return

    bot_memory = updated_memory
    with open(BOT_MEMORY_PATH, "w", encoding="utf-8") as f:
        f.write(updated_memory)
    print("長期記憶已更新")


@client.event
async def on_ready():
    print(f"Bot 已上線：{client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_id = message.author.id

    pending_buffer.setdefault(user_id, []).append({
        "text": message.content,
        "images": extract_image_urls(message),
    })

    status = get_schedule_status()

    # 已經在「等待忙碌時段結束」模式時，新訊息只需要進緩衝區，不用動計時器——
    # 因為等待時間是算到固定的時段結束點，不是每則訊息重新倒數
    if pending_mode.get(user_id) == "waiting" and status["blocking"]:
        return

    if user_id in pending:
        pending[user_id].cancel()

    if status["blocking"]:
        pending_mode[user_id] = "waiting"
        wait_seconds = max((status["available_at"] - datetime.now()).total_seconds(), 0)
        busy_started_at = datetime.now()
        task = asyncio.create_task(
            wait_then_reply(user_id, message, wait_seconds,
                             busy_reason=status["reason"], busy_started_at=busy_started_at)
        )
    else:
        pending_mode[user_id] = "debounce"
        task = asyncio.create_task(wait_then_reply(user_id, message, REPLY_DELAY))

    pending[user_id] = task


async def wait_then_reply(user_id, message, wait_seconds, busy_reason=None, busy_started_at=None):
    global current_state

    await asyncio.sleep(wait_seconds)

    buffered = pending_buffer.pop(user_id, [])
    if not buffered:
        pending.pop(user_id, None)
        pending_mode.pop(user_id, None)
        return

    merged_text = "\n".join(item["text"] for item in buffered if item["text"])
    merged_images = [url for item in buffered for url in item["images"]]
    user_content = build_user_content(merged_text, merged_images)

    catch_up_note = None
    if busy_reason:
        elapsed = format_elapsed(busy_started_at, datetime.now())
        busy_label = "睡覺" if busy_reason == "sleep" else "健身"
        catch_up_note = (
            f"【剛結束忙碌時段】你剛結束{busy_label}，這段期間（約{elapsed}）沒看訊息，"
            "現在才剛看到累積的訊息，語氣自然反映剛回神／剛醒來／剛練完的狀態即可，"
            "不用逐句回應每一則，抓重點自然回應。"
        )

    async with message.channel.typing():
        response = await asyncio.to_thread(
            llm_client.chat.completions.create,
            model=MODEL,
            messages=build_messages(user_id, user_content, catch_up_note=catch_up_note)
        )
        raw_text = response.choices[0].message.content
        reply_text, updated_state, image_prompt = split_model_output(raw_text)

        image_bytes = None
        if image_prompt:
            image_bytes = await generate_image(image_prompt)

    print("回傳內容：", reply_text)
    if image_bytes:
        discord_file = discord.File(io.BytesIO(image_bytes), filename="annie.png")
        await message.channel.send(content=reply_text, file=discord_file)
    else:
        await message.channel.send(reply_text)

    if updated_state:
        current_state = updated_state
        with open(CURRENT_STATE_PATH, "w", encoding="utf-8") as f:
            f.write(updated_state)
    else:
        print("警告：模型沒有依格式輸出現況小抄，本輪狀態維持不變")

    update_history(user_id, user_content, reply_text)
    pending.pop(user_id, None)
    pending_mode.pop(user_id, None)


async def flush_remaining_history():
    """程式關閉前，把還沒被截斷整理過的短期記憶強制存入長期記憶，避免遺失"""
    for user_id, history in list(conversation_history.items()):
        if history:
            await update_bot_memory(history)
            history.clear()
    print("關閉前已將剩餘短期記憶存入長期記憶")


try:
    client.run(TOKEN)
finally:
    asyncio.run(flush_remaining_history())
