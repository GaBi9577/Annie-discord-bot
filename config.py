# config.py

# 一般設定與參數，不含實際敏感值，可自由分享
# 敏感值（Token、API Key）在 apikey.py，那個檔案禁止分享

from apikey import TOKEN, NANOGPT_API_KEY

# ----- 方便調參用(模型) -----

text_model = "x-ai/grok-4.5"
img_model_list = ["wai-illustrious-sdxl", "nsfw-gen-illustrious",
                  "persona:376130@2456367", "infinite-illustrious",
                  "animagine-xl-31", "crystal-clear-xl"]

image_model = img_model_list[0]
# 1 sucks



# --- Discord bot token ---


# --- NanoGPT(OpenAI-compatible endpoint)---

NANOGPT_BASE_URL = "https://nano-gpt.com/api/v1"
MODEL = text_model
# 模型 ID 請上 NanoGPT 官方模型列表確認最新可用名稱 https://nano-gpt.com/api (或 docs.nano-gpt.com)


# --- 人設 ---
SYSTEM_PROMPT_PATH = "annie/persona.md"  # 人設本體(文字互動用)
PIC_PROMPT_PATH = "annie/pic_prompt.md"  # 外觀描述(圖片生成用)加在圖片生成 prompt 前面(未接 LoRA)
FEW_SHOT_PATH = "annie/few_shot.md"

# --- 記憶 ---
MAX_HISTORY_MESSAGES = 10
BOT_MEMORY_PATH = "bot_mem.md"
# 長期記憶檔案，由 bot 自動整理與更新
# 短期記憶則數上限(user+assistant 合計，約 4 輪)
# 超過的部分會自動整理進 bot_mem.md 再從短期記憶砍掉
# 數字大小請憑實測手感調整，不用算得太精確

CURRENT_STATE_PATH = "current_state.md"  # 現況小抄：短期內持續變動的狀態(情緒/動作/身體狀態等)
STATE_DELIMITER = "\n===STATE===\n"      # 用來從模型輸出切開「回覆」與「更新後的現況小抄」

# --- 對話行為 ---
REPLY_DELAY = 2  # 秒，訊息緩衝等待時間

# --- 生活作息／時間感知(硬性阻擋回覆的時段)---
WAKE_HOUR = 10          # 早上 10 點起床
SLEEP_START_HOUR = 0    # 睡眠開始(午夜 0 點，「凌晨不熬夜」沒給精確時間，先抓午夜，可自行調整)
GYM_START_HOUR = 17     # 健身開始
GYM_END_HOUR = 19       # 健身結束(17-19 點健身時段內不回訊息)


# --- 圖片生成(NanoGPT，備案為本機 SDXL Turbo，待筆電空間足夠再切換)---
NANOGPT_IMAGE_URL = "https://nano-gpt.com/v1/images/generations"
IMAGE_MODEL = image_model  # 模型 ID 請對照 NanoGPT 圖片模型列表確認
IMAGE_RESOLUTION = "1088x1920"  # 對應 9:16 直式構圖，用 API 參數硬性指定，比純文字提示可靠
IMAGE_DELIMITER = "\n===IMAGE===\n"  # 用來從模型輸出切出圖片生成 prompt(可選段落)

# --- 主動發訊(bot 主控性，不需使用者輸入也能主動開口)---
# 兩階段判斷：第一關(便宜，不呼叫 LLM)先用機率過濾，過關才呼叫 LLM 問她真正的意願
PROACTIVE_CHECK_INTERVAL_MIN_MINUTES = 30  # 每次檢查間隔下限(隨機，避免太規律)
PROACTIVE_CHECK_INTERVAL_MAX_MINUTES = 60  # 每次檢查間隔上限
PROACTIVE_MIN_QUIET_MINUTES = 20           # 距離上次互動(使用者發訊或她主動發訊都算)至少要這麼久，才考慮主動發訊
PROACTIVE_GATE_BASE_PROBABILITY = 0.15     # 第一關基礎機率(剛過最短安靜時間時)
PROACTIVE_GATE_MAX_PROBABILITY = 0.6       # 第一關機率上限(閒置越久機率越高，但封頂)
PROACTIVE_GATE_GROWTH_PER_HOUR = 0.15      # 每多閒置一小時，機率增加多少
NO_PROACTIVE_TOKEN = "NO_PROACTIVE"        # LLM 判斷這次不想主動發訊時，固定回覆這串字
