# Annie Discord Bot

單一使用者的個人 Discord Bot。
透過 LLM 驅動人設對話，並具備長期記憶、
生活作息感知、主動發訊、圖片生成等能力。

這是個人專案，不是給多人使用的服務，架構上刻意保持精簡（YAGNI），
沒有導入資料庫、Vector DB 或 Agent framework 等超出實際需求的東西。

## 目前功能

- **人設對話**：以 system prompt + few-shot 範例驅動的角色扮演，回應透過
  LLM 原生 Structured Output（JSON Schema）取得，不依賴文字 delimiter 切割。
- **短期／長期記憶**：近期對話存在記憶體中（sliding window），超過上限的
  部分由背景 worker 依序摘要進 `bot_mem.md`，並在內容過長時自動再壓縮一次。
- **現況小抄**：每輪回應會附帶「這輪之後仍會持續成立的狀態」，讓角色的
  情緒／動作／身體狀態能延續到下一輪對話。
- **生活作息感知**：睡覺、健身等時段會阻擋即時回覆，訊息會等時段結束後
  一次性回覆，並附上「剛結束忙碌時段」的自然語氣提示。
- **Debounce 合併訊息**：短時間內連續傳送的多則訊息會合併成一輪處理，
  避免逐則搶答。
- **主動發訊**：背景迴圈會定期評估要不要主動開口傳訊息給使用者，採兩階段
  判斷（機率過濾 → LLM 決策），避免每次都花錢問 LLM。
- **圖片生成**：可依對話場景自動生成一張角色照片，支援 NanoGPT 與 PixAI
  兩種 provider，透過 `config.IMAGE_PROVIDER` 切換，呼叫端程式碼不需更動。

## Architecture

```
main.py              進入點：Discord client、訊息收發、debounce/schedule 阻擋
session.py           UserSession / SessionManager：每位使用者的暫存狀態
schedule.py          生活作息判斷（睡覺／健身／上課等時段）
prompt_builder.py     組出送給 LLM 的完整 messages
llm_client.py         封裝 NanoGPT (OpenAI-compatible) API 呼叫，含 retry
response_parser.py    定義 Structured Output JSON Schema 並解析回應
state.py              現況小抄（CurrentStateHolder）
memory.py             長期記憶讀寫，asyncio.Queue + 背景 worker 序列化更新
proactive.py          主動發訊背景迴圈
image_gen.py           圖片生成 facade，依 config 切換 provider
image_provider_nanogpt.py  NanoGPT 圖片生成實作
image_provider_pixai.py    PixAI 圖片生成實作（task-based：建立任務→輪詢→下載）
config.py             一般設定與參數（不含敏感值）
apikey.py             Token / API Key（不進版本控制，需自行建立）
tests/                純邏輯單元測試，不碰 Discord、不打真的 LLM API
```

角色的 context / memory / schedule / decision / prompt / LLM 呼叫都在各自的
模組裡；`main.py` 只負責 Discord 層的收發與串接。

## Installation

需要 Python 3.11 以上。

```bash
git clone https://github.com/GaBi9577/Annie-discord-bot.git
cd Annie-discord-bot
pip install -e .
```

開發／測試另外需要：

```bash
pip install -e ".[dev]"
```

## API Key 設定

`apikey.py` 不進版本控制，需要自行在專案根目錄建立：

```python
# apikey.py
TOKEN = "你的 Discord Bot Token"
NANOGPT_API_KEY = "你的 NanoGPT API Key"
PIXAI_API_KEY = "你的 PixAI API Key"  # 若不使用 PixAI 圖片生成，可留空字串
```

以下檔案／目錄同樣不進版本控制，需要自行準備：

| 路徑 | 用途 |
|---|---|
| `annie/persona.md` | 人設本體（文字互動用） |
| `annie/pic_prompt.md` | 外觀描述（圖片生成用，加在生成 prompt 前面） |
| `annie/few_shot.md` | Few-shot 對話範例 |
| `bot_mem.md` | 長期記憶檔案，第一次執行會自動建立空白檔案 |
| `current_state.md` | 現況小抄，第一次執行會自動建立預設內容 |

其餘一般參數（模型選擇、作息時段、debounce 秒數、主動發訊機率等）都在
`config.py` 中，附有中文註解說明用途。

## 啟動方式

```bash
python main.py
```

Windows 下也可使用 `start_bot.bat`（使用相對路徑，於專案根目錄下雙擊執行
或透過終端機呼叫皆可，會依系統預設的 `python` 指令啟動）。

## Testing

```bash
pytest
```

測試涵蓋範圍為純邏輯（schedule 邊界判斷、Structured Output 解析、
pending buffer snapshot 行為、proactive 發送容錯、長期記憶序列化與字數
上限壓縮等），不會實際連線 Discord 或呼叫真正的 LLM API。
