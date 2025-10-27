Telegram LLM Bot Setup

- Prereqs: Python 3.10+, pip, a Telegram bot token, and a DeepSeek API key.

Steps

- Copy `.env.example` to `.env` and fill:
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID` (单个或以逗号分隔的多个 chat id)
  - `DEEPSEEK_API_KEY`
  - Optional: `SYSTEM_PROMPT`, `HISTORY_TURNS`
- Install deps: `pip install -r requirements.txt`
- Run the bot: `python -m bot.telegram_bot`

Notes

- Conversation memory keeps the latest N turns per chat (default 6).
- Replies longer than Telegram’s limit are split into chunks.
- Use `/reset` to clear the current chat memory; `/start` for help.
- To change model behavior, set `SYSTEM_PROMPT` in `.env`.

Push from deepseek_modular

- `deepseek_modular.py` 在每次 LLM 调用完成并解析为非 HOLD 决策时，才会将原始 LLM 返回内容通过 Bot API 推送到 `TELEGRAM_CHAT_ID` 指定的会话（HOLD 不推送）。
- 若未设置 `TELEGRAM_BOT_TOKEN` 或 `TELEGRAM_CHAT_ID`，该推送将自动跳过，不影响交易主流程。
- 当决策为非 HOLD 且通过数量计算后，会额外推送一条精简决策摘要（action/side/qty/SL/TP）。
- 订单执行后将推送成交结果：
  - 成功（MARKET/LIMIT）：包含类型、方向、成交数量、平均成交价、SL/TP 与订单 ID。
  - 失败或限价未成交：推送失败原因。
- 对于已下发的保护单（SL/TP），程序会在每轮循环中轮询其状态，一旦触发成交，会推送“Protection Triggered”通知，包含类型（SL/TP）、方向、成交数量、均价与订单 ID。
