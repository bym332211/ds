import os
import asyncio
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from .llm import DeepSeekClient


# Load environment variables from .env if present
load_dotenv()


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Optional system prompt for the assistant behavior
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Reply in the user's language.",
)

# Max number of past turns to keep per chat
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))


class ConversationMemory:
    """In-memory per-chat conversation buffer."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._store: Dict[int, Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=max_turns)
        )

    def add(self, chat_id: int, role: str, content: str) -> None:
        self._store[chat_id].append((role, content))

    def messages(self, chat_id: int) -> List[Tuple[str, str]]:
        return list(self._store[chat_id])

    def clear(self, chat_id: int) -> None:
        self._store.pop(chat_id, None)


memory = ConversationMemory(max_turns=HISTORY_TURNS)


def _build_prompt_from_memory(user_text: str, chat_id: int) -> str:
    """Render a simple chat-style prompt from memory plus new user text."""
    lines: List[str] = []
    for role, content in memory.messages(chat_id):
        lines.append(f"{role.upper()}: {content}")
    lines.append(f"USER: {user_text}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _split_for_telegram(text: str, limit: int = 4000) -> List[str]:
    chunks: List[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split on paragraph or sentence boundaries
        cut = text.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = text.rfind(". ", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(text[:cut].strip())
        text = text[cut:].lstrip()
    return [c for c in chunks if c]


async def _call_llm(prompt: str) -> str:
    # DeepSeekClient is synchronous; run in thread to avoid blocking
    loop = asyncio.get_running_loop()
    client = DeepSeekClient(api_key=DEEPSEEK_API_KEY)

    def _do_call() -> str:
        return client.complete(system_prompt=DEFAULT_SYSTEM_PROMPT, user_prompt=prompt)

    return await loop.run_in_executor(None, _do_call)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "你好！我是你的 LLM 助手机器人。直接发送消息即可与我对话。\n"
        "指令:\n"
        "/start - 开始\n"
        "/reset - 清空当前会话上下文"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    memory.clear(chat_id)
    await update.message.reply_text("已重置会话记忆。")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    if not DEEPSEEK_API_KEY:
        await update.message.reply_text("服务器未配置 DEEPSEEK_API_KEY。")
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    # Add user message to memory
    memory.add(chat_id, "user", user_text)

    # Show typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        prompt = _build_prompt_from_memory(user_text, chat_id)
        reply = await _call_llm(prompt)
    except Exception as e:
        await update.message.reply_text(f"调用 LLM 失败: {e}")
        return

    # Save assistant reply
    memory.add(chat_id, "assistant", reply)

    # Send in chunks if too long
    for chunk in _split_for_telegram(reply):
        await update.message.reply_text(chunk)


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment.")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

