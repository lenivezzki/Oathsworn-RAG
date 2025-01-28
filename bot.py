import telebot
import asyncio
from src.embedder import LaBSEEmbedder
from src.knowledgebase import QdrantKnowledgeBase
from src.ranker import ChunksReranker
from src.assistant import RulesAssistant
from src.utils import response
from dotenv import load_dotenv
import os

load_dotenv()

telegram_token = os.getenv('TELEGRAM_TOKEN')
assistant_name = os.getenv('ASSISTANT_NAME')
system_prompt = os.getenv('SYSTEM_PROMPT')
collection_name = os.getenv('COLLECTION_NAME')
qdrant_host = os.getenv('QDRANT_HOST')
qdrant_port = int(os.getenv('QDRANT_PORT'))
embedder_name = os.getenv('EMBEDDER_NAME')
ranker_alpha = float(os.getenv('RANKER_ALPHA'))


bot = telebot.TeleBot(telegram_token)
embedder = LaBSEEmbedder(model_name=embedder_name)
ranker = ChunksReranker(embedder, alpha=ranker_alpha)
kb = QdrantKnowledgeBase(embedder, host=qdrant_host, port=qdrant_port)
assistant = RulesAssistant(model_name=assistant_name)

@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(
        message,
        "Привет! Я бот, отвечаю на вопросы по правилам настольной игры Oathsworn: Верные клятве.\n"
        "Напиши мне свой вопрос, и я сделаю все возможное, чтобы найти на него ответ."
    )

@bot.message_handler(func=lambda message: True)
def handle_question(message):
    user_question = message.text
    bot.reply_to(message, "Ваш вопрос обрабатывается...")
    try:
        answer = asyncio.run(response(user_question, system_prompt, embedder, kb, collection_name, ranker, assistant, max_new_tokens=300))        
        bot.reply_to(message, f"Ответ: {answer}", parse_mode="Markdown")
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {e}")

print("Бот запущен и готов к работе.")
bot.polling()
