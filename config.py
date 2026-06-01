import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your_api_key_here":
    raise ValueError("请在 .env 文件中配置有效的 DASHSCOPE_API_KEY")

EMBEDDING_MODEL = "text-embedding-v1"
LLM_MODEL = "qwen-turbo"

CHROMA_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
