"""配置模块：从 .env 读取敏感信息，定义项目全局常量。"""

import os  # 读取系统环境变量
from dotenv import load_dotenv  # 加载 .env 文件里的配置

load_dotenv()  # 执行加载，把 .env 里的键值对变成环境变量

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 从环境变量读取通义千问的 API Key

# 安全检查：Key 不能为空，也不能是模板里的占位符
if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your_api_key_here":
    raise ValueError("请在 .env 文件中配置有效的 DASHSCOPE_API_KEY")  # 没配 Key 就报错，防止运行时才发现

EMBEDDING_MODEL = "text-embedding-v1"  # 向量化模型：把文字转成向量（1024维）
LLM_MODEL = "qwen-turbo"  # 对话模型：回答问题用，qwen-turbo 速度快成本低

CHROMA_PERSIST_DIR = "./chroma_db"  # 向量数据库本地存储目录（存到当前项目文件夹下）
CHUNK_SIZE = 1000  # 文本分块大小：每块最多 1000 个字符
CHUNK_OVERLAP = 200  # 相邻块重叠 200 个字符，避免一句话被切断在两块之间