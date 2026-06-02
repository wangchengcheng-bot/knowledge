"""RAG 问答模块：创建 LLM、提示词模板和文档格式化器。"""

from langchain_openai import ChatOpenAI  # OpenAI兼容的对话模型（通义千问通过兼容接口调用）
from langchain_core.prompts import ChatPromptTemplate  # 对话模板：定义 system/human 消息格式
from config import DASHSCOPE_API_KEY, LLM_MODEL  # 导入API Key和模型名

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 通义千问的OpenAI兼容端点地址

SYSTEM_PROMPT = """你是一个专业的知识助手。请根据以下参考资料回答问题。要求：
1. 优先使用参考资料中的信息回答。
2. 如果参考资料中找不到相关信息，请如实说明"参考资料中未包含此问题的相关信息"。
3. 回答要准确、简洁、有条理。

参考资料：
{context}"""  # {context} 是占位符，运行时会被替换成检索到的文档内容


def format_docs(docs):
    """将多个文档块拼接为一个字符串，用分隔线隔开。

    Args:
        docs: Document 对象列表（检索结果）。

    Returns:
        str: 拼接后的文档内容字符串，各块之间用 "\\n\\n---\\n\\n" 分隔。
    """
    return "\n\n---\n\n".join([d.page_content for d in docs])  # 把多个文档块拼成一段，用 --- 分隔


def create_llm():
    """创建通义千问对话模型实例。

    通过 OpenAI 兼容接口调用通义千问，使用 qwen-turbo 模型，
    temperature=0.3 保证回答稳定且有一定灵活性。

    Returns:
        ChatOpenAI: LLM 实例。
    """
    return ChatOpenAI(  # 创建对话模型实例
        model=LLM_MODEL,  # 模型名，如 qwen-turbo
        api_key=DASHSCOPE_API_KEY,  # 通义千问的API Key
        base_url=DASHSCOPE_BASE_URL,  # 指定通义千问兼容端点
        temperature=0.3,  # 温度：0~1，越低回答越稳定，越高越有创意
    )


def create_prompt():
    """创建 RAG 问答的提示词模板。

    包含 system 消息（角色设定 + 参考资料）和 human 消息（用户问题），
    其中 {context} 和 {question} 是运行时动态填充的占位符。

    Returns:
        ChatPromptTemplate: 提示词模板。
    """
    return ChatPromptTemplate.from_messages([  # 用消息列表构建提示词模板
        ("system", SYSTEM_PROMPT),  # system 消息：告诉模型它的角色和规则
        ("human", "{question}"),  # human 消息：用户的提问，{question} 是占位符
    ])