from langchain_core.output_parsers import StrOutputParser  # 输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示词模板
from rag_qa import create_llm  # 复用一个轻量LLM做改写

_rewrite_llm = None  # 全局缓存：改写用的LLM只初始化一次

# 查询改写提示词：让LLM把带有"它""这个"等指代词的问句补全为完整问题
REWRITE_SYSTEM_PROMPT = (
    "你是一个问题改写助手。你的任务是根据对话历史，把用户当前含有指代词"
    "（如'它'、'这个'、'那种'、'上面提到的'等）的模糊问题，改写成完整、独立、"
    "不依赖上下文就可以理解的问题。如果用户问题本身已经足够清晰，不需要改写，"
    "就原样返回。只返回改写后的问题，不要加任何解释或额外文字。"
)


def rewrite_query(question: str, chat_history: list = None) -> str:
    if not chat_history or len(chat_history) == 0:
        return question  # 没有历史记录就原样返回

    global _rewrite_llm
    if _rewrite_llm is None:
        _rewrite_llm = create_llm()  # 复用一个轻量LLM（qwen-turbo）

    history_text = ""  # 把最近几轮对话拼成纯文本
    for msg in chat_history[-4:]:  # 只取最近2轮（4条消息）避免上下文过长
        role_label = "用户" if msg["role"] == "user" else "助手"
        history_text += f"{role_label}：{msg['content']}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_SYSTEM_PROMPT),
        ("user", "对话历史：\n{history}\n当前问题：{question}\n\n请给出改写后的问题："),
    ])

    try:
        chain = prompt | _rewrite_llm | StrOutputParser()
        rewritten = chain.invoke({"history": history_text.strip(), "question": question})
        rewritten = rewritten.strip().strip("。").strip()  # 去掉多余标点和空白
        return rewritten if rewritten else question  # 如果模型返回空就原文兜底
    except Exception:
        return question  # 改写失败时原样返回，不影响主流程
