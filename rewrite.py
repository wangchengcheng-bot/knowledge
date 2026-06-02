"""查询改写模块：利用 LLM 将含指代词的模糊问题补全为独立完整的问题。

典型场景：
- 用户先问"通义千问是什么？"，再追问"它有什么特点？"
- 改写后将"它有什么特点？"变成"通义千问有什么特点？"
- 这样向量检索时才能命中相关文档。
"""

from langchain_core.output_parsers import StrOutputParser  # 输出解析器：把 LLM 的输出转成纯字符串
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示词模板：支持 system/user 消息
from rag_qa import create_llm  # 复用一个轻量 LLM (qwen-turbo) 做改写

_rewrite_llm = None  # 全局缓存：改写用的 LLM 只初始化一次，避免重复创建

# 查询改写提示词：让 LLM 把带有"它""这个"等指代词的问句补全为完整问题
REWRITE_SYSTEM_PROMPT = (
    "你是一个问题改写助手。你的任务是根据对话历史，把用户当前含有指代词"
    "（如'它'、'这个'、'那种'、'上面提到的'等）的模糊问题，改写成完整、独立、"
    "不依赖上下文就可以理解的问题。如果用户问题本身已经足够清晰，不需要改写，"
    "就原样返回。只返回改写后的问题，不要加任何解释或额外文字。"
)


def rewrite_query(question: str, chat_history: list = None) -> str:
    """根据对话历史，将用户含指代词的模糊问题改写为完整问题。

    改写逻辑：
    1. 如果没有对话历史，直接返回原问题。
    2. 取最近 2 轮（4 条消息）作为上下文，避免 token 过长。
    3. 调用 LLM 进行指代消解，失败时回退到原问题。

    Args:
        question: 用户当前输入的问题。
        chat_history: 对话历史，格式为 [{"role": "user"/"assistant", "content": "..."}, ...]。

    Returns:
        str: 改写后的完整问题，或原问题（无需改写 / 改写失败时）。
    """
    if not chat_history or len(chat_history) == 0:  # 首轮对话，没有历史
        return question  # 不需要改写，原样返回

    global _rewrite_llm  # 使用全局缓存
    if _rewrite_llm is None:  # 第一次调用
        _rewrite_llm = create_llm()  # 初始化改写专用 LLM（qwen-turbo）

    history_text = ""  # 把最近几轮对话拼成纯文本
    for msg in chat_history[-4:]:  # 只取最近 2 轮（4 条消息）避免上下文过长
        role_label = "用户" if msg["role"] == "user" else "助手"  # 标注发言人
        history_text += f"{role_label}：{msg['content']}\n"  # 逐条拼接

    prompt = ChatPromptTemplate.from_messages([  # 构建改写提示词
        ("system", REWRITE_SYSTEM_PROMPT),  # 系统指令：告诉 LLM 改写规则
        ("user", "对话历史：\n{history}\n当前问题：{question}\n\n请给出改写后的问题："),  # 用户消息：传入历史和当前问题
    ])

    try:
        chain = prompt | _rewrite_llm | StrOutputParser()  # 构建 LCEL 链：提示词 → LLM → 字符串解析
        rewritten = chain.invoke({"history": history_text.strip(), "question": question})  # 执行改写
        rewritten = rewritten.strip().strip("。").strip()  # 去掉 LLM 可能多带的句号和首尾空白
        return rewritten if rewritten else question  # 如果模型返回空就原文兜底
    except Exception:  # LLM 调用失败（网络、API 等）
        return question  # 改写失败时原样返回，不影响主流程