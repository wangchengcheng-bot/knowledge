from langchain_openai import ChatOpenAI  # OpenAI兼容的对话模型（通义千问通过兼容接口调用）
from langchain_core.output_parsers import StrOutputParser  # 输出解析器：把模型返回的对象转成纯文本字符串
from langchain_core.runnables import RunnablePassthrough  # 管道直通器：把用户问题原样传递下去
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
    return "\n\n---\n\n".join([d.page_content for d in docs])  # 把多个文档块拼成一段，用 --- 分隔


def create_llm():
    return ChatOpenAI(  # 创建对话模型实例
        model=LLM_MODEL,  # 模型名，如 qwen-turbo
        api_key=DASHSCOPE_API_KEY,  # 通义千问的API Key
        base_url=DASHSCOPE_BASE_URL,  # 指定通义千问兼容端点
        temperature=0.3,  # 温度：0~1，越低回答越稳定，越高越有创意
    )


def create_prompt():
    return ChatPromptTemplate.from_messages([  # 用消息列表构建提示词模板
        ("system", SYSTEM_PROMPT),  # system 消息：告诉模型它的角色和规则
        ("human", "{question}"),  # human 消息：用户的提问，{question} 是占位符
    ])


def create_rag_chain(retriever, llm=None):
    if llm is None:  # 如果没有传入模型
        llm = create_llm()  # 自动创建一个

    prompt = create_prompt()  # 创建提示词模板

    rag_chain = (  # LCEL 管道式构建 RAG 链
        {"context": retriever | format_docs, "question": RunnablePassthrough()}  # 第1步：检索文档并格式化 + 传递问题
        | prompt  # 第2步：把检索结果和问题填入提示词模板
        | llm  # 第3步：调用大模型生成回答
        | StrOutputParser()  # 第4步：解析输出，提取纯文本
    )

    print("[RAG链] LCEL 检索增强生成链已构建")  # 提示链构建完成
    return rag_chain  # 返回构建好的链


def ask(rag_chain, question: str) -> str:
    print(f"\n{'='*60}")  # 打印分隔线
    print(f"[用户问题] {question}")  # 打印用户的问题
    print(f"{'='*60}")  # 打印分隔线
    answer = rag_chain.invoke(question)  # 调用链执行问答，invoke 是同步执行
    print(f"[AI回答] {answer}")  # 打印AI的回答
    print(f"{'='*60}")  # 打印分隔线
    return answer  # 返回回答内容
