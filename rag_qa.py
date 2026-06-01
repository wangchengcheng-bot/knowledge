from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from config import DASHSCOPE_API_KEY, LLM_MODEL

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SYSTEM_PROMPT = """你是一个专业的知识助手。请根据以下参考资料回答问题。要求：
1. 优先使用参考资料中的信息回答。
2. 如果参考资料中找不到相关信息，请如实说明"参考资料中未包含此问题的相关信息"。
3. 回答要准确、简洁、有条理。

参考资料：
{context}"""


def format_docs(docs):
    return "\n\n---\n\n".join([d.page_content for d in docs])


def create_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        temperature=0.3,
    )


def create_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])


def create_rag_chain(retriever, llm=None):
    if llm is None:
        llm = create_llm()

    prompt = create_prompt()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("[RAG链] LCEL 检索增强生成链已构建")
    return rag_chain


def ask(rag_chain, question: str) -> str:
    print(f"\n{'='*60}")
    print(f"[用户问题] {question}")
    print(f"{'='*60}")
    answer = rag_chain.invoke(question)
    print(f"[AI回答] {answer}")
    print(f"{'='*60}")
    return answer
