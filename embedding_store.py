from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL, CHROMA_PERSIST_DIR


def create_embeddings():
    return DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )


def create_vectorstore(chunks, embeddings=None):
    if embeddings is None:
        embeddings = create_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"[向量存储] 已存储 {vectorstore._collection.count()} 条向量到 {CHROMA_PERSIST_DIR}")
    return vectorstore


def load_vectorstore(embeddings=None):
    if embeddings is None:
        embeddings = create_embeddings()

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"[向量加载] 已从 {CHROMA_PERSIST_DIR} 加载 {vectorstore._collection.count()} 条向量")
    return vectorstore
