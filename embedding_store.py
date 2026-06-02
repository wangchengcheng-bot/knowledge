"""向量存储模块：创建和管理 Chroma 向量数据库，支持按集合隔离存储。"""

from langchain_community.embeddings import DashScopeEmbeddings  # 通义千问的向量化工具
from langchain_community.vectorstores import Chroma  # Chroma向量数据库
from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL, CHROMA_PERSIST_DIR  # 导入配置


def create_embeddings():
    """创建通义千问的向量化模型实例。

    返回的 DashScopeEmbeddings 实例用于将文本转为 1024 维向量。

    Returns:
        DashScopeEmbeddings: 向量化模型实例。
    """
    return DashScopeEmbeddings(  # 创建向量化模型实例
        model=EMBEDDING_MODEL,  # 使用 text-embedding-v1 模型
        dashscope_api_key=DASHSCOPE_API_KEY,  # 传入API Key用于鉴权
    )


def create_vectorstore(chunks, embeddings=None, collection_name="langchain"):
    """将文本块向量化并存入 Chroma 指定集合。

    每个集合对应一个文件（以文件哈希命名），实现不同文档的隔离存储。
    索引使用余弦相似度（hnsw:space=cosine）而非默认的 L2 距离。

    Args:
        chunks: 待入库的文本块（Document 对象列表）。
        embeddings: 向量化模型，为 None 时自动创建。
        collection_name: Chroma 集合名，默认 "langchain"。

    Returns:
        Chroma: 向量数据库对象，数据已持久化到硬盘。
    """
    if embeddings is None:  # 如果没有传入向量化模型
        embeddings = create_embeddings()  # 就自动创建一个
    vectorstore = Chroma.from_documents(  # 把文本块转成向量并存入Chroma
        documents=chunks,  # 要存储的文本块
        embedding=embeddings,  # 用哪个向量化模型
        persist_directory=CHROMA_PERSIST_DIR,  # 存到本地哪个目录（数据会持久化到硬盘）
        collection_name=collection_name,  # 集合名：用文件哈希区分不同文档
        collection_metadata={"hnsw:space": "cosine"},  # 用余弦相似度代替默认的L2距离
    )
    return vectorstore  # 返回向量库对象，同时也保存到硬盘了


def load_vectorstore(embeddings=None, collection_name="langchain"):
    """从硬盘加载已有的向量库（指定集合）。

    加载时必须使用与存储时相同的向量化模型，否则查询结果无效。

    Args:
        embeddings: 向量化模型，为 None 时自动创建。
        collection_name: 要加载的 Chroma 集合名，默认 "langchain"。

    Returns:
        Chroma: 加载好的向量数据库对象。
    """
    if embeddings is None:  # 如果没有传入向量化模型
        embeddings = create_embeddings()  # 自动创建一个
    vectorstore = Chroma(  # 从本地硬盘加载已有的向量库
        embedding_function=embeddings,  # 必须和存储时用同一个模型
        persist_directory=CHROMA_PERSIST_DIR,  # 从哪个目录读取
        collection_name=collection_name,  # 指定要加载哪个集合
        collection_metadata={"hnsw:space": "cosine"},  # 与存储时一致，使用余弦相似度
    )
    return vectorstore  # 返回加载好的向量库