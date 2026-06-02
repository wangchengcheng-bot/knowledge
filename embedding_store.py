from langchain_community.embeddings import DashScopeEmbeddings  # 通义千问的向量化工具
from langchain_community.vectorstores import Chroma  # Chroma向量数据库
from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL, CHROMA_PERSIST_DIR  # 导入配置

# 向量化模型
def create_embeddings():
    return DashScopeEmbeddings(  # 创建向量化模型实例
        model=EMBEDDING_MODEL,  # 使用 text-embedding-v1 模型
        dashscope_api_key=DASHSCOPE_API_KEY,  # 传入API Key用于鉴权
    )

# 向量存储
def create_vectorstore(chunks, embeddings=None):
    if embeddings is None:  # 如果没有传入向量化模型
        embeddings = create_embeddings()  # 就自动创建一个

    vectorstore = Chroma.from_documents(  # 把文本块转成向量并存入Chroma
        documents=chunks,  # 要存储的文本块
        embedding=embeddings,  # 用哪个向量化模型
        persist_directory=CHROMA_PERSIST_DIR,  # 存到本地哪个目录（数据会持久化到硬盘）
        collection_metadata={"hnsw:space": "cosine"},  # 用余弦相似度代替默认的L2距离
    )
    return vectorstore  # 返回向量库对象，同时也保存到硬盘了

# 从硬盘加载向量库
def load_vectorstore(embeddings=None):
    if embeddings is None:  # 如果没有传入向量化模型
        embeddings = create_embeddings()  # 自动创建一个

    vectorstore = Chroma(  # 从本地硬盘加载已有的向量库
        embedding_function=embeddings,  # 必须和存储时用同一个模型
        persist_directory=CHROMA_PERSIST_DIR,  # 从哪个目录读取
        collection_metadata={"hnsw:space": "cosine"},  # 与存储时一致，使用余弦相似度
    )
    return vectorstore  # 返回加载好的向量库
