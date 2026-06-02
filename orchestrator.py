from config import CHUNK_OVERLAP, CHUNK_SIZE  # 导入分块配置常量
from config import CHROMA_PERSIST_DIR  # 向量库存储目录
from embedding_store import create_embeddings, create_vectorstore, load_vectorstore  # 向量存储模块
from pdf_loader import load_and_split_pdf  # PDF加载+分割模块
from rag_qa import create_llm, create_prompt, format_docs  # RAG问答模块
from rewrite import rewrite_query  # 查询改写：消解指代词
from langchain_core.output_parsers import StrOutputParser  # 输出解析器
from langchain_core.runnables import RunnablePassthrough  # 管道直通器

_embeddings_instance = None  # 全局缓存：向量化模型只初始化一次

"""
编排层
"""

def _get_embeddings():
    global _embeddings_instance  # 使用全局变量缓存
    if _embeddings_instance is None:  # 第一次调用时创建
        _embeddings_instance = create_embeddings()  # 初始化向量化模型
    return _embeddings_instance  # 返回缓存实例

# 向量索引构建
def build_index(pdf_path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> int:
    import config  # 动态修改运行时的配置值
    config.CHUNK_SIZE = chunk_size  # 覆盖分块大小
    config.CHUNK_OVERLAP = chunk_overlap  # 覆盖重叠字符数

    chunks = load_and_split_pdf(pdf_path)  # 阶段1：加载PDF并按段落/句子切块
    embeddings = _get_embeddings()  # 阶段2：获取向量化模型
    vectorstore = create_vectorstore(chunks, embeddings)  # 阶段3：向量化并持久化到 chroma_db/

    return vectorstore._collection.count()  # 返回入库的向量条数


def query(question: str, stream: bool = False, chat_history: list = None):
    rewritten_question = rewrite_query(question, chat_history)  # 查询改写：消解指代词

    embeddings = _get_embeddings()  # 加载向量化模型
    vectorstore = load_vectorstore(embeddings)  # 从硬盘加载向量库
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # 转为检索器（每次取最相关4条）
    llm = create_llm()  # 创建大模型实例
    prompt = create_prompt()  # 创建提示词模板

    chain_head = {  # 链的前半部分：检索+格式化+参数传递
        "context": retriever | format_docs,  # 检索到的文档
        "question": RunnablePassthrough(),  # 用户问题原样传递
    }

    if stream:  # 流式输出：不挂 StrOutputParser，保留 chunk 对象以便逐 token 提取
        rag_chain = chain_head | prompt | llm
        for chunk in rag_chain.stream(rewritten_question):  # 用改写后的问题检索+生成
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield token
    else:  # 同步输出：完整执行后一次性返回
        rag_chain = chain_head | prompt | llm | StrOutputParser()
        return rag_chain.invoke(rewritten_question)  # 用改写后的问题


# 获取向量数
def get_vector_count() -> int:
    try:
        import chromadb  # 直接操作 Chroma 底层客户端
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)  # 连本地的持久化目录
        collection = client.get_or_create_collection("langchain")  # 获取集合
        return collection.count()  # 返回有多少条向量
    except Exception:  # 目录还不存在或连接失败
        return 0  # 返回 0 表示还没有向量数据

# 判断Chroma库是否就绪
def is_ready() -> bool:
    return get_vector_count() > 0  # 向量数 > 0 就是就绪
