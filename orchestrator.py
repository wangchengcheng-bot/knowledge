"""编排层：协调 PDF 加载、向量化、检索、LLM 生成各模块，完成 RAG 全流程。

核心流程：
1. build_index: PDF → 分块 → 向量化 → 存入 Chroma（按文件哈希隔离集合）
2. query: 问题改写 → 跨所有集合检索 → LLM 生成回答
"""

from config import CHUNK_OVERLAP, CHUNK_SIZE  # 导入分块配置常量
from config import CHROMA_PERSIST_DIR  # 向量库存储目录
from embedding_store import create_embeddings, create_vectorstore, load_vectorstore  # 向量存储模块
from pdf_loader import load_and_split_pdf  # PDF加载+分割模块
from rag_qa import create_llm, create_prompt, format_docs  # RAG问答模块
from rewrite import rewrite_query  # 查询改写：消解指代词
from langchain_core.output_parsers import StrOutputParser  # 输出解析器：把 LLM 输出转成字符串
from langchain_core.runnables import RunnablePassthrough, RunnableLambda  # 管道直通器 + 自定义可运行节点

_embeddings_instance = None  # 全局缓存：向量化模型只初始化一次，避免重复创建


def _get_embeddings():
    """获取全局缓存的向量化模型实例，首次调用时创建。

    Returns:
        DashScopeEmbeddings: 向量化模型实例。
    """
    global _embeddings_instance  # 使用全局变量缓存
    if _embeddings_instance is None:  # 第一次调用时创建
        _embeddings_instance = create_embeddings()  # 初始化向量化模型
    return _embeddings_instance  # 返回缓存实例


def _get_all_collection_names():
    """获取 Chroma 数据库中所有集合的名称列表。

    Returns:
        list[str]: 集合名列表，如 ["pdf_a1b2c3d4", "pdf_e5f6g7h8"]。
    """
    import chromadb  # 直接操作 Chroma 底层客户端
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)  # 连接本地持久化目录
    return [c.name for c in client.list_collections()]  # 遍历所有集合取名字


def _search_all_collections(query_text: str, k: int = 4):
    """跨所有 Chroma 集合执行相似度检索，合并各集合结果后取 top-k。

    流程：
    1. 列出所有集合名
    2. 对每个集合分别做 similarity_search_with_score
    3. 合并所有结果，按余弦距离升序排序
    4. 返回前 k 个文档

    Args:
        query_text: 查询文本（改写后的完整问题）。
        k: 每个集合内检索的条数，默认 4。

    Returns:
        list[Document]: 合并去重排序后的 top-k 文档列表。
    """
    embeddings = _get_embeddings()  # 获取向量化模型
    all_names = _get_all_collection_names()  # 列出所有集合
    if not all_names:  # 没有任何集合
        return []  # 返回空列表

    all_results = []  # 存放所有集合的检索结果，每个元素是 (Document, distance)
    for coll_name in all_names:  # 遍历每个集合
        try:
            vs = load_vectorstore(embeddings, coll_name)  # 加载该集合的向量库
            results = vs.similarity_search_with_score(query_text, k=k)  # 在该集合内检索 top-k
            all_results.extend(results)  # 收集结果，每个元素是 (Document, score)
        except Exception:  # 某个集合加载失败（如已损坏）
            continue  # 跳过该集合，继续处理其他集合

    if not all_results:  # 所有集合都没有结果
        return []

    all_results.sort(key=lambda x: x[1])  # 按余弦距离升序排列，距离越小越相似
    return [doc for doc, _score in all_results[:k]]  # 只返回文档本身，丢弃分数，取前 k 个


def build_index(pdf_path: str, collection_name: str = "langchain",
                chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> int:
    """构建向量索引：加载 PDF → 分块 → 向量化 → 存入 Chroma 指定集合。

    Args:
        pdf_path: PDF 文件的本地路径。
        collection_name: 目标 Chroma 集合名（通常为文件内容哈希）。
        chunk_size: 文本分块大小，默认 1000。
        chunk_overlap: 相邻块重叠字符数，默认 200。

    Returns:
        int: 入库的向量条数。
    """
    import config  # 动态修改运行时的配置值
    config.CHUNK_SIZE = chunk_size  # 覆盖分块大小
    config.CHUNK_OVERLAP = chunk_overlap  # 覆盖重叠字符数

    chunks = load_and_split_pdf(pdf_path)  # 阶段1：加载PDF并按段落/句子切块
    embeddings = _get_embeddings()  # 阶段2：获取向量化模型
    vectorstore = create_vectorstore(chunks, embeddings, collection_name)  # 阶段3：向量化并持久化，用文件哈希做集合名

    return vectorstore._collection.count()  # 返回入库的向量条数


def query(question: str, stream: bool = False, chat_history: list = None):
    """执行 RAG 问答流程：改写 → 跨集合检索 → LLM 生成。

    流程：
    1. 利用对话历史改写问题（消解指代词）
    2. 跨所有 Chroma 集合检索最相关的 k 个文档块
    3. 将文档块拼接为上下文，与问题一起送入 LLM 生成回答
    4. 支持流式输出（stream=True）和同步输出（stream=False）

    Args:
        question: 用户输入的问题。
        stream: 是否流式输出，True 时为生成器逐 token 产出。
        chat_history: 对话历史，用于查询改写。

    Yields:
        str: 流式模式下逐个 token 产出。

    Returns:
        str: 同步模式下的完整回答。
    """
    rewritten_question = rewrite_query(question, chat_history)  # 查询改写：消解指代词

    custom_retriever = RunnableLambda(lambda q: _search_all_collections(q, k=4))  # 跨所有集合检索，合并后取 top-4
    llm = create_llm()  # 创建大模型实例
    prompt = create_prompt()  # 创建提示词模板

    chain_head = {  # 链的前半部分：检索后的文档 + 用户问题
        "context": custom_retriever | format_docs,  # 跨全部知识库检索 → 格式化拼接
        "question": RunnablePassthrough(),  # 用户问题原样传递，不经过任何处理
    }

    if stream:  # 流式输出：不挂 StrOutputParser，保留 chunk 对象以便逐 token 提取
        rag_chain = chain_head | prompt | llm  # 组装完整链：检索 + 提示词 + LLM
        for chunk in rag_chain.stream(rewritten_question):  # 用改写后的问题流式执行
            token = chunk.content if hasattr(chunk, "content") else str(chunk)  # 从 AIMessage 中提取文本
            yield token  # 逐个产出 token
    else:  # 同步输出：完整执行后一次性返回
        rag_chain = chain_head | prompt | llm | StrOutputParser()  # 组装链，末尾加字符串解析器
        return rag_chain.invoke(rewritten_question)  # 用改写后的问题执行，返回完整回答


def get_vector_count() -> int:
    """获取知识库中所有集合的向量总数。

    Returns:
        int: 所有集合的向量条数总和，异常时返回 0。
    """
    try:
        import chromadb  # 直接操作 Chroma 底层客户端
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)  # 连本地的持久化目录
        total = 0  # 累加计数器
        for coll in client.list_collections():  # 遍历所有集合
            total += coll.count()  # 累加每一个集合的向量条数
        return total  # 返回总和
    except Exception:  # 目录还不存在或连接失败
        return 0  # 返回 0 表示还没有向量数据


def is_ready() -> bool:
    """判断知识库是否就绪：任意集合中有数据即视为就绪。

    Returns:
        bool: 就绪返回 True，否则返回 False。
    """
    return get_vector_count() > 0  # 向量总数 > 0 就是就绪