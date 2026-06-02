"""PDF 处理模块：加载 PDF 并按语义边界切分文本块。"""

from langchain_community.document_loaders import PyMuPDFLoader  # PDF读取器：比PyPDF2更稳定，中文效果好
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 智能文本分割器：按段落→句子→字符逐级切分
from config import CHUNK_SIZE, CHUNK_OVERLAP  # 导入分块大小和重叠字符数配置


def load_pdf(file_path: str):
    """加载 PDF 文件，返回每一页的 Document 对象列表。

    Args:
        file_path: PDF 文件的本地路径。

    Returns:
        list[Document]: 每一页对应的 Document 对象。
    """
    loader = PyMuPDFLoader(file_path)  # 创建PDF加载器，传入文件路径
    documents = loader.load()  # 执行加载，每一页变成一个 Document 对象
    return documents  # 返回所有页的 Document 列表


def split_documents(documents):
    """将长文档按语义边界递归切分成小块。

    按优先级依次尝试：段落（\\n\\n）→ 换行（\\n）→ 句号（。）→ 分号（；）→ 逗号（，）→ 空格 → 字符。
    每个 chunk 不超过 CHUNK_SIZE，相邻 chunk 重叠 CHUNK_OVERLAP 个字符。

    Args:
        documents: load_pdf 返回的 Document 对象列表。

    Returns:
        list[Document]: 切分后的文本块列表。
    """
    text_splitter = RecursiveCharacterTextSplitter(  # 创建递归文本分割器
        chunk_size=CHUNK_SIZE,  # 每块最多 1000 个字符
        chunk_overlap=CHUNK_OVERLAP,  # 相邻块之间重叠 200 个字符
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],  # 分隔符优先级：先按段落→换行→句号→分号→逗号→空格→字符
        keep_separator=False,  # 不保留分隔符在文本块里
    )
    chunks = text_splitter.split_documents(documents)  # 执行分割，把长文档切成小块
    return chunks  # 返回切好的文本块列表


def load_and_split_pdf(file_path: str):
    """加载 PDF 并直接切分成文本块，一步到位。

    Args:
        file_path: PDF 文件的本地路径。

    Returns:
        list[Document]: 切分后的文本块列表。
    """
    documents = load_pdf(file_path)  # 第一步：加载PDF
    chunks = split_documents(documents)  # 第二步：分割文本
    return chunks  # 返回最终切好的文本块