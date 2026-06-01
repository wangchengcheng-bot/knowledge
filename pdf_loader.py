from langchain_community.document_loaders import PyMuPDFLoader  # PDF读取器：比PyPDF2更稳定，中文效果好
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 智能文本分割器：按段落→句子→字符逐级切分
from config import CHUNK_SIZE, CHUNK_OVERLAP  # 导入分块大小和重叠字符数配置


def load_pdf(file_path: str):
    loader = PyMuPDFLoader(file_path)  # 创建PDF加载器，传入文件路径
    documents = loader.load()  # 执行加载，每一页变成一个 Document 对象
    print(f"[PDF加载] 共加载 {len(documents)} 页文档")  # 打印加载了多少页
    return documents  # 返回所有页的 Document 列表


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(  # 创建递归文本分割器
        chunk_size=CHUNK_SIZE,  # 每块最多 1000 个字符
        chunk_overlap=CHUNK_OVERLAP,  # 相邻块之间重叠 200 个字符
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],  # 分隔符优先级：先按段落→换行→句号→分号→逗号→空格→字符
        keep_separator=False,  # 不保留分隔符在文本块里
    )
    chunks = text_splitter.split_documents(documents)  # 执行分割，把长文档切成小块
    print(f"[文本分割] 共生成 {len(chunks)} 个文本块")  # 打印切了多少块
    return chunks  # 返回切好的文本块列表


def load_and_split_pdf(file_path: str):
    documents = load_pdf(file_path)  # 第一步：加载PDF
    chunks = split_documents(documents)  # 第二步：分割文本
    return chunks  # 返回最终切好的文本块
