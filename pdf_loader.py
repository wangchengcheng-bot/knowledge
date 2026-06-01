from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path: str):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(f"[PDF加载] 共加载 {len(documents)} 页文档")
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        keep_separator=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[文本分割] 共生成 {len(chunks)} 个文本块")
    return chunks


def load_and_split_pdf(file_path: str):
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    return chunks
