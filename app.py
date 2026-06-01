import os
import tempfile
import streamlit as st
from langchain_core.runnables import RunnablePassthrough

from config import DASHSCOPE_API_KEY, CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from pdf_loader import load_and_split_pdf
from embedding_store import create_vectorstore, load_vectorstore, create_embeddings
from rag_qa import create_llm, create_prompt, format_docs


st.set_page_config(
    page_title="RAG 知识库问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 RAG 知识库问答系统")
st.caption("基于通义千问 + Chroma 向量库 | LCEL 链式构建")


@st.cache_resource
def get_embeddings():
    return create_embeddings()


def init_session():
    defaults = {
        "chat_history": [],
        "index_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_chroma_collection_count():
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_or_create_collection("langchain")
        return collection.count()
    except Exception:
        return 0


def build_index(file_path, chunk_size, chunk_overlap):
    with st.spinner("正在处理 PDF..."):
        import config
        config.CHUNK_SIZE = chunk_size
        config.CHUNK_OVERLAP = chunk_overlap
        chunks = load_and_split_pdf(file_path)

    status_text = st.empty()
    status_text.info(f"正在向量化 {len(chunks)} 个文本块...")

    embeddings = get_embeddings()
    create_vectorstore(chunks, embeddings)

    status_text.success(f"✅ 索引构建完成！共 {get_chroma_collection_count()} 条向量")
    st.session_state.index_ready = True
    st.rerun()


def create_rag_chain_from_store():
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = create_llm()
    prompt = create_prompt()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain


def main():
    init_session()

    with st.sidebar:
        st.header("📂 知识库管理")

        uploaded_file = st.file_uploader(
            "上传 PDF 文档",
            type=["pdf"],
            help="支持中文 PDF，基于 PyMuPDF 解析",
        )

        st.divider()

        st.subheader("⚙️ 分块参数")
        chunk_size = st.slider(
            "文本块大小",
            min_value=200,
            max_value=2000,
            value=CHUNK_SIZE,
            step=100,
            help="每个文本块的最大字符数",
        )
        chunk_overlap = st.slider(
            "重叠字符数",
            min_value=0,
            max_value=500,
            value=CHUNK_OVERLAP,
            step=50,
            help="相邻块之间的重叠字符数，防止语义断裂",
        )

        st.divider()

        if uploaded_file is not None:
            if st.button("🔨 构建向量索引", type="primary", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                build_index(tmp_path, chunk_size, chunk_overlap)

        st.divider()

        st.subheader("📊 当前状态")
        existing_count = get_chroma_collection_count()
        if existing_count > 0:
            st.session_state.index_ready = True

        col1, col2 = st.columns(2)
        with col1:
            st.metric("向量条数", existing_count)
        with col2:
            st.metric("向量来源",
                      "已有索引" if existing_count > 0 and uploaded_file is None
                      else ("新文档" if uploaded_file else "无"))

        if existing_count > 0:
            st.success("✅ 知识库就绪")
        else:
            st.warning("⚠️ 请先上传 PDF 构建索引")

        st.divider()

        st.caption(f"LLM: qwen-plus | Embedding: text-embedding-v1")
        if st.button("🗑️ 清除对话", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt_input := st.chat_input("输入你的问题...",
                                      disabled=not st.session_state.index_ready):
        st.session_state.chat_history.append({"role": "user", "content": prompt_input})

        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                rag_chain = create_rag_chain_from_store()

            message_placeholder = st.empty()
            full_response = ""

            for chunk in rag_chain.stream(prompt_input):
                full_response += chunk.content if hasattr(chunk, "content") else str(chunk)
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    main()
