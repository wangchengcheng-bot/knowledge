import os  # 文件操作
import tempfile  # 创建临时文件（上传的PDF暂存）
import streamlit as st  # Streamlit 前端框架
from langchain_core.runnables import RunnablePassthrough  # 管道直通器

from config import DASHSCOPE_API_KEY, CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP  # 导入所有配置项
from pdf_loader import load_and_split_pdf  # PDF加载+分割
from embedding_store import create_vectorstore, load_vectorstore, create_embeddings  # 向量存储
from rag_qa import create_llm, create_prompt, format_docs  # RAG组件


st.set_page_config(  # Streamlit 页面全局设置
    page_title="RAG 知识库问答",  # 浏览器标签页标题
    page_icon="📚",  # 标签页图标
    layout="wide",  # 宽屏布局
    initial_sidebar_state="expanded",  # 侧边栏默认展开
)

st.title("📚 RAG 知识库问答系统")  # 页面大标题
st.caption("基于通义千问 + Chroma 向量库 | LCEL 链式构建")  # 标题下方小字说明


@st.cache_resource  # Streamlit 缓存装饰器：向量模型只初始化一次，跨会话共享，不重复加载
def get_embeddings():
    return create_embeddings()  # 创建并缓存向量化模型实例


def init_session():
    defaults = {  # 默认的会话状态值
        "chat_history": [],  # 聊天记录：存用户和AI的消息列表
        "index_ready": False,  # 知识库是否就绪
    }
    for k, v in defaults.items():  # 遍历每一项
        if k not in st.session_state:  # 如果这个key还没有初始化
            st.session_state[k] = v  # 写入默认值


def get_chroma_collection_count():
    try:
        import chromadb  # 直接导入 chromadb 查询底层数据
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)  # 连接本地持久化的Chroma
        collection = client.get_or_create_collection("langchain")  # 获取或创建集合
        return collection.count()  # 返回集合里有多少条向量
    except Exception:  # 如果出错了（比如目录还不存在）
        return 0  # 返回0，表示还没有向量


def build_index(file_path, chunk_size, chunk_overlap):
    with st.spinner("正在处理 PDF..."):  # 显示加载动画
        import config  # 导入配置模块
        config.CHUNK_SIZE = chunk_size  # 用用户在侧边栏调的分块大小覆盖配置
        config.CHUNK_OVERLAP = chunk_overlap  # 同样覆盖重叠字符数
        chunks = load_and_split_pdf(file_path)  # 加载PDF并切块

    status_text = st.empty()  # 创建一个占位区域，用于动态更新状态文字
    status_text.info(f"正在向量化 {len(chunks)} 个文本块...")  # 第一步提示：正在向量化

    embeddings = get_embeddings()  # 获取缓存的向量化模型
    create_vectorstore(chunks, embeddings)  # 向量化并存入Chroma

    status_text.success(f"✅ 索引构建完成！共 {get_chroma_collection_count()} 条向量")  # 构建完成提示
    st.session_state.index_ready = True  # 标记知识库已就绪
    st.rerun()  # 刷新页面，让UI更新状态


def create_rag_chain_from_store():
    embeddings = get_embeddings()  # 获取向量化模型
    vectorstore = load_vectorstore(embeddings)  # 从硬盘加载向量库
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # 转为检索器，每次返回4条最相关内容
    llm = create_llm()  # 创建大模型实例
    prompt = create_prompt()  # 创建提示词模板

    rag_chain = (  # LCEL 管道式构建
        {"context": retriever | format_docs, "question": RunnablePassthrough()}  # 检索+格式化+传问题
        | prompt  # 填入模板
        | llm  # 调用模型生成
    )
    return rag_chain  # 返回构建好的链


def main():
    init_session()  # 初始化会话状态（首次访问创建默认值）

    # ========== 侧边栏 ==========
    with st.sidebar:  # 在侧边栏里放所有控制项
        st.header("📂 知识库管理")  # 侧边栏标题

        uploaded_file = st.file_uploader(  # 文件上传组件
            "上传 PDF 文档",  # 标签文字
            type=["pdf"],  # 只允许上传 pdf
            help="支持中文 PDF，基于 PyMuPDF 解析",  # 悬停提示
        )

        st.divider()  # 分割线

        st.subheader("⚙️ 分块参数")  # 分块参数子标题
        chunk_size = st.slider(  # 滑块：调节分块大小
            "文本块大小",  # 标签
            min_value=200,  # 最小值 200 字符
            max_value=2000,  # 最大值 2000 字符
            value=CHUNK_SIZE,  # 默认值从配置读取
            step=100,  # 步长 100
            help="每个文本块的最大字符数",  # 悬停提示
        )
        chunk_overlap = st.slider(  # 滑块：调节重叠字符数
            "重叠字符数",  # 标签
            min_value=0,  # 最小值 0
            max_value=500,  # 最大值 500
            value=CHUNK_OVERLAP,  # 默认值从配置读取
            step=50,  # 步长 50
            help="相邻块之间的重叠字符数，防止语义断裂",  # 悬停提示
        )

        st.divider()  # 分割线

        if uploaded_file is not None:  # 如果用户上传了文件
            if st.button("🔨 构建向量索引", type="primary", use_container_width=True):  # 主要操作按钮，占满宽度
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:  # 创建临时文件
                    tmp.write(uploaded_file.getvalue())  # 把上传的文件内容写进临时文件
                    tmp_path = tmp.name  # 拿到临时文件的路径
                build_index(tmp_path, chunk_size, chunk_overlap)  # 用临时文件路径构建索引

        st.divider()  # 分割线

        st.subheader("📊 当前状态")  # 状态面板子标题
        existing_count = get_chroma_collection_count()  # 查询当前向量库里有多少条
        if existing_count > 0:  # 如果有数据
            st.session_state.index_ready = True  # 标记为就绪

        col1, col2 = st.columns(2)  # 创建两列布局
        with col1:  # 第一列
            st.metric("向量条数", existing_count)  # 显示指标：向量数量
        with col2:  # 第二列
            st.metric("向量来源",  # 显示指标：数据来源
                      "已有索引" if existing_count > 0 and uploaded_file is None  # 有数据且没上传新文件→已有索引
                      else ("新文档" if uploaded_file else "无"))  # 有上传→新文档，否则→无

        if existing_count > 0:  # 有向量数据
            st.success("✅ 知识库就绪")  # 绿色提示→就绪
        else:  # 没有向量数据
            st.warning("⚠️ 请先上传 PDF 构建索引")  # 黄色警告→需要构建

        st.divider()  # 分割线

        st.caption(f"LLM: qwen-turbo | Embedding: text-embedding-v1")  # 显示当前使用的模型
        if st.button("🗑️ 清除对话", use_container_width=True):  # 清除聊天记录按钮
            st.session_state.chat_history = []  # 清空聊天记录
            st.rerun()  # 刷新页面

    # ========== 主区域：聊天界面 ==========
    for msg in st.session_state.chat_history:  # 遍历历史消息
        with st.chat_message(msg["role"]):  # 根据角色显示不同的气泡（user/assistant）
            st.markdown(msg["content"])  # 渲染消息内容

    if prompt_input := st.chat_input(  # 聊天输入框，:= 是海象运算符（一边赋值一边判断）
        "输入你的问题...",  # 输入框占位文字
        disabled=not st.session_state.index_ready  # 知识库没就绪时输入框禁用
    ):
        st.session_state.chat_history.append({"role": "user", "content": prompt_input})  # 先把用户消息加入历史

        with st.chat_message("user"):  # 渲染用户消息气泡
            st.markdown(prompt_input)

        with st.chat_message("assistant"):  # 渲染AI消息气泡
            with st.spinner("思考中..."):  # 显示加载动画
                rag_chain = create_rag_chain_from_store()  # 构建RAG链

            message_placeholder = st.empty()  # 创建占位区域，用于流式输出
            full_response = ""  # 累积完整回答

            for chunk in rag_chain.stream(prompt_input):  # stream 流式调用，每次返回一小段token
                full_response += chunk.content if hasattr(chunk, "content") else str(chunk)  # 累加内容
                message_placeholder.markdown(full_response + "▌")  # 实时更新显示（光标闪烁效果）

            message_placeholder.markdown(full_response)  # 输出完整回答（去掉光标）

        st.session_state.chat_history.append(  # 把AI回答也加入历史
            {"role": "assistant", "content": full_response}  # 角色是 assistant，内容是完整回答
        )


if __name__ == "__main__":  # 只有直接运行这个脚本时才执行
    main()  # 启动 Streamlit 应用
