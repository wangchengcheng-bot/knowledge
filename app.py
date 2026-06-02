import tempfile  # 创建临时文件（存放用户上传的 PDF）
import streamlit as st  # Streamlit 前端框架

from service import rag_service  # 服务层是前端唯一入口，不碰任何模块


st.set_page_config(  # Streamlit 页面全局设置
    page_title="RAG 知识库问答",  # 浏览器标签页标题
    page_icon="📚",  # 标签页图标
    layout="wide",  # 宽屏布局
    initial_sidebar_state="expanded",  # 侧边栏默认展开
)

st.title("📚 RAG 知识库问答系统")  # 页面大标题
st.caption("基于通义千问 + Chroma 向量库 | LCEL 链式构建")  # 标题下方小字说明


def init_session():
    defaults = {  # 页面首次加载时创建默认会话状态
        "chat_history": [],  # 聊天记录：存用户和 AI 的消息列表
        "index_ready": False,  # 知识库是否就绪
    }
    for k, v in defaults.items():  # 遍历默认项
        if k not in st.session_state:  # 如果还没初始化
            st.session_state[k] = v  # 写入默认值


def main():
    init_session()  # 每次 Streamlit rerun 时确保会话状态正确

    # ========== 侧边栏 ==========
    with st.sidebar:  # 所有控制项放在侧边栏
        st.header("📂 知识库管理")  # 侧边栏标题

        uploaded_file = st.file_uploader(  # PDF 上传组件
            "上传 PDF 文档",  # 标签
            type=["pdf"],  # 只接受 PDF
            help="支持中文 PDF，基于 PyMuPDF 解析",  # 悬停说明
        )

        st.divider()  # 分割线

        if uploaded_file is not None:  # 已上传文件才显示按钮
            if st.button("🔨 构建向量索引", type="primary", use_container_width=True):  # 构建按钮
                with st.spinner("正在处理 PDF..."):  # 加载动画
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:  # 创建临时文件
                        tmp.write(uploaded_file.getvalue())  # 写入上传内容
                        tmp_path = tmp.name  # 拿到路径
                    result = rag_service.build_index(tmp_path)  # 委托服务层构建，分块参数使用后端默认值

                if result["success"]:  # 构建成功
                    st.success(f"✅ 索引构建完成！共 {result['vector_count']} 条向量")  # 绿色提示
                    st.session_state.index_ready = True  # 标记就绪
                    st.rerun()  # 刷新页面
                else:  # 构建失败
                    st.error(f"❌ 构建失败：{result['error']}")  # 红色错误

        st.divider()  # 分割线

        st.subheader("📊 当前状态")  # 状态面板
        status = rag_service.get_status()  # 从服务层获取状态
        if status["index_ready"]:  # 更新会话状态
            st.session_state.index_ready = True

        col1, col2 = st.columns(2)  # 两列布局
        with col1:  # 左列
            st.metric("向量条数", status["vector_count"])  # 显示向量数
        with col2:  # 右列
            st.metric("向量来源",  # 显示数据来源
                      "已有索引" if status["vector_count"] > 0 and uploaded_file is None  # 已有数据
                      else ("新文档" if uploaded_file else "无"))  # 新文档 / 无

        if status["index_ready"]:  # 就绪
            st.success("✅ 知识库就绪")
        else:  # 未就绪
            st.warning("⚠️ 请先上传 PDF 构建索引")

        st.divider()  # 分割线

        st.caption(f"LLM: qwen-turbo | Embedding: text-embedding-v1")  # 模型信息
        if st.button("🗑️ 清除对话", use_container_width=True):  # 清除按钮
            st.session_state.chat_history = []  # 清空聊天记录
            st.rerun()  # 刷新

    # ========== 主区域：聊天界面 ==========
    for msg in st.session_state.chat_history:  # 渲染历史消息
        with st.chat_message(msg["role"]):  # 按角色显示气泡
            st.markdown(msg["content"])  # 渲染内容

    if prompt_input := st.chat_input(  # 聊天输入框
        "输入你的问题...",  # 占位文字
        disabled=not st.session_state.index_ready  # 知识库未就绪时禁用
    ):
        st.session_state.chat_history.append({"role": "user", "content": prompt_input})  # 保存用户消息

        with st.chat_message("user"):  # 用户气泡
            st.markdown(prompt_input)

        with st.chat_message("assistant"):  # AI 气泡
            message_placeholder = st.empty()  # 占位，用于流式更新
            full_response = ""  # 累积完整回答

            for token in rag_service.ask_stream(prompt_input, st.session_state.chat_history[:-1]):  # 通过服务层流式获取回答，传入历史消息用于查询改写
                full_response += token  # 逐 token 拼接
                message_placeholder.markdown(full_response + "▌")  # 实时显示（光标闪烁）

            message_placeholder.markdown(full_response)  # 最终显示完整回答

        st.session_state.chat_history.append(  # 保存 AI 回答
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":  # 直接运行时启动
    main()
