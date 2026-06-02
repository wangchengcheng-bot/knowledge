"""Streamlit 前端：PDF 上传、索引构建、RAG 问答交互界面。"""

import streamlit as st  # 前端框架：构建 Web 交互界面
from service import check_exists, calculate_pdf_hash, service_build  # 导入服务层函数
from orchestrator import is_ready, query  # 导编排层：状态检查 + 问答
from orchestrator import get_vector_count  # 获取知识库向量条数

st.set_page_config(page_title="知识库 RAG 问答系统", page_icon="📚", layout="wide")  # 页面标题、图标和宽屏布局

st.title("知识库 RAG 问答系统")  # 大标题

# ---------- 侧边栏：文件上传 & 索引管理 ----------
with st.sidebar:  # 用 Streamlit 的侧边栏布局
    st.header("📄 文档管理")  # 侧边栏标题

    # 显示当前知识库状态
    ready = is_ready()  # 检查知识库是否有数据
    if ready:  # 有数据
        st.success(f"知识库就绪 —— 共 {get_vector_count()} 条向量")  # 绿色提示 + 向量数量
    else:  # 无数据
        st.info("知识库为空，请上传 PDF 文件")  # 蓝色提示

    uploaded_file = st.file_uploader("选择 PDF 文件", type=["pdf"], key="pdf_uploader")  # 文件上传控件：只接受 .pdf

    if uploaded_file is not None:  # 用户已经选好了文件
        file_bytes = uploaded_file.getvalue()  # 读取文件的二进制内容
        file_hash = calculate_pdf_hash(file_bytes)  # 计算文件哈希
        collection_name, _ = check_exists(file_bytes)  # 检查是否已构建过

        if collection_name is not None:  # 该文件已经构建过
            st.warning(f"该文件已存在于知识库中，无需重复构建。\n\n文件哈希：`{file_hash}`")  # 黄色警告提示
        else:  # 新文件，未构建过
            st.info(f"新文件，哈希：`{file_hash}`")  # 显示文件哈希

        if st.button("构建索引", type="primary", use_container_width=True):  # 大大的蓝色按钮，撑满宽度
            if collection_name is not None:  # 用户点击了按钮，但文件已存在
                st.warning("该文件已构建过索引，无需重复操作。")  # 提醒不需要构建
            else:  # 新文件，可以构建
                with st.spinner("正在处理 PDF 并构建索引..."):  # 显示加载动画
                    try:
                        count = service_build(file_bytes, collection_name=file_hash)  # 执行索引构建
                        st.success(f"索引构建完成！共 {count} 条向量入库。")  # 成功绿提示
                        st.rerun()  # 刷新页面，更新知识库状态
                    except Exception as e:  # 构建过程出错
                        st.error(f"构建失败：{e}")  # 红色错误提示

# ---------- 主区域：对话问答 ----------
# 初始化 session_state 中存储的对话历史
if "messages" not in st.session_state:  # 第一次运行，还没有消息列表
    st.session_state.messages = []  # 初始化为空列表

# 渲染历史消息（从上到下显示之前的对话记录）
for msg in st.session_state.messages:  # 遍历历史消息
    with st.chat_message(msg["role"]):  # 根据角色渲染气泡（user/assistant）
        st.markdown(msg["content"])  # 用 Markdown 格式显示消息内容

if prompt := st.chat_input("请输入您的问题..."):  # 底部聊天输入框，有输入时 prompt 不为空
    st.session_state.messages.append({"role": "user", "content": prompt})  # 将用户消息存入历史
    with st.chat_message("user"):  # 渲染用户消息气泡
        st.markdown(prompt)  # 显示用户输入的内容

    with st.chat_message("assistant"):  # 渲染助手消息气泡
        if not is_ready():  # 知识库还没有数据
            response_text = "请先在侧边栏上传 PDF 文件并构建索引。"  # 提示用户先上传
            st.markdown(response_text)  # 显示提示
        else:  # 知识库就绪
            chat_history_dicts = [  # 从 session_state 提取对话历史（排除当前消息）
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]  # 排除最新一条（当前用户消息）
            ]
            with st.spinner("正在检索知识库并生成回答..."):  # 显示等待动画
                try:
                    response_placeholder = st.empty()  # 空占位符，用于流式更新内容
                    full_response = ""  # 累积完整回答

                    # 流式调用问答，逐 token 显示
                    for token in query(prompt, stream=True, chat_history=chat_history_dicts):
                        full_response += token  # 拼接 token
                        response_placeholder.markdown(full_response + "▌")  # 显示累积内容 + 闪烁光标
                    response_placeholder.markdown(full_response)  # 全部完成后去掉光标，显示最终结果
                except Exception as e:  # 问答过程出错
                    st.error(f"处理出错：{e}")  # 显示错误信息
                    full_response = f"处理出错：{e}"  # 错误信息作为回复

        st.session_state.messages.append({"role": "assistant", "content": full_response})  # 将助手回复存入历史