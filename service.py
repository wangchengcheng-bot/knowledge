"""服务层：提供文件哈希计算、索引构建和构建状态检查功能。"""

import hashlib  # 计算文件内容的 MD5 哈希值，用于唯一标识文件
import os  # 读写文件、检查文件路径
import tempfile  # 创建临时文件

import streamlit as st  # Streamlit 前端组件
from orchestrator import build_index  # 调用编排层的索引构建函数

CHROMA_DIR = "./chroma_db"  # Chroma 向量库本地持久化目录

STANDARD_COLLECTIONS = {"langchain"}  # 系统默认集合名，不算作用户上传文件


def calculate_pdf_hash(file_bytes):
    """计算 PDF 文件内容的 MD5 哈希值。

    用于判断文件是否已经构建过索引：相同内容的文件会产生相同的哈希。

    Args:
        file_bytes: PDF 文件的二进制内容（bytes）。

    Returns:
        str: 32 位十六进制 MD5 哈希值，如 "a1b2c3d4..."。
    """
    return hashlib.md5(file_bytes).hexdigest()  # 对字节内容取 MD5，转十六进制字符串


def get_user_collection_names():
    """获取用户上传文件的集合名列表（排除系统默认集合）。

    Returns:
        set[str]: 用户文件的集合名集合，如 {"pdf_a1b2c3d4", "pdf_e5f6g7h8"}。

    Raises:
        RuntimeError: 连接或读取 Chroma 数据库失败时抛出。
    """
    try:
        import chromadb  # Chroma 底层客户端
        client = chromadb.PersistentClient(path=CHROMA_DIR)  # 连接本地持久化目录
        all_names = {c.name for c in client.list_collections()}  # 获取所有集合名
        return all_names - STANDARD_COLLECTIONS  # 排除系统默认集合（如 langchain），只返回用户文件集合
    except Exception as e:  # 连接失败或目录不存在
        raise RuntimeError(f"读取 Chroma 数据库失败: {e}")  # 抛出异常，由上层决定如何处理


def check_exists(file_bytes):
    """检查给定的 PDF 文件是否已经构建过索引。

    Args:
        file_bytes: PDF 文件的二进制内容（bytes）。

    Returns:
        tuple[str | None, str | None]:
            (collection_name, pdf_hash)
            - 如果已存在：返回 (集合名, 哈希值)
            - 如果不存在：返回 (None, 哈希值)
    """
    pdf_hash = calculate_pdf_hash(file_bytes)  # 计算 MD5 哈希
    try:
        existing = get_user_collection_names()  # 获取已在库中的所有用户文件集合
    except RuntimeError:  # 数据库还没初始化，没有集合
        existing = set()  # 视为空集合

    if pdf_hash in existing:  # 这个哈希对应的集合已存在
        return (pdf_hash, pdf_hash)  # 返回集合名和哈希值，表示已构建过
    return (None, pdf_hash)  # 未构建过，返回 (None, 哈希值)


def service_build(pdf_path_or_bytes, collection_name=None):
    """执行索引构建，支持文件路径和字节内容两种输入。

    Args:
        pdf_path_or_bytes: PDF 文件路径（str）或二进制内容（bytes）。
        collection_name: 目标 Chroma 集合名，为 None 时自动用文件内容哈希。

    Returns:
        int: 入库的向量条数。

    Raises:
        ValueError: 传入参数既不是路径也不是 bytes 内容时抛出。
    """
    if isinstance(pdf_path_or_bytes, str):  # 传进来的是文件路径
        pdf_path = pdf_path_or_bytes  # 直接用这个路径
    elif isinstance(pdf_path_or_bytes, bytes):  # 传进来的是 bytes（如 st.file_uploader 返回的内容）
        pdf_hash = calculate_pdf_hash(pdf_path_or_bytes)  # 计算文件内容哈希
        if collection_name:  # 如果调用方指定了集合名
            final_collection = collection_name  # 使用指定的集合名
        else:  # 未指定时
            final_collection = pdf_hash  # 用文件哈希作为集合名
        # 将 bytes 内容写入临时文件，供 PDF 加载器读取
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")  # 不自动删除，后缀 .pdf
        tmp.write(pdf_path_or_bytes)  # 写入 PDF 二进制
        tmp.close()  # 关闭文件（但保留在磁盘上）
        pdf_path = tmp.name  # 获取临时文件路径
    else:  # 不是 str 也不是 bytes
        raise ValueError("pdf_path_or_bytes 必须是文件路径字符串或字节内容")  # 报错

    # 如果还未确定集合名（路径模式且未指定）
    if not collection_name:  # collection_name 还是 None
        collection_name = final_collection  # 使用之前计算的哈希

    vector_count = build_index(pdf_path, collection_name)  # 调用编排层构建索引

    if isinstance(pdf_path_or_bytes, bytes):  # 用了临时文件
        try:
            os.unlink(pdf_path)  # 清理临时文件
        except Exception:  # 删除失败（如文件被占用）
            pass  # 忽略，临时文件最终会被系统回收

    return vector_count  # 返回入库向量条数