from typing import Iterator  # 类型标注：流式输出的迭代器类型

from orchestrator import build_index as _build_index  # 编排层构建函数
from orchestrator import query as _query  # 编排层查询函数（统一的同步/流式入口）
from orchestrator import get_vector_count, is_ready  # 编排层状态函数
"""
对外的接口
"""

class RAGService:
    # 对外接口，传入PDF路径，进行索引构建
    def build_index(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> dict:
        if not pdf_path:  # 校验：路径不能为空
            return {"success": False, "error": "PDF 路径不能为空"}
        try:
            count = _build_index(pdf_path, chunk_size, chunk_overlap)  # 委托编排层执行构建
            return {"success": True, "vector_count": count}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # 传入问题，返回流式结果
    def ask_stream(self, question: str) -> Iterator[str]:
        if not question or not question.strip():  # 校验：问题不能为空
            yield "【错误】问题不能为空"
            return
        try:
            for token in _query(question, stream=True):  # 委托编排层流式查询
                yield token  # 逐个透传 token 给前端
        except Exception as e:
            yield f"\n【错误】{str(e)}"

    # 获取Chroma向量库向量条数、是否就绪
    def get_status(self) -> dict:
        return {  # 获取系统状态
            "vector_count": get_vector_count(),  # 向量条数
            "index_ready": is_ready(),  # 是否就绪
        }


rag_service = RAGService()  # 全局单例：前端唯一入口
