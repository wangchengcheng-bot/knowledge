import os  # 文件路径操作
import sys  # 读取命令行参数
from config import CHROMA_PERSIST_DIR  # 向量库存储目录
from pdf_loader import load_and_split_pdf  # PDF加载+分割
from embedding_store import create_vectorstore, load_vectorstore, create_embeddings  # 向量存储相关
from rag_qa import create_rag_chain, ask  # RAG链+问答


def build_index(pdf_path: str):
    chunks = load_and_split_pdf(pdf_path)  # 加载PDF并切成文本块
    create_vectorstore(chunks)  # 把文本块向量化并存入Chroma
    print("[构建完成] 向量索引已创建")  # 提示完成


def query_mode(questions: list[str]):
    embeddings = create_embeddings()  # 创建向量化模型
    vectorstore = load_vectorstore(embeddings)  # 从硬盘加载已有向量库
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # 把向量库转为检索器（每次返回最相关的4条）
    rag_chain = create_rag_chain(retriever)  # 构建RAG问答链

    for q in questions:  # 遍历问题列表
        ask(rag_chain, q)  # 逐个提问并打印回答


def main():
    if len(sys.argv) < 2:  # 没有传命令，显示帮助信息
        print("=" * 60)
        print("  RAG 知识库问答系统 - 使用说明")
        print("=" * 60)
        print()
        print("  方式1: 构建向量索引")
        print(f"    python test_rag.py build <PDF文件路径>")  # 只构建索引
        print()
        print("  方式2: 交互问答")
        print(f"    python test_rag.py query")  # 交互式对话
        print()
        print("  方式3: 构建并问答")
        print(f"    python test_rag.py all <PDF文件路径>")  # 一键构建+测试
        print()
        return  # 打印完帮助就结束

    command = sys.argv[1]  # 取第一个参数作为命令

    if command == "build":  # 构建模式
        if len(sys.argv) < 3:  # 检查是否传了PDF路径
            print("请提供PDF文件路径: python test_rag.py build <PDF文件路径>")
            return  # 没有路径就提示并退出
        pdf_path = sys.argv[2]  # 取PDF文件路径
        if not os.path.exists(pdf_path):  # 检查文件是否存在
            print(f"[错误] 文件不存在: {pdf_path}")
            return  # 文件不存在就提示并退出
        build_index(pdf_path)  # 执行构建

    elif command == "query":  # 查询模式
        if not os.path.exists(CHROMA_PERSIST_DIR):  # 检查向量库是否存在
            print(f"[错误] 向量库不存在，请先执行构建: python test_rag.py build <PDF文件路径>")
            return  # 没构建过就不能问答

        print("\n🔍 交互问答模式，输入你的问题（输入 quit 退出）\n")
        embeddings = create_embeddings()  # 创建向量化模型
        vectorstore = load_vectorstore(embeddings)  # 加载已有向量库
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # 转为检索器，每次返回4条
        rag_chain = create_rag_chain(retriever)  # 构建RAG链

        while True:  # 循环问答，直到用户退出
            question = input("\n[请输入问题] > ").strip()  # 等待用户输入问题
            if question.lower() in ("quit", "exit", "q"):  # 输入 quit/exit/q 退出
                print("再见！")
                break  # 跳出循环
            if not question:  # 输入为空
                continue  # 跳过，重新等待输入
            ask(rag_chain, question)  # 执行问答

    elif command == "all":  # 一键模式：构建+测试
        if len(sys.argv) < 3:  # 检查是否传了PDF路径
            print("请提供PDF文件路径: python test_rag.py all <PDF文件路径>")
            return
        pdf_path = sys.argv[2]  # 取PDF文件路径
        if not os.path.exists(pdf_path):  # 检查文件是否存在
            print(f"[错误] 文件不存在: {pdf_path}")
            return

        build_index(pdf_path)  # 第一步：构建向量索引
        print()  # 空行

        test_questions = [  # 预设两个测试问题
            "请用一句话总结这篇文档的主要内容。",
            "文中提到了哪些关键要点？",
        ]
        query_mode(test_questions)  # 第二步：用测试问题验证

    else:  # 未知命令
        print(f"[错误] 未知命令: {command}")  # 提示命令不支持


if __name__ == "__main__":  # 只有直接运行这个脚本时才执行（被 import 时不执行）
    main()  # 启动主函数
