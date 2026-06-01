import os
import sys
from config import CHROMA_PERSIST_DIR
from pdf_loader import load_and_split_pdf
from embedding_store import create_vectorstore, load_vectorstore, create_embeddings
from rag_qa import create_rag_chain, ask


def build_index(pdf_path: str):
    chunks = load_and_split_pdf(pdf_path)
    create_vectorstore(chunks)
    print("[构建完成] 向量索引已创建")


def query_mode(questions: list[str]):
    embeddings = create_embeddings()
    vectorstore = load_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    rag_chain = create_rag_chain(retriever)

    for q in questions:
        ask(rag_chain, q)


def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print("  RAG 知识库问答系统 - 使用说明")
        print("=" * 60)
        print()
        print("  方式1: 构建向量索引")
        print(f"    python test_rag.py build <PDF文件路径>")
        print()
        print("  方式2: 交互问答")
        print(f"    python test_rag.py query")
        print()
        print("  方式3: 构建并问答")
        print(f"    python test_rag.py all <PDF文件路径>")
        print()
        return

    command = sys.argv[1]

    if command == "build":
        if len(sys.argv) < 3:
            print("请提供PDF文件路径: python test_rag.py build <PDF文件路径>")
            return
        pdf_path = sys.argv[2]
        if not os.path.exists(pdf_path):
            print(f"[错误] 文件不存在: {pdf_path}")
            return
        build_index(pdf_path)

    elif command == "query":
        if not os.path.exists(CHROMA_PERSIST_DIR):
            print(f"[错误] 向量库不存在，请先执行构建: python test_rag.py build <PDF文件路径>")
            return

        print("\n🔍 交互问答模式，输入你的问题（输入 quit 退出）\n")
        embeddings = create_embeddings()
        vectorstore = load_vectorstore(embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        rag_chain = create_rag_chain(retriever)

        while True:
            question = input("\n[请输入问题] > ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("再见！")
                break
            if not question:
                continue
            ask(rag_chain, question)

    elif command == "all":
        if len(sys.argv) < 3:
            print("请提供PDF文件路径: python test_rag.py all <PDF文件路径>")
            return
        pdf_path = sys.argv[2]
        if not os.path.exists(pdf_path):
            print(f"[错误] 文件不存在: {pdf_path}")
            return

        build_index(pdf_path)
        print()

        test_questions = [
            "请用一句话总结这篇文档的主要内容。",
            "文中提到了哪些关键要点？",
        ]
        query_mode(test_questions)

    else:
        print(f"[错误] 未知命令: {command}")


if __name__ == "__main__":
    main()
