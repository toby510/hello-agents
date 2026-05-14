#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGTool 简易调用示例
使用 hello_agents 内置 RAGTool，支持文档添加、搜索、智能问答
"""

from dotenv import load_dotenv
load_dotenv()

from hello_agents.tools.builtin.rag_tool import RAGTool


def demo_basic():
    """最简调用：创建 → 添加文档 → 搜索 → 问答"""
    rag = RAGTool(rag_namespace="demo")

    # 1. 添加文档（支持 PDF/Office/图片/代码等 40+ 格式）
    result = rag.add_document("01_MemoryTool_Basic_Operations.py")
    print(result)

    # 2. 查看统计
    print(rag.run({"action": "stats"}))

    # 3. 搜索
    print("\n🔍 基础搜索:")
    print(rag.search("记忆系统是如何工作的？", limit=3))

    # 4. 智能问答（检索 + LLM 生成答案）
    print("\n🤖 智能问答:")
    print(rag.ask("MemoryTool的add操作怎么用？"))

    return rag


def demo_run_interface():
    """通过 run() 字典方式调用（Agent 调用 Tool 的标准方式）"""
    rag = RAGTool(rag_namespace="demo_run")

    # 所有操作统一通过 run({"action": ...}) 调用
    rag.run({"action": "add_document", "file_path": "01_MemoryTool_Basic_Operations.py"})
    rag.run({"action": "search", "query": "记忆系统", "limit": 3})
    rag.run({"action": "stats"})


def demo_multi_namespace():
    """多命名空间隔离演示"""
    # 命名空间 "python_docs"
    rag_py = RAGTool(rag_namespace="python_docs")
    rag_py.add_document("01_MemoryTool_Basic_Operations.py")

    # 命名空间 "research" — 同一集合，逻辑隔离
    rag_research = RAGTool(rag_namespace="research")
    rag_research.add_document("01_MemoryTool_Basic_Operations.py")

    # 分别搜索
    print(rag_py.search("记忆系统", limit=2))
    print(rag_research.search("记忆系统", limit=2))


if __name__ == "__main__":
    demo_basic()
