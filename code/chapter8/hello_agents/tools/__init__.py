"""HelloAgents tools package."""

import sys
import os

_CLI_PATH = "/Users/longxuebin/ai_project_reposity/pthon/hello-agents-main/Co-creation-projects/YYHDBL-HelloCodeAgentCli"
if _CLI_PATH not in sys.path:
    sys.path.insert(0, _CLI_PATH)

from tools.builtin.memory_tool import MemoryTool

# RAGTool implementation based on memory.rag.pipeline
from typing import Dict, List, Optional, Any
import tempfile
import time

from memory.rag.pipeline import (
    load_and_chunk_texts,
    index_chunks,
    search_vectors,
    search_vectors_expanded,
    merge_snippets_grouped,
    create_rag_pipeline,
)


class RAGTool:
    """RAG工具 - 基于HelloAgents RAG管道的文档问答工具"""

    def __init__(
        self,
        knowledge_base_path: Optional[str] = None,
        rag_namespace: str = "default"
    ):
        self.rag_namespace = rag_namespace
        self.knowledge_base_path = knowledge_base_path or "./rag_kb"
        self._pipeline = create_rag_pipeline(rag_namespace=rag_namespace)
        self._store = self._pipeline["store"]
        self._doc_count = 0
        self._chunk_count = 0

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行RAG操作"""
        action = parameters.get("action")

        if action == "add_document":
            return self._add_document(parameters)
        elif action == "add_text":
            return self._add_text(parameters)
        elif action == "search":
            return self._search(parameters)
        elif action == "ask":
            return self._ask(parameters)
        elif action == "stats":
            return self._stats()
        else:
            return f"不支持的操作: {action}"

    def _add_document(self, parameters: Dict[str, Any]) -> str:
        """添加文档到知识库"""
        file_path = parameters.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return f"❌ 文件不存在: {file_path}"

        chunk_size = parameters.get("chunk_size", 800)
        chunk_overlap = parameters.get("chunk_overlap", 100)

        try:
            chunks = load_and_chunk_texts(
                paths=[file_path],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                namespace=self.rag_namespace,
                source_label="rag"
            )
            if not chunks:
                return "⚠️ 未能从文档中提取内容"

            index_chunks(
                store=self._store,
                chunks=chunks,
                rag_namespace=self.rag_namespace
            )

            self._doc_count += 1
            self._chunk_count += len(chunks)

            return f"✅ 已处理文档，生成 {len(chunks)} 个文本块"
        except Exception as e:
            return f"❌ 文档处理失败: {str(e)}"

    def _add_text(self, parameters: Dict[str, Any]) -> str:
        """添加文本到知识库"""
        text = parameters.get("text", "")
        if not text:
            return "⚠️ 文本内容为空"

        doc_id = parameters.get("document_id") or parameters.get("doc_id", f"text_doc_{int(time.time())}")

        try:
            # 保存为临时文件以便复用pipeline
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(text)
                temp_path = f.name

            chunks = load_and_chunk_texts(
                paths=[temp_path],
                chunk_size=parameters.get("chunk_size", 800),
                chunk_overlap=parameters.get("chunk_overlap", 100),
                namespace=self.rag_namespace,
                source_label="rag"
            )

            # 清理临时文件
            try:
                os.unlink(temp_path)
            except Exception:
                pass

            if not chunks:
                return "⚠️ 未能从文本中提取内容"

            # 添加额外元数据
            for ch in chunks:
                meta = ch.get("metadata", {})
                meta["document_id"] = doc_id
                for key, value in parameters.items():
                    if key not in ("action", "text", "chunk_size", "chunk_overlap"):
                        meta[key] = value
                ch["metadata"] = meta

            index_chunks(
                store=self._store,
                chunks=chunks,
                rag_namespace=self.rag_namespace
            )

            self._doc_count += 1
            self._chunk_count += len(chunks)

            return f"✅ 已添加文本，生成 {len(chunks)} 个文本块"
        except Exception as e:
            return f"❌ 文本添加失败: {str(e)}"

    def _search(self, parameters: Dict[str, Any]) -> str:
        """搜索知识库"""
        query = parameters.get("query", "")
        if not query:
            return "⚠️ 查询不能为空"

        limit = parameters.get("limit", 5)
        enable_mqe = parameters.get("enable_mqe", False)
        enable_hyde = parameters.get("enable_hyde", False)

        try:
            if enable_mqe or enable_hyde:
                results = search_vectors_expanded(
                    store=self._store,
                    query=query,
                    top_k=limit,
                    rag_namespace=self.rag_namespace,
                    enable_mqe=enable_mqe,
                    enable_hyde=enable_hyde
                )
            else:
                results = search_vectors(
                    store=self._store,
                    query=query,
                    top_k=limit,
                    rag_namespace=self.rag_namespace
                )

            if not results:
                return f"🔍 未找到与 '{query}' 相关的结果"

            formatted = []
            for i, r in enumerate(results, 1):
                content = r.get("metadata", {}).get("content", "")
                score = r.get("score", 0)
                preview = content[:200] + "..." if len(content) > 200 else content
                formatted.append(f"{i}. [相似度: {score:.3f}] {preview}")

            return "\n\n".join(formatted)
        except Exception as e:
            return f"❌ 搜索失败: {str(e)}"

    def _ask(self, parameters: Dict[str, Any]) -> str:
        """问答 - 检索并生成答案"""
        question = parameters.get("question", "")
        if not question:
            return "⚠️ 问题不能为空"

        limit = parameters.get("limit", 5)
        enable_advanced = parameters.get("enable_advanced_search", False)
        enable_mqe = parameters.get("enable_mqe", False) or enable_advanced
        enable_hyde = parameters.get("enable_hyde", False) or enable_advanced
        include_citations = parameters.get("include_citations", False)

        try:
            # 检索相关上下文
            if enable_mqe or enable_hyde:
                results = search_vectors_expanded(
                    store=self._store,
                    query=question,
                    top_k=limit,
                    rag_namespace=self.rag_namespace,
                    enable_mqe=enable_mqe,
                    enable_hyde=enable_hyde
                )
            else:
                results = search_vectors(
                    store=self._store,
                    query=question,
                    top_k=limit,
                    rag_namespace=self.rag_namespace
                )

            if not results:
                return f"🔍 未在知识库中找到与 '{question}' 相关的信息。请尝试加载相关文档后再提问。"

            # 构建上下文
            if include_citations:
                context = merge_snippets_grouped(results, max_chars=2000, include_citations=True)
            else:
                context = merge_snippets_grouped(results, max_chars=2000, include_citations=False)

            # 使用LLM生成答案
            try:
                from core.llm import HelloAgentsLLM
                llm = HelloAgentsLLM()
                prompt = [
                    {"role": "system", "content": "你是一个基于知识库的问答助手。请根据提供的上下文回答问题。如果上下文不足以回答问题，请明确说明。回答要简洁、准确。"},
                    {"role": "user", "content": f"上下文：\n{context}\n\n问题：{question}\n\n请根据上下文回答问题："}
                ]
                answer = llm.invoke(prompt)
                return answer or "⚠️ 未能生成答案"
            except Exception as e:
                # LLM不可用，返回检索到的上下文
                return f"💡 基于知识库的检索结果：\n\n{context}\n\n[注：LLM生成失败，返回原始检索内容]"
        except Exception as e:
            return f"❌ 问答失败: {str(e)}"

    def _stats(self) -> str:
        """获取统计信息"""
        try:
            stats = self._pipeline.get("get_stats", lambda: {})()
            return f"📊 RAG统计\n文档数: {self._doc_count}\n文本块数: {self._chunk_count}\n命名空间: {self.rag_namespace}"
        except Exception as e:
            return f"📊 RAG统计\n文档数: {self._doc_count}\n文本块数: {self._chunk_count}\n命名空间: {self.rag_namespace}"


__all__ = ["MemoryTool", "RAGTool"]
