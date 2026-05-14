# Chapter 8 代码示例速查 (01-11)

本章覆盖两大系统：**Memory（记忆系统）** 和 **RAG（检索增强生成）**，以及两者的集成应用。

---

## 概览

| 编号 | 文件 | 模块 | 核心主题 |
|------|------|------|----------|
| 01 | MemoryTool_Basic_Operations | Memory | 基础操作：add/search/summary/stats/forget/consolidate |
| 02 | MemoryTool_Architecture | Memory | 分层架构：MemoryTool → MemoryManager → 4种Memory |
| 03 | WorkingMemory_Implementation | Memory | 工作记忆：容量管理/TTL/混合检索/时间衰减 |
| 04 | RAGTool_MarkItDown_Pipeline | RAG | MarkItDown管道：任意格式→Markdown→分块→向量化 |
| 05 | RAGTool_Advanced_Search | RAG | 高级检索：MQE多查询扩展/HyDE假设文档嵌入 |
| 06 | Memory_Consolidation_Demo | Memory | 记忆整合：working→episodic→semantic 晋升路径 |
| 07 | RAGTool_Intelligent_QA | RAG | 智能问答：检索→上下文构建→LLM生成→引用 |
| 08 | Agent_Tool_Integration | 集成 | Agent+ToolRegistry：MemoryTool与RAGTool协同编排 |
| 09 | Memory_Types_Deep_Dive | Memory | 四种记忆类型深度对比与跨类型交互 |
| 10 | RAG_Pipeline_Complete | RAG | 完整RAG管道：摄取→分块→检索→问答→性能 |
| 11 | Q&A_Assistant | 集成 | 端到端应用：PDF助手+Gradio Web UI |

---

## 01 — MemoryTool 基础操作

**核心调用**: 通过统一的 `run({"action": "..."})` 接口操作记忆系统，覆盖完整生命周期。

```
MemoryTool(user_id, memory_types)
  │
  ├─ run({"action":"add",    content, memory_type, importance})
  ├─ run({"action":"search", query, limit, min_importance})
  ├─ run({"action":"summary"})
  ├─ run({"action":"stats"})
  ├─ run({"action":"forget", strategy, threshold})
  └─ run({"action":"consolidate", from_type, to_type, importance_threshold})
```

**核心代码片段**:
```python
from hello_agents.tools import MemoryTool

memory_tool = MemoryTool(user_id="demo_user", memory_types=["working","episodic"])

# 添加
memory_tool.run({"action":"add", "content":"正在学习...", "memory_type":"working", "importance":0.7})
memory_tool.run({"action":"add", "content":"2024年开始...", "memory_type":"episodic", "importance":0.8})

# 搜索
memory_tool.run({"action":"search", "query":"记忆系统", "limit":3})

# 遗忘 + 整合
memory_tool.run({"action":"forget", "strategy":"importance_based", "threshold":0.2})
memory_tool.run({"action":"consolidate", "from_type":"working", "to_type":"episodic", "importance_threshold":0.6})
```

---

## 02 — MemoryTool 架构设计

**核心**: 展示 MemoryTool → MemoryManager → 4种Memory类型的分层组合模式。

```
MemoryTool (统一入口)
  └─ MemoryManager (组合模式)
       ├─ WorkingMemory    → 纯内存, TTL=60min, 容量50条
       ├─ EpisodicMemory   → SQLite + Qdrant, 持久化, 时间序列
       ├─ SemanticMemory   → Neo4j + Qdrant, 知识图谱, 实体关系
       └─ PerceptualMemory → 分模态Qdrant, CLIP/CLAP, 跨模态检索
```

**设计模式**: 组合模式 + 统一接口 — 所有类型通过相同的 `run({"action":...})` 调用，内部路由到对应 Memory 实现。

---

## 03 — WorkingMemory 实现详解

**核心**: 工作记忆的5个关键机制，纯内存存储，仅用 sklearn TF-IDF。

```
添加记忆 → 内存List + 最大堆(按重要性)
    │
    ├─ 容量管理: 超限时移除低重要性项
    ├─ TTL机制:   60分钟后自动过期
    ├─ 混合检索:  向量分*0.8 + 关键词 + 时间衰减*0.2 + 重要性权重
    ├─ 时间衰减:  新记忆权重高, 旧记忆权重衰减
    └─ 自动清理:  importance_based遗忘, 阈值可配置
```

**核心代码**:
```python
# 容量管理: 批量添加, 自动按重要性淘汰
for i in range(10):
    memory_tool.run({"action":"add", "content":f"测试{i}", "memory_type":"working", "importance":0.3+i*0.07})

# 混合检索: 语义+关键词+时间+重要性
memory_tool.run({"action":"search", "query":"Python编程", "memory_type":"working", "limit":2})

# 自动清理: 基于重要性阈值
memory_tool.run({"action":"forget", "strategy":"importance_based", "threshold":0.3})
```

---

## 04 — RAGTool MarkItDown 处理管道

**核心**: 任意格式文档→MarkItDown→Markdown→智能分块→嵌入预处理→Qdrant向量化。

```
文件(PDF/DOCX/HTML/JSON/CSV/代码...)
  │
  ├─ ① MarkItDown 转换 → 统一 Markdown
  │     _is_markitdown_supported_format(): 40+格式检测
  │     _convert_to_markdown(): PDF增强处理+后处理
  │
  ├─ ② 智能分块 (chunk_size=800, overlap=100)
  │     _split_paragraphs_with_headings(): 保留 ## Heading 层级路径
  │     _chunk_paragraphs(): 段落为单位+滑动窗口
  │
  ├─ ③ 嵌入预处理
  │     _preprocess_markdown_for_embedding(): 去格式符号, 保留语义
  │
  └─ ④ Qdrant 存储
        集合: hello_agents_rag_vectors, 标记 is_rag_data=True
```

**核心代码**:
```python
from hello_agents.tools import RAGTool

rag = RAGTool(knowledge_base_path="./kb", rag_namespace="demo")

# 多格式支持
rag.run({"action":"add_document", "file_path":"paper.pdf"})
rag.run({"action":"add_text", "text":"# 标题\n内容...", "document_id":"doc1"})

# 批量添加
rag.batch_add_texts(texts=["文本1","文本2"], document_ids=["id1","id2"])
```

---

## 05 — RAG 高级检索策略

**核心**: MQE（多查询扩展）和 HyDE（假设文档嵌入）两种检索增强技术。

```
用户查询 "深度学习优化"
  │
  ├─ 基础搜索 ──→ 直接 embed → Qdrant.search → top_k
  │    耗时: ~0.3s, 召回率: 基准
  │
  ├─ MQE ──→ LLM生成等价查询 ──→ 并行多路检索 ──→ 融合去重 → top_k
  │    "深度学习的优化方法"     (同memory_id取最高分)
  │    "DL模型训练优化技巧"
  │    耗时: ~1.5s, 召回率: +30%
  │
  └─ HyDE ──→ LLM生成假设答案 ──→ 用答案向量检索 → top_k
       "深度学习优化包括梯度下降..."     (答案与文档在向量空间更近)
       耗时: ~2.0s, 适合复杂长尾问题
```

**核心代码**:
```python
# 基础搜索
rag.run({"action":"search", "query":"注意力机制", "enable_advanced_search": False})

# MQE + HyDE 高级搜索
rag.run({"action":"search", "query":"注意力机制", "enable_advanced_search": True})

# 组合: 先高级搜索再问答
rag.run({"action":"ask", "question":"如何提高模型性能?", "enable_advanced_search": True, "include_citations": True})
```

---

## 06 — 记忆整合机制

**核心**: 短期记忆→长期记忆的晋升路径，模拟人类记忆固化过程。

```
Working Memory (工作记忆)             Episodic Memory (情景记忆)
  importance=0.9 ──┐                      importance=0.85
  importance=0.8 ──┤ consolidate ──→      importance=0.9×1.1≈0.99
  importance=0.7 ──┤ (threshold=0.7)      importance=0.8×1.1≈0.88
  importance=0.3 ──┘ 被过滤(不达标)       importance=0.7×1.1≈0.77

整合路径:
  working → episodic  (经历固化)    阈值: 0.6-0.8
  working → semantic  (知识提取)    阈值: 0.8-0.9
  episodic → semantic (经验抽象)    阈值: 0.7-0.85
```

**整合效果**: 重要性提升 ×1.1, 保留原始元数据, 添加 `consolidation_date` / `original_id` 标记。

**核心代码**:
```python
# 添加不同重要性的记忆
memory_tool.run({"action":"add", "content":"学习了Transformer", "memory_type":"working", "importance":0.9})
memory_tool.run({"action":"add", "content":"喝了杯咖啡",     "memory_type":"working", "importance":0.2})

# 整合: 重要性≥0.7的working记忆晋升为episodic
memory_tool.run({"action":"consolidate", "from_type":"working", "to_type":"episodic", "importance_threshold":0.7})
```

---

## 07 — RAG 智能问答系统

**核心**: 完整的 RAG Q&A 链路——问题理解→高级检索→上下文构建→LLM生成→引用溯源。

```
用户问题 "什么是机器学习？"
  │
  ├─ ① 问题分类: 概念定义/方法询问/对比分析/应用场景/实现细节
  │
  ├─ ② 高级检索: MQE扩展查询 + HyDE生成假设文档 → 多路检索融合
  │
  ├─ ③ 上下文构建:
  │     检索片段 → 相关性排序 → 清理格式化 → 智能截断(max_chars) → 引用标注
  │
  ├─ ④ System Prompt 注入:
  │     "你是一个专业的知识助手, 严格基于上下文回答, 不编造内容..."
  │
  └─ ⑤ LLM生成 + 引用:
       答案 + 📚 参考来源: [1] doc.pdf (score:0.85)
```

**答案质量评分**: 内容完整性(40%) + 答案长度(30%) + 引用完整性(20%) + 响应速度(10%)

**核心代码**:
```python
# 带引用的智能问答
rag.run({"action":"ask", "question":"什么是深度学习?", "limit":4, "include_citations":True})

# 上下文构建分步演示
rag.run({"action":"search", "query":"如何防止过拟合?", "limit":6})   # 先检索
rag.run({"action":"ask",    "question":"如何防止过拟合?", "limit":5}) # 再问答
```

---

## 08 — Agent 工具集成

**核心**: 将 MemoryTool 和 RAGTool 注册到 SimpleAgent 的 ToolRegistry，实现协同编排。

```
SimpleAgent (智能学习助手)
  │
  ├─ ToolRegistry
  │    ├─ MemoryTool (记忆管理)
  │    │    └─ add/search/stats/summary/forget/consolidate
  │    └─ RAGTool (知识检索)
  │         └─ add_document/search/ask/stats
  │
  └─ 协同场景:
       场景1: 学习新知识 → RAG存储 + Memory记录
       场景2: 回顾历程   → Memory检索 + RAG补充
       场景3: 知识应用   → RAG查询  + Memory更新
       场景4: 学习分析   → 双工具统计整合
```

**核心代码**:
```python
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool, RAGTool

# 初始化
memory_tool = MemoryTool(user_id="agent_user", memory_types=["working","episodic"])
rag_tool = RAGTool(knowledge_base_path="./kb", rag_namespace="agent_demo")

# 注册到Agent
tool_registry = ToolRegistry()
tool_registry.register_tool(memory_tool)
tool_registry.register_tool(rag_tool)

agent = SimpleAgent(name="智能助手", llm=HelloAgentsLLM())
agent.tool_registry = tool_registry
```

---

## 09 — 四种记忆类型深度解析

**核心**: 分别解析 Working/Episodic/Semantic/Perceptual 的存储、检索和交互模式。

```
信息处理流程:
  感知记忆 ──→ 工作记忆 ──→ 情景记忆 ──→ 语义记忆
  (输入)       (临时处理)   (事件记录)   (知识抽象)

各类型特点:
  Working:   纯内存, 50条容量, 60min TTL, TF-IDF检索
  Episodic:  SQLite+Qdrant, 时间序列, 会话关联, 记忆链条
  Semantic:  Neo4j+Qdrant, 实体关系, 知识图谱, 语义推理
  Perceptual: 分模态Qdrant(text/image/audio), CLIP/CLAP, 跨模态检索

交互模式:
  working → episodic (consolidate)   重要事件固化
  episodic → semantic (consolidate)  经验知识化
  semantic → working (activate)      知识激活到当前上下文
  perceptual → others (integrate)    多模态信息融入
```

---

## 10 — RAG 完整处理管道

**核心**: 端到端 RAG 管道——文档摄取→分块策略→高级检索→智能问答→性能优化。

```
完整管道:
  ① 文档摄取 (Ingestion)
     多格式 → MarkItDown → 元数据提取 → 批量处理

  ② 分块策略 (Chunking)
     语义分块 | 结构化分块 | token精确控制 | 上下文重叠

  ③ 高级检索 (Retrieval)
     MQE(多查询扩展) | HyDE(假设文档) | 分解查询 | 结果融合

  ④ 智能问答 (QA Generation)
     问题分类 → 上下文构建 → 多轮对话 → 质量评估(相关性/准确性/完整性)

  ⑤ 性能优化
     批量处理 | 缓存加速 | 吞吐量监控
```

---

## 11 — Q&A 智能文档问答助手

**核心**: 将 Memory + RAG 封装为完整的 PDF 学习助手应用，带 Gradio Web UI。

```
PDFLearningAssistant
  │
  ├─ load_document(pdf_path)
  │     → RAGTool.add_document → Memory记录加载事件
  │
  ├─ ask(question, use_advanced_search=True)
  │     → RAGTool.ask(MQE+HyDE) → Memory记录QA交互
  │
  ├─ add_note(content, concept)
  │     → MemoryTool.add(semantic, concept=xxx)
  │
  ├─ recall(query)
  │     → MemoryTool.search(跨类型检索学习历史)
  │
  ├─ get_stats()
  │     → 会话时长/文档数/提问数/笔记数
  │
  └─ generate_report()
        → Memory摘要 + RAG统计 → JSON报告

Gradio Web UI:
  Tab ① 开始使用: 初始化 + 上传PDF
  Tab ② 智能问答: Chatbot + Examples
  Tab ③ 学习笔记: 文本输入 + 概念标注
  Tab ④ 学习统计: 实时统计 + 报告生成
```

**启动方式**:
```bash
python 11_Q&A_Assistant.py
# 浏览器打开 http://localhost:7860
```

**核心代码**:
```python
assistant = PDFLearningAssistant(user_id="student1")
assistant.load_document("paper.pdf")         # 加载文档
answer = assistant.ask("什么是Transformer?")  # 智能问答
assistant.add_note("理解了注意力机制", "attention")  # 记笔记
history = assistant.recall("深度学习")        # 回顾学习历程
report = assistant.generate_report()          # 生成本次学习报告
```
