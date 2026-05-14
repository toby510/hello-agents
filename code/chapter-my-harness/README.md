# Harness Engineering 框架 — AI 驾驭工程

基于 Harness Engineering 四大核心原则的 Python 框架，让 AI Agent 执行复杂任务时可被人类/系统有效驾驭。

## 四大原则

| 原则 | 组件 | 说明 |
|------|------|------|
| **规范先行 Spec-Driven** | `Spec` | 任务执行前必须定义完整的规格（目标、输入、约束、验收标准） |
| **分层可控 Gate+Sensor** | `Gate` + `Sensor` + `Memory` | Gate 控制阶段流转，Sensor 感知状态并写入 Memory |
| **里程碑拆解 Milestone+Checkpoint** | `Milestone` + `Checkpoint` | 大任务拆分为 Milestones，每阶段 Checkpoint 验收 |
| **强制验收 Acceptance-Driven** | `AcceptanceCriteria` | 每阶段和最终输出都必须通过验收标准 |

## 快速开始

```bash
# 运行示例 1：纯函数任务（计算器）
python demo_calculator.py

# 运行示例 2：多阶段流水线（研究报告生成）
python demo_full_pipeline.py

# 运行示例 3：LLM Agent 集成（智能问答）
python demo_llm_task.py
```

## 核心架构

```
User ──► Spec ──► Harness.run(executor)
                    │
                    ▼
            ┌─────────────┐     ┌─────────────┐
            │  Milestone  │◄────│ Checkpoint  │───► Gate.can_proceed()?
            │  (阶段任务)  │     │  (分段验收)  │
            └─────────────┘     └─────────────┘
                    │                              ┌─────────┐     ┌────────┐
                    ▼                              │ Sensor  │────►│ Memory │
            ┌─────────────┘                      │ (感知)   │     │ (记忆) │
            │  Milestone N │◄────│ Checkpoint N │         └────────┘
            └─────────────┘     └─────────────┘
                    │
                    ▼
            ┌─────────────────┐
            │ Final Acceptance │───► Result
            │    (强制验收)     │
            └─────────────────┘
```

## 文件结构

```
chapter-my-harness/
├── harness/                    # 核心框架
│   ├── __init__.py             # 包导出
│   ├── spec.py                 # Spec + Milestone + AcceptanceCriteria
│   ├── memory.py               # 感知记忆存储
│   ├── sensor.py               # 状态感知器
│   ├── executor.py             # 执行器抽象（Function / LLM）
│   ├── milestone.py            # Checkpoint 验收逻辑
│   ├── gate.py                 # Gate 闸门控制策略
│   └── harness.py              # 主控制器
├── demo_calculator.py          # 示例1：纯函数任务
├── demo_full_pipeline.py       # 示例2：多阶段流水线
└── demo_llm_task.py            # 示例3：LLM Agent 集成
```

## 使用方式

### 1. 定义 Spec

```python
from harness import Spec, Milestone, AcceptanceCriteria

spec = Spec(
    goal="计算数学表达式并验证结果",
    inputs={"expression": "(100 + 200) * 3"},
    milestones=[
        Milestone(
            name="calculate",
            description="执行数学计算",
            tasks=["解析并计算表达式"],
            acceptance=AcceptanceCriteria(type="exact_match", expected="900"),
        ),
    ],
    final_acceptance=AcceptanceCriteria(type="exact_match", expected="900"),
)
```

### 2. 创建组件并执行

```python
from harness import Harness, FunctionExecutor, Gate, Sensor, Checkpoint

# 创建组件
harness = Harness(
    gate=Gate(strategy="auto"),
    sensor=Sensor(),
    checkpoint=Checkpoint(),
)

# 定义执行函数
def my_executor(task: str, context: dict) -> str:
    return str(eval(context["inputs"]["expression"]))

# 执行任务
result = harness.run(spec, FunctionExecutor(my_executor))
print(result.output)  # 900
```

### 3. 与 hello_agents LLM 集成

```python
from hello_agents import HelloAgentsLLM
from harness import LLMExecutor

llm = HelloAgentsLLM()
executor = LLMExecutor(llm, system_prompt="你是一个任务执行助手")
result = harness.run(spec, executor)
```

## Checkpoint 验收类型

- `exact_match`: 精确匹配预期输出
- `contains`: 输出包含关键内容
- `llm_judge`: 由 LLM 评估质量（需传入 LLM 实例）
- `custom_fn`: 使用自定义函数验证

## Gate 决策策略

- `auto`: 自动决策（Checkpoint 通过则 proceed，否则 retry/abort）
- `llm_eval`: LLM 综合评估后决策
- `human_confirm`: 暂停等待人工确认

## 依赖

- Python 3.10+
- 纯标准库（框架本身零外部依赖）
- 可选：`hello_agents`（用于 LLM 集成示例）
