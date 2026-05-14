"""
Harness Engineering Framework - AI 驾驭工程框架

四大核心原则：
1. 规范先行 Spec-Driven: 任何任务执行前必须有明确定义的 Spec
2. 分层可控 Gate+Sensor: Gate 控制流转，Sensor 感知状态并写入 Memory
3. 里程碑拆解 Milestone+Checkpoint: 大任务拆分为 Milestones，每阶段有 Checkpoint 验收
4. 强制验收 Acceptance-Driven: 每个阶段必须通过 Acceptance Criteria 才能继续

使用方式:
    from harness import Spec, Milestone, Harness, FunctionExecutor
    from harness import AcceptanceCriteria, Gate, Sensor, Checkpoint
"""

from .spec import Spec, Milestone, AcceptanceCriteria
from .memory import Memory
from .sensor import Sensor
from .executor import Executor, FunctionExecutor, LLMExecutor
from .milestone import Checkpoint, CheckpointResult
from .gate import Gate, GateDecision
from .harness import Harness, HarnessResult

__all__ = [
    "Spec",
    "Milestone",
    "AcceptanceCriteria",
    "Memory",
    "Sensor",
    "Executor",
    "FunctionExecutor",
    "LLMExecutor",
    "Checkpoint",
    "CheckpointResult",
    "Gate",
    "GateDecision",
    "Harness",
    "HarnessResult",
]
