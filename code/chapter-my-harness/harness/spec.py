"""
Spec-Driven: 规范先行

任何任务在执行前，必须先定义清晰的 Spec（规格）。
Spec 包含：目标、输入、约束、里程碑列表、最终验收标准。
Harness 框架的核心入口就是 Spec——没有 Spec，任务不得执行。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AcceptanceCriteria:
    """
    验收标准定义

    原则：Acceptance-Driven（验收驱动）——每个阶段必须有明确的验收标准，
    只有通过验收才能进入下一阶段。

    Attributes:
        type: 验收类型
            - "exact_match": 精确匹配预期输出
            - "contains": 输出包含关键内容
            - "llm_judge": 由 LLM 评估质量（需要配置 llm_client）
            - "custom_fn": 使用自定义函数验证
        expected: 预期值（用于 exact_match / contains）
        custom_fn: 自定义验证函数，签名 fn(output: str) -> bool
        llm_prompt: LLM 评估时的额外提示（用于 llm_judge）
    """
    type: str = "exact_match"
    expected: Any = None
    custom_fn: Optional[Callable[[str], bool]] = None
    llm_prompt: Optional[str] = None


@dataclass
class Milestone:
    """
    里程碑定义

    原则：Milestone+Checkpoint（分段验收）——将大任务拆解为多个里程碑，
    每个里程碑有独立的任务列表和验收标准。

    Attributes:
        name: 里程碑标识名
        description: 里程碑描述
        tasks: 该阶段需要完成的子任务描述列表
        acceptance: 该里程碑的验收标准
        max_retries: 验收失败时最大重试次数（默认 3）
    """
    name: str
    description: str
    tasks: List[str] = field(default_factory=list)
    acceptance: AcceptanceCriteria = field(default_factory=AcceptanceCriteria)
    max_retries: int = 3


@dataclass
class Spec:
    """
    任务规格定义

    原则：Spec-Driven（规格驱动）——任务执行前必须定义完整的 Spec，
    Harness 严格按照 Spec 执行，不得偏离规格。

    Attributes:
        goal: 任务的总体目标描述
        inputs: 任务输入参数
        expected_output: 预期最终输出描述（文本说明，非精确值）
        constraints: 约束条件列表
        milestones: 里程碑列表（至少一个）
        final_acceptance: 最终验收标准（全任务完成后执行）
    """
    goal: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    constraints: List[str] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)
    final_acceptance: AcceptanceCriteria = field(default_factory=AcceptanceCriteria)

    def validate(self) -> None:
        """验证 Spec 的完整性，不完整则抛出异常"""
        if not self.goal:
            raise ValueError("Spec.goal 不能为空")
        if not self.milestones:
            raise ValueError("Spec.milestones 不能为空")
        for ms in self.milestones:
            if not ms.name:
                raise ValueError("Milestone.name 不能为空")
