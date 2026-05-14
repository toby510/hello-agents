"""
Sensor: 状态感知器

原则：Gate+Sensor（闸门 + 感知记忆）——Sensor 是 Harness 的"感官系统"，
持续感知执行状态，将数据写入 Memory，供 Gate 决策时查询。

Sensor 本身不做决策，只负责"感知和记录"。
"""

from typing import Any, Dict, Optional
from .memory import Memory


class Sensor:
    """
    状态感知器

    职责：
    1. 记录执行过程中的各类事件（开始、结束、工具调用、错误等）
    2. 计算简单的执行指标（耗时、重试次数等）
    3. 为 Gate 提供决策所需的上下文数据
    """

    def __init__(self, memory: Optional[Memory] = None):
        self.memory = memory or Memory()

    def record(
        self,
        event_type: str,
        milestone: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录一个事件

        Args:
            event_type: 事件类型，如：
                - "execution_start" / "execution_end": 整个任务开始/结束
                - "milestone_start" / "milestone_end": 里程碑开始/结束
                - "milestone_failed": 里程碑执行失败
                - "task_execution": 单个任务执行
                - "tool_call" / "tool_result": 工具调用和结果
                - "checkpoint_result": 检查点验收结果
                - "gate_decision": 闸门决策
                - "error" / "warning": 错误或警告
            milestone: 关联的里程碑名称
            data: 事件附加数据
        """
        event = {
            "type": event_type,
            "milestone": milestone,
            "data": data or {},
        }
        self.memory.record(event)

    def get_events(
        self,
        milestone: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list:
        """查询事件"""
        return self.memory.query(milestone=milestone, event_type=event_type)

    def milestone_count(self) -> int:
        """获取已记录的里程碑数量"""
        return len(self.memory.all_milestones())

    def has_errors(self, milestone: Optional[str] = None) -> bool:
        """检查是否有错误事件"""
        errors = self.memory.query(milestone=milestone, event_type="error")
        return len(errors) > 0

    def retry_count(self, milestone: str) -> int:
        """获取指定里程碑的重试次数"""
        decisions = self.memory.query(milestone=milestone, event_type="gate_decision")
        return sum(1 for d in decisions if d.get("data", {}).get("action") == "retry")
