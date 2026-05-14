"""
Gate: 闸门控制

原则：Gate+Sensor（闸门 + 感知记忆）——Gate 是 Harness 的"决策中枢"，
基于 Checkpoint 验收结果和 Sensor 感知数据，决定下一步流向。

Gate 的决策选项：
- proceed: 继续执行下一阶段
- retry: 重试当前阶段（受 max_retries 限制）
- rollback: 回退到前一阶段（高级功能）
- abort: 终止整个任务

Gate 的策略：
- auto: 自动决策（Checkpoint 通过则 proceed，否则 retry/abort）
- llm_eval: 由 LLM 综合评估后决策
- human_confirm: 暂停等待人工确认
"""

from dataclasses import dataclass
from typing import Optional
from .spec import Milestone
from .milestone import CheckpointResult
from .sensor import Sensor


@dataclass
class GateDecision:
    """
    闸门决策结果

    Attributes:
        action: 决策动作 - "proceed" / "retry" / "rollback" / "abort"
        reason: 决策原因说明
    """
    action: str  # proceed, retry, rollback, abort
    reason: str


class Gate:
    """
    闸门控制器

    Gate 的核心职责：根据当前状态做出流转决策。
    它不执行具体任务，只回答一个问题："现在该做什么？"
    """

    def __init__(self, strategy: str = "auto", llm=None):
        """
        Args:
            strategy: 决策策略
                - "auto": 自动决策（推荐用于确定性任务）
                - "llm_eval": LLM 综合评估（推荐用于开放性任务）
                - "human_confirm": 人工确认（推荐用于高风险任务）
            llm: LLM 实例（llm_eval 策略需要）
        """
        self.strategy = strategy
        self.llm = llm

    def can_proceed(
        self,
        milestone: Milestone,
        checkpoint_result: CheckpointResult,
        sensor: Sensor,
    ) -> GateDecision:
        """
        决定是否允许进入下一阶段

        Args:
            milestone: 当前里程碑
            checkpoint_result: 检查点验收结果
            sensor: 感知器（提供历史执行数据）

        Returns:
            GateDecision: 决策结果
        """
        if self.strategy == "auto":
            return self._auto_decide(milestone, checkpoint_result, sensor)
        elif self.strategy == "llm_eval":
            return self._llm_decide(milestone, checkpoint_result, sensor)
        elif self.strategy == "human_confirm":
            return self._human_decide(milestone, checkpoint_result, sensor)
        else:
            return GateDecision(
                action="abort",
                reason=f"未知的 Gate 策略: {self.strategy}",
            )

    def _auto_decide(
        self,
        milestone: Milestone,
        checkpoint: CheckpointResult,
        sensor: Sensor,
    ) -> GateDecision:
        """自动决策策略"""
        if checkpoint.passed:
            return GateDecision(
                action="proceed",
                reason=f"里程碑 '{milestone.name}' 验收通过",
            )

        # 验收失败，检查是否可重试
        retries = sensor.retry_count(milestone.name)
        if retries < milestone.max_retries:
            return GateDecision(
                action="retry",
                reason=f"验收失败（第 {retries + 1} 次重试）: {checkpoint.message}",
            )

        # 超过最大重试次数
        return GateDecision(
            action="abort",
            reason=f"里程碑 '{milestone.name}' 验收失败且超过最大重试次数 ({milestone.max_retries})",
        )

    def _llm_decide(
        self,
        milestone: Milestone,
        checkpoint: CheckpointResult,
        sensor: Sensor,
    ) -> GateDecision:
        """LLM 综合评估策略"""
        if not self.llm:
            # 无 LLM 时降级为 auto 策略
            return self._auto_decide(milestone, checkpoint, sensor)

        # 收集上下文
        events = sensor.get_events(milestone=milestone.name)
        event_summary = "\n".join(
            f"- {e.get('type')}: {e.get('data', {})}"
            for e in events[-10:]  # 最近 10 条
        )

        prompt = f"""你是一个任务流程决策助手。请根据以下信息决定下一步动作。

## 当前里程碑
名称: {milestone.name}
描述: {milestone.description}
最大重试次数: {milestone.max_retries}
已重试次数: {sensor.retry_count(milestone.name)}

## 检查点结果
通过: {checkpoint.passed}
说明: {checkpoint.message}

## 最近事件
{event_summary}

## 可选动作
- proceed: 继续下一阶段
- retry: 重试当前阶段
- abort: 终止任务

请严格按以下格式回复（只回复这一行）：
ACTION: [proceed/retry/abort]
REASON: [简要原因]
"""

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            # 解析响应
            action = "abort"
            if "ACTION: proceed" in response:
                action = "proceed"
            elif "ACTION: retry" in response:
                action = "retry"
            elif "ACTION: abort" in response:
                action = "abort"

            # 提取原因
            reason = response
            if "REASON:" in response:
                reason = response.split("REASON:")[1].strip()

            return GateDecision(action=action, reason=reason)
        except Exception as e:
            return GateDecision(
                action="abort",
                reason=f"LLM 决策异常: {str(e)}",
            )

    def _human_decide(
        self,
        milestone: Milestone,
        checkpoint: CheckpointResult,
        sensor: Sensor,
    ) -> GateDecision:
        """人工确认策略"""
        print(f"\n{'='*60}")
        print(f"【人工确认】里程碑: {milestone.name}")
        print(f"检查点结果: {'通过' if checkpoint.passed else '失败'}")
        print(f"说明: {checkpoint.message}")
        print(f"{'='*60}")

        while True:
            choice = input("请选择: [p]roceed / [r]etry / [a]bort: ").strip().lower()
            if choice in ("p", "proceed"):
                return GateDecision(action="proceed", reason="人工确认: 继续")
            elif choice in ("r", "retry"):
                return GateDecision(action="retry", reason="人工确认: 重试")
            elif choice in ("a", "abort"):
                return GateDecision(action="abort", reason="人工确认: 终止")
            else:
                print("无效输入，请重新选择")
