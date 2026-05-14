"""
Harness: AI 驾驭工程主控制器

原则：四大原则的统一编排者
1. Spec-Driven: 严格按照 Spec 执行，执行前验证 Spec 完整性
2. Gate+Sensor: 每个阶段流转必须经过 Gate 决策，Sensor 全程记录
3. Milestone+Checkpoint: 将 Spec 拆解为 Milestones，每阶段 Checkpoint 验收
4. Acceptance-Driven: 每个里程碑 + 最终输出都必须通过验收

Harness 是框架的核心枢纽，负责 orchestrate 所有组件的协同工作。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
from .spec import Spec, Milestone
from .executor import Executor
from .sensor import Sensor
from .milestone import Checkpoint, CheckpointResult
from .gate import Gate, GateDecision
from .memory import Memory


@dataclass
class HarnessResult:
    """
    Harness 执行结果

    Attributes:
        success: 整个任务是否成功完成
        output: 最终输出内容
        milestone_results: 每个里程碑的执行记录
        final_acceptance_passed: 最终验收是否通过
        memory_summary: 执行过程摘要
    """
    success: bool
    output: str
    milestone_results: List[Dict[str, Any]] = field(default_factory=list)
    final_acceptance_passed: bool = False
    memory_summary: Dict[str, Any] = field(default_factory=dict)


class Harness:
    """
    Harness 主控制器

    使用方式:
        harness = Harness(gate=Gate(), sensor=Sensor(), checkpoint=Checkpoint())
        result = harness.run(spec, executor)

    执行流程:
        1. 验证 Spec 完整性
        2. 按顺序遍历 Milestones
        3. 对每个 Milestone:
           a. 调用 Executor 执行所有 tasks
           b. Checkpoint 验收输出
           c. Gate 决策是否 proceed / retry / abort
        4. 所有 Milestones 完成后，Final Acceptance 验收
        5. 返回 HarnessResult
    """

    def __init__(
        self,
        gate: Gate,
        sensor: Sensor,
        checkpoint: Checkpoint,
    ):
        self.gate = gate
        self.sensor = sensor
        self.checkpoint = checkpoint

    def run(self, spec: Spec, executor: Executor) -> HarnessResult:
        """
        执行 Spec 定义的任务

        Args:
            spec: 任务规格
            executor: 执行器

        Returns:
            HarnessResult: 执行结果
        """
        # === 1. Spec 验证 ===
        spec.validate()
        print(f"\n{'='*70}")
        print(f"🚀 Harness 开始执行任务: {spec.goal}")
        print(f"{'='*70}")

        # 记录任务开始
        self.sensor.record(
            event_type="execution_start",
            data={"goal": spec.goal, "milestones_count": len(spec.milestones)},
        )

        # 存储每个里程碑的输出，供后续阶段使用
        milestone_outputs: Dict[str, str] = {}
        milestone_results: List[Dict[str, Any]] = []

        # === 2. 遍历 Milestones ===
        for idx, milestone in enumerate(spec.milestones, 1):
            print(f"\n{'─'*70}")
            print(f"📍 里程碑 {idx}/{len(spec.milestones)}: {milestone.name}")
            print(f"   描述: {milestone.description}")
            print(f"   任务数: {len(milestone.tasks)}")
            print(f"{'─'*70}")

            ms_result = self._run_milestone(
                milestone=milestone,
                spec=spec,
                executor=executor,
                milestone_outputs=milestone_outputs,
                milestone_index=idx,
            )

            milestone_results.append(ms_result)

            if not ms_result["success"]:
                # 里程碑失败，终止任务
                print(f"\n❌ 里程碑 '{milestone.name}' 失败，任务终止")
                self.sensor.record(
                    event_type="execution_end",
                    data={"success": False, "failed_at": milestone.name},
                )
                return HarnessResult(
                    success=False,
                    output=ms_result.get("output", ""),
                    milestone_results=milestone_results,
                    memory_summary=self.sensor.memory.summary(),
                )

            # 保存里程碑输出
            milestone_outputs[milestone.name] = ms_result.get("output", "")

        # === 3. 最终验收 ===
        final_output = milestone_outputs.get(spec.milestones[-1].name, "") if spec.milestones else ""
        final_acceptance_passed = True

        if spec.final_acceptance.type != "exact_match" or spec.final_acceptance.expected is not None:
            print(f"\n{'='*70}")
            print("🔍 最终验收检查")
            print(f"{'='*70}")
            final_check = self.checkpoint.evaluate(
                output=final_output,
                criteria=spec.final_acceptance,
                context={"spec": spec, "milestone": {"name": "final"}},
            )
            final_acceptance_passed = final_check.passed
            status = "通过" if final_check.passed else "失败"
            print(f"   最终验收: {status} - {final_check.message}")
            self.sensor.record(
                event_type="checkpoint_result",
                milestone="final",
                data={"passed": final_check.passed, "message": final_check.message},
            )

        # 记录任务结束
        self.sensor.record(
            event_type="execution_end",
            data={"success": final_acceptance_passed, "output_length": len(final_output)},
        )

        print(f"\n{'='*70}")
        print(f"✅ Harness 任务完成: {'成功' if final_acceptance_passed else '失败'}")
        print(f"{'='*70}")

        return HarnessResult(
            success=final_acceptance_passed,
            output=final_output,
            milestone_results=milestone_results,
            final_acceptance_passed=final_acceptance_passed,
            memory_summary=self.sensor.memory.summary(),
        )

    def _run_milestone(
        self,
        milestone: Milestone,
        spec: Spec,
        executor: Executor,
        milestone_outputs: Dict[str, str],
        milestone_index: int,
    ) -> Dict[str, Any]:
        """
        执行单个里程碑（含重试循环）

        Returns:
            {"success": bool, "output": str, "retries": int}
        """
        retries = 0

        while retries <= milestone.max_retries:
            # 记录里程碑开始
            self.sensor.record(
                event_type="milestone_start",
                milestone=milestone.name,
                data={"attempt": retries + 1, "max_retries": milestone.max_retries},
            )

            try:
                # === a. 执行所有 tasks ===
                task_outputs: List[str] = []
                for task_idx, task in enumerate(milestone.tasks, 1):
                    print(f"   📝 执行任务 {task_idx}/{len(milestone.tasks)}: {task[:60]}...")

                    context = {
                        "spec": spec,
                        "milestone": milestone,
                        "inputs": spec.inputs,
                        "previous_outputs": milestone_outputs,
                        "milestone_index": milestone_index,
                        "task_index": task_idx,
                    }

                    output = executor.execute(task, context)
                    task_outputs.append(output)

                    self.sensor.record(
                        event_type="task_execution",
                        milestone=milestone.name,
                        data={"task": task, "output_length": len(output)},
                    )

                # 合并所有任务输出作为里程碑输出
                combined_output = "\n\n".join(task_outputs)

                # === b. Checkpoint 验收 ===
                print(f"   🔍 Checkpoint 验收...")
                checkpoint_result = self.checkpoint.evaluate(
                    output=combined_output,
                    criteria=milestone.acceptance,
                    context={"spec": spec, "milestone": milestone},
                )

                self.sensor.record(
                    event_type="checkpoint_result",
                    milestone=milestone.name,
                    data={
                        "passed": checkpoint_result.passed,
                        "message": checkpoint_result.message,
                        "details": checkpoint_result.details,
                    },
                )

                status_icon = "✅" if checkpoint_result.passed else "❌"
                print(f"   {status_icon} 验收结果: {checkpoint_result.message[:100]}")

                # === c. Gate 决策 ===
                decision = self.gate.can_proceed(milestone, checkpoint_result, self.sensor)

                self.sensor.record(
                    event_type="gate_decision",
                    milestone=milestone.name,
                    data={"action": decision.action, "reason": decision.reason},
                )

                print(f"   🚦 Gate 决策: {decision.action} - {decision.reason[:80]}")

                if decision.action == "proceed":
                    # 验收通过，记录里程碑完成
                    self.sensor.record(
                        event_type="milestone_end",
                        milestone=milestone.name,
                        data={"output_length": len(combined_output)},
                    )
                    return {
                        "success": True,
                        "output": combined_output,
                        "retries": retries,
                        "checkpoint": checkpoint_result,
                    }

                elif decision.action == "retry":
                    retries += 1
                    print(f"   🔄 即将第 {retries} 次重试...")
                    continue

                elif decision.action in ("abort", "rollback"):
                    # 终止或回退都视为失败
                    self.sensor.record(
                        event_type="milestone_failed",
                        milestone=milestone.name,
                        data={"reason": decision.reason, "retries": retries},
                    )
                    return {
                        "success": False,
                        "output": combined_output,
                        "retries": retries,
                        "reason": decision.reason,
                    }

            except Exception as e:
                # 执行异常
                self.sensor.record(
                    event_type="error",
                    milestone=milestone.name,
                    data={"error": str(e), "attempt": retries + 1},
                )
                self.sensor.record(
                    event_type="milestone_failed",
                    milestone=milestone.name,
                    data={"error": str(e)},
                )
                print(f"   ❌ 执行异常: {str(e)}")
                return {
                    "success": False,
                    "output": "",
                    "retries": retries,
                    "reason": f"执行异常: {str(e)}",
                }

        # 超过最大重试次数
        self.sensor.record(
            event_type="milestone_failed",
            milestone=milestone.name,
            data={"reason": "超过最大重试次数", "retries": retries},
        )
        return {
            "success": False,
            "output": "",
            "retries": retries,
            "reason": f"超过最大重试次数 ({milestone.max_retries})",
        }
