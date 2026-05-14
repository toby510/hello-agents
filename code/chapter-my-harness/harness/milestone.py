"""
Checkpoint: 分段验收

原则：Milestone+Checkpoint（分段验收）——每个里程碑完成后，
Checkpoint 负责验证输出是否符合验收标准。

Checkpoint 是 Harness 的"质检员"，确保每个阶段交付物达标。
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from .spec import AcceptanceCriteria


@dataclass
class CheckpointResult:
    """
    检查点验收结果

    Attributes:
        passed: 是否通过验收
        message: 验收说明（通过原因或失败原因）
        details: 详细数据
    """
    passed: bool
    message: str
    details: Dict[str, Any]


class Checkpoint:
    """
    检查点执行器

    根据 AcceptanceCriteria 的类型，选择对应的验证策略：
    - exact_match: 精确匹配
    - contains: 包含检查
    - llm_judge: LLM 质量评估
    - custom_fn: 自定义函数验证
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: 可选的 LLM 实例（用于 llm_judge 策略）
        """
        self.llm = llm

    def evaluate(
        self,
        output: str,
        criteria: AcceptanceCriteria,
        context: Optional[Dict[str, Any]] = None,
    ) -> CheckpointResult:
        """
        评估输出是否符合验收标准

        Args:
            output: 待验收的输出内容
            criteria: 验收标准
            context: 额外上下文（如里程碑名称、Spec 等）
        """
        eval_type = criteria.type

        if eval_type == "exact_match":
            return self._eval_exact_match(output, criteria)
        elif eval_type == "contains":
            return self._eval_contains(output, criteria)
        elif eval_type == "llm_judge":
            return self._eval_llm_judge(output, criteria, context)
        elif eval_type == "custom_fn":
            return self._eval_custom_fn(output, criteria)
        else:
            return CheckpointResult(
                passed=False,
                message=f"未知的验收类型: {eval_type}",
                details={"type": eval_type},
            )

    def _eval_exact_match(self, output: str, criteria: AcceptanceCriteria) -> CheckpointResult:
        """精确匹配验证"""
        expected = str(criteria.expected) if criteria.expected is not None else ""
        passed = output.strip() == expected.strip()
        return CheckpointResult(
            passed=passed,
            message=f"精确匹配{'通过' if passed else '失败'}: 预期 '{expected}', 实际 '{output[:200]}'",
            details={"expected": expected, "actual": output},
        )

    def _eval_contains(self, output: str, criteria: AcceptanceCriteria) -> CheckpointResult:
        """包含验证"""
        expected = str(criteria.expected) if criteria.expected is not None else ""
        passed = expected in output
        return CheckpointResult(
            passed=passed,
            message=f"包含检查{'通过' if passed else '失败'}: 需包含 '{expected}'",
            details={"expected": expected, "actual": output},
        )

    def _eval_llm_judge(
        self,
        output: str,
        criteria: AcceptanceCriteria,
        context: Optional[Dict[str, Any]],
    ) -> CheckpointResult:
        """LLM 质量评估"""
        if not self.llm:
            return CheckpointResult(
                passed=False,
                message="LLM 评估失败: 未提供 LLM 实例",
                details={},
            )

        # 构建评估 prompt
        prompt = f"""请评估以下输出是否符合要求。

## 任务目标
{context.get('spec', {}).get('goal', '未指定') if context else '未指定'}

## 当前阶段
{context.get('milestone', {}).get('name', '未指定') if context else '未指定'}

## 验收标准
{criteria.llm_prompt or '请判断输出是否符合预期'}

## 待评估输出
```
{output[:2000]}
```

请严格按以下格式回复：
PASS: [true/false]
REASON: [简要说明通过或失败的原因]
"""

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            # 解析响应
            passed = "PASS: true" in response or "PASS: True" in response
            return CheckpointResult(
                passed=passed,
                message=f"LLM 评估{'通过' if passed else '失败'}: {response[:300]}",
                details={"llm_response": response},
            )
        except Exception as e:
            return CheckpointResult(
                passed=False,
                message=f"LLM 评估异常: {str(e)}",
                details={"error": str(e)},
            )

    def _eval_custom_fn(self, output: str, criteria: AcceptanceCriteria) -> CheckpointResult:
        """自定义函数验证"""
        if not criteria.custom_fn:
            return CheckpointResult(
                passed=False,
                message="自定义验证失败: 未提供 custom_fn",
                details={},
            )
        try:
            passed = criteria.custom_fn(output)
            return CheckpointResult(
                passed=passed,
                message=f"自定义验证{'通过' if passed else '失败'}",
                details={},
            )
        except Exception as e:
            return CheckpointResult(
                passed=False,
                message=f"自定义验证异常: {str(e)}",
                details={"error": str(e)},
            )
