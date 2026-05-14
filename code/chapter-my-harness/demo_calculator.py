"""
示例 1: 纯函数任务——计算器

本示例演示 Harness 框架如何驾驭一个简单的数学计算任务。
不涉及 LLM，纯 Python 函数执行。

四大原则体现：
- Spec-Driven: 明确定义计算目标、输入、预期输出
- Gate+Sensor: Gate 自动决策，Sensor 记录执行过程
- Milestone+Checkpoint: 计算任务作为一个里程碑，精确匹配验收
- Acceptance-Driven: 结果必须通过精确匹配验收
"""

from harness import (
    Spec, Milestone, AcceptanceCriteria,
    Harness, FunctionExecutor,
    Gate, Sensor, Checkpoint,
)


def calculator_executor(task: str, context: dict) -> str:
    """
    计算器执行函数

    解析任务中的数学表达式并计算结果。
    这是一个确定性的纯函数，适合 FunctionExecutor。
    """
    import re

    # 从上下文中获取表达式
    inputs = context.get("inputs", {})
    expression = inputs.get("expression", "")

    # 安全检查：只允许数字和基本运算符
    if not re.match(r"^[\d\+\-\*\/\(\)\.\s]+$", expression):
        return f"错误: 不安全的表达式 '{expression}'"

    try:
        # 计算结果
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"


def main():
    print("=" * 70)
    print("🧮 示例 1: Harness 计算器（纯函数任务）")
    print("=" * 70)

    # === 1. 定义 Spec ===
    # 原则：Spec-Driven——任何任务执行前必须先定义清晰的 Spec
    spec = Spec(
        goal="计算数学表达式 (100 + 200) * 3 并验证结果为 900",
        inputs={"expression": "(100 + 200) * 3"},
        expected_output="900",
        constraints=[
            "只允许使用基本数学运算符",
            "结果必须是精确值",
        ],
        milestones=[
            Milestone(
                name="calculate",
                description="执行数学计算",
                tasks=["解析并计算表达式 (100 + 200) * 3"],
                # 原则：Acceptance-Driven——明确的验收标准
                acceptance=AcceptanceCriteria(
                    type="exact_match",
                    expected="900",
                ),
                max_retries=2,
            ),
        ],
        final_acceptance=AcceptanceCriteria(
            type="exact_match",
            expected="900",
        ),
    )

    # === 2. 创建组件 ===
    # 原则：Gate+Sensor——分层可控
    sensor = Sensor()           # 感知器：记录执行状态
    checkpoint = Checkpoint()   # 检查点：验收输出
    gate = Gate(strategy="auto")  # 闸门：自动决策

    # === 3. 创建 Harness ===
    harness = Harness(
        gate=gate,
        sensor=sensor,
        checkpoint=checkpoint,
    )

    # === 4. 创建 Executor ===
    executor = FunctionExecutor(calculator_executor)

    # === 5. 执行任务 ===
    result = harness.run(spec, executor)

    # === 6. 输出结果 ===
    print("\n" + "=" * 70)
    print("📊 执行结果")
    print("=" * 70)
    print(f"成功: {result.success}")
    print(f"输出: {result.output}")
    print(f"最终验收: {'通过' if result.final_acceptance_passed else '失败'}")

    # === 7. 查看 Sensor 记录 ===
    print("\n📋 执行事件记录:")
    for event in sensor.get_events():
        ms = event.get("milestone") or "global"
        print(f"   [{ms}] {event['type']}: {event.get('data', {})}")

    # === 8. 查看 Memory 摘要 ===
    print("\n🧠 Memory 摘要:")
    summary = result.memory_summary
    print(f"   总事件数: {summary['total_events']}")
    print(f"   事件分布: {summary['event_breakdown']}")

    return result


if __name__ == "__main__":
    main()
