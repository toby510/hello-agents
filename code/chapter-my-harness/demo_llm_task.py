"""
示例 3: LLM Agent 集成——智能问答任务

本示例演示 Harness 框架如何与 hello_agents 的 LLM 集成，
驾驭一个需要 LLM 推理能力的开放性任务。

场景：让 LLM 回答一个复杂问题，Harness 控制回答质量。

四大原则体现：
- Spec-Driven: 定义问题、预期回答质量、约束条件
- Gate+Sensor: LLM 输出经过 Checkpoint 验收，Gate 决定是否通过
- Milestone+Checkpoint: 问题理解 -> 推理 -> 回答，分阶段验收
- Acceptance-Driven: 回答必须通过包含检查和 LLM 自评估
"""

import os
import sys

# 将项目根目录加入路径，以便导入 harness
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import (
    Spec, Milestone, AcceptanceCriteria,
    Harness, LLMExecutor,
    Gate, Sensor, Checkpoint,
)


def main():
    print("=" * 70)
    print("🤖 示例 3: LLM Agent 集成——智能问答")
    print("=" * 70)

    # 检查是否有 LLM 配置
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("\n⚠️  未检测到 LLM_API_KEY 环境变量")
        print("   本示例需要配置 LLM 才能运行")
        print("   请设置环境变量: export LLM_API_KEY='your-key'")
        print("   或使用 .env 文件配置")
        print("\n   现在运行模拟模式（使用本地函数模拟 LLM）...\n")
        return demo_mock_mode()

    # === 有 LLM 配置，使用真实 LLM ===
    return demo_llm_mode()


def demo_llm_mode():
    """使用真实 LLM 的运行模式"""
    try:
        from hello_agents import HelloAgentsLLM
    except ImportError:
        print("❌ 未安装 hello_agents，请运行: pip install hello-agents")
        return None

    # 创建 LLM 实例
    llm = HelloAgentsLLM()

    # === 1. 定义 Spec ===
    spec = Spec(
        goal="解释量子计算的基本原理，并用通俗的语言说明其应用前景",
        inputs={"topic": "量子计算", "audience": "非技术背景读者"},
        expected_output="包含原理解释和应用前景的通俗说明，长度不少于 300 字",
        constraints=[
            "避免使用过多专业术语，必须解释术语",
            "内容必须准确，不得有科学错误",
            "回答需要结构化，分点说明",
        ],
        milestones=[
            # 阶段 1: 理解问题
            Milestone(
                name="understand_question",
                description="理解用户问题的核心要点",
                tasks=["分析问题的关键概念: 量子计算、基本原理、应用前景"],
                acceptance=AcceptanceCriteria(
                    type="contains",
                    expected="量子",
                ),
                max_retries=1,
            ),
            # 阶段 2: 生成回答
            Milestone(
                name="generate_answer",
                description="生成结构化的通俗解释",
                tasks=[
                    "解释量子计算的基本原理（叠加态、纠缠等）",
                    "说明量子计算的应用前景（药物发现、密码学、优化等）",
                    "用类比帮助非技术读者理解",
                ],
                acceptance=AcceptanceCriteria(
                    type="contains",
                    expected="应用前景",  # 验证回答包含应用前景
                ),
                max_retries=2,
            ),
            # 阶段 3: 质量检查（自评估）
            Milestone(
                name="quality_check",
                description="检查回答质量和准确性",
                tasks=["检查回答是否包含所有要求的内容", "验证语言是否通俗易懂"],
                acceptance=AcceptanceCriteria(
                    type="llm_judge",
                    llm_prompt="评估这个回答是否：1)准确解释了量子计算原理 2)说明了应用前景 3)语言通俗易懂",
                ),
                max_retries=1,
            ),
        ],
        final_acceptance=AcceptanceCriteria(
            type="contains",
            expected="原理",  # 最终回答必须包含原理说明
        ),
    )

    # === 2. 创建组件 ===
    sensor = Sensor()
    checkpoint = Checkpoint(llm=llm)  # Checkpoint 需要 LLM 进行 llm_judge
    gate = Gate(strategy="auto")       # 自动策略

    harness = Harness(gate=gate, sensor=sensor, checkpoint=checkpoint)

    # === 3. 创建 LLM Executor ===
    executor = LLMExecutor(
        llm=llm,
        system_prompt="""你是一个科普作家，擅长将复杂的技术概念用通俗易懂的语言解释给非技术背景的读者。
你的回答应该：
1. 结构清晰，使用标题和分点
2. 必要时使用类比
3. 避免未经解释的专业术语
4. 内容准确，不夸大""",
    )

    # === 4. 执行任务 ===
    result = harness.run(spec, executor)

    # === 5. 输出结果 ===
    print("\n" + "=" * 70)
    print("📊 执行结果")
    print("=" * 70)
    print(f"成功: {result.success}")
    print(f"最终验收: {'通过' if result.final_acceptance_passed else '失败'}")

    print("\n📋 里程碑执行详情:")
    for ms_result in result.milestone_results:
        status = "✅" if ms_result["success"] else "❌"
        print(f"   {status} 重试 {ms_result.get('retries', 0)} 次")

    print("\n" + "=" * 70)
    print("📝 最终回答:")
    print("=" * 70)
    print(result.output)

    return result


def demo_mock_mode():
    """模拟模式：当没有 LLM 配置时使用本地函数模拟"""

    def mock_llm_executor(task: str, context: dict) -> str:
        """模拟 LLM 执行"""
        milestone = context.get("milestone")
        milestone_name = milestone.name if milestone else "unknown"
        previous_outputs = context.get("previous_outputs", {})

        if milestone_name == "understand_question":
            return "问题分析：用户想了解量子计算的基本原理和应用前景，目标读者是非技术背景人群。"

        elif milestone_name == "generate_answer":
            return """# 量子计算：未来的计算革命

## 基本原理

量子计算利用量子力学的两个核心特性：

1. **叠加态 (Superposition)**
   传统计算机的比特只能是 0 或 1，而量子比特可以同时处于 0 和 1 的叠加状态。
   类比：想象一枚旋转中的硬币，在它落地前，它既不是正面也不是反面，而是两者的叠加。

2. **量子纠缠 (Entanglement)**
   两个量子比特可以纠缠在一起，无论相距多远，测量一个会瞬间影响另一个。
   类比：有一对心灵感应的双胞胎，无论相隔多远，一个人的感受会瞬间传递给另一个人。

## 应用前景

量子计算在以下领域有巨大潜力：

1. **药物发现**
   模拟分子相互作用，加速新药研发。

2. **密码学**
   量子计算机可以破解现有加密，但也能创造更安全的量子加密。

3. **优化问题**
   解决物流、金融组合优化等复杂问题。

## 总结

量子计算仍处于早期阶段，但它代表了计算能力的质的飞跃，未来可能彻底改变多个行业。
"""

        elif milestone_name == "quality_check":
            answer = previous_outputs.get("generate_answer", "")
            checks = []
            if "叠加态" in answer:
                checks.append("✓ 包含原理解释")
            if "应用前景" in answer:
                checks.append("✓ 包含应用前景")
            if "类比" in answer:
                checks.append("✓ 使用了类比")
            return "质量检查结果:\n" + "\n".join(checks)

        return f"模拟执行: {milestone_name}"

    # 使用 FunctionExecutor 包装模拟函数
    from harness import FunctionExecutor

    spec = Spec(
        goal="解释量子计算的基本原理，并用通俗的语言说明其应用前景",
        inputs={"topic": "量子计算"},
        expected_output="包含原理解释和应用前景的通俗说明",
        constraints=["避免使用过多专业术语", "内容必须准确"],
        milestones=[
            Milestone(
                name="understand_question",
                description="理解用户问题的核心要点",
                tasks=["分析问题的关键概念"],
                acceptance=AcceptanceCriteria(type="contains", expected="量子"),
                max_retries=1,
            ),
            Milestone(
                name="generate_answer",
                description="生成结构化的通俗解释",
                tasks=["解释基本原理", "说明应用前景", "使用类比"],
                acceptance=AcceptanceCriteria(type="contains", expected="应用前景"),
                max_retries=2,
            ),
            Milestone(
                name="quality_check",
                description="检查回答质量和准确性",
                tasks=["检查内容完整性"],
                acceptance=AcceptanceCriteria(type="contains", expected="质量检查"),
                max_retries=1,
            ),
        ],
        final_acceptance=AcceptanceCriteria(type="contains", expected="原理"),
    )

    sensor = Sensor()
    checkpoint = Checkpoint()
    gate = Gate(strategy="auto")

    harness = Harness(gate=gate, sensor=sensor, checkpoint=checkpoint)
    executor = FunctionExecutor(mock_llm_executor)

    result = harness.run(spec, executor)

    print("\n" + "=" * 70)
    print("📊 模拟模式执行结果")
    print("=" * 70)
    print(f"成功: {result.success}")
    print(f"最终验收: {'通过' if result.final_acceptance_passed else '失败'}")

    print("\n📋 里程碑执行详情:")
    for ms_result in result.milestone_results:
        status = "✅" if ms_result["success"] else "❌"
        print(f"   {status} 重试 {ms_result.get('retries', 0)} 次")

    print("\n" + "=" * 70)
    print("📝 模拟回答内容:")
    print("=" * 70)
    # 取 generate_answer 阶段的输出
    for ms_result in result.milestone_results:
        if "generate_answer" in str(ms_result):
            output = ms_result.get("output", "")
            print(output[:500] + "..." if len(output) > 500 else output)
            break

    print("\n💡 提示: 配置 LLM_API_KEY 后可运行真实 LLM 模式")

    return result


if __name__ == "__main__":
    main()
