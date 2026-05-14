"""
示例 2: 多阶段完整流水线——研究报告生成

本示例演示 Harness 框架如何驾驭一个多阶段复杂任务：
阶段 1: 数据收集 -> 阶段 2: 数据分析 -> 阶段 3: 报告生成

每个阶段有独立的验收标准，阶段之间通过 Memory 传递数据。

四大原则体现：
- Spec-Driven: 完整定义研究目标、各阶段任务、验收标准
- Gate+Sensor: 每个阶段流转由 Gate 控制，Sensor 全程记录
- Milestone+Checkpoint: 3 个里程碑，每个都有 Checkpoint 验收
- Acceptance-Driven: 每阶段和最终报告都必须通过验收
"""

from harness import (
    Spec, Milestone, AcceptanceCriteria,
    Harness, FunctionExecutor,
    Gate, Sensor, Checkpoint,
)


# === 模拟数据源 ===
MOCK_DATA = {
    "products": [
        {"name": "Product A", "sales": 1200, "rating": 4.5},
        {"name": "Product B", "sales": 800, "rating": 4.2},
        {"name": "Product C", "sales": 1500, "rating": 4.8},
        {"name": "Product D", "sales": 600, "rating": 3.9},
    ]
}


def research_executor(task: str, context: dict) -> str:
    """
    研究任务执行函数

    根据当前里程碑执行不同的模拟逻辑。
    实际场景中，这里可以是 API 调用、数据库查询、LLM 调用等。
    """
    milestone = context.get("milestone")
    milestone_name = milestone.name if milestone else "unknown"
    previous_outputs = context.get("previous_outputs", {})

    if milestone_name == "collect_data":
        # 阶段 1: 数据收集
        # 注意：一个 milestone 有多个 tasks，每个 task 的输出会被合并。
        # 为避免多个 JSON 拼接导致 parse 失败，只在第一个 task 返回数据。
        import json
        task_index = context.get("task_index", 1)
        if task_index == 1:
            return json.dumps(MOCK_DATA, ensure_ascii=False, indent=2)
        return f"步骤 {task_index} 完成"

    elif milestone_name == "analyze_data":
        # 阶段 2: 数据分析
        # 只在第一个 task 返回完整分析，避免合并输出时重复
        task_index = context.get("task_index", 1)
        if task_index > 1:
            return ""

        # 获取上一阶段的输出（可能包含多个任务的合并输出）
        raw_data = previous_outputs.get("collect_data", "{}")
        import json
        import re

        # 尝试从合并输出中提取 JSON 部分
        data = None
        try:
            # 先尝试直接解析
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            # 如果失败，尝试从文本中提取 JSON 对象
            json_match = re.search(r'\{.*\}', raw_data, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        if data is None:
            return "数据分析失败: 无法从上游输出中解析数据"

        products = data.get("products", [])
        total_sales = sum(p["sales"] for p in products)
        avg_rating = sum(p["rating"] for p in products) / len(products) if products else 0
        best_product = max(products, key=lambda x: x["sales"]) if products else None

        analysis = f"""数据分析结果:
- 产品总数: {len(products)}
- 总销量: {total_sales}
- 平均评分: {avg_rating:.2f}
- 销量冠军: {best_product['name'] if best_product else 'N/A'} (销量: {best_product['sales'] if best_product else 0})
"""
        return analysis

    elif milestone_name == "generate_report":
        # 阶段 3: 报告生成
        # 只在第一个 task 生成完整报告，其他 task 返回空（避免重复）
        task_index = context.get("task_index", 1)
        if task_index > 1:
            return ""

        analysis = previous_outputs.get("analyze_data", "")
        report = f"""# 产品研究报告

## 1. 执行摘要
本报告基于 {len(MOCK_DATA['products'])} 款产品的销售数据进行分析。

## 2. 数据分析
{analysis}

## 3. 关键发现
- 产品 C 表现最佳，销量领先
- 整体平均评分超过 4.0，产品质量良好
- Product D 销量较低，需关注

## 4. 建议
1. 加大 Product C 的推广力度
2. 分析 Product D 销量低的原因
3. 保持现有产品质量标准

---
报告生成时间: 2026-05-14
"""
        return report

    return f"未知里程碑: {milestone_name}"


def main():
    print("=" * 70)
    print("📊 示例 2: 多阶段流水线——研究报告生成")
    print("=" * 70)

    # === 1. 定义 Spec ===
    spec = Spec(
        goal="基于产品数据生成一份结构化的研究报告",
        inputs={"data_source": "mock_database"},
        expected_output="包含数据收集、分析、报告三个阶段的完整研究报告",
        constraints=[
            "报告必须包含执行摘要、数据分析、关键发现和建议",
            "数据必须来自指定的数据源",
        ],
        milestones=[
            # 阶段 1: 数据收集
            Milestone(
                name="collect_data",
                description="从数据源收集原始产品数据",
                tasks=["连接数据源", "提取产品信息", "验证数据完整性"],
                acceptance=AcceptanceCriteria(
                    type="contains",
                    expected="Product A",  # 验证数据是否包含预期内容
                ),
                max_retries=2,
            ),
            # 阶段 2: 数据分析
            Milestone(
                name="analyze_data",
                description="分析产品数据，计算关键指标",
                tasks=["计算总销量", "计算平均评分", "找出销量冠军"],
                acceptance=AcceptanceCriteria(
                    type="contains",
                    expected="销量冠军",  # 验证分析结果是否包含关键结论
                ),
                max_retries=2,
            ),
            # 阶段 3: 报告生成
            Milestone(
                name="generate_report",
                description="生成结构化的研究报告",
                tasks=["撰写执行摘要", "整理数据分析结果", "提出建议和结论"],
                acceptance=AcceptanceCriteria(
                    type="contains",
                    expected="建议",  # 验证报告是否包含建议部分
                ),
                max_retries=2,
            ),
        ],
        final_acceptance=AcceptanceCriteria(
            type="contains",
            expected="# 产品研究报告",  # 验证报告格式正确
        ),
    )

    # === 2. 创建组件 ===
    sensor = Sensor()
    checkpoint = Checkpoint()
    gate = Gate(strategy="auto")

    harness = Harness(gate=gate, sensor=sensor, checkpoint=checkpoint)
    executor = FunctionExecutor(research_executor)

    # === 3. 执行任务 ===
    result = harness.run(spec, executor)

    # === 4. 输出结果 ===
    print("\n" + "=" * 70)
    print("📊 执行结果")
    print("=" * 70)
    print(f"成功: {result.success}")
    print(f"最终验收: {'通过' if result.final_acceptance_passed else '失败'}")

    # === 5. 查看各里程碑结果 ===
    print("\n📋 里程碑执行详情:")
    for ms_result in result.milestone_results:
        status = "✅" if ms_result["success"] else "❌"
        print(f"   {status} {ms_result.get('milestone', 'unknown')}: "
              f"重试 {ms_result.get('retries', 0)} 次")

    # === 6. 查看最终报告 ===
    print("\n" + "=" * 70)
    print("📝 最终报告内容:")
    print("=" * 70)
    print(result.output)

    # === 7. 查看 Memory 摘要 ===
    print("\n🧠 Memory 摘要:")
    summary = result.memory_summary
    print(f"   总事件数: {summary['total_events']}")
    print(f"   里程碑状态:")
    for name, state in summary.get("milestones", {}).items():
        print(f"      - {name}: {state.get('status', 'unknown')}")

    # === 8. 导出执行记录（可选）===
    # sensor.memory.export_json("harness_execution_log.json")
    # print("\n💾 执行记录已导出到 harness_execution_log.json")

    return result


if __name__ == "__main__":
    main()
