#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立测试文件 - 不修改原代码，直接测试旅行规划主流程
"""
import sys
import os
from pathlib import Path

# 🔥 自动把项目根目录加入Python路径（解决所有导入问题）
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# 直接从原项目导入（不需要复制原代码！）
from app.models.schemas import TripRequest
from app.agents.trip_planner_agent import get_trip_planner_agent


# ======================
# 测试主函数
# ======================
def test_trip_planner():
    print("🚀 开始独立测试旅行规划系统")

    # 1. 创建请求对象（你要的 new 对象）
    request = TripRequest(
        city="杭州",
        start_date="2025-04-01",
        end_date="2025-04-03",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化", "美食"],
        free_text_input="希望多安排一些博物馆"
    )

    print("✅ 测试请求创建成功")
    print(f"城市: {request.city}")
    print(f"天数: {request.travel_days}")

    # 2. 获取规划器并执行
    try:
        planner = get_trip_planner_agent()
        result = planner.plan_trip(request)
        print("🎉 测试执行成功！")
        return result
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


# ======================
# 运行测试
# ======================
if __name__ == "__main__":
    test_trip_planner()