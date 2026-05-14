"""
Executor: 执行器抽象

原则：分层可控——Executor 负责"做什么"，Harness 负责"怎么控"。
Harness 通过统一的 Executor 接口调用具体执行逻辑，
而不关心执行器内部是函数调用还是 LLM Agent。

这种解耦使得 Harness 框架可以驾驭任何类型的执行单元。
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class Executor(ABC):
    """
    执行器抽象基类

    所有具体的执行器（函数执行器、LLM执行器、Agent执行器）都必须继承此类。
    Harness 只与这个接口交互，实现"控制层"与"执行层"的分离。
    """

    @abstractmethod
    def execute(self, task: str, context: Dict[str, Any]) -> str:
        """
        执行一个任务

        Args:
            task: 任务描述字符串（来自 Milestone.tasks 中的某一项）
            context: 执行上下文，包含：
                - spec: 完整的 Spec 对象
                - milestone: 当前 Milestone 对象
                - inputs: 任务输入参数
                - previous_outputs: 前面里程碑的输出结果

        Returns:
            任务的执行结果（字符串）
        """
        ...


class FunctionExecutor(Executor):
    """
    函数执行器

    将 Python 函数包装为 Executor。
    函数签名: fn(task: str, context: dict) -> str

    适用于：纯计算任务、API 调用、数据处理等确定性任务。
    """

    def __init__(self, fn: Callable[[str, Dict[str, Any]], str]):
        self.fn = fn

    def execute(self, task: str, context: Dict[str, Any]) -> str:
        return self.fn(task, context)


class LLMExecutor(Executor):
    """
    LLM 执行器

    使用 LLM 执行任务的 Executor。
    与 hello_agents 框架集成，接收 HelloAgentsLLM 实例。

    适用于：需要推理、生成、分析等开放性任务。
    """

    def __init__(self, llm, system_prompt: Optional[str] = None):
        """
        Args:
            llm: HelloAgentsLLM 实例（或任何有 invoke(messages) 方法的对象）
            system_prompt: 系统提示词
        """
        self.llm = llm
        self.system_prompt = system_prompt or "你是一个任务执行助手。"

    def execute(self, task: str, context: Dict[str, Any]) -> str:
        """构造 prompt 并调用 LLM 执行任务"""
        messages = []

        # 系统提示
        messages.append({"role": "system", "content": self.system_prompt})

        # 构建用户提示，包含上下文
        prompt_parts = [f"任务: {task}"]

        # 添加 Spec 上下文
        spec = context.get("spec")
        if spec:
            prompt_parts.append(f"\n整体目标: {spec.goal}")
            if spec.constraints:
                prompt_parts.append(f"约束条件: {', '.join(spec.constraints)}")

        # 添加上下文输入
        inputs = context.get("inputs", {})
        if inputs:
            prompt_parts.append(f"\n输入数据: {inputs}")

        # 添加前面里程碑的输出（如果有）
        previous = context.get("previous_outputs", {})
        if previous:
            prompt_parts.append("\n前面阶段的输出:")
            for name, output in previous.items():
                prompt_parts.append(f"  [{name}]: {output[:500]}")  # 截断避免过长

        messages.append({"role": "user", "content": "\n".join(prompt_parts)})

        # 调用 LLM
        return self.llm.invoke(messages)
