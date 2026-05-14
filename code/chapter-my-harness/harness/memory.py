"""
Memory: 感知记忆存储

原则：Gate+Sensor（闸门 + 感知记忆）——Sensor 持续感知执行状态，
所有感知数据写入 Memory，供 Gate 决策时查询。

Memory 提供结构化的存储和查询能力，是 Harness 的"记忆中枢"。
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json


class Memory:
    """
    感知记忆存储器

    记录所有执行过程中的事件，支持按里程碑、事件类型查询。
    为 Gate 的决策提供历史数据支持。
    """

    def __init__(self):
        # events: 按时间顺序记录的所有事件
        self._events: List[Dict[str, Any]] = []
        # milestone_states: 每个里程碑的当前状态快照
        self._milestone_states: Dict[str, Dict[str, Any]] = {}

    def record(self, event: Dict[str, Any]) -> None:
        """记录一个事件，自动附加时间戳"""
        event_copy = dict(event)
        event_copy["timestamp"] = datetime.now().isoformat()
        self._events.append(event_copy)

        # 如果是里程碑相关事件，更新里程碑状态
        if "milestone" in event_copy and event_copy["milestone"]:
            ms_name = event_copy["milestone"]
            if ms_name not in self._milestone_states:
                self._milestone_states[ms_name] = {}
            # 根据事件类型更新状态
            event_type = event_copy.get("type", "")
            if event_type == "milestone_start":
                self._milestone_states[ms_name]["status"] = "running"
                self._milestone_states[ms_name]["started_at"] = event_copy["timestamp"]
            elif event_type == "milestone_end":
                self._milestone_states[ms_name]["status"] = "completed"
                self._milestone_states[ms_name]["ended_at"] = event_copy["timestamp"]
            elif event_type == "milestone_failed":
                self._milestone_states[ms_name]["status"] = "failed"
                self._milestone_states[ms_name]["error"] = event_copy.get("data", {}).get("error", "")
            elif event_type == "checkpoint_result":
                self._milestone_states[ms_name]["checkpoint"] = event_copy.get("data", {})
            elif event_type == "gate_decision":
                self._milestone_states[ms_name]["gate_decision"] = event_copy.get("data", {})

    def query(
        self,
        milestone: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """按条件查询事件"""
        results = self._events
        if milestone:
            results = [e for e in results if e.get("milestone") == milestone]
        if event_type:
            results = [e for e in results if e.get("type") == event_type]
        return results[-limit:]

    def get_milestone_status(self, name: str) -> Optional[Dict[str, Any]]:
        """获取指定里程碑的当前状态"""
        return self._milestone_states.get(name)

    def all_milestones(self) -> Dict[str, Dict[str, Any]]:
        """获取所有里程碑状态"""
        return dict(self._milestone_states)

    def summary(self) -> Dict[str, Any]:
        """生成执行摘要"""
        total = len(self._events)
        by_type: Dict[str, int] = {}
        for e in self._events:
            t = e.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_events": total,
            "event_breakdown": by_type,
            "milestones": self.all_milestones(),
        }

    def export_json(self, path: str) -> None:
        """导出所有事件到 JSON 文件（用于调试和复盘）"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "events": self._events,
                    "milestone_states": self._milestone_states,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def clear(self) -> None:
        """清空所有记忆（用于复用 Harness 实例）"""
        self._events.clear()
        self._milestone_states.clear()
