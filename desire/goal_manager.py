# goal_manager.py
"""
GoalManager: 학습 목표(Desire) 생성 및 관리
- Belief/Interface에서 들어온 입력을 바탕으로 목표 생성
- 우선순위 기반으로 목표를 정렬/선택
"""

from typing import List, Dict, Optional

class GoalManager:
    def __init__(self):
        self.goals = []

    def add(self, goal: str, priority: int = 1):
        self.goals.append({"goal": goal, "priority": priority, "status": "pending"})

    def add_goal(self, goal: str, priority: int = 1):
        return self.add(goal, priority)

    def complete(self, goal: str):
        for g in self.goals:
            if g["goal"] == goal:
                g["status"] = "done"

    # === 별칭 메서드 ===
    def complete_goal(self, goal: str):
        return self.complete(goal)

    def get_highest_priority(self):
        if not self.goals:
            return None
        return max(self.goals, key=lambda g: g["priority"])

    def highest_priority(self):
        return self.get_highest_priority()

    def show(self):
        return self.goals
