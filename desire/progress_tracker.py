# progress_tracker.py
"""
ProgressTracker: 학습 진행 상황 추적
- GoalManager, Curriculum과 연결
- 목표 달성률, 진행 상황 관리
"""

from typing import Dict

class ProgressTracker:
    """
    학습 진행 상황 추적
    """

    def __init__(self, state_manager=None):
        self.progress = {}
        self.state_manager = state_manager  # Belief 연동용 (선택)

    def set_progress(self, topic: str, status: str):
        """주제별 진행 상황 기록"""
        self.progress[topic] = status
        self._update_state_progress()

    def get_progress(self, topic: str):
        return self.progress.get(topic, "pending")

    def overall_progress(self):
        """전체 완료율 (%) 계산"""
        if not self.progress:
            return 0.0
        done = sum(1 for v in self.progress.values() if v == "done")
        return (done / len(self.progress)) * 100

    def _update_state_progress(self):
        """Belief(StateManager)와 연동해서 progress 업데이트"""
        if self.state_manager:
            self.state_manager.update("progress", self.overall_progress())
