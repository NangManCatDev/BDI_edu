# Note: 전체 플로우 제어 모듈


"""
intention_selector.py
---------------------
현재 Beliefs를 바탕으로 실행할 학습 전략(Intention)을 선택 (더미 버전)
"""

class IntentionSelector:
    def __init__(self):
        pass

    def select(self, belief_state: dict) -> str:
        """
        단순 규칙 기반 선택
        - 정답률 50% 미만: 'explain' (개념 설명)
        - 정답률 50% 이상: 'quiz' (문제 출제)
        """
        total = belief_state["total"]
        correct = belief_state["correct"]

        if total == 0:
            return "explain"

        accuracy = correct / total

        if accuracy < 0.5:
            return "explain"
        else:
            return "quiz"
