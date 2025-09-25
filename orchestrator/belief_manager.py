# Note: Belief 관리 모듈

"""
belief_manager.py
-----------------
학습자의 상태(Beliefs)를 관리하는 모듈 (더미 버전)
"""

class BeliefManager:
    def __init__(self):
        # 단순 상태 저장 (정답률, 시도 횟수 등)
        self.state = {
            "correct": 0,
            "total": 0
        }

    def update(self, user_input: str) -> dict:
        """
        학습자의 입력을 바탕으로 상태 업데이트 (더미)
        실제 구현에서는 정답/오답 판정, 반응시간 기록 등이 필요함
        """
        self.state["total"] += 1

        # 아주 단순하게 "정답"이라는 단어가 있으면 맞은 걸로 간주
        if "정답" in user_input:
            self.state["correct"] += 1

        return self.state
