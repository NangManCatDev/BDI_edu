"""
desire_manager.py
-----------------
현재 세션의 학습 목표(Desires)를 설정하고 관리하는 모듈
"""

class DesireManager:
    def __init__(self):
        # 단순히 세션 목표 하나만 저장하는 구조
        self.current_goal = None

    def set_goal(self, topic: str) -> str:
        """
        학습 목표(Desire)를 설정한다.
        예: "일제 강점기 경제 수탈 정책 이해하기"
        """
        self.current_goal = topic
        return self.current_goal

    def get_goal(self) -> str:
        """
        현재 설정된 목표 반환
        """
        return self.current_goal or "아직 목표가 설정되지 않았습니다."
