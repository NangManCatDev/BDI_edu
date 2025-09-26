class StateManager:
    """
    학습자의 상태를 관리하는 모듈
    예: 현재 수준, 진행 상황, 최근 대화 맥락
    """

    def __init__(self):
        self.state = {
            "student_level": "beginner",
            "progress": 0,
            "last_topic": None,
        }

    def load_initial_state(self) -> dict:
        """최초 실행 시 불러올 상태"""
        return self.state

    def update_state(self, key: str, value):
        """특정 상태를 갱신"""
        self.state[key] = value
        return self.state

    def get_state(self) -> dict:
        """현재 상태 조회"""
        return self.state
