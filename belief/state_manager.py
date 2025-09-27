class StateManager:
    """
    학습자의 상태를 관리하는 모듈
    예: 현재 수준, 진행 상황, 최근 대화 맥락
    """

    def __init__(self):
        self.state = {
            "progress": 0,
            "level": "beginner",
            "last_topic": None
        }

    def update(self, key, value):
        self.state[key] = value

    def get(self, key, default=None):
        return self.state.get(key, default)

    def all(self):
        return self.state

    # === last_topic 전용 헬퍼 ===
    def set_last_topic(self, topic: str):
        self.state["last_topic"] = topic

    def get_last_topic(self):
        return self.state.get("last_topic", None)

    # === 별칭 ===
    def show(self):
        return self.all()