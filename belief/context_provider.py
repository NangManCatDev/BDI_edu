class Context:
    """
    대화 및 학습에 필요한 환경적 맥락 제공
    예: 과목, 학습 단계, 사용자 정보
    """

    def __init__(self):
        self.context_info = {
            "domain": "education",
            "subject": "math",
            "language": "ko",
        }

    def get_context(self) -> dict:
        return self.context_info

    def update_context(self, key: str, value):
        self.context_info[key] = value
        return self.context_info
