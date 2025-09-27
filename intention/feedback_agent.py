class FeedbackAgent:
    """
    실행된 결과를 평가하고, 필요하면 Planner/Executor로 피드백을 전달하는 모듈
    """

    def evaluate(self, results: list):
        """
        실행 결과에 대해 피드백 수행
        """
        # 단순 예시: 결과를 모두 잘 수행했다고 가정
        feedback = {
            "status": "ok",
            "comments": "모든 단계가 잘 수행되었습니다.",
            "issues": []
        }
        return feedback
