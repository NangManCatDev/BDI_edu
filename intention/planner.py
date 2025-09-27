class Planner:
    """
    목표(Desire)를 받아서 실행 가능한 단계(Intention Plan)로 분해하는 모듈
    """

    def make_plan(self, goal: str):
        """
        주어진 목표를 작은 학습 단계로 분해
        예: "조선 건국 이해" -> ["위화도 회군 학습", "태조 즉위 과정 학습", "조선 건국 정리"]
        """
        # 임시 예시: 단순히 goal을 2~3개 단계로 나누기
        steps = [
            f"{goal} - 배경 학습",
            f"{goal} - 핵심 사건 분석",
            f"{goal} - 정리 및 복습"
        ]
        return steps
