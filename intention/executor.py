class Executor:
    """
    Planner가 만든 학습 단계를 실제로 실행하는 모듈
    Belief 모듈(KB 조회) 또는 외부 자료 호출이 여기에 들어감
    """

    def execute(self, plan_steps: list):
        """
        계획된 step 리스트를 실행하고 결과 반환
        """
        results = []
        for step in plan_steps:
            # 실제 구현에서는 Belief KB 질의나 LLM 호출이 들어감
            results.append(f"Executed: {step}")
        return results
