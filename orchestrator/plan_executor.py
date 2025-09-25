# Note: 계획 실행 (LLM+RAG)


"""
plan_executor.py
----------------
선택된 Intention에 따라 실제 응답을 생성하는 모듈 (더미 버전)
LLM + RAG 자리에 단순한 텍스트 응답을 리턴함
"""

class PlanExecutor:
    def __init__(self):
        pass

    def execute(self, plan: str, user_input: str) -> str:
        """
        단순 실행기 (LLM 호출 대신 더미 응답)
        """
        if plan == "explain":
            return "📘 개념 설명: 일제 강점기 경제 수탈 정책은 회사령, 산미 증식 계획, 공업 정책 등이 있습니다."
        elif plan == "quiz":
            return "❓ 문제: 다음 중 1920년대 산미 증식 계획의 결과로 옳지 않은 것은?"
        else:
            return "🤔 아직 지원하지 않는 계획입니다."
