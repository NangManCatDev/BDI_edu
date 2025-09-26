"""
BDI-EDU 실행 엔트리포인트
--------------------------------
Belief → Desire → Intention → Interface → Agent
사이클을 한 번 돌려보는 샘플 파이프라인.
"""

# === Belief (지식/상태) ===
from belief.context_provider import Context
from belief.state_manager import StateManager

# === Desire (목표/커리큘럼) ===
from desire.curriculum import Curriculum
from desire.goal_manager import GoalManager
from desire.progress_tracker import ProgressTracker

# === Intention (계획/실행) ===
from intention.planner import Planner
from intention.executor import Executor
from intention.feedback_agent import FeedbackAgent

# === Interface (LLM, KQML ↔ NL 변환) ===
from interface.llm_connector import LLMConnector
from interface.nl2kqml import NL2KQML
from interface.kqml2nl import KQML2NL
from interface.chat_ui import ChatUI

# === Agents (JaCaMo 연동용) ===
# 실제 실행은 MAS 환경 필요. 여기서는 단순 placeholder.
# from agents.tutor_agent import TutorAgent


def main():
    print("=== [BDI-EDU: AI Tutor Cycle] 시작 ===")

    # -------------------------
    # 1) Belief: 초기 상태 로딩
    # -------------------------
    context = Context()
    state_manager = StateManager()
    state = state_manager.load_initial_state()
    print("[Belief] 초기 상태:", state)

    # -------------------------
    # 2) Desire: 목표 설정
    # -------------------------
    curriculum = Curriculum()
    goal_manager = GoalManager()
    progress_tracker = ProgressTracker()

    learning_goal = goal_manager.define_goal("수학_집합론_기초")
    curriculum_plan = curriculum.get_curriculum(learning_goal)
    progress_tracker.start_tracking(learning_goal)

    print("[Desire] 학습 목표:", learning_goal)
    print("[Desire] 커리큘럼:", curriculum_plan)

    # -------------------------
    # 3) Intention: 계획 수립 & 실행
    # -------------------------
    planner = Planner()
    executor = Executor()
    feedback = FeedbackAgent()

    plan = planner.create_plan(learning_goal, state)
    print("[Intention] 생성된 계획:", plan)

    result = executor.execute(plan)
    print("[Intention] 실행 결과:", result)

    feedback_msg = feedback.evaluate(result)
    print("[Intention] 피드백:", feedback_msg)

    # -------------------------
    # 4) Interface: 사용자 ↔ 에이전트 대화
    # -------------------------
    llm = LLMConnector()
    nl2kqml = NL2KQML(llm)
    kqml2nl = KQML2NL(llm)
    chat = ChatUI()

    user_msg = "집합의 원소란 무엇인가?"
    print("\n[Interface] 사용자 질문:", user_msg)

    kqml_msg = nl2kqml.translate(user_msg, context)
    print("[Interface] NL → KQML:", kqml_msg)

    # (여기서 KQML 메시지가 Tutor Agent로 전달된다고 가정)
    agent_reply_kqml = "(tell :content (element-of ?x ?set))"
    agent_reply_nl = kqml2nl.translate(agent_reply_kqml, context)

    chat.display(agent_reply_nl)

    # -------------------------
    # 5) Agent (Placeholder)
    # -------------------------
    # tutor_agent = TutorAgent()
    # tutor_agent.handle_message(kqml_msg)

    print("\n=== [BDI-EDU: Cycle 완료] ===")


if __name__ == "__main__":
    main()
