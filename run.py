from belief.state_manager import StateManager
from belief.knowledge_base import build_kb

from desire.curriculum import Curriculum
from desire.goal_manager import GoalManager
from desire.progress_tracker import ProgressTracker

from intention.planner import Planner
from intention.executor import Executor
from intention.feedback_agent import FeedbackAgent


def main():
    print("=== BDI Orchestrator ===\n")

    # ------------------------
    # Belief 단계
    # ------------------------
    state = StateManager()
    kb = build_kb()

    print("[Belief 상태 초기화]")
    print(state.all(), "\n")

    # ------------------------
    # Desire 단계
    # ------------------------
    curriculum = Curriculum()
    curriculum.add("short_term", "삼국 통일")
    curriculum.add("mid_term", "조선 건국")

    goal_manager = GoalManager()
    goal_manager.add("삼국 통일 학습", priority=2)
    goal_manager.add("조선 건국 이해", priority=1)

    tracker = ProgressTracker(state_manager=state)
    tracker.set_progress("삼국 통일", "done")   # 예: 삼국 통일 학습 완료

    print("[Curriculum]")
    print(curriculum.show(), "\n")

    print("[목표 리스트]")
    print(goal_manager.show(), "\n")

    highest_goal = goal_manager.get_highest_priority()
    print("[우선순위가 가장 높은 목표]")
    print(highest_goal, "\n")

    # Belief 갱신 (last_topic)
    if highest_goal:
        state.set_last_topic(highest_goal["goal"])

    print("[Belief 상태 업데이트]")
    print(state.all(), "\n")

    # ------------------------
    # Intention 단계
    # ------------------------
    if highest_goal:
        goal = highest_goal["goal"]

        planner = Planner()
        plan = planner.make_plan(goal)

        print("[계획 수립]")
        for step in plan:
            print("-", step)
        print()

        executor = Executor()
        results = executor.execute(plan)

        print("[실행 결과]")
        for r in results:
            print("-", r)
        print()

        feedback_agent = FeedbackAgent()
        feedback = feedback_agent.evaluate(results)

        print("[피드백]")
        print(feedback)


if __name__ == "__main__":
    main()
