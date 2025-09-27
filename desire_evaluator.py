# desire_evaluator.py
"""
Desire Evaluator with Belief 연동
- Belief 모듈의 last_topic을 GoalManager로 연결
- Curriculum, ProgressTracker와 함께 테스트 실행
"""

from belief.state_manager import StateManager
from desire.goal_manager import GoalManager
from desire.curriculum import Curriculum
from desire.progress_tracker import ProgressTracker


def main():
    print("=== Desire Evaluator (with Belief) ===\n")

    # Belief
    state = StateManager()

    # Desire
    goal_manager = GoalManager()
    curriculum = Curriculum()
    progress_tracker = ProgressTracker(state_manager=state)  # Belief 연동

    # Curriculum 등록
    curriculum.add("short_term", "삼국 통일")
    curriculum.add("mid_term", "조선 건국")
    curriculum.add("long_term", "근대 개혁")
    print("[Curriculum 전체]")
    print(curriculum.show(), "\n")

    # Goal 등록
    goal_manager.add_goal("삼국 통일 학습", priority=2)
    goal_manager.add_goal("조선 건국 이해", priority=1)
    print("[목표 리스트]")
    print(goal_manager.show(), "\n")

    # 우선순위가 가장 높은 목표
    print("[우선순위가 가장 높은 목표]")
    print(goal_manager.highest_priority(), "\n")

    # 진행 상황 (삼국 통일 완료 처리)
    progress_tracker.set_progress("삼국 통일", "done")
    progress_tracker.set_progress("조선 건국", "pending")
    print("[진행 상황]")
    for t, s in progress_tracker.progress.items():
        print(f"{t}: {s}")
    print("전체 완료율:", progress_tracker.overall_progress(), "%\n")

    # 목표 완료 처리
    goal_manager.complete_goal("삼국 통일 학습")
    print("[목표 완료 후 상태]")
    print(goal_manager.show(), "\n")

    # Belief 상태 확인
    state.set_last_topic("삼국 통일")
    print("[Belief 상태 확인]")
    print(state.show())


if __name__ == "__main__":
    main()