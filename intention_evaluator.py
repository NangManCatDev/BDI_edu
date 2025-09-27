from intention.planner import Planner
from intention.executor import Executor
from intention.feedback_agent import FeedbackAgent


def main():
    print("=== Intention Evaluator ===\n")

    # 1) 목표 예시 (보통 Desire 모듈에서 넘어옴)
    goal = "조선 건국 이해"

    # === Planner 단계 ===
    planner = Planner()
    plan = planner.make_plan(goal)
    print("[계획 수립]")
    for step in plan:
        print("-", step)
    print()

    # === Executor 단계 ===
    executor = Executor()
    results = executor.execute(plan)
    print("[실행 결과]")
    for r in results:
        print("-", r)
    print()

    # === Feedback 단계 ===
    feedback_agent = FeedbackAgent()
    feedback = feedback_agent.evaluate(results)
    print("[피드백]")
    print(feedback)


if __name__ == "__main__":
    main()
