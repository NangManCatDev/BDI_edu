# --- Python 3.10+ 호환성 패치 ---
import collections
import collections.abc

if not hasattr(collections, "Hashable"):
    collections.Hashable = collections.abc.Hashable
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable
if not hasattr(collections, "Iterator"):
    collections.Iterator = collections.abc.Iterator
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
    
# FastAPI app 객체 직접 정의
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from Dockerized FastAPI!"}





# interface/chat_ui.py
from interface.nl2kqml import nl_to_kqml
from interface.kqml2nl import kqml_to_nl
from belief.knowledge_base import build_kb
from belief.state_manager import StateManager
from desire.goal_manager import GoalManager
from desire.curriculum import Curriculum
from desire.progress_tracker import ProgressTracker
from intention.planner import Planner
from intention.executor import Executor
from intention.feedback_agent import FeedbackAgent

def main():
    print("=== BDI Tutor Chat ===\n")
    state = StateManager()
    eng = build_kb()

    # 사용자 질문
    question = input("[You] ")
    atom = nl_to_kqml(question)

    print("[NL → KQML]", atom)

    # Belief 질의
    results = eng.query(atom)
    if not results:
        print("[Belief] 결과 없음")
        return

    # 첫 번째 결과를 자연어로 변환
    answer = kqml_to_nl(atom.substitute(results[0]))
    print("[KQML → NL]", answer, "\n")

    # Desire 단계: 목표 생성
    goal_manager = GoalManager()
    curriculum = Curriculum()
    progress = ProgressTracker(state)

    curriculum.add("short_term", question)
    goal_manager.add(answer, priority=2)

    print("[Desire: 목표]", goal_manager.show())

    # Intention 단계: 계획 → 실행 → 피드백
    planner = Planner()
    executor = Executor()
    feedback = FeedbackAgent()

    goal = goal_manager.get_highest_priority()
    plan = planner.make_plan(goal["goal"])
    exec_results = executor.execute(plan)
    fb = feedback.evaluate(exec_results)

    print("\n[Intention: 계획 수립]")
    for step in plan:
        print("-", step)

    print("\n[Intention: 실행 결과]")
    for step in exec_results:
        print("-", step)

    print("\n[Intention: 피드백]")
    print(fb)

if __name__ == "__main__":
    main()
