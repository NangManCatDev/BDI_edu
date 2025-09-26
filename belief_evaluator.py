"""
belief_evaluator.py
-------------------
Belief 모듈 단독 실행/테스트 스크립트
"""

from belief import Context, StateManager, PrologEngine
import os


def main():
    print("=== Belief Evaluator 실행 ===")

    # -------------------------
    # Context 테스트
    # -------------------------
    context = Context()
    print("[Context] 초기값:", context.get_context())
    context.update_context("subject", "discrete_math")
    print("[Context] 업데이트 후:", context.get_context())

    # -------------------------
    # StateManager 테스트
    # -------------------------
    state_manager = StateManager()
    state = state_manager.load_initial_state()
    print("[State] 초기 상태:", state)

    state_manager.update_state("progress", 1)
    print("[State] 업데이트 후:", state_manager.get_state())

    # -------------------------
    # Prolog KB 테스트
    # -------------------------
    kb_path = os.path.join("belief", "knowledge_base.pl")
    prolog_engine = PrologEngine(kb_path)

    print("\n[Prolog] example/2 질의 결과:")
    for sol in prolog_engine.query("example(Q, A)"):
        print(f"  Q: {sol['Q']}, A: {sol['A']}")

    print("\n[Prolog] definition/2 질의 결과:")
    for sol in prolog_engine.query("definition(element_of, Def)."):
        print("  element_of 정의:", sol["Def"])

    print("\n=== Belief Evaluator 완료 ===")


if __name__ == "__main__":
    main()
