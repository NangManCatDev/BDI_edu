from belief.knowledge_base import build_kb
from belief.state_manager import StateManager
from pylo.language.lp import Atom, c_const, c_var

def pretty_results(results):
    return [{k: v for k, v in r.items()} if r else {} for r in results]

def main():
    eng = build_kb()
    sm = StateManager()

    print("=== Belief Query Examples ===")

    # 1) 삼국통일 조회
    q1 = Atom(eng.predicates["event"], [c_var("X"), c_const("'668'"), c_var("Y")])
    res1 = eng.query(q1)
    print("event(X, '668', Y):", pretty_results(res1))

    # 최근 학습 주제 갱신
    if res1:
        sm.set_last_topic(res1[0].get("X"))  # '삼국통일'

    # 2) 세종 조회
    q2 = Atom(eng.predicates["person"], [c_const("'세종'"), c_var("B"), c_var("D"), c_var("R")])
    res2 = eng.query(q2)
    print("person('세종', B, D, R):", pretty_results(res2))

    if res2:
        sm.set_last_topic("세종")

    # 3) 동학농민운동 원인 조회
    q3 = Atom(eng.predicates["cause"], [c_const("'동학농민운동'"), c_var("X")])
    res3 = eng.query(q3)
    print("cause('동학농민운동', X):", pretty_results(res3))

    if res3:
        sm.set_last_topic("동학농민운동")

    # === 최종 상태 출력 ===
    print("\n=== Current State ===")
    print(sm.all())

if __name__ == "__main__":
    main()
