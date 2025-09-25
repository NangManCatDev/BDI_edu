"""
orchestrator package
--------------------
본 패키지는 BDI(Belief–Desire–Intention) 아키텍처 기반 AI 튜터의 핵심 로직을 담당한다.

구성 모듈:
- belief_manager.py      : 학습자의 상태(Beliefs) 추적 및 업데이트
- intention_selector.py  : 학습 전략(Intentions) 선택 (규칙 기반 + NLI 검증)
- plan_executor.py       : 선택된 계획 실행 (LLM + RAG 기반 응답 생성)
- orchestrator.py        : 전체 Reasoning Engine을 조율하는 상위 컨트롤러
- utils.py               : 로깅, 포맷 검증 등 공용 유틸리티

역할:
1. 학습자 입력을 수집하고 Belief Manager로 상태를 갱신한다.
2. Intention Selector를 통해 실행 가능한 학습 전략을 판별한다.
3. Plan Executor를 호출하여 최종 출력(설명, 문제, 힌트 등)을 생성한다.
4. 로그 및 감사 기록을 남겨 재현성과 품질을 보장한다.

외부 사용 예시:
    from orchestrator import BeliefManager, IntentionSelector, PlanExecutor

주의:
- 본 패키지는 Prolog 기반 Knowledge Base(kb/)와 연동하도록 설계됨
- 테스트 코드는 tests/ 디렉토리 참고
"""
-