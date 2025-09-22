# BDI_EDU

## 1) 개요
본 프로젝트는 **BDI(Belief–Desire–Intention)** 아키텍처를 활용해 학생의 상태(Beliefs)를 추적하고, 명시적 학습목표(Desires)를 기반으로 학습전략(Intentions)을 **선택·생성·실행**하는 AI 튜터를 구축하는 것을 목적으로 한다.
정확한 의사결정과 설명 가능성을 확보하기 위해 **Knowledge Base + RAG + LLM**을 병행하고, 계획 선택에는 **NLI(Natural Language Inference, 자연어 추론) 기반 검증**, 계획 생성에는 **LLM**을 사용한다.

---

## 2) 범위
- **대상 과목/수준**: 한국사/고등학교 국사 6차(하)(https://contents.history.go.kr/mobile/ta/list.do?levelId=ta_h62_0040)
- **지원 시나리오**: 개념 설명, 예제 생성, 복습(출처) 경로 안내
- **디바이스/플랫폼**: 웹(학습자용)

---

## 3) 시스템 아키텍처

### 3.1 상위 구조
```
[Client]
      │
      ▼
[Reasoning Engine]
      │
      ┼───────────────────────┐
      │                       ┼────────────────────────┐
      ▼                       ▼                        ▼
 [Belief Manager]      [Intention Selector]       [Plan Executor]
 (학생 상태/이력/진단)     (규칙+NLI로 계획선택)       (LLM·RAG 실행)
      │                       │                        │
      ▼                       ▼                        ▼
 [Client DB]            [Manager DB]          [Vector DB/KB/콘텐츠]
                                                        │
                                                        ▼
                                            [Content Store(교재/문서/문제)]
```
> **Explanation**
> NLI(Natural Language Inference, 자연어 추론): 두 문장 간의 의미 관계를 판별하는 NLP 태스크
>
> 1. [Client]
> - User Side UI
> - Gradio UI로 작성
> - 질문 입력, 답안 제출, 피드백 수신
>
> 2. [Reasoning Engine]
> - 전체 플로우를 조율
> - Client의 입력을 수신
>   1. Belief Manager 호출 -> Client 상태 업데이트
>   2. Intention Selector 호출 -> '이번에는' 어떤 계획을 실행할지 결정
>   3. Plan Executor 호출 -> 실제 응답 생성
>   4. 결과 값을 Client에 return
>
> 3. [Belief Manager]
> - Client의 상태(Belief)를 추적 및 갱신하는 모듈
> - 정답률, 반응속도(풀이속도), 학습이력 등을 저장
>   - 'XX단원에서 특정 주제 X번 연속 오답 같은 정보' 업데이트
> - DataBase: Clinet_DB(Client별 프로파일)
>
> 4. [Intention Selector]
> - 현 상황에서 어떤 행동(Intention)을 취할지 선택하는 모듈
> - 규칙 기반: 예) "정답률 50% 미만 -> 설명 필요"
> - NLI 기반: 예) "Client가 이해한 개념과 계획 전제 조건이 맞는지 확인"
> - DataBase: Manager_DB(학습전략 규칙 저장)
>
> 5. [Plan Executor]
> - 실제로 응답(계획)을 실행하는 모듈
> - LLM 호출 -> 설명, 예시, 힌트 등을 생성
> - RAG 호출 -> 교재/문서/문제 가져오기
> - 최종 응답 조립 후 반환
> - DataBase
>   - Vector DB: 교재/문서 임베딩 검색용
>   - Knowledge Base: NEO 사용
>   - Content Store: 실제 학습 자료(교재, 문서, 문제)

### 3.2 핵심 모듈
- **Belief Manager**
  - 입력: 학생 답안/행동 로그/체류 시간/시도 횟수
  - 출력: 지식 상태 추정, 오개념, ~~난이도 프로파일(계획중)~~
  - 구현: 이벤트 스트림 수집 → ETL → 특징 추출 → 마스터리 모델(베이지안/IRT/간단한 규칙)
- **Intention Selector**
  - 후보: {개념 설명, 예제 제시, 힌트, 유사문항, 난이도 조정, 복습 재방문}
  - 선택 로직: (1) 규칙(정답률/반응시간/시도횟수) (2) **NLI로 “계획 조건 ⇐ 현재 Beliefs” 엔테일** 여부 확인
- **Plan Executor**
  - LLM로 텍스트/예제/해설 생성, **RAG**로 교재·문항 근거 회수, **템플릿 강제**(step-by-step·힌트·해설 포맷)

---

## 4) 데이터/지식 관리

- **지식베이스(KB)**: 이전 NEO 프로젝트에서 참고 예정
- **콘텐츠 소스**: EBS 등의 신뢰기관 자료
- **RAG 파이프라인**
  - 문서 청크, 메타데이터(학년·난이도·개념ID), 임베딩 인덱싱(Vector DB)
  - 검색 결과 재랭킹 → Context 융합 → Prompt 삽입

---

## 5) 모델 전략

- **LLM 역할 분리**
  1) **생성 모델**: 설명/예제/힌트 생성 (온프리미스 또는 폐쇄형 API 선택)
  2) **판정 모델(NLI/분류)**: 계획 조건 엔테일/난이도 판정/품질 평가지표
  3) **평가 모델**: 반환 전, 생성된 답변을 검사하여 적합한 결과 값인지 판단
     - 판단기준: RAG 근거와 불일치, 출력 포맷 오류
     - 합격: 결과 값 반환
     - 불합격: 평가 모델이 반환한 불합격 사유를 생성 모델에 추가하여 재생성(최대 3번까지 시도)
- **프롬프트 엔지니어링**
  - 관련없는 답변 거절, LLM이 Client에 대해 Question 가능하도록 설계
  - 고정 시스템 프롬프트 + 역할/톤 가이드 + 컨텍스트(학년/개념/오답패턴)
  - 출력 스키마 강제

---

## 6) BDI 튜터링 사이클(운영 로직)

1. **관찰(Observation)**: Clinet 입력/행동(풀이 시간, 정답률 등) 수집  
2. **Belief 업데이트**: Clint의 학습수준/오개념/난이도 추정 갱신  
3. **Desire 설정**: 현재 세션 목표(예: 일제 강점기 경제 수탈 정책의 특징을 이해한다.)  
4. **Intention 선택**: 규칙 + NLI로 계획 후보 필터링, 우선순위 결정  
5. **계획 실행**: LLM 생성+RAG 근거 제시, 템플릿 출력  
6. **피드백/적응**: 정답/오답/반응시간으로 Belief 재갱신, 필요 시 난이도/전략 전환 
7. **로그/감사**: 입력/컨텍스트/출력/의사결정 경로 저장(재현성·감사 대응)

---

## 7) 예상 리스크 & 대응

| 리스크 | 설명 | 대응책 |
|---|---|---|
| 사실성 저하 | LLM 환각 | RAG 근거 강제, 근거없는 진술 차단, 후처리 검증기 |
| 편향·부적절 | 유해/편향 응답 | 안전필터+교사 검수, 금칙 규칙, 히트맵 모니터링 |
| 개인정보 노출 | 입력/출력에 포함 | PII 탐지 마스킹, 프롬프트 정책, 로깅 익명화 |
| 성능 변동 | 모델/데이터 변화 | 회귀테스트, A/B, 버전 고정/핀닝 |

---

## 12) 체크리스트

- [ ] 도메인 KB 적재  
- [ ] 임베딩/벡터 DB 색인 + 메타데이터(학년·난이도·개념ID 등)  
- [ ] 규칙 기반 의사결정표(정답률/반응시간/시도)  
- [ ] NLI 모델 선택 및 엔테일 테스트(문장쌍 벤치)  
- [ ] LLM 프롬프트 템플릿/출력 스키마
- [ ] 추적로그



flowchart TD
    subgraph ClientSide[Client Side]
        A[학생 클라이언트\n(Gradio UI)]
    end

    subgraph ServerSide[Reasoning Engine]
        B[오케스트레이터\n(Reasoning Engine)]
        BM[Belief Manager\n학생 상태/이력/진단]
        IS[Intention Selector\n규칙+NLI로 계획 선택]
        PE[Plan Executor\nLLM+RAG 실행]
    end

    subgraph DBs[데이터 저장소]
        LDB[(Learner DB\n학생 프로파일)]
        MDB[(Strategy DB\n학습전략 규칙)]
        VDB[(Vector DB\n임베딩 검색)]
        KB[(Knowledge Base\nNEO 연계)]
        CS[(Content Store\n교재/문제/문서)]
    end

    %% Client → Engine
    A -->|입력(질문/답변)| B
    B --> BM
    B --> IS
    B --> PE

    %% Belief Manager 연결
    BM -->|업데이트/조회| LDB

    %% Intention Selector 연결
    IS -->|전략 규칙 조회| MDB

    %% Plan Executor 연결
    PE -->|검색| VDB
    PE -->|추론| KB
    PE -->|콘텐츠 호출| CS

    %% Engine → Client
    B -->|응답(설명/문제/피드백)| A
