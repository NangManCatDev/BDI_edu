# config/domain_config.py
"""
도메인별 설정 관리 모듈
다양한 과목과 언어를 지원하는 일반화된 설정
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class Domain(Enum):
    HISTORY = "history"
    MATH = "math"
    SCIENCE = "science"
    LANGUAGE = "language"
    LITERATURE = "literature"
    GEOGRAPHY = "geography"

class Language(Enum):
    KOREAN = "ko"
    ENGLISH = "en"
    CHINESE = "zh"
    JAPANESE = "ja"

@dataclass
class DomainConfig:
    """도메인별 설정"""
    domain: Domain
    language: Language
    predicates: Dict[str, int]  # predicate_name -> arity
    keywords: Dict[str, List[str]]  # question_type -> keywords
    system_prompt: str
    examples: List[str]

# 도메인별 설정 정의
DOMAIN_CONFIGS = {
    Domain.HISTORY: DomainConfig(
        domain=Domain.HISTORY,
        language=Language.KOREAN,
        predicates={
            "event": 3,      # event(Name, Year, Description)
            "person": 4,     # person(Name, Birth, Death, Role)
            "cause": 2,      # cause(Event, Result)
            "period": 3,     # period(Name, Start, End)
            "battle": 4,     # battle(Name, Year, Winner, Loser)
        },
        keywords={
            "temporal": ["언제", "몇 년", "년도", "시기", "when", "year", "date"],
            "person": ["누구", "누가", "인물", "who", "person"],
            "causal": ["왜", "원인", "이유", "why", "cause", "reason"],
            "process": ["어떻게", "방법", "과정", "how", "process", "method"],
            "location": ["어디서", "장소", "where", "location", "place"],
        },
        system_prompt="""당신은 역사 전문가입니다. 
KQML 형태의 논리적 표현을 자연스러운 문장으로 변환해주세요.
- 정확하고 이해하기 쉬운 문장으로 변환
- 역사 관련 용어는 정확히 사용
- 문장은 간결하고 명확하게 작성""",
        examples=[
            "삼국통일이 언제인가요?",
            "세종대왕은 누구인가요?",
            "임진왜란의 원인은 무엇인가요?"
        ]
    ),
    
    Domain.MATH: DomainConfig(
        domain=Domain.MATH,
        language=Language.KOREAN,
        predicates={
            "formula": 3,    # formula(Name, Expression, Description)
            "theorem": 3,    # theorem(Name, Statement, Proof)
            "concept": 2,    # concept(Name, Definition)
            "example": 3,    # example(Concept, Problem, Solution)
        },
        keywords={
            "temporal": ["언제", "몇 년", "년도", "when"],
            "person": ["누구", "누가", "who"],
            "causal": ["왜", "원인", "이유", "why"],
            "process": ["어떻게", "방법", "과정", "how", "solve", "calculate"],
            "definition": ["무엇", "정의", "what", "definition"],
        },
        system_prompt="""당신은 수학 전문가입니다.
KQML 형태의 논리적 표현을 자연스러운 문장으로 변환해주세요.
- 수학적 정확성을 유지
- 이해하기 쉬운 설명 제공
- 수식과 개념을 명확히 표현""",
        examples=[
            "피타고라스 정리는 무엇인가요?",
            "이차방정식을 어떻게 풀나요?",
            "미분의 정의는 무엇인가요?"
        ]
    ),
    
    Domain.SCIENCE: DomainConfig(
        domain=Domain.SCIENCE,
        language=Language.KOREAN,
        predicates={
            "phenomenon": 3, # phenomenon(Name, Description, Cause)
            "law": 3,        # law(Name, Statement, Application)
            "experiment": 4,  # experiment(Name, Purpose, Method, Result)
            "element": 4,    # element(Symbol, Name, AtomicNumber, Properties)
        },
        keywords={
            "temporal": ["언제", "when"],
            "person": ["누구", "who"],
            "causal": ["왜", "원인", "이유", "why"],
            "process": ["어떻게", "방법", "과정", "how"],
            "definition": ["무엇", "정의", "what"],
        },
        system_prompt="""당신은 과학 전문가입니다.
KQML 형태의 논리적 표현을 자연스러운 문장으로 변환해주세요.
- 과학적 정확성을 유지
- 복잡한 개념을 쉽게 설명
- 실험과 관찰을 명확히 표현""",
        examples=[
            "광합성은 무엇인가요?",
            "뉴턴의 운동법칙은 무엇인가요?",
            "DNA의 구조는 어떻게 되어있나요?"
        ]
    )
}

def get_domain_config(domain: Domain) -> DomainConfig:
    """도메인 설정 반환"""
    return DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS[Domain.HISTORY])

def get_supported_domains() -> List[Domain]:
    """지원되는 도메인 목록 반환"""
    return list(DOMAIN_CONFIGS.keys())

def get_supported_languages() -> List[Language]:
    """지원되는 언어 목록 반환"""
    return list(Language)

def create_custom_domain_config(
    domain_name: str,
    language: Language,
    predicates: Dict[str, int],
    keywords: Dict[str, List[str]],
    system_prompt: str,
    examples: List[str]
) -> DomainConfig:
    """사용자 정의 도메인 설정 생성"""
    return DomainConfig(
        domain=Domain(domain_name),
        language=language,
        predicates=predicates,
        keywords=keywords,
        system_prompt=system_prompt,
        examples=examples
    )
