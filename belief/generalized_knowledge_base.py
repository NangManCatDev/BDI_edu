# belief/generalized_knowledge_base.py
"""
일반화된 지식베이스 모듈
다양한 도메인과 언어를 지원하는 동적 지식베이스
"""

import logging
from typing import Dict, List, Optional, Any
from pylo.language.lp import c_const
from .prolog_engine import PrologEngine
from config.domain_config import Domain, DomainConfig, get_domain_config

logger = logging.getLogger(__name__)

class GeneralizedKnowledgeBase:
    """일반화된 지식베이스 클래스"""
    
    def __init__(self, domain: Domain = Domain.HISTORY):
        self.domain = domain
        self.config = get_domain_config(domain)
        self.engine = PrologEngine()
        self._setup_predicates()
        self._load_domain_facts()
        
    def _setup_predicates(self):
        """도메인에 따른 predicate 설정"""
        self.predicates = {}
        for pred_name, arity in self.config.predicates.items():
            self.predicates[pred_name] = self.engine.pred(pred_name, arity)
        logger.info(f"Predicates 설정 완료: {list(self.predicates.keys())}")
    
    def _load_domain_facts(self):
        """도메인별 기본 사실들 로드"""
        if self.domain == Domain.HISTORY:
            self._load_history_facts()
        elif self.domain == Domain.MATH:
            self._load_math_facts()
        elif self.domain == Domain.SCIENCE:
            self._load_science_facts()
        else:
            logger.warning(f"도메인 {self.domain}에 대한 기본 사실이 없습니다.")
    
    def _load_history_facts(self):
        """역사 도메인 사실들"""
        # Events
        self.engine.fact(self.predicates["event"], [
            c_const("'삼국통일'"), c_const("'668'"), c_const("'신라가 삼국을 통일함'")
        ])
        self.engine.fact(self.predicates["event"], [
            c_const("'훈민정음반포'"), c_const("'1446'"), c_const("'세종이 훈민정음을 반포함'")
        ])
        
        # Persons
        self.engine.fact(self.predicates["person"], [
            c_const("'세종'"), c_const("'1397'"), c_const("'1450'"), c_const("'조선의 4대 왕'")
        ])
        self.engine.fact(self.predicates["person"], [
            c_const("'이순신'"), c_const("'1545'"), c_const("'1598'"), c_const("'조선 수군 장군'")
        ])
        
        # Causes
        self.engine.fact(self.predicates["cause"], [
            c_const("'동학농민운동'"), c_const("'갑오개혁'")
        ])
        
        logger.info("역사 도메인 사실 로드 완료")
    
    def _load_math_facts(self):
        """수학 도메인 사실들"""
        # Formulas
        self.engine.fact(self.predicates["formula"], [
            c_const("'피타고라스정리'"), c_const("'a²+b²=c²'"), c_const("'직각삼각형의 빗변 길이 공식'")
        ])
        
        # Theorems
        self.engine.fact(self.predicates["theorem"], [
            c_const("'중간값정리'"), c_const("'연속함수는 중간값을 가진다'"), c_const("'해석학의 기본정리'")
        ])
        
        # Concepts
        self.engine.fact(self.predicates["concept"], [
            c_const("'미분'"), c_const("'함수의 순간변화율'")
        ])
        
        logger.info("수학 도메인 사실 로드 완료")
    
    def _load_science_facts(self):
        """과학 도메인 사실들"""
        # Phenomena
        self.engine.fact(self.predicates["phenomenon"], [
            c_const("'광합성'"), c_const("'식물이 빛을 이용해 포도당을 만드는 과정'"), c_const("'햇빛과 엽록소'")
        ])
        
        # Laws
        self.engine.fact(self.predicates["law"], [
            c_const("'뉴턴의운동법칙'"), c_const("'F=ma'"), c_const("'물체의 운동과 힘의 관계'")
        ])
        
        logger.info("과학 도메인 사실 로드 완료")
    
    def add_fact(self, predicate_name: str, arguments: List[str]):
        """새로운 사실 추가"""
        if predicate_name not in self.predicates:
            logger.error(f"Unknown predicate: {predicate_name}")
            return False
        
        try:
            pred = self.predicates[predicate_name]
            const_args = [c_const(f"'{arg}'") for arg in arguments]
            self.engine.fact(pred, const_args)
            logger.info(f"Fact 추가: {predicate_name}({', '.join(arguments)})")
            return True
        except Exception as e:
            logger.error(f"Fact 추가 실패: {str(e)}")
            return False
    
    def query(self, atom):
        """쿼리 실행"""
        try:
            return self.engine.query(atom)
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {str(e)}")
            return []
    
    def get_available_predicates(self) -> List[str]:
        """사용 가능한 predicate 목록 반환"""
        return list(self.predicates.keys())
    
    def get_domain_info(self) -> Dict[str, Any]:
        """도메인 정보 반환"""
        return {
            "domain": self.domain.value,
            "language": self.config.language.value,
            "predicates": self.config.predicates,
            "keywords": self.config.keywords,
            "examples": self.config.examples
        }
    
    def switch_domain(self, new_domain: Domain):
        """도메인 변경"""
        self.domain = new_domain
        self.config = get_domain_config(new_domain)
        self.engine = PrologEngine()
        self._setup_predicates()
        self._load_domain_facts()
        logger.info(f"도메인 변경: {new_domain.value}")

def build_generalized_kb(domain: Domain = Domain.HISTORY) -> GeneralizedKnowledgeBase:
    """일반화된 지식베이스 생성"""
    return GeneralizedKnowledgeBase(domain)

# 기존 호환성을 위한 래퍼 함수
def build_kb():
    """기존 코드와의 호환성을 위한 래퍼"""
    return build_generalized_kb(Domain.HISTORY).engine
