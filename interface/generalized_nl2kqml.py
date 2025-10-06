# interface/generalized_nl2kqml.py
"""
일반화된 자연어-KQML 변환 모듈
다양한 도메인과 언어를 지원하는 동적 변환 시스템
"""

import logging
from typing import Optional, List, Dict, Any
from pylo.language.lp import Variable, Atom, c_const
from belief.generalized_knowledge_base import build_generalized_kb
from config.domain_config import Domain, DomainConfig, get_domain_config

logger = logging.getLogger(__name__)

class GeneralizedNL2KQML:
    """일반화된 NL→KQML 변환기"""
    
    def __init__(self, domain: Domain = Domain.HISTORY):
        self.domain = domain
        self.config = get_domain_config(domain)
        self.kb = build_generalized_kb(domain)
        
    def convert(self, question: str) -> Optional[Atom]:
        """
        자연어 질문을 KQML Atom 쿼리로 변환
        
        Args:
            question: 자연어 질문
            
        Returns:
            변환된 Atom 객체 또는 None (매칭 실패 시)
        """
        try:
            logger.info(f"NL→KQML 변환 시작 [{self.domain.value}]: {question}")
            
            if not question or len(question.strip()) < 2:
                logger.warning("질문이 너무 짧습니다.")
                return None
                
            q = question.strip().lower()
            
            # 매칭 점수를 저장할 리스트
            matches = []
            
            for fact in self.kb.engine.facts:
                try:
                    args = [str(arg).strip("'") for arg in fact.arguments]
                    match_score = self._calculate_match_score(q, args)
                    
                    if match_score > 0:
                        matches.append((fact, match_score))
                        logger.debug(f"매칭 발견: {fact.predicate.name} (점수: {match_score})")
                        
                except Exception as e:
                    logger.warning(f"Fact 처리 중 오류: {str(e)}")
                    continue
            
            if not matches:
                logger.warning("매칭되는 fact가 없습니다.")
                return None
                
            # 가장 높은 점수의 매칭 선택
            best_match = max(matches, key=lambda x: x[1])
            fact, score = best_match
            
            logger.info(f"최적 매칭 선택: {fact.predicate.name} (점수: {score})")
            
            # Atom 생성
            pred = fact.predicate
            new_args = []
            
            for i, arg in enumerate(fact.arguments):
                arg_str = str(arg).strip("'")
                if arg_str.lower() in q:
                    new_args.append(c_const(f"'{arg_str}'"))
                else:
                    new_args.append(Variable(f"X{i}"))
            
            result = Atom(pred, new_args)
            logger.info(f"KQML 변환 완료: {result}")
            return result
            
        except Exception as e:
            logger.error(f"NL→KQML 변환 오류: {str(e)}")
            return None
    
    def _calculate_match_score(self, question: str, args: List[str]) -> float:
        """
        질문과 인수들 간의 매칭 점수 계산
        
        Args:
            question: 소문자로 변환된 질문
            args: fact의 인수들
            
        Returns:
            매칭 점수 (0.0 ~ 1.0)
        """
        if not args:
            return 0.0
            
        matched_args = 0
        total_args = len(args)
        
        for arg in args:
            arg_lower = arg.lower()
            if arg_lower in question:
                matched_args += 1
                # 더 긴 매칭에 더 높은 점수
                if len(arg_lower) > 3:
                    matched_args += 0.5
        
        base_score = matched_args / total_args
        
        # 질문 길이에 따른 가중치
        length_weight = min(len(question) / 50, 1.0)
        
        return base_score * length_weight
    
    def analyze_question_type(self, question: str) -> str:
        """질문 유형 분석 (도메인별 키워드 사용)"""
        question_lower = question.lower()
        
        for question_type, keywords in self.config.keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return question_type
        
        return "general"
    
    def get_available_patterns(self) -> List[str]:
        """현재 KB에서 사용 가능한 패턴들을 반환"""
        try:
            patterns = []
            
            for fact in self.kb.engine.facts:
                args = [str(arg).strip("'") for arg in fact.arguments]
                pattern = f"{fact.predicate.name}({', '.join(args)})"
                patterns.append(pattern)
                
            return patterns
            
        except Exception as e:
            logger.error(f"패턴 조회 오류: {str(e)}")
            return []
    
    def switch_domain(self, new_domain: Domain):
        """도메인 변경"""
        self.domain = new_domain
        self.config = get_domain_config(new_domain)
        self.kb = build_generalized_kb(new_domain)
        logger.info(f"도메인 변경: {new_domain.value}")
    
    def get_domain_info(self) -> Dict[str, Any]:
        """현재 도메인 정보 반환"""
        return self.kb.get_domain_info()

# 기존 호환성을 위한 래퍼 함수
def nl_to_kqml(question: str) -> Optional[Atom]:
    """기존 코드와의 호환성을 위한 래퍼"""
    converter = GeneralizedNL2KQML(Domain.HISTORY)
    return converter.convert(question)

# 전역 변환기 인스턴스 (기본값: 역사)
_global_converter = GeneralizedNL2KQML(Domain.HISTORY)

def get_converter() -> GeneralizedNL2KQML:
    """전역 변환기 인스턴스 반환"""
    return _global_converter

def set_domain(domain: Domain):
    """전역 도메인 설정"""
    global _global_converter
    _global_converter.switch_domain(domain)
