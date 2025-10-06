# interface/generalized_kqml2nl.py
"""
일반화된 KQML-자연어 변환 모듈
다양한 도메인과 언어를 지원하는 동적 변환 시스템
"""

import logging
from typing import Optional, Dict, Any, List
from interface.llm_connector import LLMConnector
from config.domain_config import Domain, DomainConfig, get_domain_config

logger = logging.getLogger(__name__)

class GeneralizedKQML2NL:
    """일반화된 KQML→NL 변환기"""
    
    def __init__(self, domain: Domain = Domain.HISTORY):
        self.domain = domain
        self.config = get_domain_config(domain)
        self.connector = None  # 지연 초기화
        
    def _get_connector(self) -> LLMConnector:
        """LLMConnector 인스턴스를 지연 초기화로 반환"""
        if self.connector is None:
            try:
                self.connector = LLMConnector()
                logger.info("LLMConnector 인스턴스 생성 완료")
            except Exception as e:
                logger.error(f"LLMConnector 초기화 실패: {str(e)}")
                raise
        return self.connector
    
    def convert(self, atom, context: Optional[str] = None) -> str:
        """
        KB 응답(Atom)을 자연어 문장으로 변환
        
        Args:
            atom: KQML Atom 객체
            context: 추가 컨텍스트 정보 (선택사항)
            
        Returns:
            자연어로 변환된 문자열
        """
        try:
            logger.info(f"KQML→NL 변환 시작 [{self.domain.value}]: {atom}")
            
            # Atom을 문자열로 변환
            if hasattr(atom, "predicate") and hasattr(atom, "arguments"):
                pred = atom.predicate.name
                args = [str(a).strip("'") for a in atom.arguments]
                query_str = f"{pred}({', '.join(args)})"
            else:
                query_str = str(atom)
            
            logger.debug(f"변환할 KQML: {query_str}")
            
            # 도메인별 시스템 프롬프트 사용
            system_prompt = self.config.system_prompt
            
            # 사용자 프롬프트 구성
            user_prompt = f"다음 KQML 표현을 자연스러운 문장으로 변환해주세요:\n{query_str}"
            
            if context:
                user_prompt += f"\n\n추가 컨텍스트: {context}"
            
            # LLM 호출
            connector = self._get_connector()
            result = connector.ask_with_fallback(
                prompt=user_prompt,
                system=system_prompt,
                fallback_response=f"KQML 변환 결과: {query_str}"
            )
            
            logger.info(f"KQML→NL 변환 완료: {result[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"KQML→NL 변환 오류: {str(e)}")
            # 폴백: 원본 KQML 문자열 반환
            fallback = str(atom) if atom else "변환할 수 없습니다."
            return f"변환 오류: {fallback}"
    
    def convert_batch(self, atoms: List, context: Optional[str] = None) -> List[str]:
        """
        여러 KQML Atom을 일괄 변환
        
        Args:
            atoms: KQML Atom 객체 리스트
            context: 공통 컨텍스트
            
        Returns:
            자연어 문자열 리스트
        """
        results = []
        
        for i, atom in enumerate(atoms):
            try:
                logger.info(f"배치 변환 {i+1}/{len(atoms)}: {atom}")
                result = self.convert(atom, context)
                results.append(result)
            except Exception as e:
                logger.error(f"배치 변환 오류 (항목 {i+1}): {str(e)}")
                results.append(f"변환 실패: {str(atom)}")
        
        return results
    
    def format_for_display(self, atom) -> str:
        """
        KQML을 사용자에게 보여주기 위한 형태로 포맷팅
        
        Args:
            atom: KQML Atom 객체
            
        Returns:
            포맷팅된 문자열
        """
        try:
            if hasattr(atom, "predicate") and hasattr(atom, "arguments"):
                pred = atom.predicate.name
                args = [str(a).strip("'") for a in atom.arguments]
                return f"{pred}({', '.join(args)})"
            else:
                return str(atom)
        except Exception as e:
            logger.error(f"KQML 포맷팅 오류: {str(e)}")
            return "포맷팅 오류"
    
    def switch_domain(self, new_domain: Domain):
        """도메인 변경"""
        self.domain = new_domain
        self.config = get_domain_config(new_domain)
        logger.info(f"도메인 변경: {new_domain.value}")
    
    def get_domain_info(self) -> Dict[str, Any]:
        """현재 도메인 정보 반환"""
        return {
            "domain": self.domain.value,
            "language": self.config.language.value,
            "system_prompt": self.config.system_prompt,
            "examples": self.config.examples
        }

# 기존 호환성을 위한 래퍼 함수
def kqml_to_nl(atom, context: Optional[str] = None) -> str:
    """기존 코드와의 호환성을 위한 래퍼"""
    converter = GeneralizedKQML2NL(Domain.HISTORY)
    return converter.convert(atom, context)

# 전역 변환기 인스턴스 (기본값: 역사)
_global_converter = GeneralizedKQML2NL(Domain.HISTORY)

def get_converter() -> GeneralizedKQML2NL:
    """전역 변환기 인스턴스 반환"""
    return _global_converter

def set_domain(domain: Domain):
    """전역 도메인 설정"""
    global _global_converter
    _global_converter.switch_domain(domain)
