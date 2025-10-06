# interface/kqml2nl.py
import logging
from typing import Optional, Dict, Any
from interface.llm_connector import LLMConnector

# 로깅 설정
logger = logging.getLogger(__name__)

# 전역 connector 인스턴스 (지연 초기화)
_connector: Optional[LLMConnector] = None

def get_connector() -> LLMConnector:
    """LLMConnector 인스턴스를 지연 초기화로 반환"""
    global _connector
    if _connector is None:
        try:
            _connector = LLMConnector()
            logger.info("LLMConnector 인스턴스 생성 완료")
        except Exception as e:
            logger.error(f"LLMConnector 초기화 실패: {str(e)}")
            raise
    return _connector

def kqml_to_nl(atom, context: Optional[str] = None) -> str:
    """
    KB 응답(Atom)을 자연어 문장으로 변환
    
    Args:
        atom: KQML Atom 객체
        context: 추가 컨텍스트 정보 (선택사항)
        
    Returns:
        자연어로 변환된 문자열
    """
    try:
        logger.info(f"KQML→NL 변환 시작: {atom}")
        
        # Atom을 문자열로 변환
        if hasattr(atom, "predicate") and hasattr(atom, "arguments"):
            pred = atom.predicate.name
            args = [str(a).strip("'") for a in atom.arguments]
            query_str = f"{pred}({', '.join(args)})"
        else:
            query_str = str(atom)
        
        logger.debug(f"변환할 KQML: {query_str}")
        
        # 시스템 프롬프트 설정 (더 구체적이고 상세한 정보 제공)
        system_prompt = """당신은 한국사 전문가입니다. 
KQML 형태의 논리적 표현을 자연스러운 한국어 문장으로 변환해주세요.

중요한 변환 규칙:
1. person(Name, Birth, Death, Role) → "Name은 Birth년~Death년에 살았던 Role입니다"
2. event(Name, Year, Desc) → "Name은 Year년에 발생한 사건으로, Desc입니다"
3. cause(Event, Result) → "Event는 Result의 원인이 되었습니다"
4. 변수명(X1, X2, X3 등)은 구체적인 정보로 대체하거나 생략
5. 한국사 관련 용어는 정확히 사용
6. 문장은 간결하고 명확하게 작성
7. 가능한 한 구체적이고 유용한 정보를 제공"""
        
        # 사용자 프롬프트 구성
        user_prompt = f"다음 KQML 표현을 자연스러운 한국어 문장으로 변환해주세요:\n{query_str}"
        
        if context:
            user_prompt += f"\n\n추가 컨텍스트: {context}"
        
        # LLM 호출
        connector = get_connector()
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

def kqml_to_nl_batch(atoms: list, context: Optional[str] = None) -> list:
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
            result = kqml_to_nl(atom, context)
            results.append(result)
        except Exception as e:
            logger.error(f"배치 변환 오류 (항목 {i+1}): {str(e)}")
            results.append(f"변환 실패: {str(atom)}")
    
    return results

def format_kqml_for_display(atom) -> str:
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
