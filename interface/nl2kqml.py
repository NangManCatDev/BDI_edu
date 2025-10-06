# interface/nl2kqml.py
import logging
from typing import Optional, List, Tuple
# Prolog 라이브러리 제거 - 순수 NEO 엔진 사용
from belief.knowledge_base import build_kb

# 로깅 설정
logger = logging.getLogger(__name__)

def nl_to_kqml(question: str):
    """
    자연어 질문을 KQML Atom 쿼리로 변환
    
    Args:
        question: 자연어 질문
        
    Returns:
        변환된 쿼리 객체 또는 None (매칭 실패 시)
    """
    try:
        logger.info(f"NL→KQML 변환 시작: {question}")
        
        if not question or len(question.strip()) < 2:
            logger.warning("질문이 너무 짧습니다.")
            return None
            
        eng = build_kb()
        q = question.strip().lower()
        
        # 매칭 점수를 저장할 리스트
        matches = []
        
        for fact in eng.facts:
            try:
                args = [str(arg).strip("'") for arg in fact.arguments]
                match_score = calculate_match_score(q, args)
                
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
        
        # 쿼리 객체 생성 (Prolog 없이)
        pred = fact.predicate
        new_args = []
        
        for i, arg in enumerate(fact.arguments):
            arg_str = str(arg).strip("'")
            if arg_str.lower() in q:
                new_args.append(f"'{arg_str}'")
            else:
                # 간단한 Variable 객체 생성
                var_obj = type('Variable', (), {'name': f"X{i}"})()
                new_args.append(var_obj)
        
        # 간단한 Atom 객체 생성
        result = type('Atom', (), {
            'predicate': pred,
            'arguments': new_args
        })()
        logger.info(f"KQML 변환 완료: {result}")
        return result
        
    except Exception as e:
        logger.error(f"NL→KQML 변환 오류: {str(e)}")
        return None

def calculate_match_score(question: str, args: List[str]) -> float:
    """
    질문과 인수들 간의 매칭 점수 계산 (개선된 버전)
    
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
    bonus_score = 0.0
    
    # 한국사 관련 키워드 가중치
    korean_history_keywords = {
        '왕', '대왕', '장군', '독립운동가', '성리학자', '예술가', '건국자', '개국공신',
        '통일', '왜란', '운동', '광복', '창제', '반포', '농민', '봉기'
    }
    
    for arg in args:
        arg_lower = arg.lower()
        if arg_lower in question:
            matched_args += 1
            # 더 긴 매칭에 더 높은 점수
            if len(arg_lower) > 3:
                matched_args += 0.3
            # 한국사 키워드 보너스
            if any(keyword in arg_lower for keyword in korean_history_keywords):
                bonus_score += 0.2
        # 부분 매칭도 고려
        elif any(word in arg_lower for word in question.split() if len(word) > 2):
            matched_args += 0.5
    
    base_score = matched_args / total_args
    total_score = base_score + bonus_score
    
    # 질문 길이에 따른 가중치
    length_weight = min(len(question) / 50, 1.0)
    
    return min(total_score * length_weight, 1.0)

def get_available_patterns() -> List[str]:
    """
    현재 KB에서 사용 가능한 패턴들을 반환
    
    Returns:
        패턴 문자열 리스트
    """
    try:
        eng = build_kb()
        patterns = []
        
        for fact in eng.facts:
            args = [str(arg).strip("'") for arg in fact.arguments]
            pattern = f"{fact.predicate.name}({', '.join(args)})"
            patterns.append(pattern)
            
        return patterns
        
    except Exception as e:
        logger.error(f"패턴 조회 오류: {str(e)}")
        return []
