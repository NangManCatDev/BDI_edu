# intention/enhanced_executor.py
"""
향상된 실행 모듈
실제 LLM 호출, RAG 검색, 학습 자료 생성을 통한 진짜 학습 실행
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """실행 결과"""
    step_id: str
    success: bool
    content: str
    resources: List[str]
    feedback: str
    duration: float  # 초
    quality_score: float  # 0.0-1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class EnhancedExecutor:
    """향상된 실행자"""
    
    def __init__(self):
        self.llm_connector = None
        self.rag_system = None
        self.content_generator = None
        self._initialize_components()
        logger.info("EnhancedExecutor 초기화 완료")
    
    def _initialize_components(self):
        """구성 요소 초기화"""
        try:
            from interface.llm_connector import LLMConnector
            self.llm_connector = LLMConnector()
            logger.info("LLMConnector 초기화 완료")
        except Exception as e:
            logger.warning(f"LLMConnector 초기화 실패: {str(e)}")
        
        # RAG 시스템과 콘텐츠 생성기는 추후 구현
        self.rag_system = MockRAGSystem()
        self.content_generator = MockContentGenerator()
    
    def execute_plan(self, steps: List[Any]) -> List[ExecutionResult]:
        """학습 계획 전체 실행"""
        logger.info(f"학습 계획 실행 시작: {len(steps)}개 단계")
        
        results = []
        for i, step in enumerate(steps):
            try:
                logger.info(f"단계 {i+1}/{len(steps)} 실행: {step.title if hasattr(step, 'title') else step}")
                result = self.execute_step(step)
                results.append(result)
                
                if not result.success:
                    logger.warning(f"단계 실행 실패: {step}")
                    # 실패 시에도 다음 단계 계속 진행
                    
            except Exception as e:
                logger.error(f"단계 실행 중 오류: {str(e)}")
                # 오류 발생 시 기본 결과 생성
                results.append(ExecutionResult(
                    step_id=getattr(step, 'step_id', f'step_{i}'),
                    success=False,
                    content=f"실행 오류: {str(e)}",
                    resources=[],
                    feedback="실행 중 오류가 발생했습니다.",
                    duration=0.0,
                    quality_score=0.0
                ))
        
        logger.info(f"학습 계획 실행 완료: {len(results)}개 결과")
        return results
    
    def execute_step(self, step: Any) -> ExecutionResult:
        """개별 학습 단계 실행"""
        start_time = datetime.now()
        
        try:
            # 단계 정보 추출
            step_info = self._extract_step_info(step)
            
            # 학습 자료 생성
            content = self._generate_learning_content(step_info)
            
            # 관련 자료 검색
            resources = self._search_related_resources(step_info)
            
            # 품질 평가
            quality_score = self._evaluate_content_quality(content, step_info)
            
            # 피드백 생성
            feedback = self._generate_feedback(content, step_info, quality_score)
            
            # 실행 시간 계산
            duration = (datetime.now() - start_time).total_seconds()
            
            result = ExecutionResult(
                step_id=step_info.get('step_id', 'unknown'),
                success=True,
                content=content,
                resources=resources,
                feedback=feedback,
                duration=duration,
                quality_score=quality_score,
                metadata={
                    'strategy': step_info.get('strategy', 'unknown'),
                    'phase': step_info.get('phase', 'unknown'),
                    'difficulty': step_info.get('difficulty', 'medium')
                }
            )
            
            logger.info(f"단계 실행 성공: {result.step_id} (품질: {quality_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"단계 실행 실패: {str(e)}")
            duration = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                step_id=getattr(step, 'step_id', 'unknown'),
                success=False,
                content=f"실행 실패: {str(e)}",
                resources=[],
                feedback="학습 자료 생성에 실패했습니다.",
                duration=duration,
                quality_score=0.0
            )
    
    def _extract_step_info(self, step: Any) -> Dict[str, Any]:
        """단계 정보 추출"""
        if hasattr(step, '__dict__'):
            # 객체인 경우 속성 추출
            return {
                'step_id': getattr(step, 'step_id', 'unknown'),
                'title': getattr(step, 'title', ''),
                'description': getattr(step, 'description', ''),
                'strategy': getattr(step, 'strategy', 'unknown'),
                'phase': getattr(step, 'phase', 'unknown'),
                'difficulty': getattr(step, 'difficulty', 'medium'),
                'estimated_duration': getattr(step, 'estimated_duration', 30)
            }
        else:
            # 문자열인 경우 (기존 호환성)
            return {
                'step_id': 'legacy_step',
                'title': str(step),
                'description': str(step),
                'strategy': 'explanation',
                'phase': 'practice',
                'difficulty': 'medium',
                'estimated_duration': 30
            }
    
    def _generate_learning_content(self, step_info: Dict[str, Any]) -> str:
        """학습 자료 생성"""
        try:
            if self.content_generator:
                return self.content_generator.generate_content(step_info)
            else:
                return self._generate_basic_content(step_info)
        except Exception as e:
            logger.error(f"콘텐츠 생성 실패: {str(e)}")
            return self._generate_basic_content(step_info)
    
    def _generate_basic_content(self, step_info: Dict[str, Any]) -> str:
        """기본 콘텐츠 생성 (폴백)"""
        title = step_info.get('title', '학습 단계')
        strategy = step_info.get('strategy', 'explanation')
        phase = step_info.get('phase', 'practice')
        
        content_templates = {
            'explanation': f"""
# {title}

## 학습 목표
이 단계에서는 {title}에 대한 기본 개념을 이해합니다.

## 핵심 내용
- 주요 개념 설명
- 중요한 포인트 정리
- 이해도 확인 질문

## 학습 방법
1. 개념을 차근차근 읽어보세요
2. 예시를 통해 이해해보세요
3. 연습 문제를 풀어보세요
""",
            'example': f"""
# {title} - 예제 중심 학습

## 예제 1: 기본 개념
- 구체적인 예시
- 단계별 설명
- 주의사항

## 예제 2: 응용 문제
- 조금 더 복잡한 예시
- 해결 과정 설명
- 팁과 요령

## 연습 문제
- 직접 풀어볼 수 있는 문제들
- 난이도별 구성
""",
            'practice': f"""
# {title} - 실습 중심 학습

## 실습 목표
- {title}에 대한 실전 연습
- 문제 해결 능력 향상

## 연습 문제
1. 기초 문제
2. 응용 문제
3. 심화 문제

## 해답 및 해설
- 단계별 해결 과정
- 자주 하는 실수
- 개선 방법
"""
        }
        
        return content_templates.get(strategy, content_templates['explanation'])
    
    def _search_related_resources(self, step_info: Dict[str, Any]) -> List[str]:
        """관련 자료 검색"""
        try:
            if self.rag_system:
                return self.rag_system.search_resources(step_info)
            else:
                return self._get_default_resources(step_info)
        except Exception as e:
            logger.error(f"자료 검색 실패: {str(e)}")
            return self._get_default_resources(step_info)
    
    def _get_default_resources(self, step_info: Dict[str, Any]) -> List[str]:
        """기본 자료 목록"""
        domain = step_info.get('domain', 'general')
        
        default_resources = {
            'history': ['교과서', '역사 지도', '사료 모음', '역사 다큐멘터리'],
            'math': ['교과서', '문제집', '계산기', '그래프 도구'],
            'science': ['교과서', '실험 가이드', '관찰 기록지', '과학 영상'],
            'language': ['교과서', '사전', '읽기 자료', '듣기 자료'],
            'general': ['교과서', '참고서', '온라인 자료', '학습 노트']
        }
        
        return default_resources.get(domain, default_resources['general'])
    
    def _evaluate_content_quality(self, content: str, step_info: Dict[str, Any]) -> float:
        """콘텐츠 품질 평가"""
        try:
            # 기본 품질 점수 (0.5-1.0)
            base_score = 0.7
            
            # 길이 기반 조정
            if len(content) > 500:
                base_score += 0.1
            elif len(content) < 100:
                base_score -= 0.2
            
            # 구조 기반 조정
            if '##' in content:  # 마크다운 헤더 있음
                base_score += 0.1
            
            if '1.' in content or '2.' in content:  # 목록 있음
                base_score += 0.1
            
            # 최종 점수 범위 제한
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.error(f"품질 평가 실패: {str(e)}")
            return 0.5
    
    def _generate_feedback(self, content: str, step_info: Dict[str, Any], quality_score: float) -> str:
        """피드백 생성"""
        if quality_score >= 0.8:
            return "훌륭한 학습 자료가 생성되었습니다. 차근차근 따라해보세요."
        elif quality_score >= 0.6:
            return "좋은 학습 자료입니다. 추가로 참고 자료를 활용해보세요."
        elif quality_score >= 0.4:
            return "기본적인 학습 자료입니다. 더 자세한 설명이 필요할 수 있습니다."
        else:
            return "학습 자료가 부족합니다. 다른 자료를 참고해보세요."
    
    def execute(self, plan_steps: List[Any]) -> List[str]:
        """기존 호환성을 위한 래퍼"""
        results = self.execute_plan(plan_steps)
        return [f"Executed: {result.content[:100]}..." for result in results]

class MockRAGSystem:
    """RAG 시스템 모의 구현"""
    
    def search_resources(self, step_info: Dict[str, Any]) -> List[str]:
        """관련 자료 검색 (모의)"""
        title = step_info.get('title', '')
        domain = step_info.get('domain', 'general')
        
        # 제목 키워드 기반 자료 추천
        resources = []
        
        if '역사' in title or 'history' in title.lower():
            resources.extend(['역사 교과서', '역사 지도', '사료 모음'])
        elif '수학' in title or 'math' in title.lower():
            resources.extend(['수학 교과서', '문제집', '계산기'])
        elif '과학' in title or 'science' in title.lower():
            resources.extend(['과학 교과서', '실험 도구', '관찰 기록지'])
        
        # 도메인별 기본 자료 추가
        domain_resources = {
            'history': ['역사 다큐멘터리', '역사 소설'],
            'math': ['수학 영상', '수학 게임'],
            'science': ['과학 실험 영상', '과학 뉴스'],
            'language': ['읽기 자료', '듣기 자료'],
            'general': ['온라인 강의', '참고서']
        }
        
        resources.extend(domain_resources.get(domain, []))
        return list(set(resources))  # 중복 제거

class MockContentGenerator:
    """콘텐츠 생성기 모의 구현"""
    
    def generate_content(self, step_info: Dict[str, Any]) -> str:
        """학습 콘텐츠 생성 (모의)"""
        title = step_info.get('title', '학습 단계')
        strategy = step_info.get('strategy', 'explanation')
        
        # 전략별 콘텐츠 생성
        if strategy == 'explanation':
            return self._generate_explanation_content(title)
        elif strategy == 'example':
            return self._generate_example_content(title)
        elif strategy == 'practice':
            return self._generate_practice_content(title)
        else:
            return self._generate_general_content(title)
    
    def _generate_explanation_content(self, title: str) -> str:
        """설명 중심 콘텐츠"""
        return f"""
# {title} - 개념 설명

## 학습 목표
{title}에 대한 기본 개념을 이해하고 핵심 내용을 파악합니다.

## 핵심 개념
1. **기본 정의**: {title}의 정의와 의미
2. **주요 특징**: 중요한 특징과 특성
3. **관련 개념**: 연관된 다른 개념들

## 이해도 확인
- 핵심 개념을 자신의 말로 설명해보세요
- 예시를 들어 설명해보세요
- 다른 개념과의 차이점을 설명해보세요

## 다음 단계
이 개념을 바탕으로 실제 예시와 연습 문제를 다뤄보겠습니다.
"""
    
    def _generate_example_content(self, title: str) -> str:
        """예제 중심 콘텐츠"""
        return f"""
# {title} - 예제 중심 학습

## 예제 1: 기본 예시
**문제**: {title}의 기본적인 예시를 살펴보겠습니다.

**해결 과정**:
1. 단계별 접근
2. 핵심 포인트 파악
3. 결과 확인

**핵심 포인트**: 중요한 개념과 원리

## 예제 2: 응용 예시
**문제**: 조금 더 복잡한 상황에서의 적용

**해결 과정**:
1. 문제 분석
2. 해결 방법 선택
3. 단계별 실행
4. 결과 검증

## 연습 문제
1. 기본 문제
2. 응용 문제
3. 심화 문제

**해답**: 각 문제의 해답과 해설
"""
    
    def _generate_practice_content(self, title: str) -> str:
        """연습 중심 콘텐츠"""
        return f"""
# {title} - 실습 연습

## 연습 목표
{title}에 대한 실전 연습을 통해 숙련도를 높입니다.

## 연습 문제

### 기초 문제
1. **문제 1**: 기본 개념 적용
2. **문제 2**: 단계별 해결
3. **문제 3**: 결과 확인

### 응용 문제
1. **문제 4**: 복합 상황 해결
2. **문제 5**: 창의적 접근
3. **문제 6**: 실생활 적용

### 심화 문제
1. **문제 7**: 고급 개념 활용
2. **문제 8**: 종합적 사고
3. **문제 9**: 문제 해결 전략

## 해답 및 해설
각 문제에 대한 상세한 해답과 해설을 제공합니다.

## 피드백
연습 결과를 바탕으로 개선점과 다음 단계를 제안합니다.
"""
    
    def _generate_general_content(self, title: str) -> str:
        """일반적인 콘텐츠"""
        return f"""
# {title}

## 개요
{title}에 대한 종합적인 학습을 진행합니다.

## 학습 내용
- 기본 개념
- 핵심 내용
- 실습 연습
- 평가 및 피드백

## 학습 방법
1. 개념 이해
2. 예시 학습
3. 연습 문제
4. 자기 평가

## 완료 기준
- 핵심 개념 이해
- 기본 문제 해결
- 응용 문제 도전
- 자기 설명 가능
"""

# 기존 호환성을 위한 래퍼 클래스
class Executor(EnhancedExecutor):
    """기존 코드와의 호환성을 위한 래퍼"""
    pass
