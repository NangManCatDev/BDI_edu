# intention/enhanced_planner.py
"""
향상된 계획 수립 모듈
학습자 프로파일과 목표를 분석하여 개인화된 학습 계획 생성
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random

logger = logging.getLogger(__name__)

class LearningStrategy(Enum):
    """학습 전략"""
    EXPLANATION = "explanation"      # 설명 중심
    EXAMPLE = "example"              # 예제 중심
    PRACTICE = "practice"            # 연습 중심
    DISCUSSION = "discussion"        # 토론 중심
    VISUALIZATION = "visualization"  # 시각화 중심
    GAMIFICATION = "gamification"    # 게임화 중심

class LearningPhase(Enum):
    """학습 단계"""
    INTRODUCTION = "introduction"     # 도입
    EXPLORATION = "exploration"      # 탐색
    PRACTICE = "practice"           # 연습
    APPLICATION = "application"      # 적용
    SYNTHESIS = "synthesis"         # 종합
    ASSESSMENT = "assessment"        # 평가

@dataclass
class LearningStep:
    """학습 단계"""
    step_id: str
    title: str
    description: str
    phase: LearningPhase
    strategy: LearningStrategy
    estimated_duration: int  # 분
    difficulty: str
    prerequisites: List[str] = None
    resources: List[str] = None
    assessment_criteria: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.resources is None:
            self.resources = []
        if self.assessment_criteria is None:
            self.assessment_criteria = []

class EnhancedPlanner:
    """향상된 계획 수립자"""
    
    def __init__(self):
        self.learning_templates = self._initialize_templates()
        logger.info("EnhancedPlanner 초기화 완료")
    
    def _initialize_templates(self) -> Dict[str, Dict]:
        """학습 템플릿 초기화"""
        return {
            "knowledge": {
                "phases": [LearningPhase.INTRODUCTION, LearningPhase.EXPLORATION, LearningPhase.ASSESSMENT],
                "strategies": [LearningStrategy.EXPLANATION, LearningStrategy.EXAMPLE],
                "duration_ratio": [0.3, 0.5, 0.2]
            },
            "skill": {
                "phases": [LearningPhase.INTRODUCTION, LearningPhase.PRACTICE, LearningPhase.APPLICATION],
                "strategies": [LearningStrategy.EXAMPLE, LearningStrategy.PRACTICE],
                "duration_ratio": [0.2, 0.6, 0.2]
            },
            "understanding": {
                "phases": [LearningPhase.EXPLORATION, LearningPhase.DISCUSSION, LearningPhase.SYNTHESIS],
                "strategies": [LearningStrategy.EXPLANATION, LearningStrategy.DISCUSSION],
                "duration_ratio": [0.4, 0.4, 0.2]
            },
            "application": {
                "phases": [LearningPhase.PRACTICE, LearningPhase.APPLICATION, LearningPhase.ASSESSMENT],
                "strategies": [LearningStrategy.PRACTICE, LearningStrategy.EXAMPLE],
                "duration_ratio": [0.3, 0.5, 0.2]
            }
        }
    
    def create_plan(
        self,
        goal: str,
        goal_type: str = "knowledge",
        user_profile: Dict[str, Any] = None,
        domain: str = "general",
        difficulty: str = "medium",
        estimated_duration: int = 30
    ) -> List[LearningStep]:
        """개인화된 학습 계획 생성"""
        
        logger.info(f"계획 생성 시작: {goal} ({goal_type})")
        
        # 사용자 프로파일 기본값 설정
        if user_profile is None:
            user_profile = {
                "learning_style": "visual",
                "difficulty_preference": "medium",
                "pace_preference": "normal"
            }
        
        # 목표 유형에 따른 템플릿 선택
        template = self.learning_templates.get(goal_type, self.learning_templates["knowledge"])
        
        # 학습자 스타일에 따른 전략 조정
        strategies = self._adjust_strategies_for_learning_style(
            template["strategies"], 
            user_profile.get("learning_style", "visual")
        )
        
        # 난이도에 따른 계획 조정
        adjusted_duration = self._adjust_duration_for_difficulty(
            estimated_duration, 
            difficulty
        )
        
        # 단계별 계획 생성
        steps = []
        step_counter = 0
        
        for i, phase in enumerate(template["phases"]):
            # 단계별 소요 시간 계산
            phase_duration = int(adjusted_duration * template["duration_ratio"][i])
            
            # 전략 선택 (학습자 스타일 고려)
            strategy = strategies[i % len(strategies)]
            
            # 단계 생성
            step = LearningStep(
                step_id=f"step_{step_counter:02d}",
                title=self._generate_step_title(goal, phase, strategy),
                description=self._generate_step_description(goal, phase, strategy, domain),
                phase=phase,
                strategy=strategy,
                estimated_duration=phase_duration,
                difficulty=difficulty,
                resources=self._suggest_resources(domain, phase, strategy),
                assessment_criteria=self._generate_assessment_criteria(goal, phase)
            )
            
            steps.append(step)
            step_counter += 1
        
        # 단계 간 의존성 설정
        self._set_step_dependencies(steps)
        
        logger.info(f"계획 생성 완료: {len(steps)}개 단계")
        return steps
    
    def make_plan(self, goal: str) -> List[str]:
        """기존 호환성을 위한 래퍼"""
        steps = self.create_plan(goal)
        return [f"{step.title} ({step.estimated_duration}분)" for step in steps]
    
    def _adjust_strategies_for_learning_style(
        self, 
        base_strategies: List[LearningStrategy], 
        learning_style: str
    ) -> List[LearningStrategy]:
        """학습자 스타일에 따른 전략 조정"""
        adjusted = base_strategies.copy()
        
        if learning_style == "visual":
            # 시각적 학습자에게 시각화 전략 추가
            if LearningStrategy.VISUALIZATION not in adjusted:
                adjusted.append(LearningStrategy.VISUALIZATION)
        elif learning_style == "auditory":
            # 청각적 학습자에게 토론 전략 강화
            if LearningStrategy.DISCUSSION not in adjusted:
                adjusted.append(LearningStrategy.DISCUSSION)
        elif learning_style == "kinesthetic":
            # 체감형 학습자에게 연습 전략 강화
            if LearningStrategy.PRACTICE not in adjusted:
                adjusted.append(LearningStrategy.PRACTICE)
        
        return adjusted
    
    def _adjust_duration_for_difficulty(self, base_duration: int, difficulty: str) -> int:
        """난이도에 따른 소요 시간 조정"""
        multipliers = {
            "easy": 0.7,
            "medium": 1.0,
            "hard": 1.5,
            "expert": 2.0
        }
        return int(base_duration * multipliers.get(difficulty, 1.0))
    
    def _generate_step_title(self, goal: str, phase: LearningPhase, strategy: LearningStrategy) -> str:
        """단계 제목 생성"""
        phase_titles = {
            LearningPhase.INTRODUCTION: "도입",
            LearningPhase.EXPLORATION: "탐색",
            LearningPhase.PRACTICE: "연습",
            LearningPhase.APPLICATION: "적용",
            LearningPhase.SYNTHESIS: "종합",
            LearningPhase.ASSESSMENT: "평가"
        }
        
        strategy_titles = {
            LearningStrategy.EXPLANATION: "설명",
            LearningStrategy.EXAMPLE: "예제",
            LearningStrategy.PRACTICE: "연습",
            LearningStrategy.DISCUSSION: "토론",
            LearningStrategy.VISUALIZATION: "시각화",
            LearningStrategy.GAMIFICATION: "게임화"
        }
        
        phase_title = phase_titles.get(phase, "학습")
        strategy_title = strategy_titles.get(strategy, "활동")
        
        return f"{goal} - {phase_title} 및 {strategy_title}"
    
    def _generate_step_description(
        self, 
        goal: str, 
        phase: LearningPhase, 
        strategy: LearningStrategy, 
        domain: str
    ) -> str:
        """단계 설명 생성"""
        descriptions = {
            LearningPhase.INTRODUCTION: f"{goal}에 대한 기본 개념을 소개하고 학습 목표를 명확히 합니다.",
            LearningPhase.EXPLORATION: f"{goal}의 핵심 내용을 자세히 탐구하고 이해합니다.",
            LearningPhase.PRACTICE: f"{goal}과 관련된 문제를 직접 풀어보며 실력을 향상시킵니다.",
            LearningPhase.APPLICATION: f"학습한 {goal} 내용을 실제 상황에 적용해봅니다.",
            LearningPhase.SYNTHESIS: f"{goal} 학습 내용을 정리하고 다른 개념과 연결합니다.",
            LearningPhase.ASSESSMENT: f"{goal} 학습 성과를 평가하고 개선점을 파악합니다."
        }
        
        base_description = descriptions.get(phase, f"{goal} 학습을 진행합니다.")
        
        # 전략별 추가 설명
        strategy_additions = {
            LearningStrategy.EXPLANATION: "이론적 설명을 중심으로 진행합니다.",
            LearningStrategy.EXAMPLE: "구체적인 예시를 통해 이해를 돕습니다.",
            LearningStrategy.PRACTICE: "실습과 연습 문제를 통해 숙련도를 높입니다.",
            LearningStrategy.DISCUSSION: "질문과 답변을 통해 깊이 있는 이해를 추구합니다.",
            LearningStrategy.VISUALIZATION: "도표, 그래프, 이미지를 활용하여 시각적으로 학습합니다.",
            LearningStrategy.GAMIFICATION: "게임 요소를 활용하여 재미있게 학습합니다."
        }
        
        strategy_addition = strategy_additions.get(strategy, "")
        
        return f"{base_description} {strategy_addition}"
    
    def _suggest_resources(self, domain: str, phase: LearningPhase, strategy: LearningStrategy) -> List[str]:
        """학습 자료 추천"""
        resources = []
        
        # 도메인별 기본 자료
        domain_resources = {
            "history": ["교과서", "역사 지도", "사료", "다큐멘터리"],
            "math": ["교과서", "문제집", "계산기", "그래프 도구"],
            "science": ["교과서", "실험 도구", "관찰 기록지", "과학 영상"],
            "language": ["교과서", "사전", "읽기 자료", "듣기 자료"],
            "general": ["교과서", "참고서", "온라인 자료", "학습 노트"]
        }
        
        base_resources = domain_resources.get(domain, domain_resources["general"])
        resources.extend(base_resources)
        
        # 전략별 추가 자료
        strategy_resources = {
            LearningStrategy.VISUALIZATION: ["도표", "그래프", "이미지", "영상"],
            LearningStrategy.PRACTICE: ["연습 문제", "실습 도구", "체크리스트"],
            LearningStrategy.DISCUSSION: ["토론 주제", "질문 목록", "피드백 양식"],
            LearningStrategy.GAMIFICATION: ["퀴즈", "게임", "보상 시스템", "진행률 표시"]
        }
        
        additional_resources = strategy_resources.get(strategy, [])
        resources.extend(additional_resources)
        
        return list(set(resources))  # 중복 제거
    
    def _generate_assessment_criteria(self, goal: str, phase: LearningPhase) -> List[str]:
        """평가 기준 생성"""
        criteria = []
        
        if phase == LearningPhase.INTRODUCTION:
            criteria.extend([
                "기본 개념 이해도",
                "학습 목표 인식도"
            ])
        elif phase == LearningPhase.EXPLORATION:
            criteria.extend([
                "핵심 내용 파악도",
                "개념 간 연결 이해도"
            ])
        elif phase == LearningPhase.PRACTICE:
            criteria.extend([
                "문제 해결 능력",
                "적용 정확도"
            ])
        elif phase == LearningPhase.APPLICATION:
            criteria.extend([
                "실제 상황 적용 능력",
                "창의적 사고력"
            ])
        elif phase == LearningPhase.SYNTHESIS:
            criteria.extend([
                "종합적 이해도",
                "다른 개념과의 연결"
            ])
        elif phase == LearningPhase.ASSESSMENT:
            criteria.extend([
                "전체 학습 성과",
                "개선점 파악"
            ])
        
        return criteria
    
    def _set_step_dependencies(self, steps: List[LearningStep]):
        """단계 간 의존성 설정"""
        for i, step in enumerate(steps):
            if i > 0:
                # 이전 단계를 선행 조건으로 설정
                step.prerequisites = [steps[i-1].step_id]
    
    def get_plan_summary(self, steps: List[LearningStep]) -> Dict[str, Any]:
        """계획 요약 정보 반환"""
        total_duration = sum(step.estimated_duration for step in steps)
        
        phase_counts = {}
        strategy_counts = {}
        
        for step in steps:
            phase = step.phase.value
            strategy = step.strategy.value
            
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "total_steps": len(steps),
            "total_duration": total_duration,
            "phase_distribution": phase_counts,
            "strategy_distribution": strategy_counts,
            "difficulty_levels": list(set(step.difficulty for step in steps))
        }

# 기존 호환성을 위한 래퍼 클래스
class Planner(EnhancedPlanner):
    """기존 코드와의 호환성을 위한 래퍼"""
    pass
