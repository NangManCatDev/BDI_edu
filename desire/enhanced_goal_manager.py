# desire/enhanced_goal_manager.py
"""
향상된 목표 관리 모듈
학습자 수준과 도메인에 맞는 지능적 목표 설정
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class GoalType(Enum):
    """목표 유형"""
    KNOWLEDGE = "knowledge"  # 지식 습득
    SKILL = "skill"         # 기술 습득
    UNDERSTANDING = "understanding"  # 이해도 향상
    APPLICATION = "application"     # 적용 능력
    ANALYSIS = "analysis"           # 분석 능력
    SYNTHESIS = "synthesis"         # 종합 능력

class GoalStatus(Enum):
    """목표 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class DifficultyLevel(Enum):
    """난이도 수준"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class LearningGoal:
    """학습 목표"""
    id: str
    title: str
    description: str
    goal_type: GoalType
    difficulty: DifficultyLevel
    priority: int  # 1-10
    status: GoalStatus
    created_at: datetime
    due_date: Optional[datetime] = None
    prerequisites: List[str] = None  # 선행 목표 ID들
    estimated_duration: int = 0  # 예상 소요 시간 (분)
    progress: float = 0.0  # 진행률 (0.0-1.0)
    tags: List[str] = None
    domain: str = "general"
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.tags is None:
            self.tags = []

class EnhancedGoalManager:
    """향상된 목표 관리자"""
    
    def __init__(self, user_level: str = "beginner"):
        self.goals: Dict[str, LearningGoal] = {}
        self.user_level = user_level
        self.goal_counter = 0
        
        logger.info(f"EnhancedGoalManager 초기화: {user_level}")
    
    def create_goal(
        self,
        title: str,
        description: str,
        goal_type: GoalType = GoalType.KNOWLEDGE,
        difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
        priority: int = 5,
        due_date: Optional[datetime] = None,
        prerequisites: List[str] = None,
        estimated_duration: int = 30,
        tags: List[str] = None,
        domain: str = "general"
    ) -> str:
        """새로운 학습 목표 생성"""
        goal_id = f"goal_{self.goal_counter:04d}"
        self.goal_counter += 1
        
        goal = LearningGoal(
            id=goal_id,
            title=title,
            description=description,
            goal_type=goal_type,
            difficulty=difficulty,
            priority=priority,
            status=GoalStatus.PENDING,
            created_at=datetime.now(),
            due_date=due_date,
            prerequisites=prerequisites or [],
            estimated_duration=estimated_duration,
            tags=tags or [],
            domain=domain
        )
        
        self.goals[goal_id] = goal
        logger.info(f"목표 생성: {title} ({goal_type.value})")
        return goal_id
    
    def add_goal(self, goal: str, priority: int = 5, **kwargs) -> str:
        """기존 호환성을 위한 래퍼"""
        return self.create_goal(
            title=goal,
            description=goal,
            priority=priority,
            **kwargs
        )
    
    def get_goal(self, goal_id: str) -> Optional[LearningGoal]:
        """목표 조회"""
        return self.goals.get(goal_id)
    
    def update_goal_status(self, goal_id: str, status: GoalStatus):
        """목표 상태 업데이트"""
        if goal_id in self.goals:
            self.goals[goal_id].status = status
            logger.info(f"목표 상태 변경: {goal_id} -> {status.value}")
    
    def update_goal_progress(self, goal_id: str, progress: float):
        """목표 진행률 업데이트"""
        if goal_id in self.goals:
            self.goals[goal_id].progress = max(0.0, min(1.0, progress))
            if progress >= 1.0:
                self.update_goal_status(goal_id, GoalStatus.COMPLETED)
            elif progress > 0.0:
                self.update_goal_status(goal_id, GoalStatus.IN_PROGRESS)
            logger.info(f"목표 진행률 업데이트: {goal_id} -> {progress:.1%}")
    
    def complete_goal(self, goal_id: str):
        """목표 완료"""
        self.update_goal_status(goal_id, GoalStatus.COMPLETED)
        self.update_goal_progress(goal_id, 1.0)
    
    def get_available_goals(self) -> List[LearningGoal]:
        """실행 가능한 목표들 반환 (선행 조건 만족)"""
        available = []
        
        for goal in self.goals.values():
            if goal.status == GoalStatus.PENDING:
                # 선행 조건 확인
                prerequisites_met = True
                for prereq_id in goal.prerequisites:
                    prereq = self.get_goal(prereq_id)
                    if not prereq or prereq.status != GoalStatus.COMPLETED:
                        prerequisites_met = False
                        break
                
                if prerequisites_met:
                    available.append(goal)
        
        return available
    
    def get_highest_priority_goal(self) -> Optional[LearningGoal]:
        """최고 우선순위 목표 반환"""
        available_goals = self.get_available_goals()
        if not available_goals:
            return None
        
        return max(available_goals, key=lambda g: g.priority)
    
    def get_goals_by_domain(self, domain: str) -> List[LearningGoal]:
        """도메인별 목표 조회"""
        return [goal for goal in self.goals.values() if goal.domain == domain]
    
    def get_goals_by_difficulty(self, difficulty: DifficultyLevel) -> List[LearningGoal]:
        """난이도별 목표 조회"""
        return [goal for goal in self.goals.values() if goal.difficulty == difficulty]
    
    def get_goals_by_type(self, goal_type: GoalType) -> List[LearningGoal]:
        """유형별 목표 조회"""
        return [goal for goal in self.goals.values() if goal.goal_type == goal_type]
    
    def get_overdue_goals(self) -> List[LearningGoal]:
        """기한이 지난 목표들 반환"""
        now = datetime.now()
        overdue = []
        
        for goal in self.goals.values():
            if (goal.due_date and 
                goal.due_date < now and 
                goal.status not in [GoalStatus.COMPLETED, GoalStatus.FAILED]):
                overdue.append(goal)
        
        return overdue
    
    def get_learning_recommendations(self, user_profile: Dict[str, Any] = None) -> List[str]:
        """학습자 프로파일에 따른 목표 추천"""
        recommendations = []
        
        # 현재 진행 중인 목표가 적으면 추천
        in_progress_count = len([g for g in self.goals.values() 
                               if g.status == GoalStatus.IN_PROGRESS])
        if in_progress_count < 2:
            recommendations.append("새로운 학습 목표 설정을 고려해보세요")
        
        # 기한이 지난 목표가 있으면 알림
        overdue_goals = self.get_overdue_goals()
        if overdue_goals:
            recommendations.append(f"{len(overdue_goals)}개의 기한이 지난 목표가 있습니다")
        
        # 난이도별 균형 추천
        beginner_goals = len(self.get_goals_by_difficulty(DifficultyLevel.BEGINNER))
        advanced_goals = len(self.get_goals_by_difficulty(DifficultyLevel.ADVANCED))
        
        if beginner_goals > advanced_goals * 2:
            recommendations.append("고급 목표 도전을 고려해보세요")
        elif advanced_goals > beginner_goals * 2:
            recommendations.append("기초 목표로 돌아가서 복습해보세요")
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """목표 통계 정보 반환"""
        total_goals = len(self.goals)
        completed_goals = len([g for g in self.goals.values() 
                             if g.status == GoalStatus.COMPLETED])
        in_progress_goals = len([g for g in self.goals.values() 
                               if g.status == GoalStatus.IN_PROGRESS])
        
        # 도메인별 통계
        domain_stats = {}
        for goal in self.goals.values():
            domain = goal.domain
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "completed": 0}
            domain_stats[domain]["total"] += 1
            if goal.status == GoalStatus.COMPLETED:
                domain_stats[domain]["completed"] += 1
        
        return {
            "total_goals": total_goals,
            "completed_goals": completed_goals,
            "in_progress_goals": in_progress_goals,
            "completion_rate": (completed_goals / total_goals * 100) if total_goals > 0 else 0,
            "domain_stats": domain_stats,
            "overdue_count": len(self.get_overdue_goals())
        }
    
    def show(self) -> List[Dict[str, Any]]:
        """목표 목록 반환 (기존 호환성)"""
        return [
            {
                "goal": goal.title,
                "priority": goal.priority,
                "status": goal.status.value,
                "progress": goal.progress,
                "difficulty": goal.difficulty.value,
                "domain": goal.domain
            }
            for goal in self.goals.values()
        ]
    
    def get_highest_priority(self) -> Optional[Dict[str, Any]]:
        """최고 우선순위 목표 반환 (기존 호환성)"""
        goal = self.get_highest_priority_goal()
        if goal:
            return {
                "goal": goal.title,
                "priority": goal.priority,
                "status": goal.status.value,
                "progress": goal.progress
            }
        return None

# 기존 호환성을 위한 래퍼 클래스
class GoalManager(EnhancedGoalManager):
    """기존 코드와의 호환성을 위한 래퍼"""
    
    def __init__(self):
        super().__init__("beginner")
    
    def add(self, goal: str, priority: int = 1):
        return self.add_goal(goal, priority)
    
    def complete(self, goal: str):
        # 제목으로 목표 찾아서 완료 처리
        for goal_id, goal_obj in self.goals.items():
            if goal_obj.title == goal:
                self.complete_goal(goal_id)
                break
    
    def complete_goal(self, goal: str):
        return self.complete(goal)
    
    def highest_priority(self):
        return self.get_highest_priority()
