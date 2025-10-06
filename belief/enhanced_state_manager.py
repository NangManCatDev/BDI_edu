# belief/enhanced_state_manager.py
"""
향상된 학습자 상태 관리 모듈
개인화된 학습자 프로파일과 학습 이력 추적
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

@dataclass
class LearningProfile:
    """학습자 프로파일"""
    learning_style: str = "visual"  # visual, auditory, kinesthetic, reading
    difficulty_preference: str = "medium"  # easy, medium, hard
    pace_preference: str = "normal"  # slow, normal, fast
    interests: List[str] = None
    strengths: List[str] = None
    weaknesses: List[str] = None
    
    def __post_init__(self):
        if self.interests is None:
            self.interests = []
        if self.strengths is None:
            self.strengths = []
        if self.weaknesses is None:
            self.weaknesses = []

@dataclass
class LearningSession:
    """학습 세션 기록"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    topics: List[str] = None
    questions_asked: List[str] = None
    responses_given: List[str] = None
    satisfaction_score: Optional[float] = None
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = []
        if self.questions_asked is None:
            self.questions_asked = []
        if self.responses_given is None:
            self.responses_given = []

@dataclass
class TopicMastery:
    """주제별 숙련도"""
    topic: str
    mastery_level: float  # 0.0 ~ 1.0
    last_practiced: datetime
    practice_count: int = 0
    success_rate: float = 0.0
    difficulty_level: str = "medium"

class EnhancedStateManager:
    """향상된 학습자 상태 관리자"""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.profile = LearningProfile()
        self.sessions: List[LearningSession] = []
        self.topic_mastery: Dict[str, TopicMastery] = {}
        self.current_session: Optional[LearningSession] = None
        
        # 기본 상태
        self.state = {
            "progress": 0.0,
            "level": "beginner",
            "last_topic": None,
            "total_learning_time": 0.0,
            "questions_asked": 0,
            "satisfaction_avg": 0.0
        }
        
        logger.info(f"EnhancedStateManager 초기화: {user_id}")
    
    def start_session(self, session_id: str = None) -> str:
        """새 학습 세션 시작"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = LearningSession(
            session_id=session_id,
            start_time=datetime.now()
        )
        
        logger.info(f"학습 세션 시작: {session_id}")
        return session_id
    
    def end_session(self, satisfaction_score: float = None):
        """현재 세션 종료"""
        if self.current_session:
            self.current_session.end_time = datetime.now()
            if satisfaction_score is not None:
                self.current_session.satisfaction_score = satisfaction_score
            
            self.sessions.append(self.current_session)
            self._update_learning_stats()
            self.current_session = None
            
            logger.info(f"학습 세션 종료: 만족도 {satisfaction_score}")
    
    def add_interaction(self, question: str, response: str, topic: str = None):
        """학습자 상호작용 기록"""
        if self.current_session:
            self.current_session.questions_asked.append(question)
            self.current_session.responses_given.append(response)
            if topic:
                self.current_session.topics.append(topic)
        
        # 상태 업데이트
        self.state["questions_asked"] += 1
        if topic:
            self.state["last_topic"] = topic
            self._update_topic_mastery(topic)
        
        logger.debug(f"상호작용 기록: {topic} - {question[:50]}...")
    
    def update_profile(self, **kwargs):
        """학습자 프로파일 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.profile, key):
                setattr(self.profile, key, value)
                logger.info(f"프로파일 업데이트: {key} = {value}")
    
    def get_learning_recommendations(self) -> List[str]:
        """학습자 수준에 맞는 추천사항 반환"""
        recommendations = []
        
        # 약점 기반 추천
        for weakness in self.profile.weaknesses:
            recommendations.append(f"{weakness} 영역 보완 학습 추천")
        
        # 학습 스타일 기반 추천
        if self.profile.learning_style == "visual":
            recommendations.append("시각적 자료 활용 학습 추천")
        elif self.profile.learning_style == "auditory":
            recommendations.append("음성 설명 중심 학습 추천")
        
        # 난이도 조절 추천
        if self.profile.difficulty_preference == "easy":
            recommendations.append("기초 개념부터 단계적 학습 추천")
        elif self.profile.difficulty_preference == "hard":
            recommendations.append("고급 문제와 심화 학습 추천")
        
        return recommendations
    
    def get_topic_difficulty(self, topic: str) -> str:
        """주제별 적절한 난이도 반환"""
        if topic in self.topic_mastery:
            mastery = self.topic_mastery[topic].mastery_level
            if mastery < 0.3:
                return "easy"
            elif mastery < 0.7:
                return "medium"
            else:
                return "hard"
        else:
            return self.profile.difficulty_preference
    
    def _update_topic_mastery(self, topic: str):
        """주제별 숙련도 업데이트"""
        if topic not in self.topic_mastery:
            self.topic_mastery[topic] = TopicMastery(
                topic=topic,
                mastery_level=0.1,
                last_practiced=datetime.now()
            )
        else:
            # 간단한 숙련도 증가 로직
            self.topic_mastery[topic].mastery_level = min(
                self.topic_mastery[topic].mastery_level + 0.05, 1.0
            )
            self.topic_mastery[topic].last_practiced = datetime.now()
            self.topic_mastery[topic].practice_count += 1
    
    def _update_learning_stats(self):
        """학습 통계 업데이트"""
        if not self.sessions:
            return
        
        # 총 학습 시간 계산
        total_time = 0.0
        total_satisfaction = 0.0
        satisfaction_count = 0
        
        for session in self.sessions:
            if session.end_time:
                duration = (session.end_time - session.start_time).total_seconds() / 3600
                total_time += duration
                
                if session.satisfaction_score is not None:
                    total_satisfaction += session.satisfaction_score
                    satisfaction_count += 1
        
        self.state["total_learning_time"] = total_time
        if satisfaction_count > 0:
            self.state["satisfaction_avg"] = total_satisfaction / satisfaction_count
        
        # 전체 진행률 계산
        total_topics = len(self.topic_mastery)
        if total_topics > 0:
            mastered_topics = sum(1 for tm in self.topic_mastery.values() 
                                if tm.mastery_level >= 0.8)
            self.state["progress"] = (mastered_topics / total_topics) * 100
    
    def get_state_summary(self) -> Dict[str, Any]:
        """상태 요약 정보 반환"""
        return {
            "user_id": self.user_id,
            "profile": asdict(self.profile),
            "state": self.state,
            "sessions_count": len(self.sessions),
            "topics_learned": len(self.topic_mastery),
            "recommendations": self.get_learning_recommendations()
        }
    
    def save_to_file(self, filename: str):
        """상태를 파일로 저장"""
        data = {
            "user_id": self.user_id,
            "profile": asdict(self.profile),
            "state": self.state,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "start_time": s.start_time.isoformat(),
                    "end_time": s.end_time.isoformat() if s.end_time else None,
                    "topics": s.topics,
                    "questions_asked": s.questions_asked,
                    "responses_given": s.responses_given,
                    "satisfaction_score": s.satisfaction_score
                }
                for s in self.sessions
            ],
            "topic_mastery": {
                topic: {
                    "mastery_level": tm.mastery_level,
                    "last_practiced": tm.last_practiced.isoformat(),
                    "practice_count": tm.practice_count,
                    "success_rate": tm.success_rate,
                    "difficulty_level": tm.difficulty_level
                }
                for topic, tm in self.topic_mastery.items()
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"상태 저장 완료: {filename}")
    
    def load_from_file(self, filename: str):
        """파일에서 상태 로드"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.user_id = data.get("user_id", "default")
            self.state = data.get("state", {})
            
            # 프로파일 로드
            profile_data = data.get("profile", {})
            self.profile = LearningProfile(**profile_data)
            
            # 세션 로드
            self.sessions = []
            for session_data in data.get("sessions", []):
                session = LearningSession(
                    session_id=session_data["session_id"],
                    start_time=datetime.fromisoformat(session_data["start_time"]),
                    end_time=datetime.fromisoformat(session_data["end_time"]) if session_data["end_time"] else None,
                    topics=session_data.get("topics", []),
                    questions_asked=session_data.get("questions_asked", []),
                    responses_given=session_data.get("responses_given", []),
                    satisfaction_score=session_data.get("satisfaction_score")
                )
                self.sessions.append(session)
            
            # 주제 숙련도 로드
            self.topic_mastery = {}
            for topic, mastery_data in data.get("topic_mastery", {}).items():
                self.topic_mastery[topic] = TopicMastery(
                    topic=topic,
                    mastery_level=mastery_data["mastery_level"],
                    last_practiced=datetime.fromisoformat(mastery_data["last_practiced"]),
                    practice_count=mastery_data["practice_count"],
                    success_rate=mastery_data["success_rate"],
                    difficulty_level=mastery_data["difficulty_level"]
                )
            
            logger.info(f"상태 로드 완료: {filename}")
            
        except Exception as e:
            logger.error(f"상태 로드 실패: {str(e)}")

# 기존 호환성을 위한 래퍼 클래스
class StateManager(EnhancedStateManager):
    """기존 코드와의 호환성을 위한 래퍼"""
    
    def __init__(self):
        super().__init__("default")
    
    def all(self):
        return self.state
    
    def show(self):
        return self.all()
