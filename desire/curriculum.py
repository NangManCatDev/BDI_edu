# curriculum.py
"""
Curriculum: 장기 학습 로드맵 관리
- 목표를 단기/중기/장기로 나누어 관리
- GoalManager와 연결
"""

from typing import Dict, List

class Curriculum:
    """
    장기/중기/단기 학습 커리큘럼 관리
    """

    def __init__(self):
        self.curriculum = {
            "short_term": [],
            "mid_term": [],
            "long_term": []
        }

    def add(self, term: str, topic: str):
        """단기/중기/장기 커리큘럼에 주제 추가"""
        if term not in self.curriculum:
            raise ValueError(f"Unknown term: {term}")
        self.curriculum[term].append(topic)

    def get(self, term: str):
        """특정 범주의 커리큘럼 반환"""
        return self.curriculum.get(term, [])

    def show(self):
        """전체 커리큘럼 구조 반환"""
        return self.curriculum
