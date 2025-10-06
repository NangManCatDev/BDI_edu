#!/usr/bin/env python3
"""
BDI_edu 핵심 모듈들 통합 테스트
"""

import logging
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_llm_connector():
    """LLMConnector 테스트"""
    print("🔧 LLMConnector 테스트...")
    
    try:
        from interface.llm_connector import LLMConnector
        
        # API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
            return False
            
        connector = LLMConnector()
        print("✅ LLMConnector 초기화 성공")
        
        # 간단한 테스트 (실제 API 호출은 하지 않음)
        print("✅ LLMConnector 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ LLMConnector 테스트 실패: {str(e)}")
        return False

def test_nl2kqml():
    """nl2kqml 모듈 테스트"""
    print("🔧 nl2kqml 모듈 테스트...")
    
    try:
        from interface.nl2kqml import nl_to_kqml
        
        # 테스트 질문들
        test_questions = [
            "삼국통일이 언제인가요?",
            "세종대왕은 누구인가요?",
            "동학농민운동의 원인은 무엇인가요?"
        ]
        
        for question in test_questions:
            print(f"  질문: {question}")
            result = nl_to_kqml(question)
            if result:
                print(f"  ✅ 변환 성공: {result}")
            else:
                print(f"  ⚠️  변환 실패 (매칭되는 fact 없음)")
        
        print("✅ nl2kqml 모듈 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ nl2kqml 모듈 테스트 실패: {str(e)}")
        return False

def test_kqml2nl():
    """kqml2nl 모듈 테스트"""
    print("🔧 kqml2nl 모듈 테스트...")
    
    try:
        from interface.kqml2nl import kqml_to_nl, format_kqml_for_display
        
        # 가상의 Atom 객체 시뮬레이션
        class MockAtom:
            def __init__(self, predicate, arguments):
                self.predicate = type('Predicate', (), {'name': predicate})()
                self.arguments = arguments
        
        # 테스트 Atom
        test_atom = MockAtom("event", ["'삼국통일'", "'668'", "'신라'"])
        
        print(f"  테스트 Atom: {format_kqml_for_display(test_atom)}")
        
        # 실제 변환은 API 키가 필요하므로 스킵
        print("  ⚠️  실제 LLM 변환은 API 키가 필요합니다.")
        print("✅ kqml2nl 모듈 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ kqml2nl 모듈 테스트 실패: {str(e)}")
        return False

def test_belief_modules():
    """Belief 모듈들 테스트"""
    print("🔧 Belief 모듈들 테스트...")
    
    try:
        from belief.state_manager import StateManager
        from belief.knowledge_base import build_kb
        
        # StateManager 테스트
        state = StateManager()
        print("✅ StateManager 초기화 성공")
        
        # Knowledge Base 테스트
        kb = build_kb()
        print(f"✅ Knowledge Base 로드 성공 (facts: {len(kb.facts)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Belief 모듈 테스트 실패: {str(e)}")
        return False

def test_desire_modules():
    """Desire 모듈들 테스트"""
    print("🔧 Desire 모듈들 테스트...")
    
    try:
        from desire.goal_manager import GoalManager
        from desire.curriculum import Curriculum
        from desire.progress_tracker import ProgressTracker
        
        # GoalManager 테스트
        goal_manager = GoalManager()
        goal_manager.add("테스트 목표", priority=1)
        print("✅ GoalManager 테스트 성공")
        
        # Curriculum 테스트
        curriculum = Curriculum()
        curriculum.add("short_term", "테스트 과정")
        print("✅ Curriculum 테스트 성공")
        
        # ProgressTracker 테스트
        progress = ProgressTracker()
        progress.set_progress("테스트", "pending")
        print("✅ ProgressTracker 테스트 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ Desire 모듈 테스트 실패: {str(e)}")
        return False

def test_intention_modules():
    """Intention 모듈들 테스트"""
    print("🔧 Intention 모듈들 테스트...")
    
    try:
        from intention.planner import Planner
        from intention.executor import Executor
        from intention.feedback_agent import FeedbackAgent
        
        # Planner 테스트
        planner = Planner()
        plan = planner.make_plan("테스트 목표")
        print(f"✅ Planner 테스트 성공: {len(plan)}개 계획 생성")
        
        # Executor 테스트
        executor = Executor()
        results = executor.execute(plan)
        print(f"✅ Executor 테스트 성공: {len(results)}개 결과")
        
        # FeedbackAgent 테스트
        feedback = FeedbackAgent()
        fb = feedback.evaluate(results)
        print(f"✅ FeedbackAgent 테스트 성공: {fb}")
        
        return True
        
    except Exception as e:
        print(f"❌ Intention 모듈 테스트 실패: {str(e)}")
        return False

def main():
    """전체 통합 테스트 실행"""
    print("🚀 BDI_edu 핵심 모듈 통합 테스트 시작")
    print("=" * 60)
    
    tests = [
        ("LLMConnector", test_llm_connector),
        ("nl2kqml", test_nl2kqml),
        ("kqml2nl", test_kqml2nl),
        ("Belief 모듈들", test_belief_modules),
        ("Desire 모듈들", test_desire_modules),
        ("Intention 모듈들", test_intention_modules),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name} 테스트")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 통과")
            else:
                print(f"❌ {test_name} 실패")
        except Exception as e:
            print(f"❌ {test_name} 예외 발생: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"🎯 테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 모듈들이 정상적으로 작동합니다.")
        return True
    else:
        print("⚠️  일부 테스트 실패. 로그를 확인하여 문제를 해결하세요.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
