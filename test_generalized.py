#!/usr/bin/env python3
"""
일반화된 BDI_edu 시스템 테스트
다양한 도메인과 언어 지원 테스트
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

def test_domain_configs():
    """도메인 설정 테스트"""
    print("🔧 도메인 설정 테스트...")
    
    try:
        from config.domain_config import Domain, get_domain_config, get_supported_domains
        
        # 지원되는 도메인 확인
        domains = get_supported_domains()
        print(f"✅ 지원되는 도메인: {[d.value for d in domains]}")
        
        # 각 도메인별 설정 테스트
        for domain in domains:
            config = get_domain_config(domain)
            print(f"  📋 {domain.value}: {len(config.predicates)}개 predicates, {len(config.keywords)}개 키워드 유형")
        
        return True
        
    except Exception as e:
        print(f"❌ 도메인 설정 테스트 실패: {str(e)}")
        return False

def test_generalized_knowledge_base():
    """일반화된 지식베이스 테스트"""
    print("🔧 일반화된 지식베이스 테스트...")
    
    try:
        from belief.generalized_knowledge_base import build_generalized_kb
        from config.domain_config import Domain
        
        # 각 도메인별 지식베이스 테스트
        for domain in [Domain.HISTORY, Domain.MATH, Domain.SCIENCE]:
            print(f"  📚 {domain.value} 도메인 테스트...")
            kb = build_generalized_kb(domain)
            
            info = kb.get_domain_info()
            print(f"    - Predicates: {list(info['predicates'].keys())}")
            print(f"    - Facts: {len(kb.engine.facts)}개")
            
            # 도메인 전환 테스트
            if domain != Domain.HISTORY:
                kb.switch_domain(Domain.HISTORY)
                print(f"    - 도메인 전환 테스트 완료")
        
        return True
        
    except Exception as e:
        print(f"❌ 일반화된 지식베이스 테스트 실패: {str(e)}")
        return False

def test_generalized_nl2kqml():
    """일반화된 NL→KQML 변환 테스트"""
    print("🔧 일반화된 NL→KQML 변환 테스트...")
    
    try:
        from interface.generalized_nl2kqml import GeneralizedNL2KQML
        from config.domain_config import Domain
        
        # 각 도메인별 변환 테스트
        test_questions = {
            Domain.HISTORY: [
                "삼국통일이 언제인가요?",
                "세종대왕은 누구인가요?",
                "임진왜란의 원인은 무엇인가요?"
            ],
            Domain.MATH: [
                "피타고라스 정리는 무엇인가요?",
                "이차방정식을 어떻게 풀나요?",
                "미분의 정의는 무엇인가요?"
            ],
            Domain.SCIENCE: [
                "광합성은 무엇인가요?",
                "뉴턴의 운동법칙은 무엇인가요?",
                "DNA의 구조는 어떻게 되어있나요?"
            ]
        }
        
        for domain, questions in test_questions.items():
            print(f"  📝 {domain.value} 도메인 질문 테스트...")
            converter = GeneralizedNL2KQML(domain)
            
            for question in questions:
                result = converter.convert(question)
                if result:
                    print(f"    ✅ '{question}' → {result}")
                else:
                    print(f"    ⚠️  '{question}' → 변환 실패")
                
                # 질문 유형 분석 테스트
                q_type = converter.analyze_question_type(question)
                print(f"      질문 유형: {q_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ 일반화된 NL→KQML 변환 테스트 실패: {str(e)}")
        return False

def test_generalized_kqml2nl():
    """일반화된 KQML→NL 변환 테스트"""
    print("🔧 일반화된 KQML→NL 변환 테스트...")
    
    try:
        from interface.generalized_kqml2nl import GeneralizedKQML2NL
        from config.domain_config import Domain
        
        # 가상의 Atom 객체 시뮬레이션
        class MockAtom:
            def __init__(self, predicate, arguments):
                self.predicate = type('Predicate', (), {'name': predicate})()
                self.arguments = arguments
        
        # 각 도메인별 변환 테스트
        test_atoms = {
            Domain.HISTORY: MockAtom("event", ["'삼국통일'", "'668'", "'신라'"]),
            Domain.MATH: MockAtom("formula", ["'피타고라스정리'", "'a²+b²=c²'", "'직각삼각형'"]),
            Domain.SCIENCE: MockAtom("phenomenon", ["'광합성'", "'식물이 빛을 이용'", "'햇빛'"])
        }
        
        for domain, atom in test_atoms.items():
            print(f"  📝 {domain.value} 도메인 변환 테스트...")
            converter = GeneralizedKQML2NL(domain)
            
            # 실제 변환은 API 키가 필요하므로 스킵
            print(f"    ⚠️  실제 LLM 변환은 API 키가 필요합니다.")
            print(f"    📋 포맷팅 테스트: {converter.format_for_display(atom)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 일반화된 KQML→NL 변환 테스트 실패: {str(e)}")
        return False

def test_domain_switching():
    """도메인 전환 테스트"""
    print("🔧 도메인 전환 테스트...")
    
    try:
        from interface.generalized_nl2kqml import GeneralizedNL2KQML
        from interface.generalized_kqml2nl import GeneralizedKQML2NL
        from config.domain_config import Domain
        
        # NL→KQML 변환기 도메인 전환
        nl_converter = GeneralizedNL2KQML(Domain.HISTORY)
        print("✅ NL→KQML 변환기 초기화 (역사 도메인)")
        
        nl_converter.switch_domain(Domain.MATH)
        print("✅ 도메인 전환: 역사 → 수학")
        
        # KQML→NL 변환기 도메인 전환
        kqml_converter = GeneralizedKQML2NL(Domain.HISTORY)
        print("✅ KQML→NL 변환기 초기화 (역사 도메인)")
        
        kqml_converter.switch_domain(Domain.SCIENCE)
        print("✅ 도메인 전환: 역사 → 과학")
        
        return True
        
    except Exception as e:
        print(f"❌ 도메인 전환 테스트 실패: {str(e)}")
        return False

def main():
    """전체 일반화 테스트 실행"""
    print("🚀 BDI_edu 일반화 시스템 테스트 시작")
    print("=" * 60)
    
    tests = [
        ("도메인 설정", test_domain_configs),
        ("일반화된 지식베이스", test_generalized_knowledge_base),
        ("일반화된 NL→KQML", test_generalized_nl2kqml),
        ("일반화된 KQML→NL", test_generalized_kqml2nl),
        ("도메인 전환", test_domain_switching),
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
        print("🎉 모든 일반화 테스트 통과! 시스템이 다양한 도메인을 지원합니다.")
        return True
    else:
        print("⚠️  일부 테스트 실패. 로그를 확인하여 문제를 해결하세요.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
