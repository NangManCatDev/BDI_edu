#!/usr/bin/env python3
"""
NEO Query RAG 시스템 테스트 스크립트
"""

import os
import sys
sys.path.append('.')

from interface.nl_to_neo_query import NEOQueryRAG

def test_neo_query_system():
    """NEO Query RAG 시스템 테스트"""
    print("🧪 NEO Query RAG 시스템 테스트 시작")
    print("=" * 50)
    
    # RAG 시스템 초기화
    try:
        rag = NEOQueryRAG()
        print("✅ NEOQueryRAG 초기화 완료")
    except Exception as e:
        print(f"❌ NEOQueryRAG 초기화 실패: {str(e)}")
        return False
    
    # .nkb 파일 찾기
    nkb_files = [
        "sample_history.nkb",
        "data/history.nkb", 
        "kb_generator/sample_history.nkb"
    ]
    
    nkb_file = None
    for file_path in nkb_files:
        if os.path.exists(file_path):
            nkb_file = file_path
            break
    
    if not nkb_file:
        print("❌ .nkb 파일을 찾을 수 없습니다.")
        print("먼저 kb_generator/txt_to_kb.py로 .nkb 파일을 생성하세요.")
        return False
    
    print(f"📁 NEO KB 파일 발견: {nkb_file}")
    
    # NEO KB 파일 로드
    if not rag.load_nkb_file(nkb_file):
        print("❌ NEO KB 파일 로드 실패")
        return False
    
    # KB 통계 출력
    stats = rag.get_kb_stats()
    print(f"📊 KB 통계: {stats['total_facts']}개 사실, {stats['total_rules']}개 규칙")
    print(f"📊 사용 가능한 predicate: {', '.join(stats['predicates'])}")
    
    # 테스트 질문들
    test_questions = [
        "이순신은 누구야?",
        "임진왜란은 언제 일어났어?",
        "세종대왕이 한 일은?",
        "독립운동가들은 누구야?",
        "동학농민운동의 결과는?"
    ]
    
    print("\n🔍 테스트 질문들:")
    print("-" * 30)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. 질문: {question}")
        
        try:
            result = rag.convert_to_neo_query(question)
            
            if result["success"]:
                print(f"   ✅ NEO Query: {result['query']}")
                print(f"   📊 사용된 predicate: {', '.join(result['predicates_used'])}")
            else:
                print(f"   ❌ 변환 실패: {result['error']}")
                
        except Exception as e:
            print(f"   💥 오류 발생: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎉 NEO Query RAG 시스템 테스트 완료")
    return True

if __name__ == "__main__":
    test_neo_query_system()
