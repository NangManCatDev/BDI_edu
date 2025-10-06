#!/usr/bin/env python3
"""
자연어 질의 → NEO KB 쿼리 변환 모듈
사용법: python nl_to_nkb_query.py "질문" kb_file.nkb
"""

import os
import sys
import logging
from typing import Optional, List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NLToNKBQueryConverter:
    """자연어 질의를 NEO KB 쿼리로 변환하는 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """OpenAI API 클라이언트 초기화"""
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        self.client = OpenAI(api_key=api_key)
        
        # 시스템 프롬프트 로드
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """시스템 프롬프트 파일을 로드합니다"""
        try:
            prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'system_prompt', 'nl_to_nkb_query_prompt.txt')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning("시스템 프롬프트 파일을 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
            return """당신은 NEO 지식베이스 쿼리 전문가입니다. 자연어 질의를 NEO KB 쿼리 형식으로 변환해주세요."""

    def _load_kb_schema(self, kb_file: str) -> str:
        """KB 파일의 스키마 정보를 로드합니다"""
        try:
            with open(kb_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # KB 파일에서 predicate 정보 추출
            predicates = set()
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith(';') and not line.startswith('%'):
                    # predicate(arg1, arg2, ...) 형식에서 predicate 추출
                    if '(' in line:
                        predicate = line.split('(')[0].strip()
                        predicates.add(predicate)
            
            schema_info = f"사용 가능한 predicates: {', '.join(sorted(predicates))}"
            logger.info(f"KB 스키마 로드: {len(predicates)}개 predicates")
            return schema_info
            
        except Exception as e:
            logger.warning(f"KB 스키마 로드 실패: {str(e)}")
            return "사용 가능한 predicates: event, person, cause"

    def convert_nl_to_query(self, question: str, kb_file: str) -> Optional[str]:
        """자연어 질의를 NEO KB 쿼리로 변환"""
        logger.info(f"자연어 질의를 NEO KB 쿼리로 변환: {question}")
        
        # KB 스키마 정보 로드
        schema_info = self._load_kb_schema(kb_file)
        
        user_prompt = f"""다음 자연어 질의를 NEO KB 쿼리 형식으로 변환해주세요.

질의: {question}

{schema_info}

변환 규칙:
1. event(Name, Year, Desc) - 사건 정보 조회
2. person(Name, Birth, Death, Role) - 인물 정보 조회  
3. cause(Event, Result) - 인과관계 조회

예시:
- "이순신은 누구야?" → person(X, Y, Z, W)
- "1392년에 무슨 일이 있었어?" → event(X, '1392', Y)
- "조선 건국의 원인은?" → cause(X, '조선 건국')

변환된 쿼리만 출력해주세요."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            query = response.choices[0].message.content.strip()
            
            # 코드 블록 마크다운 제거
            if query.startswith("```") and query.endswith("```"):
                query = "\n".join(query.split('\n')[1:-1]).strip()
            
            logger.info(f"변환된 쿼리: {query}")
            return query
            
        except Exception as e:
            logger.error(f"쿼리 변환 실패: {str(e)}")
            return None

    def execute_query(self, query: str, kb_file: str) -> List[Dict[str, Any]]:
        """NEO KB 쿼리를 실행합니다"""
        logger.info(f"NEO KB 쿼리 실행: {query}")
        
        try:
            # NEO 엔진을 사용한 쿼리 실행
            from belief.neo_engine_wrapper import NEOEngine
            from belief.neo_kb_loader import NEOKBLoader
            
            # KB 파일 로드
            engine = NEOEngine()
            loader = NEOKBLoader(engine)
            
            if not loader.load_kb_from_file(kb_file):
                logger.error("KB 파일 로드 실패")
                return []
            
            # 쿼리 실행
            results = engine.query(query)
            logger.info(f"쿼리 결과: {len(results)}개")
            
            return results
            
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {str(e)}")
            return []

    def query_to_nl(self, question: str, results: List[Dict[str, Any]]) -> str:
        """쿼리 결과를 자연어로 변환"""
        if not results:
            return "해당 질문에 대한 정보를 찾을 수 없습니다."
        
        try:
            # LLM을 사용한 결과 변환
            user_prompt = f"""다음 질문과 쿼리 결과를 자연스러운 한국어로 변환해주세요.

질문: {question}
결과: {results}

자연스럽고 이해하기 쉬운 한국어로 답변해주세요."""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 한국어 답변 전문가입니다. 쿼리 결과를 자연스러운 한국어로 변환해주세요."},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("쿼리 결과를 자연어로 변환 완료")
            return answer
            
        except Exception as e:
            logger.error(f"자연어 변환 실패: {str(e)}")
            return f"쿼리 결과: {results}"

    def process_question(self, question: str, kb_file: str) -> Dict[str, Any]:
        """전체 질의 처리 파이프라인"""
        logger.info(f"질의 처리 시작: {question}")
        
        # 1. 자연어 → 쿼리 변환
        query = self.convert_nl_to_query(question, kb_file)
        if not query:
            return {"error": "쿼리 변환 실패"}
        
        # 2. 쿼리 실행
        results = self.execute_query(query, kb_file)
        if not results:
            return {"error": "쿼리 실행 실패 또는 결과 없음"}
        
        # 3. 결과 → 자연어 변환
        answer = self.query_to_nl(question, results)
        
        return {
            "question": question,
            "query": query,
            "results": results,
            "answer": answer
        }

def main():
    """메인 함수"""
    if len(sys.argv) != 3:
        print("사용법: python nl_to_nkb_query.py <질문> <kb파일.nkb>")
        print("예시: python nl_to_nkb_query.py '이순신은 누구야?' history.nkb")
        sys.exit(1)
    
    question = sys.argv[1]
    kb_file = sys.argv[2]
    
    # 파일 존재 확인
    if not os.path.exists(kb_file):
        print(f"❌ KB 파일을 찾을 수 없습니다: {kb_file}")
        sys.exit(1)
    
    # 질의 처리
    converter = NLToNKBQueryConverter()
    result = converter.process_question(question, kb_file)
    
    if "error" in result:
        print(f"❌ 오류: {result['error']}")
        sys.exit(1)
    else:
        print(f"✅ 질문: {result['question']}")
        print(f"🔍 쿼리: {result['query']}")
        print(f"📊 결과: {len(result['results'])}개")
        print(f"💬 답변: {result['answer']}")

if __name__ == "__main__":
    main()
