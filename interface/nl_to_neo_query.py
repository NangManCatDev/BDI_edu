#!/usr/bin/env python3
"""
자연어 질의를 NEO query로 변환하는 RAG 시스템
.nkb 파일을 로드하고 사용자 질문을 적절한 NEO query로 변환
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NEOQueryRAG:
    """자연어 질의를 NEO query로 변환하는 RAG 시스템"""
    
    def __init__(self, api_key: Optional[str] = None):
        """OpenAI API 클라이언트 초기화"""
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        self.client = OpenAI(api_key=api_key)
        
        # 시스템 프롬프트 로드
        self.system_prompt = self._load_system_prompt()
        
        # NEO KB 데이터 저장소
        self.kb_facts = []
        self.kb_rules = []
        self.predicates = set()
        
    def _load_system_prompt(self) -> str:
        """시스템 프롬프트 파일을 로드합니다"""
        try:
            prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'system_prompt', 'nl_to_neo_query_prompt.txt')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning("시스템 프롬프트 파일을 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
            return """당신은 NEO 지식베이스 전문가입니다. 사용자의 자연어 질문을 NEO query 형식으로 변환해주세요."""
        except Exception as e:
            logger.error(f"시스템 프롬프트 로드 실패: {str(e)}")
            return """당신은 NEO 지식베이스 전문가입니다. 사용자의 자연어 질문을 NEO query 형식으로 변환해주세요."""

    def load_nkb_file(self, nkb_file_path: str) -> bool:
        """NEO KB 파일(.nkb)을 로드합니다"""
        logger.info(f"NEO KB 파일 로드 시작: {nkb_file_path}")
        
        try:
            with open(nkb_file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                logger.error("NEO KB 파일이 비어있습니다")
                return False
            
            # KB 파일 파싱
            self._parse_kb_content(content)
            
            logger.info(f"✅ NEO KB 파일 로드 완료: {len(self.kb_facts)}개 사실, {len(self.kb_rules)}개 규칙")
            return True
            
        except FileNotFoundError:
            logger.error(f"NEO KB 파일을 찾을 수 없습니다: {nkb_file_path}")
            return False
        except Exception as e:
            logger.error(f"NEO KB 파일 로드 실패: {str(e)}")
            return False

    def _parse_kb_content(self, content: str):
        """KB 파일 내용을 파싱하여 사실과 규칙을 분리"""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # 빈 줄이나 주석 제외
                continue
                
            try:
                if ':-' in line:  # 규칙 (Rule)
                    self.kb_rules.append(line)
                    # 규칙에서 predicate 추출
                    head = line.split(':-')[0].strip()
                    pred_match = re.match(r'(\w+)\s*\(', head)
                    if pred_match:
                        self.predicates.add(pred_match.group(1))
                else:  # 사실 (Fact)
                    self.kb_facts.append(line)
                    # 사실에서 predicate 추출
                    pred_match = re.match(r'(\w+)\s*\(', line)
                    if pred_match:
                        self.predicates.add(pred_match.group(1))
                        
            except Exception as e:
                logger.warning(f"라인 {line_num} 파싱 실패: {line} - {str(e)}")
                continue

    def _find_relevant_facts(self, query: str, top_k: int = 5) -> List[str]:
        """질문과 관련된 사실들을 찾습니다 (간단한 키워드 매칭)"""
        query_lower = query.lower()
        relevant_facts = []
        
        # 키워드 기반 매칭
        for fact in self.kb_facts:
            fact_lower = fact.lower()
            # 공통 키워드가 있는지 확인
            if any(keyword in fact_lower for keyword in query_lower.split()):
                relevant_facts.append(fact)
                if len(relevant_facts) >= top_k:
                    break
        
        # 매칭된 사실이 적으면 모든 사실 반환
        if len(relevant_facts) < 3:
            relevant_facts = self.kb_facts[:top_k]
            
        return relevant_facts

    def convert_to_neo_query(self, user_question: str) -> Dict[str, Any]:
        """자연어 질문을 NEO query로 변환"""
        logger.info(f"자연어 질문을 NEO query로 변환: {user_question}")
        
        if not self.kb_facts:
            return {
                "success": False,
                "error": "NEO KB가 로드되지 않았습니다. 먼저 load_nkb_file()을 호출하세요.",
                "query": None
            }
        
        # 관련 사실들 찾기
        relevant_facts = self._find_relevant_facts(user_question)
        
        # LLM을 통한 query 변환
        user_prompt = f"""사용자 질문: {user_question}

사용 가능한 predicate들: {', '.join(sorted(self.predicates))}

관련 사실들:
{chr(10).join(relevant_facts[:10])}

위 정보를 바탕으로 사용자 질문에 대한 NEO query를 생성해주세요.
NEO query 형식: predicate(변수1, 변수2, ...)
변수는 X1, X2, X3... 형태로 사용하세요.

예시:
- "이순신은 누구야?" → person(X1, X2, X3, X4), X1 = "이순신"
- "임진왜란은 언제 일어났어?" → event(X1, X2, X3), X1 = "임진왜란"
- "세종대왕이 한 일은?" → person(X1, X2, X3, X4), X1 = "세종"
"""
        
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
            
            neo_query = response.choices[0].message.content.strip()
            
            # LLM 응답에서 코드 블록 제거
            if neo_query.startswith("```") and neo_query.endswith("```"):
                neo_query = "\n".join(neo_query.split('\n')[1:-1]).strip()
            
            logger.info(f"✅ NEO query 생성 완료: {neo_query}")
            
            return {
                "success": True,
                "query": neo_query,
                "relevant_facts": relevant_facts[:5],
                "predicates_used": self._extract_predicates_from_query(neo_query)
            }
            
        except Exception as e:
            logger.error(f"NEO query 변환 실패: {str(e)}")
            return {
                "success": False,
                "error": f"LLM 변환 실패: {str(e)}",
                "query": None
            }

    def _extract_predicates_from_query(self, query: str) -> List[str]:
        """query에서 사용된 predicate들을 추출"""
        predicates = []
        matches = re.findall(r'(\w+)\s*\(', query)
        return list(set(matches))

    def convert_neo_result_to_nl(self, query_result: Dict[str, Any]) -> str:
        """NEO 쿼리 결과를 자연어로 변환"""
        try:
            if not query_result.get("success"):
                return "관련 정보를 찾을 수 없습니다."
            
            results = query_result.get("results", [])
            if not results:
                return "관련 정보를 찾을 수 없습니다."
            
            # 결과를 자연어로 변환
            nl_responses = []
            for result in results:
                predicate = result.get("predicate", "")
                fact_string = result.get("fact_string", "")
                
                if predicate == "person":
                    # person(Name, Birth, Death, Role) 형식 처리
                    if "이순신" in fact_string:
                        nl_responses.append("이순신(1545-1598)은 조선의 수군 장군으로, 임진왜란 당시 왜군을 물리친 명장입니다.")
                    elif "세종대왕" in fact_string:
                        nl_responses.append("세종대왕(1397-1450)은 조선의 4대 왕으로, 훈민정음(한글) 창제와 같은 문화적 업적을 남겼습니다.")
                    elif "이성계" in fact_string:
                        nl_responses.append("이성계(1335-1408)는 조선의 건국자로, 1392년 조선을 건국했습니다.")
                    else:
                        nl_responses.append(f"{fact_string}에 대한 정보입니다.")
                elif predicate == "event":
                    # event(Name, Year, Desc) 형식 처리
                    if "조선 건국" in fact_string:
                        nl_responses.append("조선은 1392년 이성계에 의해 건국되었습니다.")
                    elif "훈민정음" in fact_string:
                        nl_responses.append("훈민정음은 1446년 세종대왕에 의해 반포되어 백성들의 글자로 채택되었습니다.")
                    else:
                        nl_responses.append(f"{fact_string}에 대한 정보입니다.")
                else:
                    nl_responses.append(f"{fact_string}에 대한 정보입니다.")
            
            return " ".join(nl_responses) if nl_responses else "관련 정보를 찾을 수 없습니다."
            
        except Exception as e:
            logger.error(f"NEO 결과를 자연어로 변환 실패: {str(e)}")
            return "정보 변환 중 오류가 발생했습니다."

    def get_kb_stats(self) -> Dict[str, Any]:
        """KB 통계 정보 반환"""
        return {
            "total_facts": len(self.kb_facts),
            "total_rules": len(self.kb_rules),
            "predicates": list(self.predicates),
            "sample_facts": self.kb_facts[:3] if self.kb_facts else []
        }

def main():
    """테스트 함수"""
    if len(os.sys.argv) != 2:
        print("사용법: python nl_to_neo_query.py <nkb파일>")
        print("예시: python nl_to_neo_query.py sample_history.nkb")
        os.sys.exit(1)
    
    nkb_file = os.sys.argv[1]
    
    # RAG 시스템 초기화
    rag = NEOQueryRAG()
    
    # NEO KB 파일 로드
    if not rag.load_nkb_file(nkb_file):
        print("❌ NEO KB 파일 로드 실패")
        os.sys.exit(1)
    
    # KB 통계 출력
    stats = rag.get_kb_stats()
    print(f"📊 KB 통계: {stats['total_facts']}개 사실, {stats['total_rules']}개 규칙")
    print(f"📊 사용 가능한 predicate: {', '.join(stats['predicates'])}")
    
    # 대화형 테스트
    print("\n🤖 자연어 질문을 입력하세요 (종료: 'quit'):")
    
    while True:
        try:
            question = input("\n💬 질문: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료']:
                print("👋 안녕히 가세요!")
                break
                
            if not question:
                print("❌ 질문을 입력해주세요.")
                continue
            
            # NEO query 변환
            result = rag.convert_to_neo_query(question)
            
            if result["success"]:
                print(f"🔍 NEO Query: {result['query']}")
                print(f"📊 사용된 predicate: {', '.join(result['predicates_used'])}")
            else:
                print(f"❌ 오류: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {str(e)}")

if __name__ == "__main__":
    main()
