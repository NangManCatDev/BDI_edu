# belief/text_to_kb_converter.py
"""
자연어 텍스트를 NEO 지식베이스로 변환하는 모듈
"""

import os
import logging
from typing import List, Dict, Any, Optional
from interface.llm_connector import LLMConnector

logger = logging.getLogger(__name__)

class TextToKBConverter:
    """
    자연어 텍스트를 NEO 지식베이스로 변환하는 클래스
    """
    
    def __init__(self):
        self.llm_connector = LLMConnector()
        self.conversion_system_prompt = """당신은 자연어 텍스트를 NEO 지식베이스 형식으로 변환하는 전문가입니다.

변환 규칙:
1. person(Name, Birth, Death, Role) - 인물 정보
2. event(Name, Year, Description) - 사건 정보  
3. cause(Event, Result) - 인과관계
4. location(Place, Region, Type) - 장소 정보
5. concept(Term, Definition, Category) - 개념 정보

출력 형식:
- 각 사실을 한 줄씩 작성
- 변수는 구체적인 값으로 대체
- 따옴표로 문자열 감싸기
- 예: person('세종', '1397', '1450', '조선의 4대 왕')
- 예: event('훈민정음반포', '1446', '세종이 훈민정음을 반포함')

중요:
- 정확한 연도와 정보만 추출
- 추측하지 말고 명확한 정보만 변환
- 한국사 관련 정보에 집중
- 각 사실을 독립적으로 작성"""
    
    def convert_text_file(self, file_path: str, output_path: str = None) -> bool:
        """
        텍스트 파일을 NEO 지식베이스로 변환
        
        Args:
            file_path: 입력 텍스트 파일 경로
            output_path: 출력 KB 파일 경로 (기본값: 입력파일명.kb)
            
        Returns:
            변환 성공 여부
        """
        try:
            logger.info(f"텍스트 파일 변환 시작: {file_path}")
            
            # 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            logger.info(f"텍스트 파일 크기: {len(text_content)} 문자")
            
            # LLM을 통한 변환
            kb_content = self._convert_with_llm(text_content)
            
            if not kb_content:
                logger.error("LLM 변환 실패")
                return False
            
            # 출력 파일 경로 설정
            if output_path is None:
                base_name = os.path.splitext(file_path)[0]
                output_path = f"{base_name}.kb"
            
            # KB 파일 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(kb_content)
            
            logger.info(f"KB 파일 저장 완료: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"텍스트 파일 변환 실패: {str(e)}")
            return False
    
    def _convert_with_llm(self, text: str) -> Optional[str]:
        """
        LLM을 사용하여 텍스트를 KB 형식으로 변환
        
        Args:
            text: 변환할 텍스트
            
        Returns:
            변환된 KB 내용 또는 None
        """
        try:
            # 텍스트가 너무 길면 청크로 나누기
            max_chunk_size = 3000
            chunks = self._split_text(text, max_chunk_size)
            
            all_kb_content = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"청크 {i+1}/{len(chunks)} 처리 중...")
                
                # LLM 변환 요청
                user_prompt = f"""다음 텍스트를 NEO 지식베이스 형식으로 변환해주세요:

{chunk}

위 텍스트에서 추출할 수 있는 사실들을 NEO 지식베이스 형식으로 변환해주세요.
각 사실을 한 줄씩 작성하고, 정확한 정보만 포함해주세요."""
                
                response = self.llm_connector.ask_with_fallback(
                    prompt=user_prompt,
                    system=self.conversion_system_prompt
                )
                
                if response:
                    all_kb_content.append(response)
                    logger.info(f"청크 {i+1} 변환 완료")
                else:
                    logger.warning(f"청크 {i+1} 변환 실패")
            
            # 모든 청크 결과 합치기
            if all_kb_content:
                return "\n".join(all_kb_content)
            else:
                return None
                
        except Exception as e:
            logger.error(f"LLM 변환 실패: {str(e)}")
            return None
    
    def _split_text(self, text: str, max_size: int) -> List[str]:
        """
        텍스트를 지정된 크기로 나누기
        
        Args:
            text: 나눌 텍스트
            max_size: 최대 청크 크기
            
        Returns:
            나뉜 텍스트 청크 리스트
        """
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_size
            
            # 문장 경계에서 자르기
            if end < len(text):
                # 마지막 문장 끝 찾기
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                
                if last_period > start:
                    end = last_period + 1
                elif last_newline > start:
                    end = last_newline + 1
            
            chunks.append(text[start:end])
            start = end
        
        return chunks
    
    def convert_directory(self, input_dir: str, output_dir: str = None) -> Dict[str, bool]:
        """
        디렉토리 내 모든 텍스트 파일을 변환
        
        Args:
            input_dir: 입력 디렉토리 경로
            output_dir: 출력 디렉토리 경로 (기본값: 입력디렉토리)
            
        Returns:
            파일별 변환 성공 여부 딕셔너리
        """
        results = {}
        
        if output_dir is None:
            output_dir = input_dir
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 텍스트 파일 찾기
        text_files = []
        for file in os.listdir(input_dir):
            if file.endswith(('.txt', '.md')):
                text_files.append(file)
        
        logger.info(f"변환할 텍스트 파일 {len(text_files)}개 발견")
        
        # 각 파일 변환
        for file in text_files:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.kb')
            
            logger.info(f"변환 중: {file}")
            success = self.convert_text_file(input_path, output_path)
            results[file] = success
            
            if success:
                logger.info(f"✅ {file} 변환 성공")
            else:
                logger.error(f"❌ {file} 변환 실패")
        
        return results
    
    def validate_kb_file(self, kb_file_path: str) -> bool:
        """
        KB 파일의 형식 검증
        
        Args:
            kb_file_path: 검증할 KB 파일 경로
            
        Returns:
            형식이 올바른지 여부
        """
        try:
            with open(kb_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            valid_predicates = ['person', 'event', 'cause', 'location', 'concept']
            valid_lines = 0
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                
                # 기본 형식 검증
                if '(' in line and ')' in line:
                    predicate = line.split('(')[0].strip()
                    if predicate in valid_predicates:
                        valid_lines += 1
                    else:
                        logger.warning(f"알 수 없는 predicate: {predicate}")
                else:
                    logger.warning(f"잘못된 형식: {line}")
            
            logger.info(f"KB 파일 검증 완료: {valid_lines}개 유효한 사실")
            return valid_lines > 0
            
        except Exception as e:
            logger.error(f"KB 파일 검증 실패: {str(e)}")
            return False

# 사용 예시
if __name__ == "__main__":
    converter = TextToKBConverter()
    
    # 단일 파일 변환
    success = converter.convert_text_file("sample_history.txt")
    if success:
        print("✅ 텍스트 파일 변환 성공")
    else:
        print("❌ 텍스트 파일 변환 실패")
    
    # 디렉토리 변환
    results = converter.convert_directory("text_files/", "kb_files/")
    print(f"변환 결과: {results}")
