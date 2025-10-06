#!/usr/bin/env python3
"""
독립적인 텍스트 → NEO KB 변환 모듈
사용법: python txt_to_kb.py input.txt output.nkb
"""

import os
import sys
import logging
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextToKBConverter:
    """자연어 텍스트를 NEO KB 형식으로 변환하는 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """OpenAI API 클라이언트 초기화"""
        # .env 파일에서 API 키 로드
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        self.client = OpenAI(api_key=api_key)
        self.chunk_size = 1000  # LLM 입력 제한 고려
        self.overlap = 200
        
        # 시스템 프롬프트 로드
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """시스템 프롬프트 파일을 로드합니다"""
        try:
            # 상위 디렉토리의 system_prompt 폴더에서 로드
            prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'system_prompt', 'txt_to_kb_prompt.txt')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning("시스템 프롬프트 파일을 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
            return """당신은 한국사 전문가입니다. 주어진 텍스트에서 사실을 추출하여 NEO 지식베이스 형식으로 변환해주세요."""
        except Exception as e:
            logger.error(f"시스템 프롬프트 로드 실패: {str(e)}")
            return """당신은 한국사 전문가입니다. 주어진 텍스트에서 사실을 추출하여 NEO 지식베이스 형식으로 변환해주세요."""

    def _chunk_text(self, text: str) -> list[str]:
        """텍스트를 LLM 처리를 위해 청크로 나눕니다"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # 단어 경계 찾기
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                chunks.append(text[start:last_space])
                start = last_space + 1
            else:  # 공백이 없으면 강제로 자름
                chunks.append(text[start:end])
                start = end
            
            # 오버랩 적용
            if start < len(text):
                start -= self.overlap
                if start < 0: 
                    start = 0
                    
        return chunks

    def convert_text_to_kb(self, text: str) -> Optional[str]:
        """자연어 텍스트를 NEO KB 형식으로 변환"""
        logger.info("자연어 텍스트를 NEO KB 형식으로 변환 시작...")
        
        chunks = self._chunk_text(text)
        all_kb_content = []

        for i, chunk in enumerate(chunks):
            logger.info(f"청크 {i+1}/{len(chunks)} 처리 중...")
            
            user_prompt = f"""다음 텍스트에서 한국사 관련 사실들을 추출하여 NEO 지식베이스 형식으로 변환해주세요.

텍스트:
{chunk}

위 텍스트에서 추출할 수 있는 사실들을 NEO 지식베이스 형식으로 변환해주세요.
각 사실을 한 줄씩 작성하고, 정확한 정보만 포함해주세요."""
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.1
                )
                
                content = response.choices[0].message.content.strip()
                
                # LLM이 코드 블록으로 응답하는 경우 제거
                if content.startswith("```") and content.endswith("```"):
                    content = "\n".join(content.split('\n')[1:-1]).strip()
                    logger.warning("LLM 응답에서 코드 블록 마크다운 제거")
                
                if content:
                    all_kb_content.append(content)
                    logger.info(f"청크 {i+1} 변환 완료")
                else:
                    logger.warning(f"청크 {i+1} 빈 응답")
                    
            except Exception as e:
                logger.error(f"청크 {i+1} 변환 실패: {str(e)}")
                return None

        final_kb_content = "\n".join(all_kb_content).strip()
        logger.info("자연어 텍스트를 NEO KB 형식으로 변환 완료")
        return final_kb_content

    def convert_file(self, input_file: str, output_file: str) -> bool:
        """텍스트 파일을 읽어 NEO KB 파일(.nkb)로 변환하여 저장"""
        logger.info(f"파일 변환 시작: {input_file} -> {output_file}")
        
        try:
            # 입력 파일 읽기
            with open(input_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            if not text_content.strip():
                logger.error("입력 파일이 비어있습니다")
                return False
            
            # 텍스트를 KB 형식으로 변환
            kb_content = self.convert_text_to_kb(text_content)
            
            if not kb_content:
                logger.error("LLM 변환 실패")
                return False
            
            # 출력 파일에 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(kb_content)
            
            logger.info(f"✅ 파일 변환 성공: {output_file}")
            return True
            
        except FileNotFoundError:
            logger.error(f"입력 파일을 찾을 수 없습니다: {input_file}")
            return False
        except Exception as e:
            logger.error(f"파일 변환 중 오류 발생: {str(e)}")
            return False

def main():
    """메인 함수"""
    if len(sys.argv) != 3:
        print("사용법: python txt_to_kb.py <입력파일.txt> <출력파일.nkb>")
        print("예시: python txt_to_kb.py sample.txt history.nkb")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # API 키 확인 (이미 .env에서 로드됨)
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        print(".env 파일에 OPENAI_API_KEY='your-api-key' 추가해주세요.")
        sys.exit(1)
    
    # 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        sys.exit(1)
    
    # 변환 실행
    converter = TextToKBConverter()
    success = converter.convert_file(input_file, output_file)
    
    if success:
        print(f"✅ 변환 완료: {input_file} -> {output_file}")
    else:
        print("❌ 변환 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()
