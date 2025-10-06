# interface/llm_connector.py
import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# .env 파일 로드
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)

class LLMConnector:
    def __init__(self, model: str = "gpt-4o-mini", max_retries: int = 3, timeout: int = 30):
        """
        LLM 연결자 초기화
        
        Args:
            model: 사용할 모델명
            max_retries: 최대 재시도 횟수
            timeout: 타임아웃 (초)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        
        logger.info(f"LLMConnector 초기화 완료 - 모델: {model}")

    def ask(self, prompt: str, system: Optional[str] = None) -> str:
        """
        LLM에 프롬프트를 전달하고 응답을 문자열로 반환
        
        Args:
            prompt: 사용자 프롬프트
            system: 시스템 프롬프트 (선택사항)
            
        Returns:
            LLM 응답 문자열
            
        Raises:
            OpenAIError: API 호출 실패 시
            TimeoutError: 타임아웃 발생 시
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                logger.info(f"LLM API 호출 시도 {attempt + 1}/{self.max_retries}")
                logger.debug(f"프롬프트: {prompt[:100]}...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    timeout=self.timeout
                )
                
                result = response.choices[0].message.content.strip()
                logger.info(f"LLM 응답 성공 (길이: {len(result)})")
                return result
                
            except OpenAIError as e:
                logger.error(f"OpenAI API 오류 (시도 {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 지수 백오프
                
            except Exception as e:
                logger.error(f"예상치 못한 오류 (시도 {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        raise Exception("최대 재시도 횟수 초과")

    def ask_with_fallback(self, prompt: str, system: Optional[str] = None, fallback_response: str = "죄송합니다. 답변을 생성할 수 없습니다.") -> str:
        """
        LLM 호출 실패 시 대체 응답을 반환하는 안전한 메서드
        
        Args:
            prompt: 사용자 프롬프트
            system: 시스템 프롬프트
            fallback_response: 실패 시 반환할 대체 응답
            
        Returns:
            LLM 응답 또는 대체 응답
        """
        try:
            return self.ask(prompt, system)
        except Exception as e:
            logger.error(f"LLM 호출 실패, 대체 응답 사용: {str(e)}")
            return fallback_response
