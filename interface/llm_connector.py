# interface/llm_connector.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일 로드
load_dotenv()

class LLMConnector:
    def __init__(self, model="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def ask(self, prompt: str, system: str = None) -> str:
        """
        LLM에 프롬프트를 전달하고 응답을 문자열로 반환
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
