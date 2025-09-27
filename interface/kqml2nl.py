# interface/kqml2nl.py
from interface.llm_connector import LLMConnector

connector = LLMConnector()

def kqml_to_nl(atom):
    """
    KB 응답(Atom)을 LLM-friendly 문자열로 변환 후 자연어 문장으로 생성
    """
    if hasattr(atom, "predicate") and hasattr(atom, "arguments"):
        pred = atom.predicate.name
        args = [str(a).strip("'") for a in atom.arguments]
        query_str = f"{pred}({', '.join(args)})"
    else:
        query_str = str(atom)

    prompt = f"다음 KQML 스타일의 답변을 한국어 자연어 문장으로 변환해줘:\n{query_str}"
    return connector.ask(prompt)
