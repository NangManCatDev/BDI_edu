class KQML2NL:
    def __init__(self, llm):
        self.llm = llm

    def translate(self, kqml_message, context):
        return f"자연어 변환된 메시지: {kqml_message}"
