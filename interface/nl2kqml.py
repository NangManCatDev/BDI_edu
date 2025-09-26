class NL2KQML:
    def __init__(self, llm):
        self.llm = llm

    def translate(self, nl_sentence, context):
        return f"(ask :content {nl_sentence})"
