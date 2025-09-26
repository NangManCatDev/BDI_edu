from pyswip import Prolog
from pathlib import Path


class PrologEngine:
    def __init__(self, kb_path: str = None):
        self.prolog = Prolog()
        self.kb_loaded = False
        if kb_path:
            self.load_kb(kb_path)

    def load_kb(self, kb_path: str):
        path = Path(kb_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {kb_path}")

        # 항상 '/' 경로로 변환
        kb_path = path.as_posix()

        self.prolog.consult(kb_path)
        self.kb_loaded = True
        return True

    def query(self, query_str: str, max_results: int = 5):
        if not self.kb_loaded:
            raise RuntimeError("Knowledge base not loaded. Call load_kb() first.")
        return list(self.prolog.query(query_str, maxresults=max_results))
