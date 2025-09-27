# interface/nl2kqml.py
from pylo.language.lp import Variable, Atom, c_const
from belief.knowledge_base import build_kb

def nl_to_kqml(question: str):
    """
    NL 질문을 KB 기반 Atom 쿼리로 일반화 변환
    knowledge_base.py만 수정하면 자동 반영됨
    """
    eng = build_kb()
    q = question.strip()

    for fact in eng.facts:
        args = [str(arg).strip("'") for arg in fact.arguments]
        for arg in args:
            if arg in q:
                pred = fact.predicate
                new_args = []
                for i, a in enumerate(args):
                    if a in q:
                        new_args.append(c_const(f"'{a}'"))
                    else:
                        new_args.append(Variable(f"X{i}"))
                return Atom(pred, new_args)

    return None
