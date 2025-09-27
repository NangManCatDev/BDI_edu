from pylo.language.lp import c_const
from .prolog_engine import PrologEngine

def build_kb():
    eng = PrologEngine()

    # === Predicates ===
    event = eng.pred("event", 3)    # event(Name, Year, Desc)
    person = eng.pred("person", 4)  # person(Name, Birth, Death, Role)
    cause = eng.pred("cause", 2)    # cause(Event, Result)

    # === Events ===
    eng.fact(event, [c_const("'삼국통일'"), c_const("'668'"), c_const("'신라가 삼국을 통일함'")])
    eng.fact(event, [c_const("'훈민정음반포'"), c_const("'1446'"), c_const("'세종이 훈민정음을 반포함'")])
    eng.fact(event, [c_const("'임진왜란'"), c_const("'1592'"), c_const("'왜군이 조선을 침략함'")])

    # === Persons ===
    eng.fact(person, [c_const("'세종'"), c_const("'1397'"), c_const("'1450'"), c_const("'조선의 4대 왕'")])
    eng.fact(person, [c_const("'이순신'"), c_const("'1545'"), c_const("'1598'"), c_const("'조선 수군 장군'")])

    # === Causes ===
    eng.fact(cause, [c_const("'동학농민운동'"), c_const("'갑오개혁'")])

    return eng
