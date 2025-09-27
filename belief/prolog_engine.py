from pylo.language import lp


class PrologEngine:
    def __init__(self):
        self.predicates = {}
        self.facts = []   # Atom 리스트
        self.rules = []   # Clause 리스트

    # === Predicate 정의 ===
    def pred(self, name, arity):
        p = lp.Predicate(name, arity)
        self.predicates[name] = p
        return p

    # === Fact 추가 ===
    def fact(self, predicate_or_atom, args=None):
        """
        Fact 추가 방법:
          1) Atom 직접 전달:
             eng.fact(lp.Atom(p, [c_const("one"), c_const("a")]))
          2) Predicate + args 전달:
             eng.fact(p, [c_const("one"), c_const("a")])
        """
        if args is None:
            # Atom을 직접 전달받은 경우
            self.facts.append(predicate_or_atom)
        else:
            # Predicate + args 전달받은 경우
            atom = lp.Atom(predicate_or_atom, args)
            self.facts.append(atom)

    # === Rule 추가 ===
    def rule(self, head_or_clause, body_atoms=None):
        """
        Rule 추가 방법:
          1) Clause 직접 전달:
             eng.rule(lp.Clause(head, [body]))
          2) head + body_atoms 전달:
             eng.rule(head, [body1, body2])
        """
        if body_atoms is None:
            # Clause 객체 직접 추가
            self.rules.append(head_or_clause)
        else:
            clause = lp.Clause(head_or_clause, body_atoms)
            self.rules.append(clause)

    # === Query 실행 (간단한 매칭 엔진) ===
    def query(self, atom):
        results = []

        # --- Fact 검사 ---
        for f in self.facts:
            if f.predicate == atom.predicate:
                subst = {}
                matched = True
                for fa, qa in zip(f.arguments, atom.arguments):
                    if isinstance(qa, lp.Variable):
                        subst[qa.name] = fa
                    elif fa != qa:
                        matched = False
                        break
                if matched:
                    results.append(subst)

        # --- Rule 검사 ---
        for r in self.rules:
            head = r._head  # Clause 내부 head
            if head.predicate == atom.predicate:
                subst = {}
                matched = True
                for ha, qa in zip(head.arguments, atom.arguments):
                    if isinstance(qa, lp.Variable):
                        subst[qa.name] = ha
                    elif ha != qa:
                        matched = False
                        break
                if matched:
                    results.append(subst)

        # 중복 제거
        unique_results = []
        for r in results:
            if r not in unique_results:
                unique_results.append(r)

        return unique_results
