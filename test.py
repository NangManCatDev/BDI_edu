from pylo.language.lp import c_pred, c_var, c_const, Atom

father = c_pred("father", 2)
X, Y = c_var("X"), c_var("Y")
fact = Atom(father, [c_const("anakin"), c_const("luke")])

print(fact)   # father(anakin,luke)
