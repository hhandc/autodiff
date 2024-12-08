from autodiff.variable import Variable
from autodiff.ops import *
from pytest import approx
import random

def test_random_exprs():
    ops = [Add, Sub, Mul, Div]
    n_vars = random.randrange(5, 20)
    vars = [Variable(f"var_{index}", value) for index, value in enumerate([random.random() for _ in range(n_vars)])]
    expr = ""
    true_expr = ""
    for index, var in enumerate(vars):
        expr += f"vars[{index}]"
        true_expr += str(var.value)
        if index != n_vars - 1:
            op = random.choice(("+", "-", "*", "/"))
            expr += op
            true_expr += op

    ast_val = eval(expr, locals()).eval()
    true_val = eval(true_expr, locals())

    assert ast_val == approx(true_val)