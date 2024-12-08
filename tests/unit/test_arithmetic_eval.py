from autodiff.variable import Variable
from autodiff.ops import *
from pytest import approx
import random

def test_add():
    a_true = 1.442
    a = Variable("a", a_true)

    b_true = 3.141
    b = Variable("b", b_true)
    
    add_expr = a + b

    assert add_expr.eval() == approx(a_true + b_true)

def test_sub():
    a_true = 4.6123
    a = Variable("a", a_true)

    b_true = 3.141
    b = Variable("b", b_true)
    
    sub_expr = a - b

    assert sub_expr.eval() == approx(a_true - b_true)

def test_mul():
    a_true = random.random()
    a = Variable("a", a_true)

    b_true = random.random()
    b = Variable("b", b_true)

    expr = a * b

    assert expr.eval() == approx(a_true * b_true)
