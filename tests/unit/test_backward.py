from autodiff.variable import Variable
from autodiff.ops import *
from pytest import approx
import random

def test_backward_rosenbrock():
    """
    The rosenbrock function is:
    f(x, y) = (a - x) ^ 2 + b * (y - x^2)^2
    for some a and b
    Which has f(a, a^2) = 0

    Its partial derivatives are,
    dx = -2a + 2x -4bx +4bx^3
    dy = 2by - 2bx^2
    """
    a = 1
    b = 1

    x = Variable("x", 1)
    y = Variable("y", 1)

    expr = (a - x) ** 2 + b * (y - x ** 2) ** 2
    print(expr)
    expr.eval()
    expr.backward()

    grad_x = -2 * a + 2 * x.value -4 * b * x.value + 4 * b * x.value ** 3
    grad_y = 2 * b * y.value - 2 * b * x.value ** 2

    assert x.adjoint == grad_x and y.adjoint == grad_y