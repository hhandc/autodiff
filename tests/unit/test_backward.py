from autodiff.variable import Variable
from autodiff.ops import *
from pytest import approx
import random
import math

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
    expr.eval()
    expr.backward()

    grad_x = -2 * a + 2 * x.value -4 * b * x.value + 4 * b * x.value ** 3
    grad_y = 2 * b * y.value - 2 * b * x.value ** 2

    assert x.adjoint == grad_x and y.adjoint == grad_y

def test_backward_matyas():
    x = Variable("x", 0)
    y = Variable("y", 0)

    expr = 0.26 * (x**2 + y**2) - 0.48 * x * y

    expr.eval()
    expr.backward()

    assert expr.value == 0 and x.adjoint == 0 and y.adjoint == 0

def test_backward_schaffer2():
    x = Variable("x", 0)
    y = Variable("y", 0)

    expr = 0.5 + (Sin(x ** 2 - y ** 2) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2

    expr.eval()
    expr.backward()

    assert expr.value == approx(0) and x.adjoint == approx(0) and y.adjoint == approx(0)

def test_backward_easom():
    x = Variable("x", math.pi)
    y = Variable("y", math.pi)

    expr = -Cos(x) * Cos(y) * Exp(-((x - math.pi) ** 2 + (y - math.pi) ** 2))

    expr.eval()
    expr.backward()

    assert expr.value == approx(-1) and x.adjoint == approx(0) and y.adjoint == approx(0)
