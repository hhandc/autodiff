from autodiff.variable import Variable
from autodiff.ops import *
from autodiff.codegen import value_and_grad

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

    def rosenbrock(x, y):
        a = 3
        b = 2
        return (a - x) ** 2 + b * (y - x ** 2) ** 2

    val, dx, dy = value_and_grad(rosenbrock)(3, 9)
    assert val == 0 and dx == 0 and dy == 0