import typing
from autodiff.ast import Node
import math

class EvalOverloader:
    """
    This class is used to manage operator overloading for Ops from a single class. 
    """
    def __add__(self, other):
        match other:
            case int(val):
                return Add(self, Const(other))
            case _:
                return Add(self, other)

    def __radd__(self, other):
        match other:
            case int(val):
                return Add(Const(other), self)
            case _:
                return Add(other, self)

    def __sub__(self, other):
        match other:
            case int(val):
                return Sub(self, Const(other))
            case _:
                return Sub(self, other)

    def __rsub__(self, other):
        match other:
            case int(val):
                return Sub(Const(other), self )
            case _:
                return Sub(self, other)

    def __mul__(self, other):
        match other:
            case int(val):
                return Mul(self, Const(other))
            case _:
                return Mul(self, other)

    def __rmul__(self, other):
        match other:
            case int(val):
                return Mul(Const(other), self)
            case _:
                return Mul(other, self)

    def __truediv__(self, other):
        """
        standard division
        """
        match other:
            case int(val):
                return Div(self, Const(other))
            case _:
                return Div(self, other)

    def __rtruediv__(self, other):
        match other:
            case int(val):
                return Div(Const(other), self)
            case _:
                return Div(other, self)

    def __pow__(self, other):
        match other:
            case int(val):
                return Pow(self, Const(other))
            case _:
                return Pow(self, other)

    def __rpow__(self, other):
        match other:
            case int(val):
                return Pow(Const(other), self)
            case _:
                return Pow(other, self)

    def __neg__(self):
        raise NotImplementedError()


class Const(Node, EvalOverloader):
    """
    Holds a constant value
    """
    def __init__(self, value : float = float("nan")):
        self.value = value
        self.adjoint = 0

    def eval(self) -> float:
        return self.value
    
    def backward(self, adjoint: float = None):
        return


class Op(Node, EvalOverloader):
    """
    Base class for operator AST node
    """
    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class BinaryOp(Op, EvalOverloader):
    def __init__(self, left: Node, right: Node):
        super().__init__()
        self.left = left
        self.right = right

    def backward(self, adjoint: float = None):
        if adjoint is not None:
            self.adjoint = adjoint
        else:
            self.adjoint = 1.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.left}, {self.right})"


class Add(BinaryOp):
    def eval(self) -> float:
        self.value = self.left.eval() + self.right.eval()
        return self.value

    def backward(self, adjoint: float = None):
        """
        Let self = left + right
        1. Compute n = adjoint * dself/dleft and call left.backward(n)
        2. Compute n = adjoint * dself/dright and call right.backward(n)
        """
        super().backward(adjoint)

        # da/dleft = 1. We need to invoke left.backward()
        self.left.backward(self.adjoint)

        # da/dright = 1. We need to invoke right.backward()
        self.right.backward(self.adjoint)


class Sub(BinaryOp):
    def eval(self) -> float:
        self.value = self.left.eval() - self.right.eval()
        return self.value

    def backward(self, adjoint: float = None):
        """
        Let self = left - right
        1. Compute n = adjoint * dself/dleft and call left.backward(n)
        2. Compute n = adjoint * dself/dright and call right.backward(n)
        """
        super().backward(adjoint)

        # da/dleft = 1. We need to invoke left.backward()
        self.left.backward(self.adjoint)

        # da/dright = -1. We need to invoke right.backward()
        self.right.backward(-self.adjoint)


class Mul(BinaryOp):
    def eval(self) -> float:
        self.value = self.left.eval() * self.right.eval()
        return self.value

    def backward(self, adjoint: float = None):
        """
        Let self = left * right
        1. Compute n = adjoint * dself/dleft and call left.backward(n)
        2. Compute n = adjoint * dself/dright and call right.backward(n)
        dself/dleft = right.value
        dself/dright = left.value
        """
        super().backward(adjoint)

        # dself/dleft = right.value
        self.left.backward(self.adjoint * self.right.value)

        # dself/dright = left.value
        self.right.backward(self.adjoint * self.left.value)

class Div(BinaryOp):
    def eval(self) -> float:
        self.value = self.left.eval() / self.right.eval()
        return self.value


class Pow(BinaryOp):
    def eval(self) -> float:
        self.value = self.left.eval() ** self.right.eval()
        return self.value

    def backward(self, adjoint: float = None):
        """
        Let self = left ** right
        1. Compute n = adjoint * dself/dleft and call left.backward(n)
        2. Compute n = adjoint * dself/dright and call right.backward(n)
        outer = power, inner = left
        dself/dleft = adjoint * right * left  ** (right - 1)
        dself/dright = adjoint * left ** right ln left
        """

        super().backward(adjoint)
        # dself/dleft = right * left ** (right - 1)
        self.left.backward(self.adjoint * self.right.value * self.left.value ** (self.right.value - 1))

        # dself/dright = left ** right ln left
        self.right.backward(self.adjoint * self.left.value ** self.right.value * math.log(self.left.value) if self.left.value != 0 else 0)