import typing
from autodiff.ast import Node


class EvalOverloader:
    """
    This class is used to manage operator overloading for Ops from a single class.
    """
    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __truediv__(self, other):
        """
        standard division
        """
        return Div(self, other)

    def __neg__(self):
        raise NotImplementedError()


class Op(Node, EvalOverloader):
    """
    Base class for operator AST node
    """
    def __init__(self):
        super().__init__()


    def eval(self) -> float:
        """
        Evaluate the value of the operator
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class BinaryOp(Op):
    def __init__(self, left: Node, right: Node):
        super().__init__()
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.left}, {self.right})"


class Add(BinaryOp):
    def eval(self) -> float:
        return self.left.eval() + self.right.eval()


class Sub(BinaryOp):
    def eval(self) -> float:
        return self.left.eval() - self.right.eval()


class Mul(BinaryOp):
    def eval(self) -> float:
        return self.left.eval() * self.right.eval()


class Div(BinaryOp):
    def eval(self) -> float:
        return self.left.eval() / self.right.eval()