import typing
from autodiff.computational_graph import Node
import math

class EvalOverloader:
    """
    This class is used to manage operator overloading for Ops from a single class. 
    """
    def __add__(self, other):
        match other:
            case int(val) | float(val):
                return Add(self, Const(val))
            case _:
                return Add(self, other)

    def __radd__(self, other):
        match other:
            case int(val) | float(val):
                return Add(Const(val), self)
            case _:
                return Add(other, self)

    def __sub__(self, other):
        match other:
            case int(val) | float(val):
                return Sub(self, Const(val))
            case _:
                return Sub(self, other)

    def __rsub__(self, other):
        match other:
            case int(val) | float(val):
                return Sub(Const(val), self )
            case _:
                return Sub(self, other)

    def __mul__(self, other):
        match other:
            case int(val) | float(val):
                return Mul(self, Const(val))
            case _:
                return Mul(self, other)

    def __rmul__(self, other):
        match other:
            case int(val) | float(val):
                return Mul(Const(val), self)
            case _:
                return Mul(other, self)

    def __truediv__(self, other):
        """
        standard division
        """
        match other:
            case int(val) | float(val):
                return Div(self, Const(val))
            case _:
                return Div(self, other)

    def __rtruediv__(self, other):
        match other:
            case int(val) | float(val):
                return Div(Const(val), self)
            case _:
                return Div(other, self)

    def __pow__(self, other):
        match other:
            case int(val) | float(val):
                return Pow(self, Const(val))
            case _:
                return Pow(self, other)

    def __rpow__(self, other):
        match other:
            case int(val) | float(val):
                return Pow(Const(val), self)
            case _:
                return Pow(other, self)

    def __neg__(self):
        return Neg(self)


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"


class Op(Node, EvalOverloader):
    """
    Base class for operator AST node
    """
    def __init__(self):
        super().__init__()

    def backward(self, adjoint: float = None):
        if adjoint is not None:
            self.adjoint = adjoint
        else:
            self.adjoint = 1.0

    """
        adjoint = chain rule에서 자기 왼쪽항에서 계산된 값 갖고오기. 
        when top node, adjacent is 1.0 because ex) partial z / partial z = 1
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class BinaryOp(Op, EvalOverloader):
    def __init__(self, left: Node, right: Node):
        super().__init__()
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_n{self.node_index}({self.left}, {self.right})"


class UnaryOp(Op, EvalOverloader):
    def __init__(self, operand: Node):
        super().__init__()
        self.operand = operand

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_n{self.node_index}({self.operand})"

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

    def forward_codegen(self, walk_function: callable):
        return f"{walk_function(self.left)} + {walk_function(self.right)}"

    def backward_forward_dependents(self):
        return []

    def backward_codegen(self, adjoint_var_name: str, adjoint_target_variables: set[str], callback: callable, code_callback: callable):
        # 1. emit adjoint_n = ... for myself
        if adjoint_var_name:
            code_callback(f"adjoint_n{self.node_index} = {adjoint_var_name}")
        else:
            code_callback(f"adjoint_n{self.node_index} = 1.0")

        # 2. make left and right codegen with 1 as adjoint
        callback(self.left, f"adjoint_n{self.node_index}", adjoint_target_variables, callback, code_callback)
        callback(self.right, f"adjoint_n{self.node_index}", adjoint_target_variables, callback, code_callback)

 
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

    def forward_codegen(self, walk_function: callable):
        return f"{walk_function(self.left)} - {walk_function(self.right)}"

    def backward_forward_dependents(self):
        return []

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

    def forward_codegen(self, walk_function: callable):
        return f"{walk_function(self.left)} * {walk_function(self.right)}"

    def backward_forward_dependents(self):
        return [self.right, self.left]

class Div(BinaryOp):
    def eval(self) -> float:
        self.value = self.left.eval() / self.right.eval()
        return self.value

    def backward(self, adjoint: float = None):
        """
        Let self = left / right
        dself/dleft = adjoint * right^(-1)
        dself/dright = adjoint * -left * right^(-2)
        """
        super().backward(adjoint)

        self.left.backward(self.adjoint * self.right.value ** -1)

        self.right.backward(self.adjoint * -self.left.value * self.right.value ** -2)

    def forward_codegen(self, walk_function: callable):
        return f"{walk_function(self.left)} / {walk_function(self.right)}"

    def backward_forward_dependents(self):
        return [self.right ** -1, -self.left * self.right ** -2]

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

    def forward_codegen(self, walk_function: callable):
        return f"{walk_function(self.left)} ** {walk_function(self.right)}"

    def backward_forward_dependents(self):
        return [self.right * self.left ** (self.right - 1), self.left ** self.right]

class Cos(UnaryOp):
    def eval(self) -> float:
        self.value = math.cos(self.operand.eval())
        return self.value

    def backward(self, adjoint: float = None):
        super().backward(adjoint)

        self.operand.backward(self.adjoint * -math.sin(self.operand.value))

    def forward_codegen(self, walk_function: callable):
        return f"math.cos({walk_function(self.operand)})"

    def backward_forward_dependents(self):
        return [-Sin(self.operand)]

class Sin(UnaryOp):
    def eval(self) -> float:
        self.value = math.sin(self.operand.eval())
        return self.value
    
    def backward(self, adjoint: float = None):
        super().backward(adjoint)

        self.operand.backward(self.adjoint * math.cos(self.operand.value))

    def forward_codegen(self, walk_function: callable):
        return f"math.sin({walk_function(self.operand)})"

    def backward_forward_dependents(self):
        return [Cos(self.operand)]

class Exp(UnaryOp):
    def eval(self) -> float:
        self.value = math.exp(self.operand.eval())
        return self.value

    def backward(self, adjoint: float = None):
        super().backward(adjoint)

        self.operand.backward(self.adjoint * math.exp(self.operand.value))

    def forward_codegen(self, walk_function: callable):
        return f"math.exp({walk_function(self.operand)})"

    def backward_forward_dependents(self):
        return [Exp(self.operand)]

class Neg(UnaryOp):
    def eval(self) -> float:
        self.value = -self.operand.eval()
        return self.value

    def backward(self,adjoint: float = None):
        super().backward(adjoint)

        self.operand.backward(-self.adjoint)

    def forward_codegen(self, walk_function: callable):
        return f"-{walk_function(self.operand)}"

    def backward_forward_dependents(self):
        return []


class sqrt(BinaryOp):
    def eval(self) -> float:
        self.value = math.sqrt(self.operand.eval())
        return self.value
    
    def backward(self, adjoint:float = None):
        super().backward(adjoint)

        self.operand.backward(self.adjoint * math.sqrt(self.operand.value()))

    def forward_codegen(self, walk_function: callable):
        return f"math.sqrt({walk_function(self.operand)})"        

    def backward_forward_dependents(self):
        return [sqrt(self.operand)]