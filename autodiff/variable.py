import typing
from autodiff.ast import Node
import autodiff.ops as ops

class Variable(Node, ops.EvalOverloader):
    """
    Base class for variables
    """

    def __init__(self, variable_name : str, value : float = float("nan")):
        self.variable_name = variable_name
        self.value = value
        
    def eval(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"
