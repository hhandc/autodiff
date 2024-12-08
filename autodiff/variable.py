import typing
from autodiff.computational_graph import Node
import autodiff.ops as ops
from typing import Union
from numbers import Number


class Variable(Node, ops.EvalOverloader):
    """
    Base class for variables
    """

    def __init__(self, name : str, value : Union[float, Node] = float("nan")):
        super().__init__()
        self.name = name
        self.value = value
        
    def eval(self) -> float:
        if isinstance(self.value, Number):
            return self.value
        else:
            return self.value.eval()

    def backward(self, adjoint: float = None):
        """
        Since variables accumulate adjoints according to the multivariate chain rule, we add up all the adjoints.
        """
        if self.adjoint == None:
            self.adjoint = adjoint
        else:
            self.adjoint += adjoint

        """
        If the variable is assigned to an expression, we propagate the backward pass and the current adjoint to its assigned expression
        """
        if not isinstance(self.value, Number):
            self.value.backward(adjoint)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.value}, {id(self)})"