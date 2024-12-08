import typing
from autodiff.computational_graph import Node
import autodiff.ops as ops

class Variable(Node, ops.EvalOverloader):
    """
    Base class for variables
    """

    def __init__(self, variable_name : str, value : float = float("nan")):
        super().__init__()
        self.variable_name = variable_name
        self.value = value
        
    def eval(self) -> float:
        return self.value

    def backward(self, adjoint: float = None):
        """
        Since variables accumulate adjoints according to the multivariate chain rule, we add up all the adjoints.
        """
        if self.adjoint == None:
            self.adjoint = adjoint
        else:
            self.adjoint += adjoint

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.variable_name}, {self.value})"