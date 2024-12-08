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

    def backward_codegen(self, adjoint_var_name: str, adjoint_target_variables: set[str], callback: callable, code_callback: callable):
        if self.name in adjoint_target_variables:
            if adjoint_var_name:
                code_callback(f"adjoint_{self.name} += {adjoint_var_name}")
            else:
                raise Exception(f"edge case for backward_codegen for variable {self.name}")

        else:
            if adjoint_var_name:
                code_callback(f"adjoint_n{self.node_index} = {adjoint_var_name}")
            else:
                code_callback(f"adjoint_n{self.node_index} = 1.0")

        if not isinstance(self.value, Number):
            callback(self.value, adjoint_var_name, adjoint_target_variables, callable, code_callback)

    def get_value_var_name(self):
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_n{self.node_index}({self.name}, {self.value})"