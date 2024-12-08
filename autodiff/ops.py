import typing

class Op:
    """
    Base class for operator AST node
    """
    def __init__(self):
        raise NotImplementedError()

    def eval(self):
        """
        Evaluate the value of the operator
        """
        raise NotImplementedError()

    def grad_backward(self, left_adjoint : float):
        """
        Backward-model AD
        left_adjoint : left adjoint value. This is the leftward-partial derivative value
        """
        raise NotImplementedError()