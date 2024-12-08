from typing import Union


class Node:
    def __init__(self):
        self.adjoint: float = None
        self.value: float = float("nan")
    
    def eval(self) -> float:
        """
        Evaluate the value of the node
        """
        raise NotImplementedError()
    
    def backward(self, adjoint: float):
        """
        Suppose the following computation graph exists:
                 A
                 |
                 B
                / \
               C   D
        B.backward() should be receiving A's adjoint value as the argument adjoint. Then, B.backward() should do the following:
        1. Compute adjoint * dB/dC and pass it as the adjoint for C.backward()
        2. Compute adjoint * dB/dD and pass it as the adjoint for D.backward()
        """
        raise NotImplementedError()