from typing import Union


class Node:
    def __init__(self):
        self.adjoint: float = None
        self.value: float = float("nan")
        self.node_index: int = None
    
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
            a value is evaluated in C when computing dB/dC, which is dA/dB * dB/dC
        2. Compute adjoint * dB/dD and pass it as the adjoint for D.backward()
            a value is evaluated in D when computing dB/dD, which is dA/dB * dB/dD
        """
        raise NotImplementedError()

    def forward_codegen(self, walk_function: callable, intermediate_value_node_indices: set[int] = set(), current_precedence: int = 100):
        """
        walk_function : Function which should be used to walk down the tree.
                        Children of the current node should be passed as arg to this function.
        intermediate_value_node_indices : node ids which are assigned to intermediate values and can be used instead of generating code
        """
        raise NotImplementedError()

    def backward_codegen(self, adjoint_var_name: str, adjoint_target_variables: set[str], callback: callable, code_callback: callable):
        """
        adjoint_var_name: variable of the adjoint which is the received adjoint value
        adjoint_target_variables: name of the variables which adjoints need to be accumulated
        callback: function to pass children for continued iteration
        code_callback: function to pass the generated code
        """
        raise NotImplementedError()
    
    def backward_value_dependent_node_indices(self):
        """
        Return the list of indices of nodes where its values are required for computation of backward mode pass
        """
        return []

    def get_value_var_name(self):
        """
        Return the variable name that should hold the computed forward value of the node. In case of constants, it should return its value instaed
        """
        return f"value_n{self.node_index}"

class NodeCreator:
    def __init__(self):
        self.nodes_created = 0

    def create(self, node: Node, *args, **kwargs):
        node_obj = node(*args, **kwargs)
        node_obj.node_index = self.nodes_created
        self.nodes_created += 1
        return node_obj
