from autodiff.graph import Node
from autodiff.ops import UnaryOp, BinaryOp, Const
from autodiff.variable import Variable

def sort_forward_cg(parent: Node) -> list[str]:
    """
    Given an autodiff computation graph parent(toplevel) node, return the topological order of the nodes needed for forward value evaluation.
    This is done by postorder traversal.
    """
    result = []
    visited = set()

    def recurse_nodes(node: Node):
        if id(node) in visited:
            return
        visited.add(id(node))

        match node:
            case UnaryOp():
                recurse_nodes(node.operand)
            case BinaryOp():
                recurse_nodes(node.left)
                recurse_nodes(node.right)
        result.append(repr(node))

    recurse_nodes(parent)

    return result

def sort_backward_cg(parent: Node) -> list[str]:
    """
    Given an autodiff computation graph parent(toplevel) node, return the topological order of the nodes needed for backward gradient evaluation.
    This is done by preorder traversal.
    """
    result = []
    visited = set()

    def recurse_nodes(node: Node):
        if id(node) in visited:
            return
        visited.add(id(node))
        result.append(repr(node))

        match node:
            case UnaryOp():
                recurse_nodes(node.operand)
            case BinaryOp():
                recurse_nodes(node.left)
                recurse_nodes(node.right)
        

    recurse_nodes(parent)
    return result

if __name__ == "__main__":
    from autodiff.variable import Variable
    from autodiff.ops import *

    x_true = 1
    x = Variable("x", x_true)

    y_true = 3
    y = Variable("y", y_true)

    expr = Sin(x) + x * y

    print(sort_forward_cg(expr))
    print(sort_backward_cg(expr))