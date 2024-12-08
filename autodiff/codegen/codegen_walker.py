from autodiff.computational_graph import Node
from autodiff.variable import Variable
import autodiff.codegen.sorting_nodes_dfs as topsort
import autodiff.ops as ops
from collections import OrderedDict

 
class NodeMap:
    """
    A dictionary mapping of all the nodes' ids to its object
    """
    def __init__(self, parent_node: Node):
        self.node_map: dict[int, Node] = OrderedDict()

        self.populate_map()

    def populate_map(self):
        visited = set()

        def recurse_nodes(node: Node):
            if id(node) in visited:
                return
            self.node_map[id(node)] = node

            match node:
                case UnaryOp():
                    recurse_nodes(node.operand)
                case BinaryOp():
                    recurse_nodes(node.left)
                    recurse_nodes(node.right)

    def __getitem__(self, key):
        return self.node_map[key]


class ForwardCodegenWalker:
    def __init__(self):
        self.code = ""

    def walk(self, node: Node):
        match node:
            case Variable():
