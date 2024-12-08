from autodiff.computational_graph import Node
from autodiff.variable import Variable
import autodiff.codegen.sorting_nodes_dfs as topsort
import autodiff.ops as ops
from collections import OrderedDict
from ast import NodeVisitor

 
class NodeMap:
    """
    A dictionary mapping of all the node.node_index and variable names to its object
    """
    def __init__(self):
        self.node_map: dict[int, Node] = OrderedDict()
        self.variable_name_map: dict[str, Node] = {}

    def populate_map(self, parent_node: Node):
        visited = set()

        def recurse_nodes(node: Node):
            if node.node_index in visited:
                return
            self.node_map[node.node_index] = node

            match node:
                case Variable():
                    self.variable_name_map[node.name] = node
                    if isinstance(node.value, Node):
                        recurse_nodes(node.value)
                case ops.UnaryOp():
                    recurse_nodes(node.operand)
                case ops.BinaryOp():
                    recurse_nodes(node.left)
                    recurse_nodes(node.right)

        recurse_nodes(parent_node)

    def __getitem__(self, key):
        return self.node_map[key]


class ForwardCodegenWalker:
    def __init__(self, variable_alias_map: dict[str, str] = {}):
        """
        variable_alias_map : a dictionary which holds the value of aliases that should be used in place of key variable names.
        """
        self.aliases = variable_alias_map
        self.generated_exprs: list[str] = []

    def walk(self, node: Node, assignment=False):
        match node:
            case Variable():
                if node.name in self.aliases.keys():
                    node_name =  self.aliases[node.name]
                else:
                    node_name = node.name
                
                if assignment:
                    return f"{node_name} = {self.walk(node.value)}"
                else:
                    return node_name

            case ops.Const():
                return node.value
            case ops.BinaryOp():
                expr_tag = str(node)
                if expr_tag not in self.generated_exprs:
                    self.generated_exprs.append(expr_tag)
                return node.forward_codegen(self.walk)
            case ops.UnaryOp():
                expr_tag = str(node)
                if expr_tag not in self.generated_exprs:
                    self.generated_exprs.append(expr_tag)
                return node.forward_codegen(self.walk)


class ForwardDebugCodegenWalker:
    def walk(self, node: Node, assignment=False):
        match node:
            case Variable():
                if assignment:
                    return f"n{node.node_index} = {self.walk(node.value)}"
                else:
                    return f"n{node.node_index}"

            case ops.Const():
                return f"n{node.node_index}"
            case ops.BinaryOp():
                return node.forward_codegen(self.walk)
            case ops.UnaryOp():
                return node.forward_codegen(self.walk)


class BackwardCodegenWalker:
    def __init__(self, adjoint_target_variables: set[str]):
        self.adjoint_target_variables = adjoint_target_variables
        self.calculated_node_indices = set()
        self.code = []  # unindented code

    def code_callback(self, code:str):
        self.code.append(code)

    def walk_init(self, node):
        self.walk(node, None, self.adjoint_target_variables, self.walk, self.code_callback)

    def walk(self, node: Node, adjoint_var_name: str, adjoint_target_variables: set[str], callback: callable, code_callback: callable):
        node.backward_codegen(adjoint_var_name, adjoint_target_variables, self.walk, self.code_callback)
