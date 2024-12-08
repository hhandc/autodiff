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
    def __init__(self, generate_intermediate_value_node_indices: set[int] = set(), intermediate_value_code_callback: callable = (lambda x: x), variable_alias_map: dict[str, str] = {}):
        """
        generate_intermediate_value_node_indices : node indices where the forward computation of the node should be saved
        intermediate_value_code_callback : callback function that should receive the intermediate value code
        variable_alias_map : a dictionary which holds the value of aliases that should be used in place of key variable names.
        """
        self.generate_intermediate_value_node_indices = generate_intermediate_value_node_indices
        self.intermediate_value_code_callback = intermediate_value_code_callback
        self.aliases = variable_alias_map
        self.generated_exprs: list[str] = []

        self.operator_precedence = {  # lower comes first
            ops.Add: 3,
            ops.Sub: 3,
            ops.Mul: 2,
            ops.Div: 2,
            ops.Pow: 1
        }

    def get_max_precedence(self, node: Node, max_precedence: int = 0):
        match node:
            case ops.BinaryOp():
                max_precedence = max(max_precedence, self.operator_precedence[type(node)])
                max_precedence = max(max_precedence, self.get_max_precedence(node.left), max_precedence)
                max_precedence = max(max_precedence, self.get_max_precedence(node.right), max_precedence)
        
        return max_precedence


    def walk(self, node: Node, current_precedence: int = 100, assignment=False):
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


                expr_max_precedence = self.get_max_precedence(node)
                my_precedence = self.operator_precedence[type(node)]
                #code = node.forward_codegen(lambda node: self.walk(node, current_precedence=my_precedence))
                code = node.forward_codegen(self.walk, intermediate_value_node_indices=self.generate_intermediate_value_node_indices, current_precedence=my_precedence)
                if node.node_index in self.generate_intermediate_value_node_indices:
                    self.intermediate_value_code_callback(f"{node.get_value_var_name()} = {code}")

                if expr_max_precedence > current_precedence:
                    code = f"({code})"
                return code

            case ops.UnaryOp():
                expr_tag = str(node)
                if expr_tag not in self.generated_exprs:
                    self.generated_exprs.append(expr_tag)
                
                my_precedence = self.operator_precedence[type(node)]
                code = node.forward_codegen(self.walk, self.generate_intermediate_value_node_indices, my_precedence)
                if node.node_index in self.generate_intermediate_value_node_indices:
                    self.intermediate_value_code_callback(f"{node.get_value_var_name()} = {code}")
                return code


class ForwardDebugCodegenWalker:
    def __init__(self, intermediate_value_node_indices: set[int] = set()):
        self.intermediate_value_node_indices = intermediate_value_node_indices

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
                return node.forward_codegen(self.walk, self.intermediate_value_node_indices)
            case ops.UnaryOp():
                return node.forward_codegen(self.walk, self.intermediate_value_node_indices)


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
