from autodiff.computational_graph import Node
from autodiff.codegen.codegen_walker import NodeMap, ForwardCodegenWalker, ForwardDebugCodegenWalker, BackwardCodegenWalker
from autodiff.variable import Variable
from autodiff.codegen.ast_lowering import PyFuncDeclaration
import autodiff.ops as ops
from textwrap import indent

class FunctionCodeGen:
    def __init__(self, func_decl: PyFuncDeclaration, debug=False):
        self.func_decl = func_decl
        self.debug = debug
        self.debug_ljust = 120


        self.node_map = NodeMap()
        self.dependent_forward_node_indices = set()

        def collect_dependent_node_indices(node: Node):
            for index in node.backward_value_dependent_node_indices():
                if index in self.node_map.node_map and not isinstance(self.node_map[index], Variable) and not isinstance(self.node_map[index], ops.Const):
                    self.dependent_forward_node_indices.add(index)
            match node:
                case ops.BinaryOp():
                    collect_dependent_node_indices(node.left)
                    collect_dependent_node_indices(node.right)
                case ops.UnaryOp():
                    collect_dependent_node_indices(node.operand)
                case Variable():
                    if isinstance(node.value, Node):
                        collect_dependent_node_indices(node.value)

        for body in func_decl.body:
            self.node_map.populate_map(body)
            collect_dependent_node_indices(body)

        self.node_map.populate_map(func_decl.return_body)
        collect_dependent_node_indices(func_decl.return_body)

        self.dependent_variables = set(self.node_map.node_map.keys())

        self.code: list[str] = []

    def generate_declaration(self):
        code = f"def {self.func_decl.function_name}("
        argstrs = []
        for arg in self.func_decl.args:
            if arg.default:
                walker = ForwardCodegenWalker()
                argstrs.append(f"{arg.name} = {walker.walk(arg.default)}")
            else:
                argstrs.append(arg.name)
        
        code += ", ".join(argstrs)
        code += "):"

        if self.debug:
            arg_names = [arg.name for arg in self.func_decl.args]
            arg_comment = ", ".join([f"{name} = n{self.node_map.variable_name_map[name].node_index}" for name in arg_names])
            code = code.ljust(self.debug_ljust + 4, " ")
            code += "  # " + arg_comment

        self.code.append(code)

    def generate_forward(self):
        def inject_forward_code_line_callback(code: str):
            self.code.append(" " * 4 + code)
        
        walker = ForwardCodegenWalker(self.dependent_forward_node_indices, inject_forward_code_line_callback)
        debug_walker = ForwardDebugCodegenWalker(self.dependent_forward_node_indices)

        for stmt in self.func_decl.body:
            if isinstance(stmt, Variable):
                code = walker.walk(stmt, assignment=True)
            else:
                code = walker.walk(stmt)
            
            if self.debug:
                code = code.ljust(self.debug_ljust, " ")
                if isinstance(stmt, Variable):
                    code += "  # " + debug_walker.walk(stmt, assignment=True)
                else:
                    code += "  # " + debug_walker.walk(stmt)

            self.code.append(" " * 4 + code)

        forward_return_expr = f"forward_return = {walker.walk(self.func_decl.return_body)}"

        if self.debug:
            forward_return_expr = forward_return_expr.ljust(self.debug_ljust, " ")
            forward_return_expr += f"  # n{self.func_decl.return_body.node_index}: " + debug_walker.walk(self.func_decl.return_body) 
        self.code.append("    " + forward_return_expr)

    def generate_backward(self):
        adjoint_target_variables = set()
        for arg_obj in self.func_decl.args:
            arg_name = arg_obj.name
            adjoint_target_variables.add(arg_name)
            self.code.append("    " + f"adjoint_{arg_name} = 0.0")

        walker = BackwardCodegenWalker(adjoint_target_variables)
        walker.walk_init(self.func_decl.return_body)

        for code in walker.code:
            self.code.append("    " + code)

    def generate_return(self):
        adjoints = [f"adjoint_{arg.name}" for arg in self.func_decl.args]
        self.code.append(f"    return forward_return, {', '.join(adjoints)}")

    def generate(self):
        self.generate_declaration()
        self.generate_forward()
        self.code.append("")
        self.generate_backward()
        self.generate_return()