from autodiff.computational_graph import Node
from autodiff.codegen.codegen_walker import NodeMap, ForwardCodegenWalker
from autodiff.variable import Variable
from autodiff.codegen.ast_lowering import PyFuncDeclaration
import autodiff.ops as ops
from textwrap import indent

class FunctionCodeGen:
    def __init__(self, func_decl: PyFuncDeclaration):
        self.func_decl = func_decl

        self.node_map = NodeMap()
        for body in func_decl.body:
            self.node_map.populate_map(body)
        self.node_map.populate_map(func_decl.return_body)

        self.dependent_variables = set(self.node_map.node_map.keys())

        self.code: list[str] = []

    def generate_declaration(self):
        code = f"def {self.func_decl.function_name}("
        argstrs = []
        for arg in self.func_decl.args:
            if arg.default:
                argstrs.append(f"{arg.name} = {arg.default}")
            else:
                argstrs.append(arg.name)
        
        code += ", ".join(argstrs)
        code += "):"
        self.code.append(code)

    def generate_forward(self):
        walker = ForwardCodegenWalker()
        for stmt in self.func_decl.body:
            if isinstance(stmt, Variable):
                code = walker.walk(stmt, assignment=True)
            else:
                code = walker.walk(stmt)
            self.code.append("    " + code)

        forward_return_expr = walker.walk(self.func_decl.return_body)
        self.code.append("    " + f"forward_return = {forward_return_expr}")

    def generate_backward(self):
        for arg_obj in self.func_decl.args:
            arg_name = arg_obj.name
            self.code.append("    " + f"adjoint_{arg_name} = 0.0")

    def generate(self):
        self.generate_declaration()
        self.generate_forward()
        self.code.append("")
        self.generate_backward()