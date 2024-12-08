import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from numbers import Number
from ast import NodeVisitor
import ast as past
import autodiff.ops as ops
import autodiff.computational_graph as cg
from autodiff.variable import Variable


@dataclass
class PyFuncArg:
    name: str
    default: Number = field(default=None)


@dataclass
class PyFuncDeclaration:
    function_name: str
    body: list[cg.Node]
    return_body: cg.Node
    args: list[PyFuncArg] = field(default_factory=list)


def print_ast(ast_obj):
    print(ast.dump(ast_obj, indent=4))


class PyExprEvaluator(NodeVisitor):
    def __init__(self, node_creator: cg.NodeCreator):
        super().__init__()
        self.variable_nodes: dict[str, Variable] = {}
        self.node_creator = node_creator

    def visit_Constant(self, node):
        return self.node_creator.create(ops.Const, node.value)

    def visit_BinOp(self, node):
        left = node.left
        op = node.op
        right = node.right
        match op:
            case past.Add():
                op = ops.Add
            case past.Sub():
                op = ops.Sub
            case past.Mult():
                op = ops.Mul
            case past.Div():
                op = ops.Div
            case past.Pow():
                op = ops.Pow
            case past.USub():
                op = ops.Neg
            case _:
                raise Exception(f"Unsupported binary operation {op.__class__.__name__}")
        
        left = self.visit(left)
        right = self.visit(right)
        return self.node_creator.create(op, left, right)

    def visit_Name(self, node):
        if node.id not in self.variable_nodes:
            new_node = self.node_creator.create(Variable, node.id)
            self.variable_nodes[node.id] = new_node
        return self.variable_nodes[node.id]

    def visit_Call(self, node):
        func = node.func
        match func.id:
            case "sin":
                funcname = ops.Sin
            case "cos":
                funcname = ops.Cos
            case "exp":
                funcname = ops.Exp

        args = [self.visit(arg) for arg in node.args]

        return self.node_creator.create(funcname, *args)


class PyStmtLowerer(PyExprEvaluator):
    def visit_Return(self, node):
        return self.visit(node.value)

    def visit_Assign(self, node):
        value = self.visit(node.value)

        assigns = []
        for target in node.targets:
            name = target.id
            new_node = self.node_creator.create(Variable, name, value)
            self.variable_nodes[name] = new_node
            if len(node.targets) == 1:
                return new_node
            assigns.append(new_node)

        return assigns


def lower_func_decl(ast_obj: ast.AST, node_creator: cg.NodeCreator) -> PyFuncDeclaration:
    """
    Does not support *args and **kwargs
    Only supports constant default values
    """
    assert isinstance(ast_obj, ast.Module), "The python ast entry object should be of type `ast.Module`"
    ast_obj = ast_obj.body
    assert len(ast_obj) == 1 and isinstance(ast_obj[0], ast.FunctionDef), "The python ast should contain a single function declaration"
    ast_obj = ast_obj[0]
    # print_ast(ast_obj)

    function_name = ast_obj.name

    function_arg_names = [arg.arg for arg in ast_obj.args.args]
    function_arg_defaults  = ast_obj.args.defaults

    if len(function_arg_defaults) < len(function_arg_names):
        for _ in range(len(function_arg_names) - len(function_arg_defaults)):
            function_arg_defaults.insert(0, None)

    function_arg_objs = []

    for arg_name, arg_expr in zip(function_arg_names, function_arg_defaults):
        if arg_expr:
            arg_expr = PyExprEvaluator(node_creator).visit(arg_expr)

        function_arg_objs.append(PyFuncArg(arg_name, arg_expr))


    body_exprs = []
    return_expr = None

    stmt_walker = PyStmtLowerer(node_creator)

    for stmt in ast_obj.body:
        if isinstance(stmt, past.Return):
            return_expr = stmt_walker.visit(stmt)
            break
        body_exprs.append(stmt_walker.visit(stmt))

    func_declaration = PyFuncDeclaration(function_name, body_exprs, return_expr, function_arg_objs)
    
    return func_declaration
    
def get_ast(obj: object):
    sc = inspect.getsource(obj)
    f_ast = ast.parse(textwrap.dedent(sc))

    return f_ast
