import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from numbers import Number

class PyFuncDeclaration:
    def __init__(self, function_name):
        pass


@dataclass
class PyFuncArg:
    name: str
    default: Number = field(default=None)


def print_ast(ast_obj):
    print(ast.dump(ast_obj, indent=4))

def lower_ast(ast_obj: ast.AST):
    assert isinstance(ast_obj, ast.Module), "The Phthon ast entry object should be of type `ast.Module`"
    ast_obj = ast_obj.body
    assert len(ast_obj) == 1 and isinstance(ast_obj[0], ast.FunctionDef), "The python ast should contain a single function declaration"
    ast_obj = ast_obj[0]
    print_ast(ast_obj)

    function_name = ast_obj.name

    


if __name__ == "__main__":
    import math
    def test_func(x, y):
        return math.sin(x) + x * y

    from math import sin
    def test_func2(x, y):
        return sin(x) + x * y

    sc = inspect.getsource(test_func2)
    f_ast = ast.parse(textwrap.dedent(sc))
    print("*" * 10)
    lower_ast(f_ast) 