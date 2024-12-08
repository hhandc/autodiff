from autodiff.codegen.ast_lowering import lower_func_decl, get_ast
from autodiff.codegen.codegen_walker import ForwardCodegenWalker
from autodiff.codegen.codegen_function import FunctionCodeGen
from autodiff.codegen.cg_transform import SSAVariableTransformer
from autodiff.computational_graph import NodeCreator
import math
import inspect

def value_and_grad(f: callable) -> callable:
    frame = inspect.currentframe().f_back
    name, code = value_and_grad_code(f, return_func_name=True)
    scope = frame.f_locals
    exec(code, scope)
    return scope[name]

def value_and_grad_code(f: callable, return_func_name = False) -> str:
    node_creator = NodeCreator()

    decl = lower_func_decl(get_ast(f), node_creator)
    st = SSAVariableTransformer(decl)
    decl = st.transform()
    cg = FunctionCodeGen(decl, debug=True)
    cg.generate()

    code = "\n".join(cg.code)
    if return_func_name:
        return decl.function_name, code
    else:
        return code
