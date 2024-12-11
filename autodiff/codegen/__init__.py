from autodiff.codegen.ast_lowering import lower_func_decl, get_ast
from autodiff.codegen.codegen_walker import ForwardCodegenWalker
from autodiff.codegen.codegen_function import FunctionCodeGen
from autodiff.codegen.cg_transform import SSAVariableTransformer
from autodiff.computational_graph import NodeCreator
from autodiff.codegen.codemanage import manage_Code
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
        return f"{decl.function_name}_value_and_grad", code
    else:
        return code

def saved_value_and_grad(f: callable) -> callable:
    check = manage_Code()
    frame = inspect.currentframe().f_back

    if (check.retrieve(f.__name__) == -1):
        name, code = value_and_grad_code(f, return_func_name=True)
    else:
        code = check.retrieve(f.__name__)

    scope = frame.f_locals
    exec(code, scope)
    return scope[name]

def saved_value_and_grad_code(f: callable, return_func_name = False) -> str:
    node_creator = NodeCreator()

    decl = lower_func_decl(get_ast(f), node_creator)
    st = SSAVariableTransformer(decl)
    decl = st.transform()
    cg = FunctionCodeGen(decl, debug=True)
    cg.generate()

    code = "\n".join(cg.code)

    check = manage_Code()
    check.save(f, code)
    if return_func_name:
        return f"{decl.function_name}_value_and_grad", code
    else:
        return code
