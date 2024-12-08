from autodiff.codegen.ast_lowering import lower_func_decl, get_ast
from autodiff.codegen.codegen_walker import ForwardCodegenWalker
from autodiff.codegen.codegen_function import FunctionCodeGen
from autodiff.codegen.cg_transform import SSAVariableTransformer
from autodiff.computational_graph import NodeCreator
import math

def value_and_grad(f: callable) -> callable:
    node_creator = NodeCreator()

    decl = lower_func_decl(get_ast(f), node_creator)
    st = SSAVariableTransformer(decl)
    decl = st.transform()
    cg = FunctionCodeGen(decl, debug=True)
    cg.generate()

    code = "\n".join(cg.code)
    function_name = decl.function_name
    scope = {}
    exec(code, scope)
    return scope[function_name]

def value_and_grad_code(f: callable) -> str:
    node_creator = NodeCreator()

    decl = lower_func_decl(get_ast(f), node_creator)
    st = SSAVariableTransformer(decl)
    decl = st.transform()
    cg = FunctionCodeGen(decl, debug=True)
    cg.generate()

    code = "\n".join(cg.code)
    return code
