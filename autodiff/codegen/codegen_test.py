from autodiff.codegen.ast_lowering import lower_func_decl, get_ast
from autodiff.codegen.codegen_walker import ForwardCodegenWalker
from autodiff.codegen.codegen_function import FunctionCodeGen
from autodiff.codegen.cg_transform import SSAVariableTransformer
from math import sin

def myfunc(x, y = 1):
    a = sin(x)
    b = a + x
    a = y + a
    return a + b

decl = lower_func_decl(get_ast(myfunc))
print(decl)
st = SSAVariableTransformer(decl)
decl = st.transform()

cg = FunctionCodeGen(decl)
cg.generate()

print("\n".join(cg.code))

