from autodiff.codegen.ast_lowering import lower_func_decl, get_ast
from autodiff.codegen.codegen_walker import ForwardCodegenWalker
from autodiff.codegen.codegen_function import FunctionCodeGen
from autodiff.codegen.cg_transform import SSAVariableTransformer
from autodiff.computational_graph import NodeCreator
from math import sin

def myfunc(x, y = 1):
    a = x
    b = a + x
    a = y
    return a + b # x + x + y

node_creator = NodeCreator()

decl = lower_func_decl(get_ast(myfunc), node_creator)
print(decl)
st = SSAVariableTransformer(decl)
decl = st.transform()

cg = FunctionCodeGen(decl, debug=True)
cg.generate()

print("\n".join(cg.code))

