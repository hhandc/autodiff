from autodiff.codegen.ast_lowering import lower_func_decl, get_ast
from autodiff.codegen.codegen_walker import ForwardCodegenWalker
from autodiff.codegen.codegen_function import FunctionCodeGen
from autodiff.codegen.cg_transform import SSAVariableTransformer
from autodiff.computational_graph import NodeCreator
from math import sin

def sub_myfunc(x, y = 1):
    a = x
    b = a + x
    a = y
    c = 2
    return c + a - b # 2 + x - y

def add_myfunc(x, y = 1):
    a = x
    b = a + x
    a = y
    return a + b # -x + x - y

def mul_myfunc(x, y = 1):
    a = x
    b = a * x
    a = y
    return a * b # y * (x * x)

def div_myfunc(x, y = 1):
    a = x
    b = a / x
    a = y
    return a / b # y / (x / x)

def beale_func(x, y = 1):
    a = x
    b = 1.5 - a
    c = y
    d = (b + a * c) ** 2
    b = 2.25 - a
    c = y ** 2
    e = (b + a * c) ** 2
    b = 2.625 - a
    c = y ** 3
    f = (b + a * c) ** 2
    return d + e + f

def himmel_func(x, y):
    a = x ** 2 + y - 11
    b = a ** 2
    c = (x + y ** 2 - 7) 
    d = c ** 2

    return b + d

def maytas_func(x, y =1):
    a = x           
    b = y            
    c = a * a        
    d = b * b       
    e = c + d
    i = 0.26        
    f = i * e   
    g = a * b
    i = -0.48        
    h = i * g   
    return f + h 

def test_func(x):
    return (x + 1) * 2

# node_creator = NodeCreator()

# decl = lower_func_decl(get_ast(test_func), node_creator)
# print(decl)
# st = SSAVariableTransformer(decl)
# decl = st.transform()

# cg = FunctionCodeGen(decl, debug=True)
# cg.generate()

# print("\n".join(cg.code))

from autodiff.codegen import value_and_grad, value_and_grad_code

print(value_and_grad(test_func)(1))
print(value_and_grad_code(test_func))