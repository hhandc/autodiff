from autodiff.codegen.ast_lowering import lower_func_decl, get_ast, print_ast
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

def beale_func(x = 3, y = 0.5):
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

def matyas_func(x, y):
    a = x ** 2 + y ** 2
    b = a * 0.26
    c = -0.48 * x * y
    return b + c

def test_func(x):
    return (x + 0.11) * -1

# node_creator = NodeCreator()

# decl = lower_func_decl(get_ast(test_func), node_creator)
# print(decl)
# st = SSAVariableTransformer(decl)
# decl = st.transform()

# cg = FunctionCodeGen(decl, debug=True)
# cg.generate()

# print("\n".join(cg.code))

from autodiff.codegen import value_and_grad, value_and_grad_code

# print_ast(get_ast(test_func))

print("generated code:")
print(value_and_grad_code(test_func))
print("#" * 20)
value_grad = value_and_grad(test_func)(0)
print("function value eval:", value_grad[0])
print("function value actual:", test_func(0))
print("grads:", value_grad[1:])
