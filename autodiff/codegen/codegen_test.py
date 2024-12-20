from autodiff.codegen.ast_lowering import lower_func_decl, get_ast, print_ast
from autodiff.codegen.codegen_walker import ForwardCodegenWalker
from autodiff.codegen.codegen_function import FunctionCodeGen
from autodiff.codegen.cg_transform import SSAVariableTransformer
from autodiff.computational_graph import NodeCreator
from autodiff.variable import Variable
from autodiff.ops import *
from autodiff.graph import draw_computation_graph
import time
import psutil
import os

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
    a = 1.5 - x + x * y
    b = a ** 2
    a = 2.25 - x + x * y ** 2
    c = a ** 2
    a = 2.625 - x + x * y ** 3
    d = a ** 2

    return b + c + d

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

from autodiff.codegen import value_and_grad, value_and_grad_code, saved_value_and_grad

# print_ast(get_ast(test_func))

# print("generated code:")
# print(value_and_grad_code(himmel_func))
# print("#" * 20)
# value_grad = value_and_grad(himmel_func)(3, 2)
# print("function value eval:", value_grad[0])
# print("function value actual:", himmel_func(3, 2))
# print("grads:", value_grad[1:])

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

start = time.time()
mem_before = process_memory()

beale_grad_func = value_and_grad(beale_func)
himmel_grad_func = value_and_grad(himmel_func)
matyas_grad_func = value_and_grad(matyas_func)

end_codegen_time = time.time()
mem_post_codegen = process_memory()

print(f"codegen time: {round(end_codegen_time - start, 5)} codegen mem kb: {(mem_post_codegen - mem_before) / 1024}")

start = time.time()
mem_before = process_memory()

# for i in range(0, 10000):
#     value_grad_1 = value_and_grad(beale_func)(3, 0.5)
#     value_grad_2 = value_and_grad(himmel_func)(3, 2)
#     value_grad_3 = value_and_grad(matyas_func)(0, 0)

# start = time.time()
# mem_before = process_memory()

for i in range(0, 10000):
    value_grad_1 = beale_grad_func(3, 0.5)
    value_grad_2 = himmel_grad_func(3, 2)
    value_grad_3 = matyas_grad_func(0, 0)

mem_after = process_memory()
end = time.time()

print(f"eval time: {round(end - start, 5)} eval mem kb: {(mem_after - mem_before) / 1024}")

# start = time.time()
# mem_before = process_memory()

# for i in range(0,10000):
#     x = Variable("x", 3)
#     y = Variable("y", 0.5)

#     beale_expr = (1.5 - x + x * y) ** 2 + (2.25 - x + x * (y ** 2)) ** 2 + (2.625 - x + x * (y ** 3)) ** 2

#     beale_expr.eval()
#     beale_expr.backward()

#     x = Variable("x", 3)
#     y = Variable("y", 2)

#     himmel_expr = (x ** 2 + y - 11) ** 2 + (x + y**2 + 7) ** 2

#     himmel_expr.eval()
#     himmel_expr.backward()    

#     x = Variable("x", 0)
#     y = Variable("y", 0)

#     matyas_expr = 0.26 * (x**2 + y**2) - 0.48 * x * y

#     matyas_expr.eval()
#     matyas_expr.backward()

# mem_after = process_memory()
# end = time.time()
# print((mem_after - mem_before) / 1024)
# print(round((end - start), 5))