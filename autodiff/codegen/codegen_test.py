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

node_creator = NodeCreator()

decl = lower_func_decl(get_ast(beale_func), node_creator)
print(decl)
st = SSAVariableTransformer(decl)
decl = st.transform()

cg = FunctionCodeGen(decl, debug=True)
cg.generate()

print("\n".join(cg.code))

def beale_func_deriv(x, y = 1):                                                                                                     # x = n1, y = n6
    a = x                                                                                                                     # n2 = n1
    b = 1.5 - a                                                                                                               # n5 = n3 - n2
    c = y                                                                                                                     # n7 = n6
    value_n9 = b + a * c
    d = b + a * c ** 2                                                                                                        # n12 = n5 + n2 * n7 ** n10
    b1 = 2.25 - a                                                                                                             # n15 = n13 - n2
    c1 = y ** 2                                                                                                               # n18 = n6 ** n16
    value_n20 = b1 + a * c1
    e = b1 + a * c1 ** 2                                                                                                      # n23 = n15 + n2 * n18 ** n21
    b2 = 2.625 - a                                                                                                            # n26 = n24 - n2
    c2 = y ** 3                                                                                                               # n29 = n6 ** n27
    value_n31 = b2 + a * c2
    f = b2 + a * c2 ** 2                                                                                                      # n34 = n26 + n2 * n29 ** n32
    forward_return = d + e + f                                                                                                # n36: n12 + n23 + n34

    adjoint_x = 0.0
    adjoint_y = 0.0
    adjoint_n36 = 1.0
    adjoint_n35 = adjoint_n36
    adjoint_n12 = adjoint_n35
    adjoint_n11 = adjoint_n35
    adjoint_n9 = adjoint_n11 * 2 * value_n9 ** (2 - 1)
    adjoint_n5 = adjoint_n9
    adjoint_n4 = adjoint_n9
    adjoint_n2 = -1 * adjoint_n4
    adjoint_x += -1 * adjoint_n4
    adjoint_n8 = adjoint_n9
    adjoint_n2 = adjoint_n8 * c
    adjoint_x += adjoint_n8 * c
    adjoint_n7 = adjoint_n8 * a
    adjoint_y += adjoint_n8 * a
    adjoint_n23 = adjoint_n35
    adjoint_n22 = adjoint_n35
    adjoint_n20 = adjoint_n22 * 2 * value_n20 ** (2 - 1)
    adjoint_n15 = adjoint_n20
    adjoint_n14 = adjoint_n20
    adjoint_n2 = -1 * adjoint_n14
    adjoint_x += -1 * adjoint_n14
    adjoint_n19 = adjoint_n20
    adjoint_n2 = adjoint_n19 * c1
    adjoint_x += adjoint_n19 * c1
    adjoint_n18 = adjoint_n19 * a
    adjoint_n17 = adjoint_n19 * a
    adjoint_y += adjoint_n17 * 2 * y ** (2 - 1)
    adjoint_n34 = adjoint_n36
    adjoint_n33 = adjoint_n36
    adjoint_n31 = adjoint_n33 * 2 * value_n31 ** (2 - 1)
    adjoint_n26 = adjoint_n31
    adjoint_n25 = adjoint_n31
    adjoint_n2 = -1 * adjoint_n25
    adjoint_x += -1 * adjoint_n25
    adjoint_n30 = adjoint_n31
    adjoint_n2 = adjoint_n30 * c2
    adjoint_x += adjoint_n30 * c2
    adjoint_n29 = adjoint_n30 * a
    adjoint_n28 = adjoint_n30 * a
    adjoint_y += adjoint_n28 * 3 * y ** (3 - 1)
    return forward_return, adjoint_x, adjoint_y

print(beale_func_deriv(3, 0.5))