import graphviz
from autodiff.variable import Variable
from autodiff.ops import *

def draw_computation_graph(name: str, expr):
    dot = graphviz.Graph(name, engine="dot")

    recurse_expr_cgraph(expr, dot)

    dot.render(name, format="png")

def recurse_expr_cgraph(expr: Node, dot: graphviz.Digraph, parent_id: str = None):
    match expr:
        case UnaryOp():
            cid = str(id(expr))
            dot.node(cid, expr.__class__.__name__)
            if parent_id:
                dot.edge(parent_id, cid)

            recurse_expr_cgraph(expr.operand, dot, cid)

        case BinaryOp():
            cid = str(id(expr))
            dot.node(cid, expr.__class__.__name__)
            if parent_id:
                dot.edge(parent_id, cid)
            
            recurse_expr_cgraph(expr.left, dot, cid)
            recurse_expr_cgraph(expr.right, dot, cid)

        case Variable():
            cid = str(id(expr))
            dot.node(cid, expr.variable_name)
            if parent_id:
                dot.edge(parent_id, cid)

        case Const():
            cid = str(id(expr))
            dot.node(cid, str(expr.value))
            if parent_id:
                dot.edge(parent_id, cid)