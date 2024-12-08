from autodiff.codegen.ast_lowering import PyFuncDeclaration
from autodiff.computational_graph import Node
from autodiff.variable import Variable
import autodiff.ops as ops
from collections import defaultdict


class SSAVariableTransformer:
    """
    Transform the statements in a PyFuncDeclaration into a SSA form, by renaming reassignments to new variables.
    """
    def __init__(self, func_decl: PyFuncDeclaration):
        self.func_decl = func_decl

        # This dict keeps track the number of times a variable was assigned to
        self.assign_counts = defaultdict(lambda: 0)

        # This dict holds the Variable object for each variable name at the current scope
        self.scope_variables = {}

    def transform(self) -> PyFuncDeclaration:
        for stmt in self.func_decl.body:
            if isinstance(stmt, Variable):
                self.walk(stmt, assignment=True)
            else:
                self.walk(stmt)

        self.walk(self.func_decl.return_body)

        return self.func_decl

    def walk(self, node: Node, assignment=False):
        match node:
            case Variable():
                name = node.name
                if assignment:
                    if self.assign_counts[name] > 0:
                        # This means we're reassigning to an existing variable.
                        new_name = f"{name}{self.assign_counts[name]}"
                        node.name = new_name
                    self.scope_variables[name] = node
                    self.assign_counts[name] += 1
                else:
                    # We're just referencing a variable on RHS
                    self.walk(node.value)

            case ops.BinaryOp():
                if isinstance(node.left, Variable):
                    var_name = node.left.name
                    if var_name in self.scope_variables:
                        node.left = self.scope_variables[var_name]
                else:
                    self.walk(node.left)


                if isinstance(node.right, Variable):
                    var_name = node.right.name
                    if var_name in self.scope_variables:
                        node.right = self.scope_variables[var_name]
                else:
                    self.walk(node.right)
    
            case ops.UnaryOp():
                if isinstance(node.operand, Variable):
                    var_name = node.operand.name
                    if var_name in self.scope_variables:
                        node.operand = self.scope_variables[var_name]
                else:
                    self.walk(node.operand)
