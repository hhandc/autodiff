from autodiff.computational_graph import Node
from autodiff.codegen.codegen_walker import NodeMap
from autodiff.variable import Variable
import autodiff.ops as ops


class FunctionCodeGen:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

        self.node_map = NodeMap(self.parent_node)
        self.node_map.populate_map()

        self.dependent_variables = set(self.node_map.node_map.keys())

        self.code = ""

    def generate_declaration()``