import typing

class Var:
    """
    Base class for variables
    """

    def __init__(self, variable_name : str):
        self.variable_name = variable_name
