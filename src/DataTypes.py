
from typing import Union, List

class Variable:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return f"Variable({self.value})"

class Terminal:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return f"Terminal({self.value})"

class Term:
    def __init__(self, value: Union[Variable, Terminal]):
        self.value = value

    def __repr__(self):
        return f"Term({self.value})"
