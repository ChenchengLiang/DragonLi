from abc import ABC, abstractmethod
from typing import Dict, List
from DataTypes import Variable, Terminal, Term


class AbstractParser(ABC):
    @abstractmethod
    def parse(self, content):
        pass


class EqParser(AbstractParser):
    def __init__(self):
        self.variable_str = None
        self.variable_str = None
        self.variables = None
        self.terminals = None
        self.left_terms = None
        self.right_terms = None

    def wrap_to_term(self, c: str)-> Term:
        if c in self.variable_str:
            return Term(Variable(c))
        elif c in self.terminal_str:
            return Term(Terminal(c))


    def parse(self, content: Dict) -> Dict:
        self.variable_str=content["variables_str"]
        self.terminal_str=content["terminals_str"]
        self.variables = set([Variable(v) for v in content["variables_str"]])
        self.terminals = set([Terminal(t) for t in content["terminals_str"]])

        left_str, right_str = content["equation_str"].split('=')


        self.left_terms = [self.wrap_to_term(c) for c in left_str]
        self.right_terms = [self.wrap_to_term(c) for c in right_str]


        parsed_content = {"variables": self.variables, "terminals": self.terminals, "left_terms": self.left_terms,
                          "right_terms": self.right_terms}

        return parsed_content



class SMT2Parser(AbstractParser):
    def parse(self, content: Dict):
        # Implement the parsing logic here for SMT2 files
        # ...
        pass


class Parser:
    def __init__(self, parser: AbstractParser):
        self.parser = parser

    def parse(self, file_path: str) -> Dict:
        file_reader = EqReader() if type(self.parser) == EqParser else SMT2Reader()
        content = file_reader.read(file_path)
        return self.parser.parse(content)


class AbstractFileReader(ABC):
    @abstractmethod
    def read(self, file_path):
        pass


class EqReader(AbstractFileReader):
    def read(self, file_path: str) -> Dict:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        variables_str = lines[0].strip().split("{")[1].split("}")[0]
        terminals_str = lines[1].strip().split("{")[1].split("}")[0]
        equation_str = lines[2].strip().split(": ")[1].replace(" ", "")
        content = {"variables_str": variables_str, "terminals_str": terminals_str, "equation_str": equation_str}
        return content


class SMT2Reader(AbstractFileReader):
    def read(self, file_path: str) -> Dict:
        # Implement the reading logic here for SMT2 files
        # ...
        pass
