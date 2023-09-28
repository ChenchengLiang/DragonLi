from abc import ABC, abstractmethod
from typing import Dict
from DataTypes import Variable, Terminal, Term
class AbstractParser(ABC):
    @abstractmethod
    def parse(self, content):
        pass


class EqParser(AbstractParser):
    def parse(self, content:Dict)->Dict:


        variables = set([Variable(v) for v in content["variables_str"]])
        terminals = set([Terminal(t) for t in content["terminals_str"]])

        left_str, right_str = content["equation_str"].split('=')
        left_terms = [ch for ch in left_str if ch in variables or ch in terminals]
        right_terms = [ch for ch in right_str if ch in variables or ch in terminals]

        parsed_content = {"variables": variables, "terminals": terminals, "left_terms": left_terms,
                          "right_terms": right_terms}

        return parsed_content


class SMT2Parser(AbstractParser):
    def parse(self, content:Dict):
        # Implement the parsing logic here for SMT2 files
        # ...
        pass


class Parser:
    def __init__(self, parser: AbstractParser):
        self.parser = parser

    def parse(self, file_path:str) -> Dict:
        file_reader = EqReader() if self.parser == EqParser else SMT2Reader()
        content = file_reader.read(file_path)
        return self.parser.parse(self, content)


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
