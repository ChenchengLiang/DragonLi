from abc import ABC, abstractmethod
from typing import Dict

from src.solver.DataTypes import Variable, Terminal, Term, Equation, EMPTY_TERMINAL
from src.solver.independent_utils import remove_duplicates
from src.process_benchmarks.parser_utils import parse_smtlib_to_simple_format


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

    def __str__(self):
        return "EqParser"

    def wrap_to_term(self, c: str) -> Term:
        if c in self.variable_values:
            return Term(Variable(c))
        elif c in self.terminal_values:
            return Term(Terminal(c))

    def parse(self, content: Dict) -> Dict:
        self.variable_str = content["variables_str"]
        self.terminal_str = content["terminals_str"]

        # handle two differernt input format {ABCDE} and {A B C D E}
        if " " in content["variables_str"]:
            content["variables_str"] = content["variables_str"].split()
        if " " in content["terminals_str"]:
            content["terminals_str"] = content["terminals_str"].split()

        self.variables = remove_duplicates([Variable(v) for v in content["variables_str"]])
        self.terminals = remove_duplicates([EMPTY_TERMINAL] + [Terminal(t) for t in content["terminals_str"]])
        self.variable_values = [v.value for v in self.variables]
        self.terminal_values = [t.value for t in self.terminals]
        self.file_path = content["file_path"]

        def wrap_one_side_str(one_side_str):
            # dealing with "" and empty string
            if one_side_str == "\"\"":
                wrapped_terms = [self.wrap_to_term(one_side_str)]
            elif len(one_side_str) == 0:
                wrapped_terms = [Term(EMPTY_TERMINAL)]
            else:
                if " " in one_side_str:
                    wrapped_terms = [self.wrap_to_term(c) for c in one_side_str.split()]
                else:
                    wrapped_terms = [self.wrap_to_term(one_side_str)]

            return wrapped_terms

        equation_list = []
        for eq_str in content["equation_str_list"]:
            left_str, right_str = eq_str.split(' = ')
            wrapped_left_terms = wrap_one_side_str(left_str)
            wrapped_right_terms = wrap_one_side_str(right_str)

            equation_list.append(Equation(wrapped_left_terms, wrapped_right_terms))

        parsed_content = {"variables": self.variables, "terminals": self.terminals, "equation_list": equation_list,
                          "file_path": self.file_path}

        return parsed_content


class SMT2Parser(AbstractParser):
    def __init__(self):
        super().__init__()
        self.eq_parser = EqParser()  # Create an instance of EqParser

    def __str__(self):
        return "SMT2Parser"

    def parse(self, content):
        # Use the eq_parser instance to parse the content
        return self.eq_parser.parse(content)


class Parser:
    def __init__(self, parser: AbstractParser):
        self.parser = parser

    def parse(self, file_path: str, zip=None, log=False) -> Dict:
        file_reader = EqReader() if type(self.parser) == EqParser else SMT2Reader()
        content = file_reader.read(file_path, zip)
        if log == True:
            print("-" * 10, "Parsing", "-" * 10)
            print("file content: ", content)
        return self.parser.parse(content)


class AbstractFileReader(ABC):
    @abstractmethod
    def read(self, file_path):
        pass


class EqReader(AbstractFileReader):

    def read(self, file_path: str, zip=None) -> Dict:
        equation_str_list = []
        if zip == None:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        else:
            lines = []
            with zip.open(file_path) as f:
                for line in f:
                    line = line.decode('utf-8')
                    lines.append(line)

        variables_str = lines[0].strip().split("{")[1].split("}")[0]
        terminals_str = lines[1].strip().split("{")[1].split("}")[0]
        # equation_str = lines[2].strip().split(": ")[1].replace(" ", "")
        for line in lines[2:]:
            if line.startswith("Equation"):
                equation_str_list.append(line.strip().split(": ")[1])

        content = {"variables_str": variables_str, "terminals_str": terminals_str,
                   "equation_str_list": equation_str_list, "file_path": file_path}
        return content


class SMT2Reader(AbstractFileReader):
    def read(self, file_path: str) -> Dict:
        content = {}
        with open(file_path, 'r') as smtlib_input:
            parsed_format = parse_smtlib_to_simple_format(smtlib_input.read())
            content["variables_str"] = parsed_format["Variables"]
            content["terminals_str"] = parsed_format["Terminals"]
            content["equation_str_list"] = parsed_format["Equation"]
            content["file_path"] = file_path
        return content
