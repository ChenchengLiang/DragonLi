from abc import ABC, abstractmethod

class AbstractParser(ABC):
    @abstractmethod
    def parse(self, content):
        pass

class EqParser(AbstractParser):
    def parse(self, content):
        variables= set(content["variables_str"][1:-1].split(','))
        terminals = set(content["terminals_str"][1:-1].split(','))

        left, right = content["equation_str"].split(' = ')
        left_terms = [ch for ch in left if ch in variables or ch in terminals]
        right_terms = [ch for ch in right if ch in variables or ch in terminals]

        return variables, terminals, left_terms, right_terms


class SMT2Parser(AbstractParser):
    def parse(self, content):
        # Implement the parsing logic here for SMT2 files
        # ...
        pass


class Parser:
    def __init__(self, parser, content):
        self.parser = parser
        self.content = content

    def parse(self):
        return self.parser.parse(self.content)

