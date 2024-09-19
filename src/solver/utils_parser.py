
from .Parser import EqParser, Parser


def perse_eq_file(eq_str: str,log=False):
    parser_type = EqParser()
    parser = Parser(parser_type)
    return parser.parse(eq_str,log=log)