
from src.solver.Parser import Parser, EqParser, SMT2Parser
from src.solver.DataTypes import Equation, Variable, Terminal
from src.solver.independent_utils import strip_file_name_suffix,remove_duplicates
from typing import List, Dict
from src.solver.DataTypes import Term

def one_eq_file_to_smt2(file_path):
    # parse .eq
    parser_type = EqParser() if file_path.endswith(".eq") else SMT2Parser()
    parser = Parser(parser_type)
    parsed_content = parser.parse(file_path)
    #print("parsed_content:",parsed_content)


    # get smt string
    smt_str = "(set-logic QF_S) \n"
    # define variables
    variable_list = []
    for eq in parsed_content["equation_list"]:
        variable_list.extend(eq.variable_list)
    variable_list = remove_duplicates(variable_list)
    for v in variable_list:
        smt_str += f"(declare-fun {v.value} () String) \n"

    # assert eqiations
    for eq in parsed_content["equation_list"]:
        assert_str = assert_one_eq(eq)
        smt_str += assert_str + "\n"

    smt_str += "(check-sat) \n"
    smt_str += "(get-model)"

    # write smt_str to smt2 file
    smt_file_name=strip_file_name_suffix(file_path)+".smt2"
    with open(smt_file_name, "w") as f:
        f.write(smt_str)
    return smt_file_name


def assert_one_eq(eq: Equation):
    lhs = handle_one_side(eq.left_terms)
    rhs = handle_one_side(eq.right_terms)
    if len(eq.left_terms)>1:
        lhs = add_parentheses(f"str.++ {lhs}")
    else:
        lhs = f"{lhs}"
    if len(eq.right_terms)>1:
        rhs = add_parentheses(f"str.++ {rhs}")
    else:
        rhs = f"{rhs}"
    eq_str = add_parentheses(f"= {lhs} {rhs}")
    assert_str = add_parentheses(f"assert {eq_str}")
    return assert_str


def handle_one_side(one_side:List[Term]):
    first_element = one_side[0]
    concatenated = first_element.get_value_str if first_element.value_type == Variable else add_quote(
        first_element.get_value_str)
    for letter in one_side[1:]:
        letter_smt_format = letter.get_value_str if letter.value_type == Variable else add_quote(letter.get_value_str)
        concatenated = concatenated + " " + letter_smt_format
    return concatenated


def add_parentheses(s):
    return f"({s})"


def add_quote(s):
    return "\"" + s + "\""