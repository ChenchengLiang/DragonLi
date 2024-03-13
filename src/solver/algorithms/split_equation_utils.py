from typing import Tuple, List

from src.solver.Constants import SAT, UNSAT, UNKNOWN
from src.solver.DataTypes import Equation, Formula, Term, Variable, _update_term_in_eq_list, _update_term_list,Terminal
import random

from src.solver.independent_utils import color_print


def _category_formula_by_rules(f: Formula) -> List[Tuple[Equation, int]]:
    '''
    1: both sides are empty
    2: one side is empty
    3: matched prefix terminal, this should not happen due to the simplification
    4: mismatched prefix or suffix terminal
    5: first terms are variable and terminal respectively
    6: first terms are variables
    '''
    category_eq_list: List[Tuple[Equation, int]] = []
    for eq in f.eq_list:
        # both sides are empty
        if len(eq.term_list) == 0:
            category_eq_list.append((eq, 1))
        # left side is empty
        elif len(eq.left_terms) == 0 and len(eq.right_terms) > 0:
            category_eq_list.append((eq, 2))
        # right side is empty
        elif len(eq.left_terms) > 0 and len(eq.right_terms) == 0:  # right side is empty
            category_eq_list.append((eq, 2))
        # both sides are not empty
        else:
            first_left_term = eq.left_terms[0]
            first_right_term = eq.right_terms[0]
            last_left_term = eq.left_terms[-1]
            last_right_term = eq.right_terms[-1]
            # \epsilon=\epsilon \wedge \phi case
            if eq.left_terms == eq.right_terms: #this has been simplified, so will never reach here
                category_eq_list.append((eq, 1))

            # match prefix terminal #this has been simplified, so will never reach here
            elif first_left_term.value_type == Terminal and first_right_term.value_type == Terminal and first_left_term.value == first_right_term.value:
                category_eq_list.append((eq, 3))

            # mismatch prefix terminal
            elif first_left_term.value_type == Terminal and first_right_term.value_type == Terminal and first_left_term.value != first_right_term.value:
                eq.given_satisfiability = UNSAT
                category_eq_list.append((eq, 4))
            # mistmatch suffix terminal
            elif last_left_term.value_type == Terminal and last_right_term.value_type == Terminal and last_left_term.value != last_right_term.value:
                eq.given_satisfiability = UNSAT
                category_eq_list.append((eq, 4))

            # split rules
            else:
                left_term = eq.left_terms[0]
                right_term = eq.right_terms[0]
                # left side is variable, right side is terminal
                if type(left_term.value) == Variable and type(right_term.value) == Terminal:
                    category_eq_list.append((eq, 5))

                # left side is terminal, right side is variable
                elif type(left_term.value) == Terminal and type(right_term.value) == Variable:
                    category_eq_list.append((eq, 5))

                # both side are differernt variables
                elif type(left_term.value) == Variable and type(right_term.value) == Variable:
                    category_eq_list.append((eq, 6))
                else:
                    color_print(f"error: {eq.eq_str}", "red")

    return category_eq_list


def _left_variable_right_terminal_branch_1(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> \
        Tuple[
            Equation, Formula, int, str]:
    '''
    Equation: V1 [Terms] = a [Terms]
    Assume V1 = ""
    Delete V1
    Obtain [Terms] [V1/""] = a [Terms] [V1/""]
    '''

    local_eq = eq.deepcopy()
    left_term: Term = local_eq.left_terms.pop(0)
    right_term: Term = local_eq.right_terms.pop(0)

    # define old and new term
    old_term: Term = left_term
    new_term: List[Term] = []

    label_str = f"{old_term.get_value_str} = \"\" "

    # update equation
    new_left_term_list = _update_term_list(old_term, new_term, local_eq.left_terms)
    new_right_term_list = [right_term] + _update_term_list(old_term, new_term, local_eq.right_terms)
    new_eq = Equation(new_left_term_list, new_right_term_list)

    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula, fresh_variable_counter, label_str


def _left_variable_right_terminal_branch_2(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> \
        Tuple[
            Equation, Formula, int, str]:
    '''
    Equation: V1 [Terms] = a [Terms]
    Assume V1 = aV1'
    Replace V1 with aV1'
    Obtain V1' [Terms] [V1/aV1'] = [Terms] [V1/aV1']
    '''
    local_eq = eq.deepcopy()
    left_term: Term = local_eq.left_terms.pop(0)
    right_term: Term = local_eq.right_terms.pop(0)

    # create fresh variable
    (fresh_variable_term, fresh_variable_counter) = _create_fresh_variables(fresh_variable_counter)
    # define old and new term
    old_term: Term = left_term
    new_term: List[Term] = [right_term, fresh_variable_term]

    label_str = f"{old_term.get_value_str}= {new_term[0].get_value_str}{new_term[1].get_value_str}"

    # update equation
    new_left_term_list = [fresh_variable_term] + _update_term_list(old_term, new_term, local_eq.left_terms)
    new_right_term_list = _update_term_list(old_term, new_term, local_eq.right_terms)
    new_eq = Equation(new_left_term_list, new_right_term_list)

    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula, fresh_variable_counter, label_str


def _two_variables_branch_1(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[
    Equation, Formula, int, str]:
    '''
    Equation: V1 [Terms] = V2 [Terms]
    Assume |V1| > |V2|
    Replace V1 with V2V1'
    Obtain V1' [Terms] [V1/V2V1'] = [Terms] [V1/V2V1']
    '''

    local_eq = eq.deepcopy()
    left_term: Term = local_eq.left_terms.pop(0)
    right_term: Term = local_eq.right_terms.pop(0)

    # create fresh variable
    (fresh_variable_term, fresh_variable_counter) = _create_fresh_variables(fresh_variable_counter)
    # define old and new term
    new_term: List[Term] = [right_term, fresh_variable_term]
    old_term: Term = left_term

    label_str = f"{old_term.get_value_str}= {new_term[0].get_value_str}{new_term[1].get_value_str}"

    # update equation
    new_left_term_list = [fresh_variable_term] + _update_term_list(old_term, new_term, local_eq.left_terms)
    new_right_term_list = _update_term_list(old_term, new_term, local_eq.right_terms)
    new_eq = Equation(new_left_term_list, new_right_term_list)

    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula, fresh_variable_counter, label_str


def _two_variables_branch_2(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[
    Equation, Formula, int, str]:
    '''
    Equation: V1 [Terms] = V2 [Terms]
    Assume |V1| < |V2|
    Replace V2 with V1V2'
    Obtain [Terms] [V2/V1V2'] = V2' [Terms] [V2/V1V2']
    '''
    return _two_variables_branch_1(Equation(eq.right_terms, eq.left_terms), current_formula, fresh_variable_counter)


def _two_variables_branch_3(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[
    Equation, Formula, int, str]:
    '''
    Equation: V1 [Terms] = V2 [Terms]
    Assume |V1| = |V2|
    Replace V1 with V2
    Obtain [Terms] [V1/V2] = [Terms] [V1/V2]
    '''
    local_eq = eq.deepcopy()
    left_term: Term = local_eq.left_terms.pop(0)
    right_term: Term = local_eq.right_terms.pop(0)

    # define old and new term
    old_term: Term = left_term
    new_term: List[Term] = [right_term]

    label_str = f"{old_term.get_value_str}= {new_term[0].get_value_str}"

    # update equation
    new_eq = Equation(_update_term_list(old_term, new_term, local_eq.left_terms),
                      _update_term_list(old_term, new_term, local_eq.right_terms))
    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula, fresh_variable_counter, label_str


def _update_formula(f: Formula, old_term: Term, new_term: List[Term]) -> Formula:
    return Formula(_update_term_in_eq_list(f.eq_list, old_term, new_term))


def _create_fresh_variables(fresh_variable_counter) -> Tuple[Term, int]:
    fresh_variable_term = Term(Variable(f"V{fresh_variable_counter}"))  # V1, V2, V3, ...
    fresh_variable_counter += 1
    return fresh_variable_term, fresh_variable_counter
