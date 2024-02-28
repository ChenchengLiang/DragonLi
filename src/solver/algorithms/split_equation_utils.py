from typing import Tuple, List

from src.solver.Constants import SAT, UNSAT, UNKNOWN
from src.solver.DataTypes import Equation, Formula, Term, Variable
import random

def choose_an_unknown_eqiatons_random(f: Formula) -> (Equation, Formula):
    unknown_eq_index = random.randint(0, len(f.unknown_equations) - 1)
    unknown_eq: Equation = f.unknown_equations.pop(unknown_eq_index)
    _,new_formula=_update_formula_delete_eq(f,unknown_eq)
    return unknown_eq, new_formula


def choose_an_unknown_eqiatons_fixed(f: Formula) -> (Equation, Formula):
    unknown_eq: Equation = f.unknown_equations.pop(0)
    _, new_formula = _update_formula_delete_eq(f, unknown_eq)
    return unknown_eq, new_formula




def _one_variable_one_terminal_branch_1(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[
    Equation, Formula,int]:
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

    # update equation
    new_left_term_list = _update_term_list(old_term, new_term, local_eq.left_terms)
    new_right_term_list = [right_term] + _update_term_list(old_term, new_term, local_eq.right_terms)
    new_eq = Equation(new_left_term_list, new_right_term_list)

    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula,fresh_variable_counter


def _one_variable_one_terminal_branch_2(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[
    Equation, Formula,int]:
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
    (fresh_variable_term,fresh_variable_counter)= _create_fresh_variables(fresh_variable_counter)
    # define old and new term
    old_term: Term = left_term
    new_term: List[Term] = [right_term, fresh_variable_term]

    # update equation
    new_left_term_list = [fresh_variable_term] + _update_term_list(old_term, new_term, local_eq.left_terms)
    new_right_term_list = _update_term_list(old_term, new_term, local_eq.right_terms)
    new_eq = Equation(new_left_term_list, new_right_term_list)

    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula,fresh_variable_counter


def _two_variables_branch_1(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[Equation, Formula,int]:
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
    (fresh_variable_term,fresh_variable_counter) = _create_fresh_variables(fresh_variable_counter)
    # define old and new term
    new_term: List[Term] = [right_term, fresh_variable_term]
    old_term: Term = left_term

    # update equation
    new_left_term_list = [fresh_variable_term] + _update_term_list(old_term, new_term, local_eq.left_terms)
    new_right_term_list = _update_term_list(old_term, new_term, local_eq.right_terms)
    new_eq = Equation(new_left_term_list, new_right_term_list)

    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula,fresh_variable_counter


def _two_variables_branch_2(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[Equation, Formula,int]:
    '''
    Equation: V1 [Terms] = V2 [Terms]
    Assume |V1| < |V2|
    Replace V2 with V1V2'
    Obtain [Terms] [V2/V1V2'] = V2' [Terms] [V2/V1V2']
    '''
    return _two_variables_branch_1(Equation(eq.right_terms, eq.left_terms), current_formula,fresh_variable_counter)


def _two_variables_branch_3(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[Equation, Formula,int]:
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

    # update equation
    new_eq = Equation(_update_term_list(old_term, new_term, local_eq.left_terms),
                      _update_term_list(old_term, new_term, local_eq.right_terms))
    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula,fresh_variable_counter


def _update_formula(f: Formula, old_term: Term, new_term: List[Term]) -> Formula:
    new_eq_list = []
    for eq_in_formula in f.formula:
        new_left = _update_term_list(old_term, new_term, eq_in_formula.left_terms)
        new_right = _update_term_list(old_term, new_term, eq_in_formula.right_terms)
        new_eq_list.append(Equation(new_left, new_right))
    return Formula(new_eq_list)

def _update_formula_with_new_eq(f: Formula, new_eq: Equation,new_eq_satisfiability:str) -> Tuple[str,Formula]:
    new_eq_list = [new_eq]
    new_formula = Formula(new_eq_list+ f.formula)
    if new_eq_satisfiability==SAT:
        if new_eq in new_formula.unknown_equations:
            new_formula.unknown_equations.remove(new_eq)
            new_formula.sat_equations.append(new_eq)
            is_fact, fact_assignment = new_eq.is_fact()
            if is_fact:
                new_formula.facts.append((new_eq, fact_assignment))
    if new_eq_satisfiability==UNSAT:
        if new_eq in new_formula.unknown_equations:
            new_formula.unknown_equations.remove(new_eq)
            new_formula.unsat_equations.append(new_eq)
    if new_eq_satisfiability==UNKNOWN:
        if new_eq in new_formula.sat_equations:
            new_formula.sat_equations.remove(new_eq)
            new_formula.unknown_equations.append(new_eq)
        if new_eq in new_formula.unsat_equations:
            new_formula.unsat_equations.remove(new_eq)
            new_formula.unknown_equations.append(new_eq)

    return new_formula.satisfiability,new_formula

def _update_formula_delete_eq(f: Formula, eq: Equation) -> Tuple[str,Formula]:
    new_eq_list = []
    for eq_in_formula in f.formula:
        if eq_in_formula != eq:
            new_eq_list.append(eq_in_formula)
    new_formula = Formula(new_eq_list)
    return new_formula.satisfiability,new_formula



def _create_fresh_variables(fresh_variable_counter) -> Tuple[Term, int]:
    fresh_variable_term = Term(Variable(f"V{fresh_variable_counter}"))  # V1, V2, V3, ...
    fresh_variable_counter += 1
    return fresh_variable_term,fresh_variable_counter


def _update_term_list(old_term: Term, new_term: List[Term], term_list: List[Term]) -> List[Term]:
    new_term_list = []
    for t in term_list:
        if t == old_term:
            for new_t in new_term:
                new_term_list.append(new_t)
        else:
            new_term_list.append(t)
    return new_term_list
