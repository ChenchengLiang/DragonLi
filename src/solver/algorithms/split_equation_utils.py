from typing import Tuple, List, Callable, Dict, Union

from src.solver.Constants import SAT, UNSAT, UNKNOWN, HYBRID_ORDER_EQUATION_RATE, HYBRID_BRANCH_RATE
from src.solver.DataTypes import Equation, Formula, Term, Variable, _update_term_in_eq_list, _update_term_list, \
    Terminal, IsomorphicTailSymbol
import random

from src.solver.independent_utils import color_print, time_it, log_print_to_file


def get_unsat_label(satisfiability_list,label_list,back_track_count_list):
    # output one-hot encoding labels
    # mix unsat and unknown
    if satisfiability_list.count(UNSAT) >= 1 and satisfiability_list.count(UNKNOWN) >= 1:
        if satisfiability_list.count(UNSAT) == 1:
            label_list[satisfiability_list.index(UNSAT)] = 1
        else:  # satisfiability_list.count(UNSAT)>1
            unsat_back_track_counts = [(index, back_track_count_list[index]) for index, value in
                                       enumerate(satisfiability_list) if value == UNSAT]
            min_back_track_count_index = min(unsat_back_track_counts, key=lambda x: x[1])[0]
            label_list[min_back_track_count_index] = 1
    else:  # only unsat or unknown
        min_back_track_count_index = back_track_count_list.index(min(back_track_count_list))
        label_list[min_back_track_count_index] = 1

    assert sum(label_list) == 1
    return label_list

def run_summary(summary_dict):
    print(f"----- run summary -----")
    for k, v in summary_dict.items():
        print(f"{k}: {v}")


def differentiate_isomorphic_equations(eq_list: List[Equation]) -> List[Equation]:
    '''
    Add different number of '#' to the tails on both sides for isomorphic equations
    '''
    occurrence_tracker: Dict[
        Tuple[Tuple[Union[Variable, Terminal], ...], Tuple[Union[Variable, Terminal], ...]], int] = {}
    new_eq_list: List[Equation] = []

    for eq in eq_list:
        # Convert list to tuple to make them hashable
        eq_type = (tuple(eq.left_hand_side_type_list), tuple(eq.right_hand_side_type_list))

        if eq_type in occurrence_tracker:
            occurrence_tracker[eq_type] += 1
            modified_eq = Equation(
                eq.left_terms + [Term(IsomorphicTailSymbol("#"))] * occurrence_tracker[eq_type],
                eq.right_terms + [Term(IsomorphicTailSymbol("#"))] * occurrence_tracker[eq_type]
            )
            new_eq_list.append(modified_eq)
        else:
            occurrence_tracker[eq_type] = 0
            new_eq_list.append(eq)

    return new_eq_list


def order_equations_fixed(f: Formula, category_call=0) -> (Formula,int):
    return f,category_call

def order_equations_random(f: Formula, category_call=0) -> (Formula,int):
    random.shuffle(f.eq_list)
    return f,category_call

def order_equations_hybrid_fixed_random(f: Formula, category_call=0) -> (Formula,int):
    if random.random() < HYBRID_ORDER_EQUATION_RATE:
        return order_equations_fixed(f,category_call)
    else:
        return order_equations_random(f,category_call)


def order_equations_hybrid_category_fixed_random(f: Formula, category_call=0) -> (Formula, int):
    if random.random() < HYBRID_ORDER_EQUATION_RATE:
        return order_equations_category(f,category_call)
    else:
        return order_equations_category_random(f,category_call)

def order_equations_category(f: Formula, category_call=0) -> (Formula,int):
    categoried_eq_list: List[Tuple[Equation, int]] = _category_formula_by_rules(f)
    category_call += 1
    sorted_eq_list = sorted(categoried_eq_list, key=lambda x: x[1])


    return Formula([eq for eq, _ in sorted_eq_list]),category_call

def order_equations_shortest(f: Formula, category_call=0) -> (Formula,int):
    eq_list_with_length: List[Tuple[Equation, int]] = [(e,e.term_length) for e in f.eq_list]
    sorted_eq_list = sorted(eq_list_with_length, key=lambda x: x[1])

    return Formula([eq for eq, _ in sorted_eq_list]),category_call

def order_equations_longest(f: Formula, category_call=0) -> (Formula,int):
    eq_list_with_length: List[Tuple[Equation, int]] = [(e,e.term_length) for e in f.eq_list]
    sorted_eq_list = sorted(eq_list_with_length, key=lambda x: x[1],reverse=True)

    return Formula([eq for eq, _ in sorted_eq_list]),category_call

def order_equations_unsatcore(f: Formula, category_call=0) -> (Formula,int):
    condition = len([e for e in f.eq_list if e in f.unsat_core]) > 0
    unsatcore_eq_score_func=unsatcore_eq_score_random
    return order_equations_with_unsatcore_and_fixed_condition(f,condition,empty_else_func,unsatcore_eq_score_func,category_call)

def order_equations_unsatcore_first_n_iterations(f: Formula, category_call=0) -> (Formula,int):
    condition = f.current_total_split_eq_call==1 and len([e for e in f.eq_list if e in f.unsat_core]) > 0
    unsatcore_eq_score_func = unsatcore_eq_score_random
    return order_equations_with_unsatcore_and_fixed_condition(f,condition,empty_else_func,unsatcore_eq_score_func,category_call)

def order_equations_unsatcore_category(f: Formula, category_call=0) -> (Formula,int):
    condition = len([e for e in f.eq_list if e in f.unsat_core]) > 0
    unsatcore_eq_score_func = unsatcore_eq_score_random
    return order_equations_with_unsatcore_and_fixed_condition(f,condition,order_equations_category,unsatcore_eq_score_func,category_call)

def order_equations_unsatcore_first_n_iterations_category(f: Formula, category_call=0) -> (Formula,int):
    condition = f.current_total_split_eq_call==1  and len([e for e in f.eq_list if e in f.unsat_core]) > 0
    unsatcore_eq_score_func = unsatcore_eq_score_random
    return order_equations_with_unsatcore_and_fixed_condition(f,condition,order_equations_category,unsatcore_eq_score_func,category_call)


def order_equations_unsatcore_shortest(f: Formula, category_call=0) -> (Formula,int):
    condition = len([e for e in f.eq_list if e in f.unsat_core]) > 0
    unsatcore_eq_score_func=unsatcore_eq_score_shortest
    return order_equations_with_unsatcore_and_fixed_condition(f,condition,empty_else_func,unsatcore_eq_score_func,category_call)

def order_equations_unsatcore_shortest_first_n_iterations(f: Formula, category_call=0) -> (Formula,int):
    condition = f.current_total_split_eq_call==1 and len([e for e in f.eq_list if e in f.unsat_core]) > 0
    unsatcore_eq_score_func=unsatcore_eq_score_shortest
    return order_equations_with_unsatcore_and_fixed_condition(f,condition,empty_else_func,unsatcore_eq_score_func,category_call)

def order_equations_unsatcore_shortest_category(f: Formula, category_call=0) -> (Formula,int):
    condition = len([e for e in f.eq_list if e in f.unsat_core]) > 0
    unsatcore_eq_score_func=unsatcore_eq_score_shortest
    return order_equations_with_unsatcore_and_fixed_condition(f,condition,order_equations_category,unsatcore_eq_score_func,category_call)

def order_equations_unsatcore_shortest_first_n_iterations_category(f: Formula, category_call=0) -> (Formula,int):
    condition = f.current_total_split_eq_call==1  and len([e for e in f.eq_list if e in f.unsat_core]) > 0
    unsatcore_eq_score_func=unsatcore_eq_score_shortest
    return order_equations_with_unsatcore_and_fixed_condition(f,condition,order_equations_category,unsatcore_eq_score_func,category_call)


def order_equations_unsatcore_longest(f: Formula, category_call=0) -> (Formula,int):
    condition = len([e for e in f.eq_list if e in f.unsat_core]) > 0
    unsatcore_eq_score_func=unsatcore_eq_score_longest
    return order_equations_with_unsatcore_and_fixed_condition(f,condition,empty_else_func,unsatcore_eq_score_func,category_call)

def order_equations_unsatcore_longest_first_n_iterations(f: Formula, category_call=0) -> (Formula,int):
    condition = f.current_total_split_eq_call==1 and len([e for e in f.eq_list if e in f.unsat_core]) > 0
    unsatcore_eq_score_func = unsatcore_eq_score_longest
    return order_equations_with_unsatcore_and_fixed_condition(f,condition,empty_else_func,unsatcore_eq_score_func,category_call)

def order_equations_unsatcore_longest_category(f: Formula, category_call=0) -> (Formula,int):
    condition = len([e for e in f.eq_list if e in f.unsat_core]) > 0
    unsatcore_eq_score_func = unsatcore_eq_score_longest
    return order_equations_with_unsatcore_and_fixed_condition(f,condition,order_equations_category,unsatcore_eq_score_func,category_call)

def order_equations_unsatcore_longest_first_n_iterations_category(f: Formula, category_call=0) -> (Formula,int):
    condition = f.current_total_split_eq_call==1  and len([e for e in f.eq_list if e in f.unsat_core]) > 0
    unsatcore_eq_score_func = unsatcore_eq_score_longest
    return order_equations_with_unsatcore_and_fixed_condition(f,condition,order_equations_category,unsatcore_eq_score_func,category_call)



def unsatcore_eq_score_shortest(eq_length,total_formula_length) -> int:
    return eq_length
def unsatcore_eq_score_longest(eq_length,total_formula_length) -> int:
    return total_formula_length-eq_length
def unsatcore_eq_score_random(eq_length,total_formula_length) -> int:
    return random.randint(0,total_formula_length)

def order_equations_with_unsatcore_and_fixed_condition(f: Formula, condition,else_func,unsatcore_eq_score_func,category_call=0) -> (Formula,int):
    if f.unsat_core == []:
        return else_func(f, category_call)
    else:
        if condition:  # if there is at least one unsat core equation
            # Assign 0 to the unsat core equations and 1 to the others, this could be extended to assign different scores for unsatcores
            eq_list_with_unsat_core_score: List[Tuple[Equation, int]] = [(e, unsatcore_eq_score_func(e.term_length,f.total_eq_size)) if e in f.unsat_core else (e, f.total_eq_size) for e
                                                                         in f.eq_list]
            sorted_eq_list = sorted(eq_list_with_unsat_core_score, key=lambda x: x[1])

            print("sorted")
            for eq,score in sorted_eq_list:
                print(score,eq.eq_str)
            print("-"*10)

            return Formula([eq for eq, _ in sorted_eq_list]), category_call
        else:
            return else_func(f, category_call)



def empty_else_func(f: Formula, category_call=0) -> (Formula, int):
    return f, category_call

def order_equations_category_random(f: Formula, category_call=0) -> (Formula,int):
    return order_equation_with_category_and_fixed_condition(f, order_equations_random,category_call)


def order_equations_category_shortest(f: Formula, category_call=0) -> (Formula,int):
    return order_equation_with_category_and_fixed_condition(f, order_equations_shortest,category_call)


def order_equations_category_longest(f: Formula, category_call=0) -> (Formula,int):
    return order_equation_with_category_and_fixed_condition(f, order_equations_longest,category_call)

def order_equations_category_unsatcore(f: Formula, category_call=0) -> (Formula,int):
    return order_equation_with_category_and_fixed_condition(f, order_equations_unsatcore,category_call)


def order_equation_with_category_and_fixed_condition(f: Formula, order_func,category_call=0) -> (Formula,int):
    categoried_eq_list_with_score: List[Tuple[Equation, int]] = _category_formula_by_rules(f)
    sorted_eq_list = [eq for eq, _ in sorted(categoried_eq_list_with_score, key=lambda x: x[1])]

    # Check if the equation categories are only 5 and 6
    only_5_and_6: bool = all(n in [5, 6] for _, n in categoried_eq_list_with_score)

    if only_5_and_6 == True and len(categoried_eq_list_with_score) > 1:
        ordered_formula, category_call = order_func(Formula(sorted_eq_list), category_call)
        sorted_eq_list = ordered_formula.eq_list
    else:
        category_call += 1
        #sorted_eq_list = [eq for eq, _ in sorted(categoried_eq_list_with_score, key=lambda x: x[1])]

    return Formula(sorted_eq_list), category_call


order_equations_static_func_map={
                             "fixed": order_equations_fixed,
                             "shortest": order_equations_shortest,
                             "longest": order_equations_longest,
                             "random": order_equations_random,
                             "hybrid_fixed_random": order_equations_hybrid_fixed_random,
                             "category": order_equations_category,
                             "category_shortest": order_equations_category_shortest,
                             "category_longest": order_equations_category_longest,
                             "category_random": order_equations_category_random,
                             "hybrid_category_fixed_random": order_equations_hybrid_category_fixed_random,

                             "unsatcore": order_equations_unsatcore,
                             "unsatcore_first_n_iterations": order_equations_unsatcore_first_n_iterations,
                             "unsatcore_category": order_equations_unsatcore_category,
                             "unsatcore_first_n_iterations_category": order_equations_unsatcore_first_n_iterations_category,

                             "unsatcore_shortest": order_equations_unsatcore_shortest,
                             "unsatcore_shortest_first_n_iterations": order_equations_unsatcore_shortest_first_n_iterations,
                             "unsatcore_shortest_category": order_equations_unsatcore_shortest_category,
                             "unsatcore_shortest_first_n_iterations_category": order_equations_unsatcore_shortest_first_n_iterations_category,

                             "unsatcore_longest": order_equations_unsatcore_longest,
                             "unsatcore_longest_first_n_iterations": order_equations_unsatcore_longest_first_n_iterations,
                             "unsatcore_longest_category": order_equations_unsatcore_longest_category,
                             "unsatcore_longest_first_n_iterations_category": order_equations_unsatcore_longest_first_n_iterations_category,

                             "category_unsatcore": order_equations_category_unsatcore, # may not call unsatcores because category change eqs
}


def _get_unsatcore(file_name,parameters, equation_list):
    import os
    if "unsat_core_file" in parameters and parameters["unsat_core_file"] != "":
        unsat_core_file_path = parameters["unsat_core_file"]
    elif os.path.exists(file_name + ".unsatcore"):
        unsat_core_file_path = file_name + ".unsatcore"
    else:
        unsat_core_file_path = ""

    unsat_core: List[Equation] = Formula(equation_list,
                                         unsat_core_file=unsat_core_file_path).get_unsat_core()

    return unsat_core

def simplify_and_check_formula(f: Formula) -> Tuple[str, Formula]:
    # f.print_eq_list()
    f.simplify_eq_list()

    satisfiability = f.check_satisfiability_2()

    return satisfiability, f


def apply_rules_prefix(eq: Equation, f: Formula,fresh_variable_counter) -> Tuple[List[Tuple[Equation, Formula, str]],int]:
    # handle non-split rules
    eq_left_length=len(eq.left_terms)
    eq_right_length=len(eq.right_terms)

    # both sides are empty
    if eq.term_length == 0:
        children: List[Tuple[Equation, Formula, str]] = [(eq, f, " \" = \" ")]
    # left side is empty
    elif eq_left_length == 0 and eq_right_length > 0:
        children: List[Tuple[Equation, Formula, str]] = _left_side_empty(eq, f)
    # right side is empty
    elif eq_left_length > 0 and eq_right_length == 0:  # right side is empty
        children: List[Tuple[Equation, Formula, str]] = _left_side_empty(Equation(eq.right_terms, eq.left_terms), f)
    # both sides are not empty
    else:
        first_left_term = eq.left_terms[0]
        first_right_term = eq.right_terms[0]
        last_left_term = eq.left_terms[-1]
        last_right_term = eq.right_terms[-1]
        # \epsilon=\epsilon \wedge \phi case
        if eq.left_terms == eq.right_terms:
            children: List[Tuple[Equation, Formula, str]] = [(eq, f, " \" = \" ")]

        # match prefix
        elif first_left_term == first_right_term:
            local_eq=eq.deepcopy()
            local_eq.pop_same_prefix()
            children: List[Tuple[Equation, Formula, str]] = [
            (local_eq, Formula([local_eq] + f.eq_list), " t u= t v \wedge \phi")]
        # match suffix
        elif last_left_term == last_right_term:
            local_eq=eq.deepcopy()
            local_eq.pop_same_suffix()
            children: List[Tuple[Equation, Formula, str]] = [
            (local_eq, Formula([local_eq] + f.eq_list), " u t= v t \wedge \phi")]


        #
        # # match prefix terminal R_{6} in paper
        # elif first_left_term.value_type == Terminal and first_right_term.value_type == Terminal and first_left_term.value == first_right_term.value:
        #     eq.pop_same_prefix()
        #     children: List[Tuple[Equation, Formula, str]] = [
        #         (eq, Formula([eq] + f.eq_list), " a u= a v \wedge \phi")]
        # # match suffix terminal R_{6} in paper
        # elif last_left_term.value_type == Terminal and last_right_term.value_type == Terminal and last_left_term.value == last_right_term.value:
        #     eq.pop_same_suffix()
        #     children: List[Tuple[Equation, Formula, str]] = [
        #         (eq, Formula([eq] + f.eq_list), " u a= v a \wedge \phi")]

        # mismatch prefix terminal, R_{7} in paper
        elif first_left_term.value_type == Terminal and first_right_term.value_type == Terminal and first_left_term.value != first_right_term.value:
            eq.given_satisfiability = UNSAT
            children: List[Tuple[Equation, Formula, str]] = [
                (eq, Formula([eq] + f.eq_list), " a u = b v \wedge \phi")]

        # mistmatch suffix terminal, R_{7} in paper
        elif last_left_term.value_type == Terminal and last_right_term.value_type == Terminal and last_left_term.value != last_right_term.value:
            eq.given_satisfiability = UNSAT
            children: List[Tuple[Equation, Formula, str]] = [
                (eq, Formula([eq] + f.eq_list), "u a= v b \wedge \phi")]

        # split rules
        # left side is variable, right side is terminal, R_{8} prefix version in paper
        elif type(first_left_term.value) == Variable and type(first_right_term.value) == Terminal:
            rule_list: List[Callable] = [_left_variable_right_terminal_branch_1_prefix,
                                         _left_variable_right_terminal_branch_2_prefix]
            children,fresh_variable_counter= _get_split_children(eq, f, rule_list,fresh_variable_counter)

        # left side is terminal, right side is variable, R_{8} prefix version in paper
        elif type(first_left_term.value) == Terminal and type(first_right_term.value) == Variable:
            rule_list: List[Callable] = [_left_variable_right_terminal_branch_1_prefix,
                                         _left_variable_right_terminal_branch_2_prefix]
            children,fresh_variable_counter= _get_split_children(Equation(eq.right_terms, eq.left_terms), f, rule_list,fresh_variable_counter)

        # both side are differernt variables, R_{9} prefix version in paper
        elif type(first_left_term.value) == Variable and type(first_right_term.value) == Variable:
            rule_list: List[Callable] = [_two_variables_branch_3_prefix, _two_variables_branch_1_prefix, _two_variables_branch_2_prefix]
            children, fresh_variable_counter = _get_split_children(eq, f, rule_list,fresh_variable_counter)

        else:
            children: List[Tuple[Equation, Formula, str]] = []
            color_print(f"error: {eq.eq_str}", "red")

    return children,fresh_variable_counter


def apply_rules_suffix(eq: Equation, f: Formula, fresh_variable_counter) -> Tuple[
    List[Tuple[Equation, Formula, str]], int]:
    # handle non-split rules
    eq_left_length = len(eq.left_terms)
    eq_right_length = len(eq.right_terms)
    # both sides are empty
    if eq.term_length == 0:
        children: List[Tuple[Equation, Formula, str]] = [(eq, f, " \" = \" ")]
    # left side is empty
    elif eq_left_length == 0 and eq_right_length > 0:
        children: List[Tuple[Equation, Formula, str]] = _left_side_empty(eq, f)
    # right side is empty
    elif eq_left_length > 0 and eq_right_length == 0:  # right side is empty
        children: List[Tuple[Equation, Formula, str]] = _left_side_empty(Equation(eq.right_terms, eq.left_terms), f)
    # both sides are not empty
    else:
        first_left_term = eq.left_terms[0]
        first_right_term = eq.right_terms[0]
        last_left_term = eq.left_terms[-1]
        last_right_term = eq.right_terms[-1]
        # \epsilon=\epsilon \wedge \phi case
        if eq.left_terms == eq.right_terms:
            children: List[Tuple[Equation, Formula, str]] = [(eq, f, " \" = \" ")]

        #match suffix
        elif last_left_term == last_right_term:
            local_eq=eq.deepcopy()
            local_eq.pop_same_suffix()
            children: List[Tuple[Equation, Formula, str]] = [
                (local_eq, Formula([local_eq] + f.eq_list), " u t= v t \wedge \phi")]
        # match prefix
        elif first_left_term == first_right_term:
            local_eq=eq.deepcopy()
            local_eq.pop_same_prefix()
            children: List[Tuple[Equation, Formula, str]] = [
                (local_eq, Formula([local_eq] + f.eq_list), " t u= t v \wedge \phi")]


        # # match suffix terminal R_{6} in paper
        # elif last_left_term.value_type == Terminal and last_right_term.value_type == Terminal and last_left_term.value == last_right_term.value:
        #     eq.pop_same_suffix()
        #     children: List[Tuple[Equation, Formula, str]] = [
        #     (eq, Formula([eq] + f.eq_list), " u a= v a \wedge \phi")]
        #
        # # match prefix terminal R_{6} in paper
        # elif first_left_term.value_type == Terminal and first_right_term.value_type == Terminal and first_left_term.value == first_right_term.value:
        #     eq.pop_same_prefix()
        #     children: List[Tuple[Equation, Formula, str]] = [
        #     (eq, Formula([eq] + f.eq_list), " a u= a v \wedge \phi")]

        # mistmatch suffix terminal, R_{7} in paper
        elif last_left_term.value_type == Terminal and last_right_term.value_type == Terminal and last_left_term.value != last_right_term.value:
            eq.given_satisfiability = UNSAT
            children: List[Tuple[Equation, Formula, str]] = [
            (eq, Formula([eq] + f.eq_list), "u a= v b \wedge \phi")]


        # mismatch prefix terminal, R_{7} in paper
        elif first_left_term.value_type == Terminal and first_right_term.value_type == Terminal and first_left_term.value != first_right_term.value:
            eq.given_satisfiability = UNSAT
            children: List[Tuple[Equation, Formula, str]] = [
            (eq, Formula([eq] + f.eq_list), " a u = b v \wedge \phi")]
        

        # split rules
        # left side is variable, right side is terminal, R_{8} suffix version in paper
        elif type(last_left_term.value) == Variable and type(last_right_term.value) == Terminal:
            rule_list: List[Callable] = [_left_variable_right_terminal_branch_1_suffix,
                                         _left_variable_right_terminal_branch_2_suffix]
            children, fresh_variable_counter = _get_split_children(eq, f, rule_list, fresh_variable_counter)

            print("-Variable-Terminal-")
            print("parent",eq.eq_str_pretty)
            for c in children:
                print(f"{c[2]}, {c[0].eq_str_pretty}")

        # left side is terminal, right side is variable, R_{8} suffix version in paper
        elif type(last_left_term.value) == Terminal and type(last_right_term.value) == Variable:
            rule_list: List[Callable] = [_left_variable_right_terminal_branch_1_suffix,
                                         _left_variable_right_terminal_branch_2_suffix]
            children, fresh_variable_counter = _get_split_children(Equation(eq.right_terms, eq.left_terms), f,
                                                                   rule_list, fresh_variable_counter)

            print("-Terminal-Variable-")
            print("parent",eq.eq_str_pretty)
            for c in children:
                print(f"{c[2]}, {c[0].eq_str_pretty}")
        # both side are differernt variables, R_{9} suffix version in paper
        elif type(last_left_term.value) == Variable and type(last_right_term.value) == Variable:
            rule_list: List[Callable] = [_two_variables_branch_3_suffix, _two_variables_branch_1_suffix, _two_variables_branch_2_suffix]
            children, fresh_variable_counter = _get_split_children(eq, f, rule_list, fresh_variable_counter)

            print("--both side are differernt variables--")
            print("parent",eq.eq_str_pretty)
            for c in children:
                print(f"{c[2]}, {c[0].eq_str_pretty}")

        else:
            children: List[Tuple[Equation, Formula, str]] = []
            color_print(f"error: {eq.eq_str}", "red")

    return children, fresh_variable_counter

def _left_side_empty(eq: Equation, f: Formula) -> List[Tuple[Equation, Formula, str]]:
    '''
    Assume another side is empty.
    there are three conditions for one side: (1). terminals + variables (2). only terminals (3). only variables
    '''
    # (1) + (2): if there are any Terminal in the not_empty_side, then it is UNSAT
    not_empty_side = eq.right_terms
    if any(isinstance(term.value, Terminal) for term in not_empty_side):
        eq.given_satisfiability = UNSAT
        children: List[Tuple[Equation, Formula, str]] = [
            (eq, Formula([eq] + f.eq_list), " a u = \epsilon \wedge \phi")]
    # (3): if there are only Variables in the not_empty_side
    else:
        for variable_term in not_empty_side:
            f = _update_formula(f, variable_term, [])
        children: List[Tuple[Equation, Formula, str]] = [
            (eq, f, " XYZ = \epsilon \wedge \phi")]

    return children


def _get_split_children(eq: Equation, f: Formula, rule_list: List[Callable],fresh_variable_counter:int) -> Tuple[List[Tuple[Equation, Formula, str]],int]:
    children: List[Tuple[Equation, Formula, str]] = []
    for rule in rule_list:
        new_eq, new_formula, fresh_variable_counter, label_str = rule(eq, f, fresh_variable_counter)
        reconstructed_formula = Formula([new_eq] + new_formula.eq_list)
        child: Tuple[Equation, Formula, str] = (new_eq, reconstructed_formula, label_str)
        children.append(child)
    return children,fresh_variable_counter


def _category_formula_by_rules(f: Formula) -> List[Tuple[Equation, int]]:
    '''
    1: both sides are empty
    2: one side is empty
    3: mismatched prefix %or suffix terminal
    4: matched prefix terminal
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
            if eq.left_terms == eq.right_terms:  # this has been simplified, so will never reach here
                category_eq_list.append((eq, 1))


            # mismatch prefix terminal
            elif first_left_term.value_type == Terminal and first_right_term.value_type == Terminal and first_left_term.value != first_right_term.value:
                eq.given_satisfiability = UNSAT
                category_eq_list.append((eq, 3))
            # mistmatch suffix terminal
            elif last_left_term.value_type == Terminal and last_right_term.value_type == Terminal and last_left_term.value != last_right_term.value:
                eq.given_satisfiability = UNSAT
                category_eq_list.append((eq, 3))

            # match prefix terminal
            elif first_left_term.value_type == Terminal and first_right_term.value_type == Terminal and first_left_term.value == first_right_term.value:
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

def _left_variable_right_terminal_branch_1_prefix(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> \
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

def _left_variable_right_terminal_branch_1_suffix(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[Equation, Formula, int, str]:
    '''
    Equation:  [Terms] V1 = [Terms] a
    Assume V1 = ""
    Delete V1
    Obtain  [Terms] [V1/""]  = [Terms] [V1/""] a
    '''

    local_eq = eq.deepcopy()
    last_left_term: Term = local_eq.left_terms.pop(-1)
    last_right_term: Term = local_eq.right_terms.pop(-1)

    # define old and new term
    old_term: Term = last_left_term
    new_term: List[Term] = []

    label_str = f"{old_term.get_value_str} = \"\" "

    # update equation
    new_left_term_list = _update_term_list(old_term, new_term, local_eq.left_terms)
    new_right_term_list = _update_term_list(old_term, new_term, local_eq.right_terms) + [last_right_term]
    new_eq = Equation(new_left_term_list, new_right_term_list)

    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula, fresh_variable_counter, label_str

def _left_variable_right_terminal_branch_2_prefix(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> \
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

    label_str = f"{old_term.get_value_str} = {new_term[0].get_value_str} {new_term[1].get_value_str}"

    # update equation
    new_left_term_list = [fresh_variable_term] + _update_term_list(old_term, new_term, local_eq.left_terms)
    new_right_term_list = _update_term_list(old_term, new_term, local_eq.right_terms)
    new_eq = Equation(new_left_term_list, new_right_term_list)

    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula, fresh_variable_counter, label_str
def _left_variable_right_terminal_branch_2_suffix(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[Equation, Formula, int, str]:
    '''
    Equation: [Terms] V1 = [Terms] a
    Assume V1 = V1'a
    Replace V1 with V1'a
    Obtain [Terms] [V1/V1'a] V1' = [Terms] [V1/V1'a]
    '''
    local_eq = eq.deepcopy()
    last_left_term: Term = local_eq.left_terms.pop(-1)
    last_right_term: Term = local_eq.right_terms.pop(-1)

    # create fresh variable
    (fresh_variable_term, fresh_variable_counter) = _create_fresh_variables(fresh_variable_counter)
    # define old and new term
    old_term: Term = last_left_term
    new_term: List[Term] = [fresh_variable_term, last_right_term]

    label_str = f"{old_term.get_value_str} = {new_term[0].get_value_str} {new_term[1].get_value_str}"

    # update equation
    new_left_term_list = _update_term_list(old_term, new_term, local_eq.left_terms) + [fresh_variable_term]
    new_right_term_list = _update_term_list(old_term, new_term, local_eq.right_terms)
    new_eq = Equation(new_left_term_list, new_right_term_list)

    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula, fresh_variable_counter, label_str


def _two_variables_branch_1_prefix(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[
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

    label_str = f"{old_term.get_value_str} = {new_term[0].get_value_str} {new_term[1].get_value_str}"

    # update equation
    new_left_term_list = [fresh_variable_term] + _update_term_list(old_term, new_term, local_eq.left_terms)
    new_right_term_list = _update_term_list(old_term, new_term, local_eq.right_terms)
    new_eq = Equation(new_left_term_list, new_right_term_list)

    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula, fresh_variable_counter, label_str

def _two_variables_branch_1_suffix(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[Equation, Formula, int, str]:
    '''
    Equation: [Terms] V1 = [Terms] V2
    Assume |V1| > |V2|
    Replace V1 with V1'V2
    Obtain  [Terms] [V1/V1'V2] V1' = [Terms] [V1/V1'V2]
    '''
    local_eq = eq.deepcopy()
    last_left_term: Term = local_eq.left_terms.pop(-1)
    last_right_term: Term = local_eq.right_terms.pop(-1)

    # create fresh variable
    (fresh_variable_term, fresh_variable_counter) = _create_fresh_variables(fresh_variable_counter)
    # define old and new term
    new_term: List[Term] = [fresh_variable_term, last_right_term]
    old_term: Term = last_left_term

    label_str = f"{old_term.get_value_str} = {new_term[0].get_value_str} {new_term[1].get_value_str}"

    # update equation
    new_left_term_list = _update_term_list(old_term, new_term, local_eq.left_terms) + [fresh_variable_term]
    new_right_term_list = _update_term_list(old_term, new_term, local_eq.right_terms)
    new_eq = Equation(new_left_term_list, new_right_term_list)

    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula, fresh_variable_counter, label_str


def _two_variables_branch_2_prefix(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[
    Equation, Formula, int, str]:
    '''
    Equation: V1 [Terms] = V2 [Terms]
    Assume |V1| < |V2|
    Replace V2 with V1V2'
    Obtain [Terms] [V2/V1V2'] = V2' [Terms] [V2/V1V2']
    '''
    return _two_variables_branch_1_prefix(Equation(eq.right_terms, eq.left_terms), current_formula, fresh_variable_counter)

def _two_variables_branch_2_suffix(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[
    Equation, Formula, int, str]:
    '''
    Equation: [Terms] V1 = [Terms] V2
    Assume |V1| < |V2|
    Replace V2 with V2'V1
    Obtain [Terms] [V2/V2'V1] = V2' [Terms] [V2/V2'V1]
    '''
    return _two_variables_branch_1_suffix(Equation(eq.right_terms, eq.left_terms), current_formula, fresh_variable_counter)


def _two_variables_branch_3_prefix(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[
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

    label_str = f"{old_term.get_value_str} = {new_term[0].get_value_str}"

    # update equation
    new_eq = Equation(_update_term_list(old_term, new_term, local_eq.left_terms),
                      _update_term_list(old_term, new_term, local_eq.right_terms))
    # update formula
    new_formula: Formula = _update_formula(current_formula, old_term, new_term)

    return new_eq, new_formula, fresh_variable_counter, label_str

def _two_variables_branch_3_suffix(eq: Equation, current_formula: Formula, fresh_variable_counter: int) -> Tuple[Equation, Formula, int, str]:
    '''
    Equation: [Terms] V1 = [Terms] V2
    Assume |V1| = |V2|
    Replace V1 with V2
    Obtain [Terms] [V1/V2] = [Terms] [V1/V2]
    '''
    local_eq = eq.deepcopy()
    last_left_term: Term = local_eq.left_terms.pop(-1)
    last_right_term: Term = local_eq.right_terms.pop(-1)

    # define old and new term
    old_term: Term = last_left_term
    new_term: List[Term] = [last_right_term]

    label_str = f"{old_term.get_value_str} = {new_term[0].get_value_str}"

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


def _get_global_info(eq_list: List[Equation]):
    global_info = {}
    variable_global_occurrences = {}
    terminal_global_occurrences = {}
    for eq in eq_list:
        for term in eq.term_list:

            if term.value_type == Variable:
                if term.value not in variable_global_occurrences:
                    variable_global_occurrences[term.value] = 0
                variable_global_occurrences[term.value] += 1
            elif term.value_type == Terminal:
                if term.value not in terminal_global_occurrences:
                    terminal_global_occurrences[term.value] = 0
                terminal_global_occurrences[term.value] += 1

    global_info["variable_global_occurrences"] = variable_global_occurrences
    global_info["terminal_global_occurrences"] = terminal_global_occurrences

    # for eq in eq_list:
    #     print(eq.eq_str)
    # for k, v in global_info.items():
    #     print(k)
    #     print(v)
    return global_info


def order_branches_fixed(children: List[Tuple[Equation, Formula]]) -> List[Tuple[Equation, Formula]]:
    return children

def order_branches_random(children: List[Tuple[Equation, Formula]]) -> List[Tuple[Equation, Formula]]:
    random.shuffle(children)
    return children

def order_branches_hybrid_fixed_random(children: List[Tuple[Equation, Formula]]) -> List[
    Tuple[Equation, Formula]]:
    probability = random.random()
    if probability < HYBRID_BRANCH_RATE:
        return order_branches_fixed(children)
    else:
        return order_branches_random(children)