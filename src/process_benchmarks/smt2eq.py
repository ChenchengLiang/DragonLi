from src.process_benchmarks.parser_utils import parse_smtlib_to_simple_format
def main():
    # Example SMT-LIB input
    smtlib_input = """
(set-logic QF_S)
(declare-const x String)
(declare-const y String)
(declare-const z String)
(assert (= (str.++ a "hello") "testhello"))
(assert (= (str.++ x y) (str.++ "abc" z) ) )
(assert (= (str.++  A A A "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")  (str.++  A "aaaaaaaaaaaaaaaaaaa" A "aaaaaa" C "aaa") ))
(check-sat)
(get-model)
    """
    input_file="/home/cheli243/Downloads/wordbenchmarks-comparison_start-models-kaluza/models/kaluza/kaluzaWoorpje/1022.corecstrs.readable.smt2"
    with open(input_file,'r') as smtlib_input:
        parsed_format = parse_smtlib_to_simple_format(smtlib_input.read())
        print(parsed_format)


if __name__ == '__main__':
    main()