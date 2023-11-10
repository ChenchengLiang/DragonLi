from pysmt.smtlib.parser import SmtLibParser
from src.process_benchmarks.parser_utils import parse_smtlib_to_simple_format
from io import StringIO
def main():

    input_file="/home/cheli243/Downloads/wordbenchmarks-comparison_start-models-kaluza/models/kaluza/kaluzaWoorpje/1022.corecstrs.readable.smt2"

    with open(input_file, 'r') as smtlib_input:

        parser = SmtLibParser()
        script = parser.get_script(StringIO(smtlib_input.read()))
        for cmd in script:
            print(cmd.name,cmd.args)
            for a in cmd.args:
                print(a)


if __name__ == '__main__':
    main()