
from src.solver.Constants import project_folder,bench_folder
from src.process_benchmarks.utils import smt_to_eq_one_folder
def main():

    folder=bench_folder+"/smtlib/test/QF_S/2019-Jiang/slog"
    smt_to_eq_one_folder(folder)

if __name__ == '__main__':
    main()
