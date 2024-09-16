
from src.solver.Constants import project_folder,bench_folder
from src.process_benchmarks.utils import clean_eq_files

def main():
    folder=bench_folder+"/smtlib-test/small"
    clean_eq_files(folder)



if __name__ == '__main__':
    main()