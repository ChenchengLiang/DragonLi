
from src.solver.Constants import project_folder,bench_folder
from src.process_benchmarks.utils import clean_eq_files, get_clean_statistics



def main():
    benchmark="smtlib-test/small"
    folder=bench_folder+"/"+benchmark
    clean_eq_files(folder)


    get_clean_statistics(benchmark, [folder])



if __name__ == '__main__':
    main()