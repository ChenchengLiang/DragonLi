from src.process_benchmarks.utils import run_on_one_problem
from src.solver.Constants import project_folder,bench_folder
from src.solver.independent_utils import strip_file_name_suffix,time_it
import glob
import os
import shutil
import subprocess
def main():
    smt_file_folder=bench_folder+"/kaluzaWoorpje/smt2"
    eq_file_folder=bench_folder+"/kaluzaWoorpje/eq"
    ostrich_output_file=bench_folder+"/temp/output.eq"
    solver="ostrich_export"

    update_ostrich()

    for smt_file in glob.glob(smt_file_folder+"/*.smt2"):
        if os.path.exists(ostrich_output_file):
            os.remove(ostrich_output_file)

        smt_file_path=os.path.join(smt_file_folder,smt_file)
        result_dict = run_on_one_problem(file_path=smt_file_path, parameters_list=["-timeout=0"], solver=solver)
        file_name=strip_file_name_suffix(os.path.basename(smt_file_path))
        shutil.copy(ostrich_output_file,eq_file_folder+f"/{file_name}.eq")

@time_it
def update_ostrich():
    run_shell_command = ["sh", "/home/cheli243/Desktop/CodeToGit/update_ostrich_export.sh"]
    completed_process = subprocess.run(run_shell_command, capture_output=True, text=True, shell=False)


if __name__ == '__main__':
    main()
