## Build Environment

We first build a [Apptainer](https://apptainer.org/docs/admin/main/index.html) image (similar to docker) to serve as our environment.
Apptainer installation instructions can be found [here](https://apptainer.org/docs/admin/main/installation.html).

If you don't use containers, you can also follow the commands in the `eval_recipe.def` file mentioned below to install everything.



Go to the `container` folder, build an image by:

    apptainer build eval_image.sif eval_recipe.def


#### Path Configuration
Adapt the paths in the file `config.ini` to make it run correctly.


## Reproduce Instruction


#### Prepare Data for Training

Go to the folder `demo/script` and run:

    sh prepare_train_data_and_configurations.sh

This will perform the following tasks:

1. The `.eq` files represent word equation problems, and each corresponding `.unsatcore` file is one minimal unsatisfiable subset.

   For each pair of `.eq` and `.unsatcore` files found in:
   - `demo/data/train_data/divided_1/UNSAT`
   - `demo/data/train_data/valid_data/UNSAT`

   Generate the labeled training data. The output is stored in `train.zip`.


2. Draw graph representations of the training data and store them in `graph_1.zip`. Each `.graph.json` file represents a single training example.  
To improve loading performance, these graphs are also serialized in pickle format and stored in `pkl.zip`.


3. A set of training parameters and all necessary information are generated and stored in `Model/configurations/config_0.json`.


#### Train a Model

First, go to the `demo/script` folder and run the mlflow server by 

    sh run_server.sh

This will create a `mlruns` folder in the root path of this project to monitor and store all information while training. The terminal to run this command should not be closed until the training finished.

Then, we initialize the training by

    sh train_initial.sh

This 



#### Evaluation
In the project root folder, run:

    apptainer exec --nv container/eval_image.sif python3 src/example/branch_eval.py

This will evaluate the problems in the folder `benchmarks_and_experimental_results/example/01_track_eval/ALL`. The experimental results will be stored in files `results.csv` and `summary.csv` under `benchmarks_and_experimental_results/example/01_track_eval`.

By default, we run the five configurations (fixed, random, GNN, GNN+fixed, GNN+random) of DragonLi. They are named this:EliminateVariablesRecursive-config_i where $i\in \{1,2,3,4,5\}$ representing fixed, random, GNN, GNN+fixed, and GNN+random, respectively in `results.csv` and `summary.csv`.

Meanwhile, this code also runs Z3 on the same problems. Thus, you can find a comparison and consistency check in both files `results.csv` and `summary.csv`.







### Note
This demo provides a small set of word equations for both training and validation (the same set of problems is used for both). For the fully trained models used in the paper's evaluation, see the experimental_results_tables/eval_data_GNN/*/*/model/*/artifacts directory