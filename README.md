## Build Environment

We first build [Apptainer](https://apptainer.org/docs/admin/main/index.html) images (similar to docker) to serve as our environment.
Apptainer installation instructions can be found [here](https://apptainer.org/docs/admin/main/installation.html).

If you don't use containers, you can also follow the commands in .def files mentioned below to install everything.


#### For training
Go to the container folder, build an image by:

    apptainer build train_image.sif alvis_word_equation_recipe-A100.def

#### For evaluation
Go to the container folder, build an image by:

    apptainer build eval_image.sif uppmax_word_equation_recipe.def

#### Path configuration
Change the paths in file `config.ini` to make it run correctly.


## Reproduce Instruction


#### Train a Model

Go to the project root folder and run:

    apptainer exec --nv container/train_image.sif python3 src/example/branch_train.py

This will perform the following tasks:

1. Generate training data (split points) in the file `benchmarks_and_experimental_results/example/01_track_train/divided_1/train.zip` for training and `benchmarks_and_experimental_results/example/01_track_train/valid_data/train.zip` for validation.
2. Generate graph files `graph_1.zip` for each split point and store them in the folders `divided_1` and `valid_data` under `benchmarks_and_experimental_results/example/01_track_train` for training and validation respectively.
3. Train two GNN models for 2- and 3-category classification for Rule 7 and 8, respectively. These will be stored in `Models/model_2_graph_1_GCNSplit.pth` and `Models/model_3_graph_1_GCNSplit.pth` respectively.



#### Evaluation
In the project root folder, run:

    apptainer exec --nv container/eval_image.sif python3 src/example/branch_eval.py

This will evaluate the problems in the folder `benchmarks_and_experimental_results/example/01_track_eval/ALL`. The experimental results will be stored in files `results.csv` and `summary.csv` under `benchmarks_and_experimental_results/example/01_track_eval`.

By default, we run the five configurations (fixed, random, GNN, GNN+fixed, GNN+random) of DragonLi. They are named this:EliminateVariablesRecursive-config_i where $i\in \{1,2,3,4,5\}$ representing fixed, random, GNN, GNN+fixed, and GNN+random, respectively in `results.csv` and `summary.csv`.

Meanwhile, this code also runs Z3 on the same problems. Thus, you can find a comparison and consistency check in both files `results.csv` and `summary.csv`.







### Note
This demo is used to show the training process, so it only contain a few word equations for traaining and validation and they are the same set of problems. 
The trained models that used in the evaluation shown in the paper can be found in the folder experimental_results_tables