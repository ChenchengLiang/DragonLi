## DragonLi

DragonLi is a solver for string equations, which aims to find solutions to equations involving strings rather than numerical values. These equations consist of sequences of characters, and the solver works to determine how unknown strings can be combined to satisfy given relationships.

DragonLi solves string equations by recursively applying a set of calculus rules to simplify the first terms on both sides of an equation. These rules reduce the complexity of the equation by making assumptions about the lengths of the terms. This iterative process constructs a proof tree, where each node represents a string equation, and each edge represents a branch resulting from the application of a calculus rule.

A key feature of DragonLi is its use of deep learning-based heuristics to guide the branching order while constructing the proof tree. This significantly boosts the solving efficiency. Specifically, each branching point is represented as a graph, which is used to train a Graph Neural Network (GNN). The trained GNN model is then employed to guide the branching process, making DragonLi highly efficient in finding solutions.

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

#### Notice
You may need to change the paths in file `config.ini` to make it run correctly.
The train, valid, test data are exactly the same problems. So in this example as long as we train enough epochs, it will solve all problems.


## Citation of This Work

```bibtex
@InProceedings{10.1007/978-3-031-78709-6_14,
  author    = "Abdulla, Parosh Aziz
               and Atig, Mohamed Faouzi
               and Cailler, Julie
               and Liang, Chencheng
               and R{\"u}mmer, Philipp",
  editor    = "Akshay, S.
               and Niemetz, Aina
               and Sankaranarayanan, Sriram",
  title     = "Guiding Word Equation Solving Using Graph Neural Networks",
  booktitle = "Automated Technology for Verification and Analysis",
  year      = "2025",
  publisher = "Springer Nature Switzerland",
  address   = "Cham",
  pages     = "279--301",
  isbn      = "978-3-031-78709-6"
}
