#!/bin/bash

timeout_in_second=10

cd ..

if [ -f webapp/answer.txt ]; then
    rm webapp/answer.txt
fi



eq_file="webapp/user-input.eq"
order_equations_method=$1
rank_task="1"
model_path="Models/model_2_graph_1_GCNSplit.pth"

timeout $timeout_in_second apptainer exec --nv container/eval_image.sif python3 src/process_benchmarks/main_parameter.py $eq_file "fixed" --graph_type "graph_1" --gnn_model_path $model_path --gnn_task "rank_task" --rank_task $rank_task --termination_condition "termination_condition_0" --order_equations_method $order_equations_method --algorithm "SplitEquations"
status=$?

# Check if the command timed out (exit status 124)
if [ $status -eq 124 ]; then
    echo "Timeout $timeout_in_second s" > webapp/answer.txt
fi


