#!/bin/bash

timeout_in_second=10

cd ..

if [ -f webapp/answer.txt ]; then
    rm webapp/answer.txt
fi



eq_file="webapp/user-input.eq"
order_equations_method=$1
rank_task=$(( $2 - 1 )) # -1 to match the code task enumerate from 0 to 2
model_path="Models/model_2_graph_1_GCNSplit.pth"



case "$order_equations_method" in
    "category")
        echo "Command for a"
        timeout $timeout_in_second apptainer exec --nv container/eval_image.sif python3 src/process_benchmarks/main_parameter.py $eq_file "fixed" --graph_type "graph_1" --termination_condition "termination_condition_0" --order_equations_method $order_equations_method --algorithm "SplitEquations"
        status=$?
        ;;
    "hybrid_category_gnn_random")
        echo "Command for b"
        timeout $timeout_in_second apptainer exec --nv container/eval_image.sif python3 src/process_benchmarks/main_parameter.py $eq_file "fixed" --graph_type "graph_1" --gnn_model_path $model_path --gnn_task "rank_task" --rank_task "$rank_task" --termination_condition "termination_condition_0" --order_equations_method $order_equations_method --algorithm "SplitEquations"
	status=$?
        ;;
    "category_gnn_first_n_iterations")
        timeout $timeout_in_second apptainer exec --nv container/eval_image.sif python3 src/process_benchmarks/main_parameter.py $eq_file "fixed" --graph_type "graph_1" --gnn_model_path $model_path --gnn_task "rank_task" --rank_task "$rank_task" --termination_condition "termination_condition_0" --order_equations_method $order_equations_method --algorithm "SplitEquations"
	status=$?
        ;;
    *)
        echo "No matching command for: $x"
        ;;
esac





# Check if the command timed out (exit status 124)
if [ $status -eq 124 ]; then
    echo "Timeout $timeout_in_second s" > webapp/answer.txt
fi



#echo "$2" >> webapp/answer.txt
#echo "$rank_task" >> webapp/answer.txt


