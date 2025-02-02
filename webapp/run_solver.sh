#!/bin/bash

timeout_in_second=$4

cd ..

if [ -f webapp/answer.txt ]; then
    rm webapp/answer.txt
fi



eq_file="webapp/user-input.eq"
order_equations_method=$1
rank_task=$(( $2 - 1 )) # -1 to match the code task enumerate from 0 to 2
benchmark=$3
# map task number to label size
if [ "$rank_task" -eq "2" ]; then
    label_size=50
else
    label_size=2
fi
model_path="experimental_results_tables/eval_data_GNN/$benchmark/task_$2/model/artifacts/model_"$label_size"_graph_1_GCNSplit.pth"




# Check if benchmark equals "C"
if [ "$benchmark" = "C" ]; then # if it is C, then No GNN is called
    timeout $timeout_in_second apptainer exec --nv container/eval_image.sif python3 src/process_benchmarks/main_parameter.py $eq_file "fixed" --graph_type "graph_1" --termination_condition "termination_condition_0" --order_equations_method "category" --algorithm "SplitEquations"
    status=$?
else
case "$order_equations_method" in
    "category")
        timeout $timeout_in_second apptainer exec --nv container/eval_image.sif python3 src/process_benchmarks/main_parameter.py $eq_file "fixed" --graph_type "graph_1" --termination_condition "termination_condition_0" --order_equations_method "category" --algorithm "SplitEquations"
        status=$?
        ;;
    *)
        timeout $timeout_in_second apptainer exec --nv container/eval_image.sif python3 src/process_benchmarks/main_parameter.py $eq_file "fixed" --graph_type "graph_1" --gnn_model_path $model_path --gnn_task "rank_task" --rank_task "$rank_task" --termination_condition "termination_condition_0" --order_equations_method $order_equations_method --algorithm "SplitEquations"
	status=$?
        ;;
esac
fi







# Check if the command timed out (exit status 124)
if [ $status -eq 124 ]; then
    echo "Timeout $timeout_in_second s" > webapp/answer.txt
fi


# verify input from front end
#echo "$3" >> webapp/answer.txt
#echo " $test, $benchmark " >> webapp/answer.txt


