# boosting-string-equation-solving-by-GNNs

Woking pipeline:

1. Uppmax/Local: sh word_equation_submit_generate_tracks.sh #gnerate and divide tracks
For not generated data, divide eval data: divide_train_and_test_data.py


2. Uppmax: sh word_equation_submit_multiple_eval.sh #get .answer and summary


3. Uppmax: 
sh word_equation_collect_answers.sh
#sh word_equation_submit_collect_answers.sh


4. Uppmax+Local+Alvis: 
python3 divide_train_and_valid_data.py
send answered track to and Alvis


5. Alvis: sh word_equation_submit_generate_train_data_and_draw_graphs.sh
#sh word_equation_submit_generate_train_data.sh #generate train data (.eq)
#sh word_equation_submit_draw_graphs_for_train_data.sh #draw graphs for train data (.json)


6. Alvis: sh word_equation_submit_store_dataset_to_pickle.sh
set benchmark in python code src/solver/models/store_dataset_to_pickle.py

7. Alvis:
sh word_equation_submit_initialize_configurations.sh
sh word_equation_submit_multiple_train_continuously.sh #train
#sh word_equation_submit_multiple_train.sh #train


8. Alvis+Local+Uppmax: select and send back good models
    Local: cd /usr/local/bin; run_server_8888;
    Local: browser visit http://0.0.0.0:8888 select good models and find it in /home/cheli243/Desktop/CodeToGit/string-equation-solver/cluster-mlruns
    change name: such as "model_2_graph_2_GINSplit_8ac6ff7a3b614c5d95e3492216142366.pth" to "model_2_graph_2_GINSplit.pth"
    send good models to Uppmax path /home/cheli243/boosting-string-equation-solving-by-GNNs/Models

9. Local: configurate evaluation parameters in generate_eval_configurations.py
    Uppmax: sh word_equation_submit_multiple_eval.sh #evaluate track with good models
    Uppmax: sh send_summary_back.sh


10. sh merge_and_summary.sh $benchmark_name # this will call merge_summary_folders.py and summary_solutions.py

