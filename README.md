# boosting-string-equation-solving-by-GNNs

Woking pipeline:

1. Uppmax: sh word_equation_submit_generate_tracks.sh #gnerate and divide tracks


2. Uppmax: sh word_equation_submit_multiple_eval.sh #get .answer and summary


3. Uppmax: sh word_equation_submit_collect_answers.sh


4. Uppmax+Local+Alvis: send answered track to and Alvis


5. Alvis: sh word_equation_submit_generate_train_data.sh #generate train data (.eq)


6. Alvis: sh word_equation_submit_draw_graphs_for_train_data.sh #draw graphs for train data (.json)


7. Alvis: sh word_equation_submit_store_dataset_to_pickle.sh


8. Alvis: sh word_equation_submit_multiple_train.sh #train


9. Alvis+Local: select and send back good models


10. Uppmax: sh word_equation_submit_multiple_eval.sh #evaluate track with good models


11. Local: merge_summary_folders.py
Local: summary_solutions.py

