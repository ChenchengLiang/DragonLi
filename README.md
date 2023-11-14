# boosting-string-equation-solving-by-GNNs

Woking pipeline:

1. Uppmax: sh word_equation_generate_tracks.sh #gnerate and divide tracks


2. Uppmax: sh word_equation_submit_multiple_eval.sh #get .answer and summary


3. Uppmax: sh word_equation_collect_answers.sh


4. Uppmax+Local+Alvis: send answered track to and Alvis


5. Alvis: sh word_equation_generate_train_data.sh #generate train data (.eq + .json)


6. Alvis: sh word_equation_submit_multiple_train.sh #train, select and send back good model


7. Alvis+Local: select and send back good model


8. Uppmax: sh word_equation_submit_multiple_eval.sh #evaluate tack with good model



9. Local: merge_summary_folders.py
Local: summary_solutions.py

