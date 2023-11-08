# boosting-string-equation-solving-by-GNNs

Woking pipeline:

Local: generate track, send to Uppmax and Alvis

Uppmax: get .answer and summary

Local/Alvis: generate train data (.eq + .json)

Alvis: train, select and send back good model

Uppmax: evaluate tack with good model

Local: summary

