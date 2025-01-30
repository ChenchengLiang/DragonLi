#!/bin/sh

 
cd ../..


apptainer exec --nv container/uppmax_word_equation_image.sif python3 src/train_data_collection/generate_train_data_from_unsatcores.py train_data


