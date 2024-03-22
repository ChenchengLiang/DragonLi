#!/bin/bash
rm uppmax_word_equation_image.sif

apptainer build uppmax_word_equation_image.sif uppmax_word_equation_recipe.def

echo "send to uppmax"
send_to_uppmax_folder uppmax_word_equation_image.sif /home/cheli243/container



echo "done" 
