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

