import configparser
import os
import sys

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import mlflow
import argparse
import json
import datetime
import subprocess
from src.solver.independent_utils import color_print, time_it
import signal
from src.solver.models.train_util import train_one_model, train_multiple_models_separately, check_run_exists, \
    load_checkpoint, training_phase, validation_phase, log_and_save_best_model, data_loader_2, \
    save_checkpoint, training_phase_without_loader, validation_phase_without_loader, get_data_distribution
import torch.nn as nn
from src.solver.models.utils import device_info, update_config_file
from src.solver.rank_task_models.train_utils import initialize_model, MyPrintingCallback, initialize_model_lightning, \
    get_gnn_and_classifier, get_dm
from src.solver.rank_task_models.Dataset import read_dataset_from_zip, DGLDataModuleRank1, DGLDataModuleRank0
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.solver.Constants import checkpoint_folder, project_folder
from src.solver.models.Models import GraphClassifierLightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger


def main():
    # parse argument
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    arg_parser.add_argument('--configuration_file', type=str, default=None,
                            help='path to configuration json file ')
    args = arg_parser.parse_args()
    # Accessing the arguments
    configuration_file = args.configuration_file
    if configuration_file is not None:
        # read json file
        with open(configuration_file) as f:
            train_config = json.load(f)

    # Clear the cache
    torch.cuda.empty_cache()

    #with mlflow.start_run(run_id=train_config["run_id"]) as mlflow_run:
    color_print(text=f"use the existing run id {train_config['run_id']}", color="yellow")
    # pick one unfinished train
    train_config["current_train_folder"] = None
    for key, value in train_config["train_data_folder_epoch_map"].items():
        if value < train_config["num_epochs"]:
            train_config["current_train_folder"] = train_config["benchmark"] + "/" + key
            break

    if train_config["current_train_folder"] is None:
        color_print(text=f"all training folders are done", color="green")
    else:
        color_print(text=f"current training folder:{train_config['current_train_folder']}", color="yellow")
        train_a_model(train_config)



@time_it
def train_a_model(parameters):
    device_info()

    dm = get_dm(parameters)


    logger = MLFlowLogger(experiment_name=parameters["experiment_name"], run_id=parameters["run_id"],tracking_uri=f"http://127.0.0.1:{parameters['port']}")
    profiler = "simple"

    #load from checkpoint
    #check_point_model_path = f"{checkpoint_folder}/{parameters['run_id']}_model_checkpoint.ckpt" #load from checkpoint folder
    check_point_model_path = f"{project_folder}/mlruns/{parameters['experiment_id']}/{parameters['run_id']}/artifacts/last.ckpt"  # load from mlflow folder last
    color_print(f"load check point from {check_point_model_path}", color="yellow")

    gnn_model, classifier_2 = get_gnn_and_classifier(parameters)
    model = GraphClassifierLightning.load_from_checkpoint(checkpoint_path=check_point_model_path,
                                                          shared_gnn=gnn_model,
                                                          classifier=classifier_2,
                                                          model_parameters=parameters)

    # save for best model in mlflow
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{project_folder}/mlruns/{parameters['experiment_id']}/{parameters['run_id']}/artifacts",
        filename=f"model_{parameters['label_size']}_{parameters['graph_type']}_{parameters['model_type']}",
        save_top_k=1,
        save_last=True,
        enable_version_counter=False,
        verbose=True,
        monitor='best_val_accuracy',  # or another metric
        mode='min'
    )

    print(f"Resuming training from epoch {model.total_epoch}, "
          f"last best validation accuracy {model.best_val_accuracy}, best epoch {model.best_epoch}  ")
    devices_list = [i for i in range(0, torch.cuda.device_count())]

    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(accelerator="gpu",
                         devices=devices_list,
                         callbacks=[MyPrintingCallback(),checkpoint_callback],
                         logger=logger,
                         precision=32,
                         min_epochs=parameters["train_step"],
                         max_epochs=parameters["train_step"],
                         enable_progress_bar=False,
                         enable_checkpointing=True,
                         )

    # Resume training
    trainer.fit(model, datamodule=dm)
    # trainer.validate(model, dm)






if __name__ == '__main__':
    main()
