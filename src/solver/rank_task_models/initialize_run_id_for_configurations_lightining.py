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
import time
import random
from typing import Dict, List, Any
from src.solver.Constants import project_folder, checkpoint_folder, bench_folder
import mlflow
import argparse
import json
import datetime
import subprocess
from src.solver.independent_utils import color_print, get_folders
import signal
import torch.nn as nn
from src.solver.models.train_util import (initialize_model_structure, load_train_and_valid_dataset, training_phase,
                                          validation_phase,
                                          initialize_train_objects, log_and_save_best_model, save_checkpoint,
                                          data_loader_2, get_data_distribution,
                                          training_phase_without_loader, validation_phase_without_loader,
                                          log_metrics_with_lock)

from src.solver.models.utils import device_info, update_config_file
from src.solver.rank_task_models.train_utils import initialize_model, initialize_model_lightning, MyPrintingCallback, \
    get_dm
from src.solver.rank_task_models.Dataset import read_dataset_from_zip, DGLDataModuleRank1, DGLDataModuleRank0
from pytorch_lightning.loggers import MLFlowLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.profiler import profile, ProfilerActivity

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
    else:
        print("No configuration file")

    train_config["configuration_file"]=configuration_file


    train_in_parallel(train_config)







def train_in_parallel(parameters):
    ############### Initialize training parameters ################

    benchmark_folder_list = get_folders(bench_folder + "/" + parameters["benchmark"])
    train_folder_list = [folder for folder in benchmark_folder_list if "divided" in folder]

    if "divided_1" in train_folder_list:
        train_data_folder_epoch_map: Dict[str, int] = {folder: 0 for folder in train_folder_list}
    else:
        train_data_folder_epoch_map: Dict[str, int] = {parameters["benchmark"]: 0}
    parameters["train_data_folder_epoch_map"] = train_data_folder_epoch_map

    parameters["current_train_folder"] = parameters["benchmark"] + "/" + \
                                         list(parameters["train_data_folder_epoch_map"].items())[
                                             0][0]
    parameters["benchmark_folder"] = parameters["benchmark"]

    today = datetime.date.today().strftime("%Y-%m-%d")
    experiment_name = today + "-" + parameters["benchmark"]
    parameters["experiment_name"] = experiment_name

    parameters["model_save_path"] = os.path.join(project_folder, "Models",
                                                            f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")


    logger = MLFlowLogger(experiment_name=parameters["experiment_name"],tracking_uri='http://127.0.0.1:5000')

    color_print(text=f"use the existing run id {logger.run_id}", color="yellow")
    parameters["run_id"] = logger.run_id
    parameters["experiment_id"] = logger.experiment_id



    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{project_folder}/mlruns/{parameters['experiment_id']}/{parameters['run_id']}/artifacts",
        filename=f"model_{parameters['label_size']}_{parameters['graph_type']}_{parameters['model_type']}",#'best-checkpoint',
        save_top_k=1,
        save_last=True,
        enable_version_counter=False,
        verbose=True,
        monitor='best_val_accuracy',  # or another metric
        mode='min'
    )

    profiler = "simple"

    dm = get_dm(parameters)


    devices_list = [i for i in range(0, torch.cuda.device_count())]
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        #strategy="ddp",  # ddp
        # profiler=profiler,
        logger=logger,
        accelerator="gpu",
        devices=devices_list,
        min_epochs=1,
        max_epochs=1,
        precision=32,
        callbacks=[MyPrintingCallback(),checkpoint_callback],
        enable_progress_bar=False,
        enable_checkpointing=True,
    )


    model = initialize_model_lightning(parameters)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
                 record_shapes=True) as prof:
        trainer.fit(model, dm)
        #trainer.validate(model, dm)
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))



    # if "run_id" in parameters and parameters["run_id"] is not None:
    #     check_point_model_path = f"{checkpoint_folder}/{parameters['run_id']}_model_checkpoint.ckpt"
    #     trainer.save_checkpoint(check_point_model_path)
    #     color_print(f"save checkpoint to {check_point_model_path}", "green")
    #




if __name__ == '__main__':
    main()
