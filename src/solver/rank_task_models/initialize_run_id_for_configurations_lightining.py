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
from src.solver.rank_task_models.train_utils import initialize_model, initialize_model_lightning, MyPrintingCallback
from src.solver.rank_task_models.Dataset import read_dataset_from_zip, DGLDataModule
from pytorch_lightning.loggers import MLFlowLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only


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


    #train_config, mlflow_ui_process = start_mlflow(train_config)

    #model, dm, profiler, devices_list = initialize_trainer_parameters(train_config)

    train_in_parallel(train_config)

    #terminate_mlflow(mlflow_ui_process, configuration_file, train_config)



def start_mlflow(train_config):
    color_print("--- initiate_run_id_for_a_configuration ---", color="green")
    device_info()
    train_config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_folder = config['Path']['woorpje_benchmarks']
    today = datetime.date.today().strftime("%Y-%m-%d")
    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui'], preexec_fn=os.setpgrp)

    benchmark_folder_list = get_folders(benchmark_folder + "/" + train_config["benchmark"])
    train_folder_list = [folder for folder in benchmark_folder_list if "divided" in folder]
    experiment_name = today + "-" + train_config["benchmark"]
    if "divided_1" in train_folder_list:
        train_data_folder_epoch_map: Dict[str, int] = {folder: 0 for folder in train_folder_list}
    else:
        train_data_folder_epoch_map: Dict[str, int] = {train_config["benchmark"]: 0}
    train_config["train_data_folder_epoch_map"] = train_data_folder_epoch_map
    train_config["experiment_name"] = experiment_name

    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    torch.autograd.set_detect_anomaly(True)

    mlflow.start_run()
    train_config["run_id"] = mlflow.active_run().info.run_id
    train_config["experiment_id"] = mlflow.active_run().info.experiment_id
    train_config["current_train_folder"] = train_config["benchmark"] + "/" + \
                                           list(train_config["train_data_folder_epoch_map"].items())[0][0]

    train_config["model_save_path"] = os.path.join(project_folder, "Models",
                                                   f"model_{train_config['graph_type']}_{train_config['model_type']}.pth")

    train_config["benchmark_folder"] = train_config["benchmark"]
    mlflow.log_params(train_config)
    return train_config, mlflow_ui_process



def terminate_mlflow(mlflow_ui_process, configuration_file, train_config):
    mlflow.end_run()
    mlflow_ui_process.terminate()
    os.killpg(os.getpgid(mlflow_ui_process.pid), signal.SIGTERM)

    update_config_file(configuration_file, train_config)
    print("done")




def initialize_trainer_parameters(parameters):
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
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui'], preexec_fn=os.setpgrp)

    if mlflow.active_run() is not None:
        mlflow.end_run()
        mlflow.start_run()


    model = initialize_model_lightning(parameters)

    #logger = MLFlowLogger(experiment_name=parameters["experiment_name"], run_id=parameters["run_id"])
    profiler = "simple"

    dm = DGLDataModule(parameters, parameters["batch_size"], num_workers=1)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=checkpoint_folder,  # Path where the checkpoints will be saved
    #     filename=f"{parameters['run_id']}_model_checkpoint",  # Naming convention using epoch and validation loss
    #     save_top_k=1,
    #     save_last=True,  # Additionally, always save the last completed epoch checkpoint
    #     # save_top_k=1,  # Save the top 3 models according to monitored value
    #     # monitor='best_val_accuracy',  # Metric to monitor for improvement
    #     # mode='min',  # `min` mode saves the model if the monitored value decreases
    #     verbose=True
    # )

    devices_list = [i for i in range(0, torch.cuda.device_count())]

    return model, dm, profiler, devices_list


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
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #mlflow_ui_process = subprocess.Popen(['mlflow', 'ui'], preexec_fn=os.setpgrp)

    parameters["model_save_path"] = os.path.join(project_folder, "Models",
                                                            f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")

    mlflow.start_run()

    parameters["run_id"] = mlflow.active_run().info.run_id
    parameters["experiment_id"] = mlflow.active_run().info.experiment_id

    model = initialize_model_lightning(parameters)

    # logger = MLFlowLogger(experiment_name=parameters["experiment_name"], run_id=parameters["run_id"])
    profiler = "simple"

    dm = DGLDataModule(parameters, parameters["batch_size"], num_workers=1)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=checkpoint_folder,  # Path where the checkpoints will be saved
    #     filename=f"{parameters['run_id']}_model_checkpoint",  # Naming convention using epoch and validation loss
    #     save_top_k=1,
    #     save_last=True,  # Additionally, always save the last completed epoch checkpoint
    #     # save_top_k=1,  # Save the top 3 models according to monitored value
    #     # monitor='best_val_accuracy',  # Metric to monitor for improvement
    #     # mode='min',  # `min` mode saves the model if the monitored value decreases
    #     verbose=True
    # )

    devices_list = [i for i in range(0, torch.cuda.device_count())]


    trainer = pl.Trainer(
        #strategy="ddp",  # ddp
        # profiler=profiler,
        accelerator="gpu",
        devices=[0],#devices_list,
        min_epochs=1,
        max_epochs=1,
        # precision=16,
        callbacks=[MyPrintingCallback()],
        enable_progress_bar=False,
        # enable_checkpointing=True,
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)

    check_point_model_path = f"{checkpoint_folder}/{parameters['run_id']}_model_checkpoint.ckpt"
    trainer.save_checkpoint(check_point_model_path)

    print("-" * 10, "train finished", "-" * 10)

    #pid = mlflow_ui_process.pid
    mlflow.end_run()
    #mlflow_ui_process.terminate()
    #os.killpg(os.getpgid(pid), signal.SIGTERM)

    update_config_file(parameters["configuration_file"], parameters)
    print("done")



if __name__ == '__main__':
    main()
