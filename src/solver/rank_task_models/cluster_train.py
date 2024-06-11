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
    update_config_file, load_checkpoint, training_phase, validation_phase, log_and_save_best_model, data_loader_2, \
    save_checkpoint, training_phase_without_loader, validation_phase_without_loader, get_data_distribution
import torch.nn as nn
from src.solver.models.utils import device_info
from src.solver.rank_task_models.train_utils import initialize_model
from src.solver.rank_task_models.Dataset import read_dataset_from_zip
from torch.utils.data import DataLoader


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

    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui'], preexec_fn=os.setpgrp)
    mlflow.set_experiment(train_config["experiment_name"])
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    torch.autograd.set_detect_anomaly(True)
    if check_run_exists(train_config["run_id"]):  # continuos training
        with mlflow.start_run(run_id=train_config["run_id"]) as mlflow_run:
            color_print(text=f"use the existing run id {mlflow_run.info.run_id}", color="yellow")
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
                train_config = train_a_model(train_config, mlflow_run)
                train_config["train_data_folder_epoch_map"][os.path.basename(train_config["current_train_folder"])] += \
                train_config["train_step"]
                # update configuration file
                update_config_file(configuration_file, train_config)

    mlflow_ui_process.terminate()
    os.killpg(os.getpgid(mlflow_ui_process.pid), signal.SIGTERM)
    print("done")

@time_it
def train_a_model(train_config, mlflow_run):
    device_info()
    if torch.cuda.is_available():
        train_config["device"] = torch.device("cuda")
        print(f"exists {torch.cuda.device_count()} GPUs")
    else:
        device = torch.device("cpu")
    train_config["run_id"] = mlflow_run.info.run_id
    train_config["experiment_id"] = mlflow_run.info.experiment_id
    if len(mlflow_run.data.params) == 0:
        mlflow.log_params(train_config)

    train_continuously(train_config)

    return train_config

@time_it
def train_continuously(parameters):
    # read data
    get_data_statistics = False
    train_dataset = read_dataset_from_zip(parameters, os.path.basename(parameters["current_train_folder"]),get_data_statistics=get_data_statistics)
    valid_dataset = read_dataset_from_zip(parameters, "valid_data",get_data_statistics=get_data_statistics)
    dataset = {"train": train_dataset, "valid": valid_dataset}
    train_dataloader, valid_dataloader = data_loader_2(dataset, parameters)
    get_data_distribution(dataset, parameters)

    # initialize model
    model = initialize_model(parameters)

    # initialize parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    loss_function = nn.CrossEntropyLoss()
    classification_type = "multi_classification"
    best_model=None
    epoch_info_log = ""

    # read from checkpoint
    check_point_model_path = parameters["run_id"] + f"_checkpoint_model_{parameters['label_size']}.pth"
    model, optimizer, start_epoch, best_valid_loss, best_valid_accuracy = load_checkpoint(model, optimizer, parameters,
                                                                                          filename=check_point_model_path)

    # training
    for index, epoch in enumerate(range(start_epoch, parameters["num_epochs"] + 1)):


        # Training Phase
        model, avg_train_loss = training_phase(model, train_dataloader, loss_function, optimizer, parameters)

        # Validation Phase
        model, avg_valid_loss, valid_accuracy = validation_phase(model, valid_dataloader, loss_function,
                                                                 classification_type, parameters)

        # Save based on specified criterion
        best_model, best_valid_loss, best_valid_accuracy, epoch_info_log = log_and_save_best_model(parameters, epoch,
                                                                                                   best_model, model,
                                                                                                   "multi_class",
                                                                                                   parameters[
                                                                                                       "label_size"],
                                                                                                   avg_train_loss,
                                                                                                   avg_valid_loss,
                                                                                                   valid_accuracy,
                                                                                                   best_valid_loss,
                                                                                                   best_valid_accuracy,
                                                                                                   epoch_info_log,
                                                                                                   index)
        if index == parameters["train_step"] - 1 or epoch >= parameters["num_epochs"] or best_valid_accuracy == 1.0:
            save_checkpoint(model, optimizer, epoch, best_valid_loss, best_valid_accuracy, parameters,
                            filename=check_point_model_path)
            break

    # Return the trained model and the best metrics
    best_metrics = {f"best_valid_loss_multi_class_{parameters['label_size']}": best_valid_loss,
                    f"best_valid_accuracy_multi_class_{parameters['label_size']}": best_valid_accuracy}

    mlflow.log_metrics(best_metrics)


if __name__ == '__main__':
    main()
