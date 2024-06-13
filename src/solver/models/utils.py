import json
import os

import mlflow
import torch
from src.solver.Constants import project_folder
from src.solver.independent_utils import color_print

import dgl
def load_model(model_path) :
    #color_print(text=f"load model from {model_path}", color="green")
    loaded_model = torch.load(model_path,map_location=torch.device('cpu'))
    #print(loaded_model.keys())
    loaded_model.eval()  # Set the model to evaluation mode
    return loaded_model


def load_model_from_mlflow(experiment_id,run_id):
    model_path = project_folder + "/mlruns/" + experiment_id + "/" + run_id + "/artifacts/model/data/model.pth"
    return load_model(model_path)



def device_info():
    print("-" * 10)
    color_print(f"torch.cuda.is_available: {torch.cuda.is_available()}", "green")
    color_print(f"torch vesion: {torch.__version__}", "green")
    color_print(f"dgl backend: {dgl.backend.backend_name}", "green")
    print("-" * 10)


def squeeze_labels(pred, labels):
    # Convert labels to float for BCELoss
    labels = labels.float()
    pred_squeezed = torch.squeeze(pred)
    if len(labels) == 1:
        pred_final = torch.unsqueeze(pred_squeezed, 0)
    else:
        pred_final = pred_squeezed
    return pred_final, labels


def save_model_local_and_mlflow(parameters,model_index,best_model):
    best_model_path = parameters["model_save_path"].replace(".pth", "_" + parameters["run_id"] + ".pth").replace(
        "model_", f"model_{model_index}_")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        torch.save(best_model, best_model_path)
        mlflow.log_artifact(best_model_path)
        os.remove(best_model_path)

    best_model_path_save_locally = parameters["model_save_path"].replace(
        "model_", f"model_{model_index}_")
    if os.path.exists(best_model_path_save_locally):
        os.remove(best_model_path_save_locally)
    torch.save(best_model, best_model_path_save_locally)


def update_config_file(configuration_file,train_config):
    with open(configuration_file, 'w') as f:
        train_config["device"] = str(train_config["device"])  # change tensor to string so can dump it to json
        json.dump(train_config, f, indent=4)
