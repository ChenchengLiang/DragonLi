import json
import os
import shutil

import mlflow
import torch
from src.solver.Constants import project_folder
from src.solver.independent_utils import color_print

import dgl

def restore_model_structure(model_path):
    from src.solver.models.Models import GraphClassifierLightning
    from src.solver.rank_task_models.train_utils import get_gnn_and_classifier
    from src.solver.models.Models import GraphClassifier

    # Define the same model architecture
    config_list = os.path.basename(model_path).split("_")
    graph_type = config_list[2] + "_" + config_list[3]
    model_type = config_list[4].split(".")[0]
    label_size = int(config_list[1])
    configuration_file = f"{project_folder}/Models/configuration_model_{label_size}_{graph_type}_{model_type}.json"

    with open(configuration_file) as f:
        model_parameters = json.load(f)

    gnn_model, classifier_2 = get_gnn_and_classifier(model_parameters)
    original_model = GraphClassifierLightning(shared_gnn=gnn_model, classifier=classifier_2,
                                              model_parameters=model_parameters)
    return original_model


def load_model_torch_script(model_path) :
    from src.solver.models.Models import GraphClassifier

    color_print(text=f"load model from {model_path}", color="green")
    loaded_model = torch.load(model_path, map_location=torch.device('cpu'))
    loaded_model.eval()  # Set the model to evaluation mode

    # Save the state dictionary
    torch.save(loaded_model.state_dict(), model_path.replace(".pth", "_state_dict.pth"))

    original_model = restore_model_structure(model_path)

    original_model.load_state_dict(torch.load(model_path.replace(".pth", "_state_dict.pth")))
    original_model.eval()

    # Wrap the model to remove Trainer dependency
    simplified_model = GraphClassifier(shared_gnn=original_model.shared_gnn,
                                                 classifier=original_model.classifier)

    # Optimize model with TorchScript
    # scripted_model = torch.jit.script(simplified_model)
    # scripted_model.save(model_path.replace(".pth", ".pt"))
    #
    # # Load the optimized model
    # optimized_model = torch.jit.load(model_path.replace(".pth", ".pt"))

    return simplified_model



def load_model(model_path) :
    color_print(text=f"load model from {model_path}", color="green")
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
    color_print(f"dgl version: {dgl.__version__}", "green")
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


    model_name = f"model_{parameters['label_size']}_{parameters['graph_type']}_{parameters['model_type']}.pth"
    local_dir = f"{project_folder}/Models"
    best_model_path_local=f"{local_dir}/{model_name}"
    torch.save(best_model, best_model_path_local)


    mlflow_dir = f"{project_folder}/mlruns/{parameters['experiment_id']}/{parameters['run_id']}/artifacts"
    best_model_path_mlflow=f"{mlflow_dir}/{model_name}"
    torch.save(best_model, best_model_path_mlflow)
    color_print(f"Save best model to {best_model_path_mlflow}\n", "green")



def update_config_file(configuration_file,train_config):
    with open(configuration_file, 'w') as f:
        train_config["device"] = str(train_config["device"])  # change tensor to string so can dump it to json
        json.dump(train_config, f, indent=4)
