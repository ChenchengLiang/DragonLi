import json
import os
import shutil
import sys

import mlflow
import torch
from src.solver.Constants import project_folder


from src.solver.independent_utils import color_print
import torch.onnx
import dgl

import onnxruntime as ort

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

    # # Optimize model with TorchScript using tracing
    # dummy_input = get_example_input(model_path)
    # traced_model = torch.jit.trace(simplified_model, dummy_input)
    # traced_model_path = model_path.replace(".pth", ".pt")
    # traced_model.save(traced_model_path)
    #
    # # Load the optimized model
    # optimized_model = torch.jit.load(traced_model_path)

    # Optimize model with TorchScript
    scripted_model = torch.jit.script(simplified_model)
    scripted_model.save(model_path.replace(".pth", ".pt"))

    # Load the optimized model
    optimized_model = torch.jit.load(model_path.replace(".pth", ".pt"))

    return optimized_model


def get_example_input(model_path):
    # Create a dummy input tensor matching the input shape
    from src.solver.algorithms.split_equation_utils import _get_global_info
    from src.solver.DataTypes import Equation, Formula, Term, Variable, Terminal
    from src.solver.utils import graph_func_map
    from src.solver.algorithms import graph_to_gnn_format
    from src.solver.models.Dataset import get_one_dgl_graph
    eq_1 = Equation([Term(Variable("X"))], [Term(Terminal("a"))])
    eq_2 = Equation([Term(Variable("X"))], [Term(Variable("YYY"))])
    eq_list = [eq_1, eq_2]
    f = Formula(eq_list)
    global_info = _get_global_info(f.eq_list)

    config_list = os.path.basename(model_path).split("_")
    graph_type = config_list[2] + "_" + config_list[3]
    graph_func = graph_func_map[graph_type]
    G_list_dgl = []
    for index, eq in enumerate(f.eq_list):
        split_eq_nodes, split_eq_edges = graph_func(eq.left_terms, eq.right_terms, global_info)
        graph_dict = graph_to_gnn_format(split_eq_nodes, split_eq_edges)
        dgl_graph, _ = get_one_dgl_graph(graph_dict)
        G_list_dgl.append(dgl_graph)

    input_eq_graph_list_dgl = []
    for index, g_dgl in enumerate(G_list_dgl):
        one_eq_data_dgl = [g_dgl] + G_list_dgl
        input_eq_graph_list_dgl.append(one_eq_data_dgl)

    batch_eqs = [dgl.batch(g_G_dgl) for g_G_dgl in input_eq_graph_list_dgl]
    return batch_eqs
def load_model_onnx(model_path) :
    color_print(text=f"load model from {model_path}", color="green")
    loaded_model = torch.load(model_path,map_location=torch.device('cpu'))
    #print(loaded_model.keys())
    loaded_model.eval()  # Set the model to evaluation mode

    dummy_input = get_example_input(model_path)

    # Specify the path for the ONNX model
    onnx_model_path = model_path.replace(".pth", ".onnx")

    # Export the model
    torch.onnx.export(loaded_model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      onnx_model_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    # Load the ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)
    sys.exit(0)

    return ort_session

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
