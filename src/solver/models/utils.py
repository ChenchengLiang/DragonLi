import torch
from src.solver.Constants import project_folder
from src.solver.independent_utils import color_print
from src.solver.models.Models import GraphClassifier
import argparse
import json
def load_model(model_path) :
    #color_print(text=f"load model from {model_path}", color="green")
    loaded_model = torch.load(model_path,map_location=torch.device('cpu'))
    #print(loaded_model.keys())
    loaded_model.eval()  # Set the model to evaluation mode
    return loaded_model


def load_model_from_mlflow(experiment_id,run_id):
    model_path = project_folder + "/mlruns/" + experiment_id + "/" + run_id + "/artifacts/model/data/model.pth"
    return load_model(model_path)

