import torch
from src.solver.Constants import project_folder
from src.solver.independent_utils import color_print
from src.solver.models.Models import GraphClassifier
import argparse
import json
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