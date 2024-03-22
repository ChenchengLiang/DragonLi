import torch
from src.solver.Constants import project_folder
def load_model(model_path,map_location=torch.device('cpu')) :
    loaded_model = torch.load(model_path,map_location=map_location)
    loaded_model.eval()  # Set the model to evaluation mode
    return loaded_model

def load_model_from_mlflow(experiment_id,run_id):
    model_path = project_folder + "/mlruns/" + experiment_id + "/" + run_id + "/artifacts/model/data/model.pth"
    return load_model(model_path)