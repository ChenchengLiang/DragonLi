import torch








def load_model(model_path) :
    loaded_model = torch.load(model_path)
    loaded_model.eval()  # Set the model to evaluation mode
    return loaded_model