import torch
import utils

def save_ckpt(path, model, optimizer=None, best_score=None):
    state = {
        "model_state": model.state_dict(), 
        "model_architecture": model
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if best_score is not None:
        state["best_score"] = best_score
    torch.save(state, path)


def load_ckpt(path):
    checkpoint = torch.load(path)

    model = checkpoint['model_architecture']
    
    model.load_state_dict(checkpoint['model_state'])
    
    optimizer_state = checkpoint.get('optimizer_state', None)
    best_score = checkpoint.get('best_score', None)
    
    print("load checkpoint from ", path)
    return model, optimizer_state, best_score
