import os
import torch
from utils.global_variables import GLOBAL_INFO

def save_model(model, optimizer, epoch, loss, path='model_checkpoint.pth'):
    """
    Save model and optimizer state to a file
    params model: model
    params optimizer: optimizer
    params epoch: epoch
    params loss: loss
    params path: path to save the model
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(GLOBAL_INFO.script_dir, path))
    print(f"Model saved to {path}")


def load_model(model, optimizer=None, path='model_checkpoint.pth'):
    """
    Load model and optionally optimizer state from a file
    params model: model
    params optimizer: optimizer
    params path: path to load the model
    Returns: model, optimizer, epoch, loss
    """
    checkpoint = torch.load(os.path.join(GLOBAL_INFO.script_dir, path))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    print(f"Model loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    return model, optimizer, epoch, loss