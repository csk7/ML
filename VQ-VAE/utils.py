import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def loss_function(output, target, commitment_loss, codebook_loss):
    """
    Loss function for VQ-VAE
    params output: output
    params target: target
    params commitment_loss: commitment loss
    params codebook_loss: codebook loss
    returns loss: loss
    """
    recontruction_loss = F.mse_loss(output, target, reduction='sum')/target.shape[0]
    total_loss = commitment_loss + codebook_loss + recontruction_loss
    loss_dict = {'reconstruction_loss': recontruction_loss, 'commitment_loss': commitment_loss, 'codebook_loss': codebook_loss}
    return total_loss, loss_dict

def save_model(model, optimizer, epoch, loss, path='model_checkpoint.pth'):
    """
    Save model and optimizer state to a file
    params model: model
    params optimizer: optimizer
    params epoch: epoch
    params loss: loss
    params path: path to save the model
    """
    path = os.path.join(os.path.dirname(__file__), 'model','model_checkpoint.pth')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model saved to {path}")

def load_model(model, optimizer, path=os.path.join(os.path.dirname(__file__), 'model','model_checkpoint.pth')):
    """
    Load model and optimizer state from a file
    params model: model
    params optimizer: optimizer
    params path: path to load the model
    returns model, optimizer, epoch, loss
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    print(f"Model loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    return model, optimizer, epoch, loss

def save_image_grid(tensor, filename, nrow=8, title=None):
    """
    Save a grid of images from a PyTorch tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (N, C, H, W)
        filename (str): Path to save the image
        nrow (int): Number of images per row in the grid
        title (str, optional): Title for the plot
    """
    # Convert tensor to numpy and denormalize if needed
    if tensor.max() > 1.0 or tensor.min() < 0.0:
        # Assuming the tensor is in [-1, 1], normalize to [0, 1]
        tensor = (tensor + 1) / 2.0
    
    # Create a grid of images
    grid = vutils.make_grid(tensor, nrow=nrow, padding=2, normalize=False)
    
    # Convert to numpy and change dimension order for matplotlib
    np_grid = grid.cpu().numpy().transpose((1, 2, 0))
    
    # Create plot
    plt.figure(figsize=(15, 15))
    plt.imshow(np_grid)
    plt.axis('off')
    if title:
        plt.title(title)
    
    # Save the figure
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()