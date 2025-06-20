import os
import torch
from model import VQVAE
from data import testloader
from utils import load_model, save_image_grid

def test(model, optimizer):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    

    model, optimizer, _, _ = load_model(model, optimizer, path=os.path.join(os.path.dirname(__file__), 'model','model_checkpoint.pth'))
    model.eval()
    with torch.no_grad():
        # Get a single random batch
        data, _ = next(iter(testloader))
        data = data.to(device)
        output, _, _, _, _ = model(data)
        
        # Ensure the output directory exists
        path = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(path, exist_ok=True)
        
        # Save input images
        save_image_grid(data, os.path.join(path, 'input_images.png'), nrow=8, title='Input Images')
        
        # Save reconstructed images
        save_image_grid(output, os.path.join(path, 'reconstructed_images.png'), nrow=8, title='Reconstructed Images')
        
        # Create side by side comparison (first 8 images)
        comparison = torch.cat([data[:8], output[:8]])
        save_image_grid(comparison, os.path.join(path, 'comparison.png'), nrow=8, title='Input (top) vs Reconstructed (bottom)')
