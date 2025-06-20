import argparse
import torch
import torch.optim as optim
from model import VQVAE
from data import trainloader
from utils import loss_function
from train import train
from test import test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQ-VAE training and testing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode to run the script in')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10

    model = VQVAE(hidden_channels=64, in_channels=3, embedding_size=64, n_layers=8)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    #Training
    if (args.mode == 'train'):
        train(model, trainloader, optimizer, loss_function, num_epochs=num_epochs, device=device)
    
    #Testing
    if (args.mode == 'test'):
        test(model = model, optimizer = optimizer)
    