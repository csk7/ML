import os
import torch
import torchvision
import torchvision.transforms as transforms
from utils.global_variables import GLOBAL_INFO
###Data loading###
transform = transforms.Compose([
    transforms.ToTensor()
])

path = os.path.join(GLOBAL_INFO.script_dir, 'data', 'mnist')
trainset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)