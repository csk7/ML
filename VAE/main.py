import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.ToTensor()
    ])
path = './data/mnist/'
trainset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Visualize input data
dataiter = iter(trainloader)
images = next(dataiter)

num_samples = 25
sample_images = [images[0][i,0] for i in range(num_samples)]
#Show all in a 5x5 grid
plt.figure(figsize=(10,10))
for i in range(num_samples):
    plt.subplot(5,5,i+1)
    plt.imshow(sample_images[i].numpy().squeeze(), cmap='gray_r')

path = os.path.join(os.getcwd(), 'img')
os.makedirs(path, exist_ok=True)
plt.savefig(os.path.join(path, 'mnist.png'))
plt.close()

#Creating a VAE class
class VAE(nn.Module):
    def __init__(self, input_size = 784, hidden_size = 256, latent_size = 200):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, latent_size),
            nn.LeakyReLU(0.2),
        )

        self.meanLayer = nn.Linear(latent_size, 2)
        self.logVarLayer = nn.Linear(latent_size, 2)
        
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mean = self.meanLayer(x)
        logVar = self.logVarLayer(x)
        return mean, logVar
        
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def reparameterize(self, mean, logVar):
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std).to(device)
        return mean + eps * std

    def forward(self, x):
        mean, logVar = self.encode(x)
        z = self.reparameterize(mean, logVar)
        x_res = self.decode(z)
        return x_res, mean, logVar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(x, x_res, mean, logVar):
    BCE = F.binary_cross_entropy(x_res, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logVar - mean.pow(2) - logVar.exp())
    return BCE + KLD

batchSize = 32
inputDim = 784
def train(model, trainloader, optimizer, loss_function, epochs=10):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0.0
        for batch_idx, (data, _) in enumerate(trainloader):
            x = data.view(batchSize, inputDim).to(device)

            optimizer.zero_grad()
            x_result, mean, logVar = model(x)
            loss = loss_function(x = x, x_res = x_result, mean = mean, logVar = logVar)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epcoh : {epoch} ; Loss : {overall_loss}")
    return overall_loss

train(model = model, trainloader= trainloader, optimizer= optimizer, 
    loss_function=loss_function)

def generate_digit(mean, variance):
    z_sample = torch.tensor([mean, variance], dtype = torch.float).to(device)
    model.eval()
    x_decoded = model.decode(z_sample)
    digit = x_decoded.detach().cpu().reshape(28,28)
    plt.open()
    plt.imshow(digit, cmap ='gray')
    plt.axis('off')
    plt.savefig('./img/mnist_reconstructed.png')
    plt.close()

generate_digit(0.0, 1.0)