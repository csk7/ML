import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor()
    ])

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'data','mnist')
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

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'img')
os.makedirs(path, exist_ok=True)
plt.savefig(os.path.join(path, 'mnist.png'))
plt.close()

#Creating a VAE class
class VAE(nn.Module):
    def __init__(self, input_size = 784, hidden_size = 256, latent_size = 200, mode = 'visualize'):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, latent_size),
            nn.LeakyReLU(0.2),
        )

        if mode == 'visualize':
            self.meanLayer = nn.Linear(latent_size, 2)
            self.logVarLayer = nn.Linear(latent_size, 2)
        else:
            self.meanLayer = nn.Linear(latent_size, latent_size)
            self.logVarLayer = nn.Linear(latent_size, latent_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(2 if mode == 'visualize' else latent_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
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


model = VAE(mode='visualize').to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(x, x_res, mean, logVar):
    BCE = F.binary_cross_entropy(x_res, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logVar - mean.pow(2) - logVar.exp())
    return BCE + KLD

batchSize = 32
inputDim = 784

def save_model(model, optimizer, epoch, loss, path='model_checkpoint.pth'):
    """
    Save model and optimizer state to a file
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model saved to {path}")

def load_model(model, optimizer=None, path='model_checkpoint.pth'):
    """
    Load model and optionally optimizer state from a file
    Returns: model, optimizer, epoch, loss
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    print(f"Model loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    return model, optimizer, epoch, loss

def train(model, trainloader, optimizer, loss_function, epochs=10, save_path='model_checkpoint.pth'):
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
        
        avg_loss = overall_loss/(batchSize*(batch_idx+1))
        print(f"Epoch: {epoch} ; Loss: {avg_loss:.4f}")
        
        # Save model after each epoch
        save_model(model, optimizer, epoch, avg_loss, save_path)
    
    return overall_loss

# To train and save:
train(model, trainloader, optimizer, loss_function, epochs=20, save_path='vae_model.pth')

def generate_digit(mean, variance, num_samples = 1, latent_size = 200, file_name='mnist_generated.png'):
    std = torch.sqrt(torch.tensor(variance, dtype = torch.float).to(device))
    normalDistribution = torch.distributions.Normal(mean, std)
    z_sample = normalDistribution.sample((num_samples, latent_size))
    z_sample = z_sample.to(device)
    
    model.eval()
    x_decoded = model.decode(z_sample)
    model.train()
    
    digit = x_decoded.detach().cpu().reshape(28,28)
    plt.imshow(digit, cmap ='gray')
    plt.axis('off')
    
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, 'img')
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, file_name)
    plt.savefig(save_path)
    plt.close()

# To load and generate:
loaded_model = VAE().to(device)
loaded_model, _, _, _ = load_model(loaded_model, path='vae_model.pth')
generate_digit(0.0, 1.0, num_samples=1, latent_size=2, file_name='generated_after_loading.png')

def plot_latent_space(model, scale = 1.0, n=25, digit_size=28, figsize=15):
    figure = np.zeros((digit_size*n, digit_size*n))
    val_y = np.linspace(-scale, scale, n)
    val_x = np.linspace(-scale, scale, n)
    for i, y_i in enumerate(val_y):
        for j, x_j in enumerate(val_x):
            z = torch.tensor([x_j, y_i], dtype=torch.float).unsqueeze(0).to(device)
            x_decoded = model.decode(z)
            digit = x_decoded.detach().cpu().reshape(28,28)
            figure[i*digit_size:(i+1)*digit_size, j*digit_size:(j+1)*digit_size] = digit
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, 'img')
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, 'latent_space.png')
    plt.savefig(save_path)
    plt.close()

plot_latent_space(model)

    