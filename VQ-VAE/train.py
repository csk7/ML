import torch
from utils import save_model

##Training##
def train(model, trainloader, optimizer, loss_function, num_epochs=10, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            input, _ = data[0].to(device), data[1].to(device)
            output, _, _, commitment_loss, codebook_loss = model(input)
            loss, loss_dict = loss_function(output, input, commitment_loss, codebook_loss)
            loss.backward()
            optimizer.step()
            if ((i+1) % 100 == 0):
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}')
                print(f"Reconstruction loss: {loss_dict['reconstruction_loss'].item():.4f}, Commitment loss: {loss_dict['commitment_loss'].item():.4f}, \
                    Codebook loss: {loss_dict['codebook_loss'].item():.4f}")
        save_model(model, optimizer, epoch, loss.item())


