from utils.nn_model_tools import save_model
from utils.global_variables import GLOBAL_INFO
def train(model, trainloader, optimizer, loss_function, epochs=10, save_path='model_checkpoint.pth'):
    """
    Train the model
    params model: model
    params trainloader: trainloader from the data
    params optimizer: optimizer
    params loss_function: loss function
    params epochs: number of epochs
    params save_path: path to save the model
    """
    model.train()
    for epoch in range(epochs):
        overall_loss = 0.0
        for batch_idx, (data, _) in enumerate(trainloader):
            x = data.view(GLOBAL_INFO.batchSize, GLOBAL_INFO.inputDim).to(GLOBAL_INFO.device)

            optimizer.zero_grad()
            x_result, mean, logVar = model(x)
            loss = loss_function(x = x, x_res = x_result, mean = mean, logVar = logVar)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_loss = overall_loss/(GLOBAL_INFO.batchSize*(batch_idx+1))
        print(f"Epoch: {epoch} ; Loss: {avg_loss:.4f}")
        
    # Save model after each epoch
    save_model(model, optimizer, epoch, avg_loss, save_path)
    
    return overall_loss