import matplotlib.pyplot as plt
from data.data import trainloader  
###Visualize input data###
def show_input_data(file_location):
    """
    Show input data
    params file_location: file location
    """
    dataiter = iter(trainloader)
    images = next(dataiter)
    num_samples = 25
    sample_images = [images[0][i,0] for i in range(num_samples)]
    #Show all in a 5x5 grid
    plt.figure(figsize=(10,10))
    for i in range(num_samples):
        plt.subplot(5,5,i+1)
        plt.imshow(sample_images[i].numpy().squeeze(), cmap='gray_r')

    plt.savefig(file_location)
    plt.close()