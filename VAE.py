import torch 
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets  
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from torch import nn


# Network parameters 
batch_size=64 # Higher bacth size
epoch=3
learning_rate=1e-3



# Set up the transformation of the data 
img_transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,),(0.5,))])

# Setup the device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download training data from open datasets.
training_data=datasets.MNIST(root="data",train=True,transform=img_transform,download=True) 
training_dataloader=DataLoader(training_data, batch_size=batch_size, shuffle=True)  # Specify batch size and shuffle
test_data=datasets.MNIST(root="data",train=False,transform=img_transform,download=True)
test_dataloader=DataLoader(test_data, batch_size=batch_size)

# Bulding the VAE model (Variational Autoencoder)

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        flat=nn.Flatten()
        self.encoder_network=nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder_network=nn.Sequential(
            nn.ConvTranspose2d(8, 16,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.Tanh() # TANH instead of sigmoid
        )
    def forward(self, x):
        for layer in self.encoder_network:
            x = layer(x)
            z = x  # Output of the encoder

        for layer in self.decoder_network:
             x = layer(x)
        return x, z

model=VAE().to(device)
print(model)

# setup the optimizer
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

# Define the loss function 
# Mean square error is taken as loss function here as the pixel intensity is continuous
loss_func = nn.MSELoss()                 # One possible loss criterion

def traning_loop(training_dataloader,optimizer,loss_func,model):
    size=len(training_dataloader.dataset)
    model.train()
    for index ,(actual_data,ground_truth) in enumerate(training_dataloader):
        actual_data,ground_truth=actual_data.to(device),ground_truth.to(device)
        #print(actual_data.shape, ground_truth.shape)
        
        # Forward Propaget 
        reconstructed_data,_=model(actual_data)
        #print(reconstructed_data.shape, actual_data.shape)
        # Resize or crop the reconstructed data to match the size of the actual input data
        # reconstructed_data = F.interpolate(reconstructed_data, size=actual_data.shape[2:])
        loss=loss_func(reconstructed_data,actual_data)
        
        # Now we will backpropagate 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if  index %100==0:
            loss,current=loss.item(),index*len(actual_data)+len(actual_data)
            print(f"loss:{loss} current:{current}")
            

for t in range(epoch): 
    print(f" Epoch {t+1}\n--------------------------------------------")
    traning_loop(training_dataloader=training_dataloader,
                 optimizer=optimizer,
                 loss_func=loss_func,
                 model=model)
print("  Done ! ")



# Save the model
torch.save(model.state_dict(),'conv_autoencoder.pth')
print("Saved PyTorch Model State to model.pth")

# Load the model
"""model=VAE()
model.load_state_dict(torch.load("conv_autoencoder.pth"))
"""
def unnormalize(img):
    img = (img * 0.5) + 0.5  # Reverse normalization
    return img

with torch.no_grad():
    model.eval()
    x, _ = next(iter(test_dataloader))
    x = x[:8]
    x = x.to(device)
    x_hat, _ = model(x)
  # x_hat = F.interpolate(x_hat, size=x.shape[2:]) # not needed
    x_hat = x_hat.cpu().numpy()

    x = unnormalize(x)
    x_hat = unnormalize(x_hat)
      
  # Plot the original and reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=len(x), figsize=(16, 4))
    for i in range(len(x)):
        axes[0, i].imshow(x[i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(x_hat[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()