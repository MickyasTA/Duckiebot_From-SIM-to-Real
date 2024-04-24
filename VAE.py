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
batch_size=4
epoch=3
learning_rate=1e-3



# Set up the transformation of the data 
img_transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,),(0.5,))])

# Setup the device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download training data from open datasets.
training_data=datasets.MNIST(root="data",train=True,transform=img_transform,download=True)
training_dataloader=DataLoader(training_data)
test_data=datasets.MNIST(root="data",train=False,transform=img_transform,download=True)
test_dataloader=DataLoader(test_data)

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
            nn.Sigmoid()
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
            
def testing_loop(test_dataloader,loss_func,model):
    size=len(test_dataloader.dataset)
    model.eval()
    num_batches =len(test_data)
    correct,test_loss=0,0
    with torch.no_grad():
        for actual_data,ground_truth in test_dataloader:
            actual_data,ground_truth=actual_data.to(device),ground_truth.to(device)
            predction,_=model(actual_data)
            # Resize the ground truth tensor to match the shape of the prediction tensor
            ground_truth = ground_truth.unsqueeze(1).expand_as(predction)
            test_loss+=loss_func(actual_data,ground_truth).item()
            correct+=(predction.argmax(1)==ground_truth).type(torch.float).sum().item()
        test_loss/=num_batches
        correct/=size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        # Move tensors to CPU for visualization
"""        actual_data = actual_data.cpu()
        predction = predction.cpu()
        
         # Visualization of the reconstructed data
        for i in range(batch_size):
            plt.figure(figsize=(2, 2))
            plt.imshow(actual_data[i].squeeze(), cmap='gray')
            plt.axis('off')
            plt.title('Original')
            plt.show()
            
            plt.figure(figsize=(2, 2))
            plt.imshow(predction[i].squeeze(), cmap='gray')
            plt.axis('off')
            plt.title('Reconstructed')
            plt.show()"""
for t in range(epoch): 
    print(f" Epoch {t+1}\n--------------------------------------------")
    traning_loop(training_dataloader=training_dataloader,
                 optimizer=optimizer,
                 loss_func=loss_func,
                 model=model)
    testing_loop(test_dataloader=test_dataloader,
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

# Visualization of the reconstructed data
with torch.no_grad():
    model.eval()
    for actual_data, _ in test_dataloader:
        actual_data = actual_data.to(device)
        reconstructed_data, _ = model(actual_data)
        reconstructed_data = F.interpolate(reconstructed_data, size=actual_data.shape[2:])
        reconstructed_data = reconstructed_data.cpu().numpy()
        
        # Plot the original and reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=batch_size, figsize=(batch_size*2, 4))
        for i in range(batch_size):
            axes[0, i].imshow(actual_data[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructed_data[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
        plt.tight_layout()
        plt.show()

