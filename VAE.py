import os
import torch 
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets  
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from torch import nn
from torch.utils.tensorboard import SummaryWriter   # Tensorboard support


# Network parameters 
batch_size=64 # Higher bacth size
epoch=10
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

# Bulding the AE model ( Autoencoder)
# Check if the trained model file exists

model_path = 'conv_autoencoder.pth'  # Adjust the path to your trained model file

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        self.flat=nn.Flatten()
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
        """        for layer in self.encoder_network:
            x = layer(x)
            lattent_img= x  # Output of the encoder"""
        x = self.encoder_network(x)
        lattent_img= x
        """for layer in self.decoder_network:
             x = layer(x)"""
        x=self.decoder_network(x)
        return x, lattent_img

model=VAE().to(device)
print(model)
# setup the optimizer
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

# Define the loss function 
# Mean square error is taken as loss function here as the pixel intensity is continuous
loss_func = nn.MSELoss()                 # One possible loss criterion

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
#-------------tensorboard --logdir=runs----------------
tb = SummaryWriter()
network = VAE()
images,lables=next(iter(training_dataloader))
grid = torchvision.utils.make_grid(images)
tb.add_image('images',grid)
tb.add_graph(network,images)
#-------------tensorboard --logdir=runs----------------
if model_path and os.path.exists(model_path):
    model = VAE()
    model.load_state_dict(torch.load(model_path))
    print("Loaded PyTorch Model State from model.pth")
    model.eval()

    # run the inference on the test data and display the results
    def unnormalize(img):
        img = (img * 0.5) + 0.5  # Reverse normalization
        return img

    with torch.no_grad():
        model.eval()
        x, _ = next(iter(test_dataloader))
        x = x[:8]
        x = x.to(device)
        x_hat, _ = model(x)
        #output, latetnt=model.forward(x).to(device)
        #print(latetnt.shape())
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
  
else:
    print("No model.pth found, training a new model...")
    print("Training the model") 



    def traning_loop(training_dataloader,optimizer,loss_func,model):
        size=len(training_dataloader.dataset)
        model.train()
      
        for index ,(actual_data,ground_truth) in enumerate(training_dataloader):
            actual_data,ground_truth=actual_data.to(device),ground_truth.to(device)
            #print(actual_data.shape, ground_truth.shape)
            total_loss=0
            total_correct=0 
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
                #loss,current=loss+loss.item(),index*len(actual_data)+len(actual_data)
                #total_loss,total_correct=total_loss+loss.item(),total_correct+get_num_correct(reconstructed_data,actual_data)
                total_loss,total_correct=total_loss+loss.item(),index*len(actual_data)+len(actual_data)
                
                tb.add_scalar("Loss",total_loss,epoch)
                tb.add_scalar("Number Correct",total_correct,epoch)
                tb.add_scalar("Accuracy",total_correct/size,epoch)
                    
                tb.add_histogram("conv1.bias",model.encoder_network[0].bias,epoch)
                tb.add_histogram("conv1.weight",model.encoder_network[0].weight,epoch)
                tb.add_histogram("conv2.bias",model.encoder_network[3].bias,epoch)
                tb.add_histogram("conv2.weight",model.encoder_network[3].weight,epoch)
                tb.add_histogram("deconv1.bias",model.decoder_network[0].bias,epoch)
                tb.add_histogram("deconv1.weight",model.decoder_network[0].weight,epoch)
                tb.add_histogram("deconv2.bias",model.decoder_network[2].bias,epoch)
                tb.add_histogram("deconv2.weight",model.decoder_network[2].weight,epoch)
                tb.close()
                #print(f"loss:{total_loss} total_correct:{total_correct}")
                print(f"loss:{total_loss} total_correct:{total_correct}")
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
model=VAE()


"""if os.path.exists("conv_autoencoder.pth                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             "):
    model.load_state_dict(torch.load("conv_autoencoder.pth"))
"""

# run the inference on the test data and display the results
def unnormalize(img):
    img = (img * 0.5) + 0.5  # Reverse normalization
    return img
with torch.no_grad():
    model.eval()
    x, _ = next(iter(test_dataloader))
    x = x[:8]
    x = x.to(device)
    x_hat, _ = model(x)
    #output, latetnt=model.forward(x).to(device)
    #print(latetnt.shape())
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
# Run tensorboard to visualize the training process 
#    tensorboard --logdir=runs
