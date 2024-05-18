import torch
import torch.nn as nn

from horseUtils import HorseDataset, HorseshoeNetwork
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize, Lambda

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Create the dataset

def resize_transform(height:int, width:int):
    return Lambda(lambda X: Resize((height,width))(X))


dataset = HorseDataset(transform=resize_transform(256,256), 
                        target_transform=resize_transform(256,256))

# Create the developement and test datasets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

training_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# create the dataloaders
BATCH_SIZE = 4

train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Declare our network
architecture = [(2,64),(2,128),(3,256),(3,512),(3,512)]
model = HorseshoeNetwork(architecture=architecture)
model.to(device)

# Loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer) -> None:
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        ## Forward pass:
        # Compute prediction error
        pred = model(X)
        # Compute loss
        loss = loss_fn(pred, y)

        ## Backpropagation:
        # Zero out the gradient
        optimizer.zero_grad()
        # Compute the gradient (backward step)
        loss.backward()
        # apply weight update
        optimizer.step()

def evaluate(dataloader, model, loss_fn) -> float:
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    #print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss

# Train the model
EPOCHS = 5
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    training_loss = evaluate(train_dataloader, model, loss_fn)
print("Done!")