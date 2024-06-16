import matplotlib.pyplot as plt

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

# Functions for training and evaluation
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

def compute_score(y_true, 
                  y_pred,
                  score_fn:str='accuracy') -> float:
    # turn the probability into classes
    y_pred_bin = (y_pred >= 0.5).float()
    
    # Legacy code:
    #accuracy = (y_true == y_pred_bin).sum().item() / y_true.numel()

    TPs = (y_true * y_pred_bin).sum().item()
    TNs = ((1 - y_true) * (1 - y_pred_bin)).sum().item()
    FPs = ((1 - y_true) * y_pred_bin).sum().item()
    FNs = (y_true * (1 - y_pred_bin)).sum().item()

    # Since I compute the average, tuurns out the formulas are the ones
    # below
    score = {}
    score['accuracy'] = (TPs + TNs) / (TPs + TNs + FPs + FNs)
    score['precision'] = TPs / (TPs + FPs) if TPs + FPs != 0 else 0
    score['recall'] = TPs / (TPs + FNs) if TPs + FNs != 0 else 0

    if score_fn == 'whole':
        return score
    else:
        return score[score_fn.lower()]
    
# To review, obviously
def evaluate(dataloader, model, loss_fn) -> float:
    num_batches = len(dataloader)
    test_loss = 0
    avg_accuracy = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            # compute the loss, and accumulate it
            test_loss += loss_fn(pred, y).item()

            # compute accuracy, and accumulate it
            avg_accuracy += compute_score(y_true=y, y_pred=pred, score_fn='accuracy')
    # Average over the number of batches
    avg_accuracy /= num_batches
    test_loss /= num_batches
    return test_loss, avg_accuracy

# Declare our network
architecture = [(2,64),(2,128),(3,256),(3,512),(3,512)]
model = HorseshoeNetwork(architecture=architecture)
model.to(device)

# Loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
train_loss, train_accuracy = [], []

EPOCHS = 20
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    # Compute the loss and accuracy
    loss, acc = evaluate(train_dataloader, model, loss_fn)

    # print
    print(f"Train Error: \n \tAvg loss: {loss:>8f}, avg accuracy: {100*acc:>2.4f}% \n")
    # Store loss and accuracy for the current epoch
    train_loss.append(loss)
    train_accuracy.append(acc)
print("Done!")

# Let's try on an image
img, mask = test_dataset[0]
img, mask = img.to(device), mask.to(device)
pred = model(img[None, ...])    # add a batch dimension


pred = pred.squeeze()           # remove the batch and channel dimensions (assuming 1 output channel for the prediction)
pred_bin = (pred >= 0.5).float() # binarize the prediction
img /= img.max()                # Normalize the image (for visualization purposes)

print(pred.shape, mask.shape, img.shape)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(img.cpu().permute(1, 2, 0))
ax[0].set_title('Image', fontsize=16, fontweight='bold')
ax[0].axis('off')

ax[1].imshow(mask.cpu().squeeze(), cmap='gray')
ax[1].set_title('Mask', fontsize=16, fontweight='bold')
ax[1].axis('off')

ax[2].imshow(pred.cpu().detach(), cmap='gray')
ax[2].set_title('Prediction', fontsize=16, fontweight='bold')
ax[2].axis('off')

plt.tight_layout()
plt.show()