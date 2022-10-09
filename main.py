import torch, tqdm
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as T
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from utils import num_params, reduce_stats, train_network, evaluate_network

# Download the data
transform = T.ToTensor()
cifar_train = datasets.CIFAR10("./data", train=True, transform=transform, download=True)

# Split into train and validation sets
splits = [0.85 * len(cifar_train), (1-0.85) * len(cifar_train)]
cifar_train, cifar_valid = random_split(cifar_train, lengths=splits)

# Dataloaders
bs = 256
train_dl = DataLoader(cifar_train, batch_size=bs, shuffle=True)
valid_dl = DataLoader(cifar_valid, batch_size=2*bs, shuffle=False)

# Define the model and hyperparameters
class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            # (N, 3, 32, 32)
            nn.Conv2d(3, 32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            # (N, 32, 30, 30)
            nn.Conv2d(32, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # (N, 32, 28, 28)
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(inplace=True),
            # (N, 64, 26, 26)
            nn.Conv2d(64, 256, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(inplace=True),
            # (N, 256, 12, 12)
            nn.Conv2d(256, 256, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(inplace=True),
            # (N, 256, 5, 5)
            nn.Conv2d(256, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            # (N, 128, 3, 3)
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # (N, 64, 3, 3)
            nn.Conv2d(64, 10, kernel_size=(3, 3)),
            # (N, 10, 1, 1)
            nn.Flatten()
        )
    
    def forward(self, images):
        return self.network(images)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

network = ConvNet()
network = network.to(device)
print(f"The network has a total of {num_params(network)}")

optimizer = optim.Adam(network.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 15   

# Train and evaluate the models
training_statistics = {"loss": [], "accuracy": []}
validation_statistics = {"loss": [], "accuracy": []}

for epoch in tqdm(n_epochs, leave=False):
    epoch_statistics = train_network(network, train_dl, loss_fn, optimizer, device)
    loss, accuracy = reduce_stats(training_statistics)
    training_statistics["loss"].append(loss)
    training_statistics["accuracy"].append(accuracy)

    if epoch % 3 == 0:
        eval_statistics, _ = evaluate_network(network, valid_dl, loss_fn, device)
        loss, accuracy = reduce_stats(eval_statistics)
        validation_statistics["loss"].append(loss)
        validation_statistics["accuracy"].append(accuracy)