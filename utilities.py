import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader


def num_params(network: nn.Module):
    """
    Takes in the neural network (subclassed nn.Module) and returns the number of 
    trainable parameters in the network
    """
    return sum([param.numel() for param in network.parameters() if param.requires_grad])


def cosine_similarity(vector_a: torch.Tensor, vector_b: torch.Tensor=None):
    """
    Takes in a (two - optional) vectors, computes and returns the normalized dot product between them
    expected input format for a vector := (m, n)
                                           m => number of vectors
                                           n => n_features of each vector
    - supplied with only a single vector/stack of vectors => computes cosine similarity among themselves
    - supplied with two vectors/stack of vectors => computes cosine similarity between each of the pairs q
    """
    # squeeze extra dimensions and add fake batch dimensions
    vector_a = vector_a.squeeze(0)
    vector_a = vector_a.unsqueeze(0) if vector_a.ndim == 1 else vector_a
    normalized_a = (vector_a.T / torch.norm(vector_a, dim=1)).T  # (m, n)

    if vector_b is not None:
        # squeeze extra dimensions and add fake batch dimensions
        vector_b = vector_b.squeeze(0)
        vector_b = vector_b.unsqueeze(0) if vector_b.ndim == 1 else vector_b
        normalied_b = vector_b.T / torch.norm(vector_b, dim=1)  # (n, m)
    else:
        normalied_b = normalized_a.T  # (n, m)
    
    similarities = torch.mm(normalized_a, normalied_b)

    return similarities


def multiclass_accuracy(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Takes in the raw logits and ground truth labels and 
    computes the multiclass accuracy
    """
    predicted_labels = torch.argmax(predictions, dim=1)
    return (predicted_labels == targets).float().mean()


def binary_accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    Takes in the logits and ground truth labels and 
    computes the binary accuracy
    """
    predicted_labels = (predictions > threshold).float()
    return (predicted_labels == targets).float().mean()


def intersection_over_union(predicted_mask: torch.Tensor, true_mask: torch.Tensor):
    predicted_mask = predicted_mask.squeeze()
    true_mask = true_mask.squeeze()
    intersection = torch.bitwise_and(predicted_mask, true_mask)
    union = torch.bitwise_or(predicted_mask, true_mask)
    return intersection.sum() / union.sum()


def plot_grid(images: torch.Tensor, shape: tuple, nrow: int, title: str = None):
    shape = (-1, ) + shape
    images = images.view(shape)
    image_grid = make_grid(images, nrow=nrow, padding=2, normalize=True, pad_value=1)
    plt.grid(False)
    plt.xticks([]), plt.yticks([])
    if title:
        plt.title(title)
    plt.imshow(image_grid.permute(1, 2, 0))
    plt.show()
    return


def reduce_stats(statistics: dict):
    """
    Takes in a dictionary containing model statistics, reduces (mean value) and returns them
    """
    reduced_stats = {statistic: sum(values) / len(values) for statistic, values in statistics.items() if values}
    return list(reduced_stats.values())


def train_network(network: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str = "cpu"):
    n_batches = len(dataloader)
    loop = tqdm(dataloader, total=n_batches, leave=False)
    # Stores the epoch statistics
    epoch_statistics = {"loss": [], "accuracy": []}
    network.train()  # Set the network to train mode
    for samples, targets in loop:
        # Load data onto the device
        samples, targets = samples.to(device), targets.to(device)
        predictions = network(samples)
        loss = criterion(predictions, targets)
        accuracy = multiclass_accuracy(predictions, targets)
        # Mini batch gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record the epoch statistics
        epoch_statistics["loss"].append(loss.item())
        epoch_statistics["accuracy"].append(accuracy.item() )
        # Log the batch statistic
        loop.set_postfix(loss=loss.item(), accuracy=accuracy.item())
    
    return epoch_statistics


@torch.no_grad()
def evaluate_network(network: nn.Module, dataloader: DataLoader, criterion: nn.Module=None, device: str = "cpu"):
    network.eval()  # Set the network to eval mode
    loop = tqdm(dataloader, total=len(dataloader), leave=False)
    eval_statistics = {"loss": [], "accuracy": []}
    eval_predictions = []
    for samples, targets in loop:
        samples, targets = samples.to(device), targets.to(device)
        predictions = network(samples)
        eval_predictions.append(predictions)
        if criterion:
            loss = criterion(predictions, targets)
            eval_statistics["loss"].append(loss.item())
        accuracy = multiclass_accuracy(predictions, targets)
        eval_statistics["accuracy"].append(accuracy)
        
        loop.set_postfix(loss=loss.item(), accuracy=accuracy.item())
    
    return eval_statistics, torch.stack(eval_predictions, dim=0)


def save_network(location: str, network: nn.Module, optimizer: optim.Optimizer, **kwargs):
    """
    Wrapper over torch.save() that takes in a neural network, optimizer and other 
    statistics and saves them to the disk at the given`location`
    """
    checkpoint = {"network_state_dict": network.state_dict(), "optimizer_state_dict": optimizer.state_dict(), **kwargs}
    try:
        path, extension = location.split(".")
        extension = ".pth" if extension not in [".pt", ".pth"] else extension
        location = path + extension
    except ValueError as err:
        print(f"encountered {err} while trying to split location by `.`")
        location = location + ".pth"
    print("saving network at ", location)
    torch.save(checkpoint, location)


def load_network(location: str, network: nn.Module, optimizer: optim.Optimizer=None, device: str = "cpu"):
    """
    Wrapper over torch.load() and companion method to save_network() loads the model 
    from the location and returns the saved statistics if any
    """
    checkpoint = torch.load(location, map_location=device)
    network.load_state_dict(checkpoint["network_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    additional_stats = {statistic:value for statistic, value in checkpoint.items() if not statistic.endswith("state_dict")}
    return additional_stats
    

def train_network_mp(network: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                     optimizer: optim.Optimizer, scaler: GradScaler, device: str = "cpu"):
    n_batches = len(dataloader)
    loop = tqdm(dataloader, total=n_batches, leave=False)
    # Stores the epoch statistics
    epoch_statistics = {"loss": [], "accuracy": []}
    network.train()  # Set the network to train mode
    for samples, targets in loop:
        # shift data onto the device
        samples, targets = samples.to(device), targets.to(device)
        # mixed precision forward pass
        with autocast:
            predictions = network(samples)
            loss = criterion(predictions, targets)
            accuracy = multiclass_accuracy(predictions, targets)
        # mini batch gradient descent with mixed precision
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        # record the epoch statistics
        epoch_statistics["loss"].append(loss.item())
        epoch_statistics["accuracy"].append(accuracy.item() )
        # log the batch statistic
        loop.set_postfix(loss=loss.item(), accuracy=accuracy.item())
    
    return epoch_statistics
