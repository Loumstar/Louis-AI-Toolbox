import inspect
import os.path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .vae import VariationalAutoEncoder

module = inspect.currentframe()
assert module is not None, "Module is None"

module_filepath = inspect.getabsfile(module)
directory = os.path.dirname(module_filepath)

image_size = 28
image_channels = 1

epochs = 20
batch_size = 128
latent_dimensions = 100
beta = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.MNIST(
    "datasets", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    "datasets", train=False, download=True, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size, shuffle=True, drop_last=True
)

test_loader = DataLoader(test_dataset, batch_size)

model = VariationalAutoEncoder(1, 1, latent_dimensions).to(device)
model.train()

model_parameters = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(model)
print(f"Model has {model_parameters:,} parameters.")

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

train_losses = []
test_losses = []


def divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / 2


def criterion(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    x = x.view(-1, image_size**2)
    recon = recon.view(-1, image_size**2)

    return F.binary_cross_entropy(recon, x, reduction="sum")


def evaluate(loader: DataLoader) -> float:
    bce, div, batches = 0, 0, 0
    model.eval()

    pbar = tqdm.tqdm(loader, desc="Evaluate", unit="batch")

    with torch.no_grad():
        for images, _ in pbar:
            images = images.to(device)
            recon = model(images)

            mean = model.encoder.mean
            logvar = model.encoder.logvar

            bce += criterion(images, recon).item()
            div += divergence(mean, logvar).item()

            batches += 1

            pbar.set_postfix(
                {
                    "bce": bce / batches,
                    "div": div / batches,
                }
            )

    return bce - (beta * div) / batches


for epoch in range(epochs):
    model.train()

    pbar = tqdm.tqdm(train_loader, desc=f"Epoch #{epoch+1}", unit="batch")

    for images, _ in pbar:
        images = images.to(device)

        optimiser.zero_grad()

        recon = model(images)
        mean = model.encoder.mean
        logvar = model.encoder.logvar

        loss = criterion(images, recon) - (beta * divergence(mean, logvar))

        loss.backward()
        optimiser.step()

        train_losses.append(loss.item())

        pbar.set_postfix({"loss": loss.mean().item()})

    test_loss = evaluate(test_loader)
    test_losses.append(test_loss)

    filepath = os.path.join(directory, "checkpoints", f"epoch_{epoch}.pt")
    torch.save(model.state_dict(), filepath)

train_losses_filepath = os.path.join(directory, "checkpoints", "train_losses")
np.save(train_losses_filepath, np.array(train_losses))

test_losses_filepath = os.path.join(directory, "checkpoints", "test_losses")
np.save(test_losses_filepath, np.array(test_losses))
