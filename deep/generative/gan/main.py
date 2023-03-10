import inspect
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .models import Discriminator, Generator

module = inspect.currentframe()
assert module is not None, "Module is None"

module_filepath = inspect.getabsfile(module)
directory = os.path.dirname(module_filepath)

epochs = 20
beta = 0.5
lr = 2e-4
latent_dimensions = 20
image_channels = 3
batch_size = 128  # change that

device = "cuda" if torch.cuda.is_available() else "cpu"

mean = torch.Tensor([0.4914, 0.4822, 0.4465])
std = torch.Tensor([0.247, 0.243, 0.261])

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

train_dataset = datasets.CIFAR10("datasets", train=True, transform=transform)
test_dataset = datasets.CIFAR10("datasets", train=False, transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size, shuffle=True, drop_last=True
)

test_loader = DataLoader(test_dataset, batch_size, drop_last=True)

generator = Generator(latent_dimensions, image_channels).to(device)
generator.train()

model_parameters = sum(
    p.numel() for p in generator.parameters() if p.requires_grad
)

print(generator)
print(f"Generator has {model_parameters:,} model_parameters.")

discriminator = Discriminator(image_channels).to(device)
discriminator.train()

model_parameters = sum(
    p.numel() for p in discriminator.parameters() if p.requires_grad
)

print(discriminator)
print(f"Discriminator has {model_parameters:,} model_parameters.")

gen_optimiser = torch.optim.Adam(
    generator.parameters(), lr=lr, betas=(beta, 0.999)
)
disc_optimiser = torch.optim.Adam(
    discriminator.parameters(), lr=lr, betas=(beta, 0.999)
)

criterion = nn.BCELoss()

train_gen_losses = []
train_disc_losses = []

val_gen_losses = []
val_disc_losses = []


def evaluate(loader: DataLoader) -> Tuple[float, float]:
    disc_loss, gen_loss, batches = 0, 0, 0
    discriminator.eval()
    generator.eval()

    real_labels = torch.full(
        (batch_size,), 1, dtype=torch.float, device=device
    )
    fake_labels = torch.full(
        (batch_size,), 1, dtype=torch.float, device=device
    )

    pbar = tqdm.tqdm(loader, desc="Evaluate", unit="batch")

    with torch.no_grad():
        for images, _ in pbar:
            images = images.to(device)
            output = discriminator(images)

            disc_loss += criterion(output, real_labels).item()

            noise = torch.randn(batch_size, latent_dimensions, device=device)
            fake = generator(noise)

            output = discriminator(fake)

            disc_loss += criterion(output, fake_labels).item()
            gen_loss += criterion(output, real_labels).item()

            batches += 1

            pbar.set_postfix(
                {
                    "disc": disc_loss / batches,
                    "gen": gen_loss / batches,
                }
            )

    return disc_loss / batches, gen_loss / batches


for epoch in range(epochs):
    generator.train()
    discriminator.train()

    real_labels = torch.full(
        (batch_size,), 1, dtype=torch.float, device=device
    )
    fake_labels = torch.full(
        (batch_size,), 0, dtype=torch.float, device=device
    )

    pbar = tqdm.tqdm(train_loader, desc=f"Epoch #{epoch+1}", unit="batch")

    for images, _ in pbar:
        disc_optimiser.zero_grad()
        images = images.to(device)

        # Train discriminator on real images
        output = discriminator(images)
        disc_real_loss = criterion(output, real_labels)
        disc_real_loss.backward()

        noise = torch.randn(batch_size, latent_dimensions, device=device)
        fake = generator(noise)

        # Train discriminator on fake images
        output = discriminator(fake.detach())
        disc_fake_loss = criterion(output, fake_labels)
        disc_fake_loss.backward()

        disc_loss = disc_real_loss + disc_fake_loss
        disc_optimiser.step()

        gen_optimiser.zero_grad()
        output = discriminator(fake)

        gen_loss = criterion(output, real_labels)
        gen_loss.backward()

        gen_optimiser.step()

        train_gen_losses.append(gen_loss.item())
        train_disc_losses.append(disc_loss.item())

        pbar.set_postfix(
            {
                "disc": disc_loss.mean().item(),
                "gen": gen_loss.mean().item(),
            }
        )

    val_disc_loss, val_gen_loss = evaluate(test_loader)

    val_disc_losses.append(val_disc_loss)
    val_gen_losses.append(val_gen_loss)

    gen_filepath = os.path.join(
        directory, "checkpoints", f"generator_{epoch}.pt"
    )
    torch.save(generator.state_dict(), gen_filepath)

    disc_filepath = os.path.join(
        directory, "checkpoints", f"discriminator_{epoch}.pt"
    )

    torch.save(discriminator.state_dict(), disc_filepath)

gen_loss_filepath = os.path.join(directory, "checkpoints", "generator_loss")
np.save(gen_loss_filepath, np.array(train_gen_losses))

disc_loss_filepath = os.path.join(
    directory, "checkpoints", "discriminator_loss"
)
np.save(disc_loss_filepath, np.array(train_disc_losses))
