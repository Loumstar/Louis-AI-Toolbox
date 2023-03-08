if __name__ == "__main__":
    import os.path
    from typing import Tuple

    import torch
    import torch.nn.functional as F
    import tqdm
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from .vae import VariationalAutoEncoder

    path = "deep/generative/vae"

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

    train_dataset = datasets.MNIST(path, download=True, transform=transform)
    test_dataset = datasets.MNIST(path, transform=transform)

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

    optimiser = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-5
    )

    losses = []

    test_bce = None
    test_div = None

    batches = len(train_dataset) // batch_size

    def divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / 2

    def criterion(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, image_size**2)
        recon = recon.view(-1, image_size**2)

        return F.binary_cross_entropy(recon, x, reduction="sum")

    def evaluate(loader: DataLoader) -> Tuple[float, float]:
        bce, div, total = 0, 0, 0
        model.eval()

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                recon = model(images)

                mean = model.encoder.mean
                logvar = model.encoder.logvar

                bce += criterion(images, recon).item()
                div += divergence(mean, logvar).item()

                total += images.size(0)

        return bce / total, div / total

    for epoch in range(epochs):
        model.train()

        pbar = tqdm.tqdm(total=batches, desc=f"Epoch #{epoch+1}", unit="batch")

        for images, _ in train_loader:
            images = images.to(device)

            optimiser.zero_grad()

            recon = model(images)
            mean = model.encoder.mean
            logvar = model.encoder.logvar

            loss = criterion(images, recon) - (beta * divergence(mean, logvar))

            loss.backward()
            optimiser.step()

            losses.append(loss.item())

            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": loss.item() / images.size(0),
                    "bce": test_bce,
                    "div": test_div,
                }
            )

        test_bce, test_div = evaluate(test_loader)

        filepath = os.path.join(path, "checkpoints", f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), filepath)
