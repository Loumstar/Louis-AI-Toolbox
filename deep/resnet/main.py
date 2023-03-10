if __name__ == "__main__":
    import inspect
    import os.path
    from typing import Tuple

    import numpy as np
    import torch
    import torch.nn as nn
    import tqdm
    from torch.utils.data import ConcatDataset, DataLoader, random_split
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    from . import presets
    from .resnet import ResNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128

    module = inspect.currentframe()
    assert module is not None, "Module is None"

    module_filepath = inspect.getabsfile(module)
    directory = os.path.dirname(module_filepath)

    train_path = os.path.join("datasets", "NaturalImageNet", "train")
    test_path = os.path.join("datasets", "NaturalImageNet", "test")

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    transform_norm = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
    )

    transform_aug = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine((0, 45), (0.125, 0.125), (0.5, 1.5)),
            transforms.GaussianBlur(5),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
    )

    original_dataset = ImageFolder(train_path, transform=transform_norm)
    augmented_dataset = ImageFolder(train_path, transform=transform_aug)
    combined_dataset = ConcatDataset([original_dataset, augmented_dataset])

    train_dataset, val_dataset = random_split(combined_dataset, [0.9, 0.1])
    test_dataset = ImageFolder(test_path, transform=transform_norm)

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    val_loader = DataLoader(val_dataset, batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=2)

    classes = 20
    epochs = 10

    model = ResNet(presets.ResNet18, 3, classes).to(device)
    model.train()

    model_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(model)
    print(f"Model has {model_parameters:,} parameters.")

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adamax(
        model.parameters(), lr=1e-3, weight_decay=1e-5
    )

    train_losses = []
    val_losses = []
    test_losses = []

    def evaluate(
        loader: DataLoader, desc: str = "Evaluate"
    ) -> Tuple[float, float]:
        correct, total, loss, batches = 0, 0, 0, 0
        model.eval()

        pbar = tqdm.tqdm(loader, desc=desc, unit="batch")

        with torch.no_grad():
            for image, labels in pbar:
                image = image.to(device)
                output = model(image)

                _, predicted = output.max(1)

                loss += criterion(output, labels).item()
                correct += (predicted.cpu() == labels.cpu()).sum().item()

                total += labels.size(0)
                batches += 1

                pbar.set_postfix(
                    {"acc": correct / total, "loss": loss / batches}
                )

        return correct / total, loss / batches

    for epoch in range(epochs):
        model.train()

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch #{epoch+1}")

        for image, labels in pbar:
            image = image.to(device)
            labels = labels.to(device, dtype=torch.long)

            optimiser.zero_grad()

            output = model(image)
            loss = criterion(output, labels)

            loss.backward()
            optimiser.step()

            train_losses.append(loss.item())

            pbar.set_postfix({"loss": loss.item()})

        val_loss = evaluate(val_loader)
        test_loss = evaluate(test_loader, "Test")

        val_losses.append(val_loss)
        test_losses.append(test_loss)

        filepath = os.path.join(directory, "checkpoints", f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), filepath)

    train_losses_filepath = os.path.join(
        directory, "checkpoints", "train_losses"
    )
    np.save(train_losses_filepath, np.array(train_losses))

    val_losses_filepath = os.path.join(directory, "checkpoints", "val_losses")
    np.save(val_losses_filepath, np.array(val_losses))

    test_losses_filepath = os.path.join(
        directory, "checkpoints", "test_losses"
    )
    np.save(test_losses_filepath, np.array(test_losses))
