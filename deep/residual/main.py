if __name__ == "__main__":
    import os.path

    import torch
    import torch.nn as nn
    import tqdm
    from torch.utils.data import ConcatDataset, DataLoader, random_split
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    from . import resnet

    path = "residual/"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128

    train_path = os.path.join(path, "NaturalImageNetTrain")
    test_path = os.path.join(path, "NaturalImageNetTest")

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

    model = resnet.ResNet(resnet.ResNet18, 3, classes).to(device)
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

    losses = []

    val_accuracy = None
    test_accuracy = None

    batches = len(train_dataset) // batch_size

    def evaluate(loader: DataLoader) -> float:
        correct, total = 0, 0
        model.eval()

        with torch.no_grad():
            for image, labels in loader:
                image = image.to(device)
                output = model(image)

                _, predicted = output.max(1)
                correct += (predicted.cpu() == labels.cpu()).sum().item()
                total += labels.size(0)

        model.train()

        return 100 * correct // total

    for epoch in range(epochs):
        pbar = tqdm.tqdm(total=batches, desc=f"Epoch #{epoch+1}")

        for image, labels in train_loader:
            image = image.to(device)
            labels = labels.to(device, dtype=torch.long)

            optimiser.zero_grad()

            output = model(image)
            loss = criterion(output, labels)

            loss.backward()
            optimiser.step()

            losses.append(loss.item())

            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "val": val_accuracy,
                    "test": test_accuracy,
                }
            )

        val_accuracy = evaluate(val_loader)

        filepath = os.path.join(path, "checkpoints", f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), filepath)
