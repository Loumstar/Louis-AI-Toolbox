if __name__ == "__main__":
    import os.path

    import torch
    import torch.nn as nn
    import tqdm
    from torch.autograd import Variable
    from torch.utils.data import DataLoader

    from . import cells
    from .dataset import SpeechCommandsDataset
    from .rnn import UniDirectionalRNN

    path = "recurrent/"

    train_dataset = SpeechCommandsDataset(path, "train")
    val_dataset = SpeechCommandsDataset(path, "val")
    test_dataset = SpeechCommandsDataset(path, "test")

    batch_size = 100
    epochs = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True
    )

    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    sequence_size, in_size = train_dataset[0][0].shape
    out_size = 3
    hidden_size = 32
    num_layers = 3
    bias = True

    model = UniDirectionalRNN(
        cells.VanillaCell, in_size, out_size, hidden_size, num_layers, bias
    ).to(device)

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []

    val_accuracy = None
    test_accuracy = None

    batches = len(train_dataset) // batch_size

    def evaluate(loader: DataLoader):
        correct, total = 0, 0
        model.eval()

        with torch.no_grad():
            for audio, labels in loader:
                audio = Variable(audio.view(-1, sequence_size, in_size)).to(
                    device
                )
                output = model(audio)

                _, predicted = torch.max(output.data, 1)
                correct += (predicted.cpu() == labels.cpu()).sum().item()
                total += labels.size(0)

        model.train()

        return 100 * correct // total

    for epoch in range(epochs):
        pbar = tqdm.tqdm(total=batches, desc=f"Epoch #{epoch+1}")

        for i, (audio, labels) in enumerate(train_loader):
            audio = Variable(audio.view(-1, sequence_size, in_size)).to(device)
            labels = Variable(labels).to(device)

            optimiser.zero_grad()
            output = model(audio)

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
        test_accuracy = evaluate(test_loader)

        filepath = os.path.join(path, "checkpoints", f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), filepath)

    print("Done!")
