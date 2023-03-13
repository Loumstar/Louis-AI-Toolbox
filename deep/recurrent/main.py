import inspect
import os.path
from typing import Tuple

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from . import cells
from .loader import SpeechCommandsDataset
from .rnn import BiDirectionalRNN

module = inspect.currentframe()
assert module is not None, "Module is None"

module_filepath = inspect.getabsfile(module)
directory = os.path.dirname(module_filepath)

train_dataset = SpeechCommandsDataset("datasets", "train")
val_dataset = SpeechCommandsDataset("datasets", "val")
test_dataset = SpeechCommandsDataset("datasets", "test")

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

model = BiDirectionalRNN(
    cells.GRUCell, in_size, out_size, hidden_size, num_layers, bias
).to(device)

model.train()

model_parameters = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(model)
print(f"Model has {model_parameters:,} parameters.")

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

train_losses = []

val_accuracies = []
test_accuracies = []


def evaluate(
    loader: DataLoader, desc: str = "Evaluate"
) -> Tuple[float, float]:
    correct, total, loss, batches = 0, 0, 0, 0
    model.eval()

    pbar = tqdm.tqdm(loader, desc=desc, unit="batch")

    with torch.no_grad():
        for audio, labels in pbar:
            audio = audio.to(device)
            output = model(audio)

            _, predicted = torch.max(output.data, 1)

            loss += criterion(output, labels).item()
            correct += (predicted.cpu() == labels.cpu()).sum().item()

            total += labels.size(0)
            batches += 1

            pbar.set_postfix({"acc": correct / total, "loss": loss / batches})

    return correct / total, loss / batches


for epoch in range(epochs):
    model.train()

    pbar = tqdm.tqdm(train_loader, desc=f"Epoch #{epoch+1}", unit="batch")

    for audio, labels in pbar:
        audio = audio.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()
        output = model(audio)

        loss = criterion(output, labels)
        loss.backward()
        optimiser.step()

        train_losses.append(loss.item())

        pbar.set_postfix({"loss": loss.item()})

    val_accuracy = evaluate(val_loader, desc="Evaluate")
    test_accuracy = evaluate(test_loader, desc="Test")

    val_accuracies.append(val_accuracy)
    test_accuracies.append(test_accuracy)

    filepath = os.path.join(directory, "checkpoints", f"epoch_{epoch}.pt")
    torch.save(model.state_dict(), filepath)
