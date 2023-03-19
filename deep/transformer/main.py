from argparse import ArgumentParser
from typing import Iterator, List, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator

from .transformer import Transformer

parser = ArgumentParser()

batch_size = 64
length = 30

source_tokeniser = get_tokenizer("basic_english")
target_tokeniser = get_tokenizer("basic_english")

train_iterator: IterDataPipe = Multi30k("datasets", "train")
valid_iterator: IterDataPipe = Multi30k("datasets", "valid")
test_iterator: IterDataPipe = Multi30k("datasets", "test")


def yield_tokens(
    iterator: IterDataPipe, is_source: bool
) -> Iterator[torch.Tensor]:
    for batch in iterator:
        yield batch[0] if is_source else batch[1]


source_vocab = build_vocab_from_iterator(
    yield_tokens(train_iterator, True),
    specials=["<unk>", "<pad>", "<sos>", "<eos>"],
)

target_vocab = build_vocab_from_iterator(
    yield_tokens(train_iterator, False),
    specials=["<unk>", "<pad>", "<sos>", "<eos>"],
)


def pad(tokens, length):
    tokens = "<sos>" + tokens + "<eos>"
    return tokens[:length] + ["<pad>"] * max(0, length - len(tokens))


def collate_fn(
    batch: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    source_batch, target_batch = [], []

    for source, target in batch:
        source_batch.append(source_vocab(pad(source, length)))
        target_batch.append(target_vocab(pad(target, length)))

    tokenised_source = torch.tensor(source_batch, dtype=torch.long)
    tokenised_target = torch.tensor(target_batch, dtype=torch.long)

    return tokenised_source, tokenised_target


source_vocab.set_default_index(source_vocab["<unk>"])
target_vocab.set_default_index(target_vocab["<unk>"])

train_dataloader = DataLoader(
    train_iterator, batch_size, shuffle=True, collate_fn=collate_fn
)

valid_dataloader = DataLoader(
    valid_iterator, batch_size, shuffle=False, collate_fn=collate_fn
)

test_dataloader = DataLoader(
    test_iterator, batch_size, shuffle=False, collate_fn=collate_fn
)

model = Transformer(
    source_vocab,
    target_vocab,
    dimensions=256,
    heads=8,
    layers=6,
    feed_forward_hidden_size=1024,
)

trainer = L.Trainer(min_epochs=20)
trainer.fit(model, train_dataloader, valid_dataloader)
