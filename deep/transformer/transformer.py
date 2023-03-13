from typing import Union

import torch
import torch.nn as nn

from .layers.decoder import TransformerDecoder
from .layers.embedding import Embedding
from .layers.encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        dimensions: int,
        heads: int,
        layers: int,
        feed_forward_hidden_size: int,
        source_pad_token_index: int,
        target_pad_token_index: int,
        max_length: int = 200,
        dropout: float = 0.3,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        super().__init__()

        self.heads = heads
        self.source_pad_token_index = source_pad_token_index
        self.target_pad_token_index = target_pad_token_index
        self.device = device

        encoder_embedder = Embedding(
            source_vocab_size, dimensions, source_pad_token_index
        )

        decoder_embedder = Embedding(
            target_vocab_size, dimensions, target_pad_token_index
        )

        self.encoder = TransformerEncoder(
            encoder_embedder,
            dimensions,
            heads,
            layers,
            feed_forward_hidden_size,
            max_length,
            dropout,
        )

        self.decoder = TransformerDecoder(
            decoder_embedder,
            dimensions,
            heads,
            layers,
            feed_forward_hidden_size,
            max_length,
            dropout,
        )

        self.fc = nn.Linear(dimensions, target_vocab_size)

        self.initialise()

    def initialise(self):
        for p in self.parameters():
            if p.requires_grad:
                nn.init.xavier_uniform_(p)

    def source_mask(self, source: torch.Tensor) -> torch.Tensor:
        return (source != self.source_pad_token_index).unsqueeze(-2)

    def target_mask(self, target: torch.Tensor) -> torch.Tensor:
        target_mask = (target != self.target_pad_token_index).unsqueeze(-2)
        future_token_mask = (
            torch.ones((1, target.size(1), target.size(1)))
            .triu(1)
            .logical_not()
            .to(self.device)
        )

        return target_mask & future_token_mask

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        source_mask = self.source_mask(source)
        target_mask = self.target_mask(target)

        encoder_out = self.encoder(source, mask=source_mask)
        decoder_out = self.decoder(
            target, encoder_out, source_mask, target_mask
        )

        return self.fc(decoder_out)
