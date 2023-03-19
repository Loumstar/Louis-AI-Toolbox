import lightning as L
import torch
import torch.nn as nn
from torchtext.vocab import Vocab

from .layers.decoder import TransformerDecoder
from .layers.embedding import Embedding
from .layers.encoder import TransformerEncoder


class Transformer(L.LightningModule):
    def __init__(
        self,
        source_vocab: Vocab,
        target_vocab: Vocab,
        dimensions: int,
        heads: int,
        layers: int,
        feed_forward_hidden_size: int,
        max_length: int = 200,
        dropout: float = 0.3,
        pad_token: str = "<pad>",
        lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.dimensions = dimensions
        self.heads = heads
        self.layers = layers

        self.source_pad_index = source_vocab.lookup_indices([pad_token]).pop()
        self.target_pad_index = target_vocab.lookup_indices([pad_token]).pop()

        encoder_embedder = Embedding(
            len(source_vocab), dimensions, self.source_pad_index
        )

        decoder_embedder = Embedding(
            len(target_vocab), dimensions, self.target_pad_index
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

        self.fc = nn.Linear(dimensions, len(target_vocab))

        self.initialise()

        self.lr = lr
        self.loss = nn.CrossEntropyLoss(ignore_index=self.target_pad_index)

    def initialise(self):
        for p in self.parameters():
            if p.requires_grad:
                nn.init.xavier_uniform_(p)

    def source_mask(self, source: torch.Tensor) -> torch.Tensor:
        return (source != self.source_pad_index).unsqueeze(-2)

    def target_mask(self, target: torch.Tensor) -> torch.Tensor:
        target_mask = (target != self.target_pad_index).unsqueeze(-2)
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        source, target = batch
        source = source.to(self.device)
        target = target.to(self.device)

        logits = self.forward(source, target[:, :1])

        return self.loss(logits.permute(0, 2, 1), target[:, 1:])

    def validation_step(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        source, target = batch
        source = source.to(self.device)
        target = target.to(self.device)

        logits = self.forward(source, target[:, :1])

        return self.loss(logits.permute(0, 2, 1), target[:, 1:])
