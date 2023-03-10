from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetPreset(NamedTuple):
    blocks: Tuple[int, ...]
    conv_2: Tuple[Tuple[int, ...], Tuple[int, ...]]
    conv_3: Tuple[Tuple[int, ...], Tuple[int, ...]]
    conv_4: Tuple[Tuple[int, ...], Tuple[int, ...]]
    conv_5: Tuple[Tuple[int, ...], Tuple[int, ...]]


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Tuple[int, ...],
        kernels: Tuple[int, ...],
        stride: int = 1,
    ) -> None:
        super().__init__()

        if not channels or not kernels:
            raise ValueError("Tuple of channels/kernels must not be empty.")
        if len(channels) != len(kernels):
            raise ValueError("Number of channels and kernels do not match.")

        layers = []

        for i, (channel, kernel) in enumerate(zip(channels, kernels)):
            layers.extend(
                [
                    nn.Conv2d(
                        channels[i - 1] if i > 0 else in_channels,
                        channel,
                        kernel_size=kernel,
                        stride=stride if i == 0 else 1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channel),
                ]
            )

            if i < len(channels) - 1:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

        if stride > 1 or in_channels != channels[-1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels[-1],
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(channels[-1]),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.layers(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(
        self,
        preset: ResNetPreset,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        conv_2_in = 64
        conv_3_in = preset.conv_2[0][-1]
        conv_4_in = preset.conv_3[0][-1]
        conv_5_in = preset.conv_4[0][-1]

        conv_5_out = preset.conv_5[0][-1]

        conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        conv_2 = self.stage(2, preset.blocks[0], conv_2_in, *preset.conv_2)
        conv_3 = self.stage(3, preset.blocks[1], conv_3_in, *preset.conv_3)
        conv_4 = self.stage(4, preset.blocks[2], conv_4_in, *preset.conv_4)
        conv_5 = self.stage(5, preset.blocks[3], conv_5_in, *preset.conv_5)

        self.stages = nn.Sequential(conv1, conv_2, conv_3, conv_4, conv_5)
        self.pool = nn.AvgPool2d(7)

        self.fc = nn.Linear(conv_5_out, out_channels, bias=True)

    def stage(
        self,
        stage: int,
        blocks: int,
        in_channels: int,
        channels: Tuple[int, ...],
        kernels: Tuple[int, ...],
    ) -> nn.Module:
        layers = []

        if stage == 2:
            layers.append(nn.MaxPool2d(3, padding=1, stride=1))

        for i in range(blocks):
            layer_in_channels = channels[i - 1] if i > 0 else in_channels
            stride = 2 if i == 0 else 1

            layers.append(
                ResNetBlock(layer_in_channels, channels, kernels, stride)
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stages(x)
        features = self.pool(out).view(x.size(0), -1)

        return self.fc(features)
