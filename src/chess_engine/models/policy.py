import torch
from torch import nn

from chess_engine.features import INPUT_PLANES
from chess_engine.models.nee import ResidualBlock


class MovePolicyNet(nn.Module):
    def __init__(self, blocks: int = 3, channels: int = 48):
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])
        self.from_head = nn.Linear(channels * 8 * 8, 64)
        self.to_head = nn.Linear(channels * 8 * 8, 64)
        self.promo_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, 5),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        flat = x.flatten(1)
        return self.from_head(flat), self.to_head(flat), self.promo_head(x)
