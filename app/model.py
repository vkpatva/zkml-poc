import torch.nn as nn


class DigitMLP(nn.Module):
    """Circuit-friendly MLP for 8x8 digit classification.

    Architecture is intentionally limited to Linear + ReLU so the graph can be
    exported to ONNX and proven inside a ZK circuit (EZKL). No BatchNorm,
    Dropout, or final Softmax: argmax over raw logits equals argmax over
    softmax probabilities, and exp/softmax is expensive to prove in ZK.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),  # raw logits, no final activation
        )

    def forward(self, x):
        return self.net(x)
