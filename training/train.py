import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from app.config import ARTIFACTS_DIR, WEIGHTS_BY_TIER
from app.model import DigitMLP

SEED = 42
BATCH_SIZE = 32

# Fewer epochs → lower accuracy (free tier). More epochs → premium tier.
TIER_EPOCHS = {
    "free": 25,
    "premium": 150,
}


def train_tier(
    tier: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> float:
    epochs = TIER_EPOCHS[tier]
    model = DigitMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    n_train = X_train.shape[0]
    print(f"\n=== training {tier} tier ({epochs} epochs) ===")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(X_train[idx])
            loss = criterion(logits, y_train[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * idx.shape[0]
        epoch_loss /= n_train

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"  epoch {epoch:3d}/{epochs}  loss {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()

    out_path = WEIGHTS_BY_TIER[tier]
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"  test accuracy: {acc * 100:.2f}%")
    print(f"  saved → {out_path}")
    return acc


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    results = {}
    for tier in ("free", "premium"):
        results[tier] = train_tier(tier, X_train, y_train, X_test, y_test)

    print("\n=== summary ===")
    for tier, acc in results.items():
        print(f"  {tier:8s}  {acc * 100:.2f}%  ({TIER_EPOCHS[tier]} epochs)")


if __name__ == "__main__":
    main()
