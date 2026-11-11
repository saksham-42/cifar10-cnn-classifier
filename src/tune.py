import torch
import torch.nn as nn
import pandas as pd

from .data import make_loaders
from .model import SimpleCNN
from .train import fit


def run_experiment(lr, batch_size, dropout, epochs, device):
    train_loader, val_loader, _ = make_loaders(batch_size)

    model = SimpleCNN(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = fit(model, train_loader, val_loader, criterion, optimizer, device, epochs)

    return {
        "lr": lr,
        "batch_size": batch_size,
        "dropout": dropout,
        "best_val_acc": history["best_val_acc"],
        "best_state": history["best_state"]
    }


def grid_search(device, epochs_tune: int = 10):
    # same grid as your notebook
    lrs = [1e-2, 1e-3, 1e-4]
    batch_sizes = [32, 64]
    dropouts = [0.2, 0.5]

    results = []
    best_run = None

    for lr in lrs:
        for bs in batch_sizes:
            for do in dropouts:
                out = run_experiment(lr, bs, do, epochs_tune, device)

                results.append({
                    "lr": out["lr"],
                    "batch_size": out["batch_size"],
                    "dropout": out["dropout"],
                    "best_val_acc": out["best_val_acc"]
                })

                if (best_run is None) or (out["best_val_acc"] > best_run["best_val_acc"]):
                    best_run = out

    df_results = pd.DataFrame(results).sort_values(by="best_val_acc", ascending=False)
    return df_results, best_run
