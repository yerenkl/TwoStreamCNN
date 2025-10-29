import os
import torch
import pandas as pd

def save_metrics(results, path):
    """Save training/validation metrics to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False)


def load_metrics(path):
    """Resume training from existing CSV if available."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded previous metrics from {path}")
        return df.to_dict(orient="list"), len(df)
    return {'acc_tr': [], 'acc_val': [], 'loss_tr': [], 'loss_val': []}, 0



