import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(csv_path, save_dir, title_suffix=""):
    """
    Plot training & validation loss and accuracy curves from a metrics CSV file.
    """

    os.makedirs(save_dir, exist_ok=True)

    results = pd.read_csv(csv_path)
    epochs = np.arange(1, len(results) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, results['loss_tr'], label='Train Loss', linewidth=2)
    plt.plot(epochs, results['loss_val'], label='Val Loss', linewidth=2, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss vs Epoch {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'loss_curve{title_suffix}.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, 100 * results['acc_tr'], label='Train Acc', linewidth=2)
    plt.plot(epochs, 100 * results['acc_val'], label='Val Acc', linewidth=2, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy vs Epoch {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'accuracy_curve{title_suffix}.png'), dpi=200)
    plt.close()

    print(f"Plots saved to: {save_dir}")
