import torch
import os
from tqdm import tqdm
from utils import save_metrics

class Trainer:
    """
    Training manager for video classification models.
    """

    def __init__(self, model, optimizer, loss_fn,
                 train_loader, val_loader, device,
                 save_dir="./results", model_name="model", resume=False, save_check=True):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_check = save_check
        self.model_name = model_name
        self.resume = resume

        # Paths
        self.save_dir = save_dir
        self.best_ckpt = os.path.join(save_dir, f"{model_name}_best.pth")
        self.last_ckpt = os.path.join(save_dir, f"{model_name}_last.pth")
        self.metrics_path = os.path.join(save_dir, f"{model_name}_metrics.csv")
        os.makedirs(save_dir, exist_ok=True)

        # Training state
        self.results = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_val_loss = float('inf')
        self.start_epoch = 0

        # Resume if available
        if resume and os.path.exists(self.last_ckpt):
            self._load_checkpoint(self.last_ckpt)
            print(f"Resumed training from {self.last_ckpt}")

    # Save & Load checkpoints
    def _save_checkpoint(self, filename):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "results": self.results,
            "best_val_loss": self.best_val_loss
        }, filename)

    def _load_checkpoint(self, filename):
        ckpt = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.results = ckpt.get("results", self.results)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.start_epoch = len(self.results["train_loss"])
    
    def _save_results(self, train_loss, train_acc, val_loss, val_acc, epoch):
        self.results["train_loss"].append(train_loss)
        self.results["train_acc"].append(train_acc)
        self.results["val_loss"].append(val_loss)
        self.results["val_acc"].append(val_acc)

        print(f"[Epoch {epoch+1}/{self.epochs}] "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        # Save best + last
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if self.save_check:
                self._save_checkpoint(self.best_ckpt)
                print("Saved new best model.")
        if self.save_check:
            self._save_checkpoint(self.last_ckpt)
            save_metrics(self.results, self.metrics_path)

    def train(self, epochs):
        print(f"Start training for {epochs} epochs")
        self.epochs = epochs

        for epoch in range(self.start_epoch, epochs):
            # Training
            self.model.train()
            train_loss, train_acc, total = 0.0, 0.0, 0

            for data, target in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                loss = self.loss_fn(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * target.size(0)
                train_acc += (output.argmax(1) == target).sum().item()
                total += target.size(0)

            train_loss /= total
            train_acc /= total
            self.model.eval()
            val_loss, val_acc, total_val = 0.0, 0.0, 0

            with torch.no_grad():
                for data, target in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    data, target = data.to(self.device), target.to(self.device)
                    if data.ndim == 5 and self.model_name == 'spatial_stream': 
                        # In val mode, take the average of frames for spatial stream
                        B, N, C, H, W = data.shape
                        data = data.view(B * N, C, H, W)
                        output = self.model(data)
                        output = output.view(B, N, -1).mean(dim=1)
                    else:
                        output = self.model(data)

                    loss = self.loss_fn(output, target)
                    val_loss += loss.item() * target.size(0)
                    val_acc += (output.argmax(1) == target).sum().item()
                    total_val += target.size(0)

            val_loss /= total_val
            val_acc /= total_val
            self._save_results(train_loss, train_acc, val_loss, val_acc, epoch)

        print("Training complete. Metrics and model checkpoints saved.")
        return self.results
