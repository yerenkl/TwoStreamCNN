import torch
from tqdm import tqdm
import torch.nn as nn

class Evaluator:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def evaluate_single(self, model, loader, name="Model", is_spatial=False):
        model.eval()
        correct, total, total_loss = 0, 0, 0.0

        with torch.inference_mode():
            for data, target in tqdm(loader, desc=f"{name} [Eval]", leave=False):
                data, target = data.to(self.device), target.to(self.device)

                # Spatial stream forwarding per-frame
                if is_spatial and data.ndim == 5:
                    B, T, C, H, W = data.shape
                    data = data.view(B*T, C, H, W)
                    logits = model(data)
                    logits = logits.view(B, T, -1).mean(dim=1)
                else:
                    logits = model(data)

                loss = self.criterion(logits, target)
                total_loss += loss.item() * target.size(0)
                correct += (logits.argmax(1) == target).sum().item()
                total += target.size(0)

        return correct/total, total_loss/total

    def evaluate_fusion(self, spatial, temporal, loader):
        spatial.eval()
        temporal.eval()

        fusion_correct, fusion_total, fusion_loss = 0, 0, 0.0

        with torch.inference_mode():
            for rgb, flow, target in tqdm(loader, desc="Fusion [Eval]", leave=False):
                rgb, flow, target = rgb.to(self.device), flow.to(self.device), target.to(self.device)

                s = spatial(rgb)
                t = temporal(flow)
                logits = (s + t) / 2.0

                loss = self.criterion(logits, target)
                fusion_loss += loss.item() * target.size(0)
                fusion_correct += (logits.argmax(1) == target).sum().item()
                fusion_total += target.size(0)

        return fusion_correct/fusion_total, fusion_loss/fusion_total
