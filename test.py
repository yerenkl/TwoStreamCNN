import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import sys
import logging
from tqdm import tqdm

from datasets import SpatialStreamDataset, TemporalStreamDataset, TwoStreamVideoDataset
from model import SpatialStream, TemporalStream
from utils import save_metrics
from config import ROOT_DIR, SAVE_DIR
from transforms import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_tfms_rgb = spatial_eval_tfms()

test_tfms_flow = flow_tfms()

test_dataset_spatial = SpatialStreamDataset(
    root_dir=ROOT_DIR, split='test', transform=test_tfms_rgb
)
test_loader_spatial = DataLoader(
    test_dataset_spatial, batch_size=32, shuffle=False, num_workers=0
)

test_dataset_temporal = TemporalStreamDataset(
    root_dir=ROOT_DIR, split='test',
    transform_flow=test_tfms_flow)

test_loader_temporal = DataLoader(
    test_dataset_temporal, batch_size=32, shuffle=False, num_workers=0
)

test_dataset_twostream = TwoStreamVideoDataset(
    root_dir=ROOT_DIR, split='test',
    transform_flow=test_tfms_flow, transform_rgb=test_tfms_rgb
)
test_loader_twostream = DataLoader(
    test_dataset_twostream, batch_size=32, shuffle=False, num_workers=0
)

# Load models
models = [SpatialStream(10).to(device), TemporalStream().to(device)]

for indx, weight_name in enumerate(['spatial_stream_best.pth', 'temporal_stream_best.pth']):
    weights = os.path.join(SAVE_DIR, weight_name)
    ckpt = torch.load(weights, map_location=device)
    models[indx].load_state_dict(ckpt["model_state"])

criterion = nn.CrossEntropyLoss()

def evaluate_single(model, loader, name="Model"):
    """Evaluate single model performance."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in tqdm(loader, desc=f"{name} [Eval]", leave=False):
            data, target = data.to(device), target.to(device)
            if data.ndim == 5 and name == "Spatial Stream":
                B, N, C, H, W = data.shape
                data = data.view(B * N, C, H, W)
                outputs = model(data)
                outputs = outputs.view(B, N, -1).mean(dim=1)
            else:
                outputs = model(data)

            loss = criterion(outputs, target)
            total_loss += loss.item() * target.size(0)
            correct += (outputs.argmax(1) == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"{name:>15} | Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}%")
    return acc, avg_loss


print("Evaluating Spatial Stream...")
spatial_acc, spatial_loss = evaluate_single(models[0], test_loader_spatial, "Spatial Stream")

print("Evaluating Temporal Stream...")
temporal_acc, temporal_loss = evaluate_single(models[1], test_loader_temporal, "Temporal Stream")

# Fusion evaluation
print("Evaluating Fused Predictions")
fusion_correct, fusion_total, fusion_loss = 0, 0, 0.0
with torch.no_grad():
    for rgb, flow, target in tqdm(test_loader_twostream, desc="Fusion [Eval]", leave=False):
        rgb, flow, target = rgb.to(device), flow.to(device), target.to(device)
        logits_rgb = models[0](rgb)
        logits_flow = models[1](flow)
        logits_fused = (logits_rgb + logits_flow) / 2.0

        loss = criterion(logits_fused, target)
        fusion_loss += loss.item() * target.size(0)
        preds = logits_fused.argmax(1)
        fusion_correct += (preds == target).sum().item()
        fusion_total += target.size(0)

fusion_acc = fusion_correct / fusion_total
fusion_loss = fusion_loss / fusion_total

print(f"Spatial Stream  Acc: {100*spatial_acc:.2f}%")
print(f"Temporal Stream Acc: {100*temporal_acc:.2f}%")
print(f"Fused (Average) Acc: {100*fusion_acc:.2f}%")
