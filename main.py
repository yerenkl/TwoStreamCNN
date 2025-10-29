import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from model import SpatialStream, TemporalStream
from datasets import SpatialStreamDataset, TwoStreamVideoDataset, TemporalStreamDataset
from trainer import Trainer
from utils import load_metrics
from config import *

def get_args():
    parser = argparse.ArgumentParser(description="Train Spatial or Temporal Stream CNN")
    parser.add_argument("--stream", choices=["spatial", "temporal"], required=True,
                        help="Select which stream to train")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--resume", action="store_true", help="Resume training if checkpoints exist")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--root_dir", default=ROOT_DIR)
    parser.add_argument("--save_dir", default=SAVE_DIR)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"\nTraining {args.stream.upper()} stream.\n")

    # Data transformations
    train_tfms_rgb = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfms_rgb = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_tfms_flow = T.Compose([T.Resize(224)])

    # Loading datasets
    if args.stream == "spatial":
        train_ds = SpatialStreamDataset(args.root_dir, split="train", transform=train_tfms_rgb)
        val_ds = SpatialStreamDataset(args.root_dir, split="val", transform=eval_tfms_rgb)
        model = SpatialStream(n_class=NUM_CLASSES)
        
    else:
        train_ds = TemporalStreamDataset(args.root_dir, split="train",
                                         transform_flow=train_tfms_flow)
        val_ds = TemporalStreamDataset(args.root_dir, split="val",
                                       transform_flow=train_tfms_flow)
        model = TemporalStream(n_class=NUM_CLASSES)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        model_name=f"{args.stream}_stream",
        resume=args.resume,
        save_check=True
    )
    
    results = trainer.train(args.epochs)

    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Metrics saved at: {trainer.metrics_path}")


if __name__ == "__main__":
    main()
