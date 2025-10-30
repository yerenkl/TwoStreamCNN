import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import SpatialStreamDataset, TemporalStreamDataset, TwoStreamVideoDataset
from model import SpatialStream, TemporalStream
from config import ROOT_DIR, SAVE_DIR, NUM_CLASSES
from data.transforms import spatial_eval_tfms, flow_tfms
from engine.evaluator import Evaluator

def get_args():
    parser = argparse.ArgumentParser(description="Test Two-Stream CNN")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--root_dir", default=ROOT_DIR)
    parser.add_argument("--save_dir", default=SAVE_DIR)

    parser.add_argument("--spatial_weights", default="spatial_stream_best.pth")
    parser.add_argument("--temporal_weights", default="temporal_stream_best.pth")

    parser.add_argument("--evaluate_spatial", action="store_true")
    parser.add_argument("--evaluate_temporal", action="store_true")
    parser.add_argument("--evaluate_fusion", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device(args.device)

    # Transforms
    test_tfms_rgb = spatial_eval_tfms()
    test_tfms_flow = flow_tfms()

    # Datasets & loaders
    test_loader_spatial = DataLoader(
        SpatialStreamDataset(args.root_dir, split="test", transform=test_tfms_rgb),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    test_loader_temporal = DataLoader(
        TemporalStreamDataset(args.root_dir, split="test", transform_flow=test_tfms_flow),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    test_loader_twostream = DataLoader(
        TwoStreamVideoDataset(args.root_dir, split="test",
                              transform_rgb=test_tfms_rgb, transform_flow=test_tfms_flow),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Load models
    spatial_model = SpatialStream(NUM_CLASSES).to(device)
    temporal_model = TemporalStream().to(device)

    spatial_path = os.path.join(args.save_dir, args.spatial_weights)
    temporal_path = os.path.join(args.save_dir, args.temporal_weights)

    spatial_model.load_state_dict(torch.load(spatial_path, map_location=device)["model_state"])
    temporal_model.load_state_dict(torch.load(temporal_path, map_location=device)["model_state"])

    evaluator = Evaluator(device=device)

    # Evaluate
    if args.evaluate_spatial or (not args.evaluate_temporal and not args.evaluate_fusion):
        print("\nEvaluating Spatial Stream...")
        s_acc, _ = evaluator.evaluate_single(spatial_model, test_loader_spatial, "Spatial Stream", is_spatial=True)
        print(f"Spatial Stream Acc: {s_acc * 100:.2f}%")

    if args.evaluate_temporal:
        print("\nEvaluating Temporal Stream...")
        t_acc, _ = evaluator.evaluate_single(temporal_model, test_loader_temporal, "Temporal Stream", is_spatial=False)
        print(f"Temporal Stream Acc: {t_acc * 100:.2f}%")

    if args.evaluate_fusion:
        print("\nEvaluating Fusion...")
        f_acc, _ = evaluator.evaluate_fusion(spatial_model, temporal_model, test_loader_twostream)
        print(f"Fused Stream Acc: {f_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
