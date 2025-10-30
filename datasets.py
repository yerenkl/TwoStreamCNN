import os
import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import numpy as np
import pandas as pd
import random

class SpatialStreamDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')

        self.video_dirs = sorted(glob(f'{root_dir}/frames/{split}/*/*'))
        self.labels = []
        for v in self.video_dirs:
            video_name = os.path.basename(v)
            meta = self.df.loc[self.df['video_name'] == video_name]
            if not meta.empty:
                self.labels.append(meta['label'].item())
            else:
                self.labels.append(-1)  

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        label = self.labels[idx]
        frame_paths = sorted(glob(os.path.join(video_dir, '*.jpg')))

        if self.split == 'train': # train mode network input is a single random frame
            frame_path = random.choice(frame_paths)
            imgs = self.transform(Image.open(frame_path).convert('RGB'))
        else:  # val mode, we average predictions of all frames
            imgs = []
            for i in range(9):
                img = Image.open(frame_paths[i]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            imgs = torch.stack(imgs, dim=0)
        return imgs, label

class TwoStreamVideoDataset(Dataset):
    """Stacks RGB and Flow data for fusion operation."""
    def __init__(self, 
        root_dir='/dtu/datasets1/02516/ucf101_noleakage', 
        split='train', 
        transform_rgb=None, 
        transform_flow=None,
        n_frames=10
    ):
        self.flow_paths = sorted(glob(f'{root_dir}/flows/{split}/*/*'))
        self.rgb_paths  = sorted(glob(f'{root_dir}/frames/{split}/*/*'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.transform_rgb = transform_rgb
        self.transform_flow = transform_flow
        self.n_frames = n_frames
        self.n_sampled_flows = n_frames - 1
        assert len(self.rgb_paths) == len(self.flow_paths), "RGB/Flow mismatch!"

    def __len__(self):
        return len(self.flow_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        flow_dir = self.flow_paths[idx]
        rgb_dir  = self.rgb_paths[idx]
        video_name = os.path.basename(flow_dir)
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        # RGB frame
        rgb_frame = self.load_single_frame(rgb_dir)
        if self.transform_rgb:
            rgb_frame = self.transform_rgb(rgb_frame)

        # Flows
        flows = []
        for i in range(1, self.n_sampled_flows + 1):
            flow_file = os.path.join(flow_dir, f'flow_{i}_{i+1}.npy')
            flow = torch.from_numpy(np.load(flow_file)) 
            flows.append(flow)
        if self.transform_flow:
            flows = [self.transform_flow(f) for f in flows]
        flows = torch.stack(flows, dim=0)

        return rgb_frame, flows, label

    def load_single_frame(self, rgb_dir):
        i = random.randint(1, self.n_frames)
        path = os.path.join(rgb_dir, f'frame_{i}.jpg')
        return Image.open(path).convert('RGB')

class TemporalStreamDataset(Dataset):
    def __init__(self, 
        root_dir='/dtu/datasets1/02516/ucf101_noleakage', 
        split='train', 
        transform_flow=None,
        n_frames=10
    ):
        self.flow_paths = sorted(glob(f'{root_dir}/flows/{split}/*/*'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.transform_flow = transform_flow
        self.n_frames = n_frames
        self.n_sampled_flows = n_frames - 1

    def __len__(self):
        return len(self.flow_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        flow_dir = self.flow_paths[idx]
        video_name = os.path.basename(flow_dir)
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        flows = []
        for i in range(1, self.n_sampled_flows + 1):
            flow_file = os.path.join(flow_dir, f'flow_{i}_{i+1}.npy')
            flow = torch.from_numpy(np.load(flow_file)) 
            flows.append(flow)
        if self.transform_flow:
            flows = [self.transform_flow(f) for f in flows]
        flows = torch.stack(flows, dim=0)
        return flows, label