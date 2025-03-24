import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
# Use a pipeline as a high-level helper
from transformers import pipeline
from glob import glob

SEQ_LENGTH = 16
IMG_SIZE = (224, 224)

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        frames = self.extract_frames(self.video_paths[idx])
        label = self.labels[idx]
        return frames, label
    
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, total_frames - 1, SEQ_LENGTH, dtype=int)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_idxs:
                frame = cv2.resize(frame, IMG_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)
        cap.release()
        return torch.stack(frames)
