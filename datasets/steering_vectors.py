from __future__ import annotations

import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset


@dataclass
class SteeringVectorSample:
    foa: np.ndarray
    hoa: np.ndarray
    index: int
    folder: str


class SteeringVectorDataset(Dataset):
    """Dataset for steering vectors stored in per-room `.pkl` files.

    Each `.pkl` file is expected to contain `svect_hoa` and `svect_foa` arrays.
    Files follow the pattern `room_sim_XXXX.pkl` within subdirectories.
    """

    def __init__(
        self,
        base_data_dir: str | Path,
        get_idx: bool = False,
        num_files_per_folder: int = 1500,
        noise_std: float = 0.0,
        shuffle_index: bool = True,
        fallback_random: bool = True,
    ) -> None:
        super().__init__()
        self.base_data_dir = Path(base_data_dir)
        self.get_idx = get_idx
        self.num_files_per_folder = num_files_per_folder
        self.noise_std = noise_std
        self.shuffle_index = shuffle_index
        self.fallback_random = fallback_random

        if not self.base_data_dir.exists():
            raise FileNotFoundError(f"Base data directory not found: {self.base_data_dir}")

        ignore_folders = {
            "__pycache__",
            ".git",
            ".vscode",
            ".idea",
            "node_modules",
            ".DS_Store",
        }

        self.data_folders: List[Path] = []
        for item in os.listdir(self.base_data_dir):
            if item in ignore_folders or item.startswith("."):
                continue
            folder_path = self.base_data_dir / item
            if folder_path.is_dir():
                pkl_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
                if pkl_files:
                    self.data_folders.append(folder_path)
                else:
                    print(f"Skipping folder {item} - no .pkl files found")

        self.data_folders.sort()
        if not self.data_folders:
            raise ValueError(f"No valid data folders found in {self.base_data_dir}")

        self.total_files = len(self.data_folders) * self.num_files_per_folder
        self.index_mapping: List[Tuple[int, int]] = []
        for folder_idx in range(len(self.data_folders)):
            for file_idx in range(self.num_files_per_folder):
                self.index_mapping.append((folder_idx, file_idx))

        if self.shuffle_index:
            random.shuffle(self.index_mapping)

    def __len__(self) -> int:
        return self.total_files

    def __getitem__(self, idx: int):
        folder_idx, file_idx = self.index_mapping[idx]
        folder_path = self.data_folders[folder_idx]
        data_path = folder_path / f"room_sim_{file_idx:04d}.pkl"

        if not data_path.exists():
            if not self.fallback_random:
                raise FileNotFoundError(f"Missing data file: {data_path}")
            print(f"Warning: File {data_path} not found, selecting random alternative", file=sys.stderr)
            random_folder_idx = random.randint(0, len(self.data_folders) - 1)
            random_file_idx = random.randint(0, self.num_files_per_folder - 1)
            folder_path = self.data_folders[random_folder_idx]
            data_path = folder_path / f"room_sim_{random_file_idx:04d}.pkl"

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        if "svect_hoa" not in data or "svect_foa" not in data:
            raise KeyError(f"Missing svect_hoa/svect_foa in {data_path}")

        svect_hoa = data["svect_hoa"]
        svect_foa = data["svect_foa"]
        if self.noise_std > 0:
            svect_foa = svect_foa + np.random.randn(*svect_hoa.shape) * self.noise_std

        if self.get_idx:
            return svect_foa, svect_hoa, idx

        return svect_foa, svect_hoa
