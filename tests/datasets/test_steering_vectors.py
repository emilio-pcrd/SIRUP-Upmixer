import pickle
from pathlib import Path

import numpy as np

from datasets.steering_vectors import SteeringVectorDataset


def test_steering_vector_dataset(tmp_path: Path) -> None:
    room_dir = tmp_path / "room_000"
    room_dir.mkdir(parents=True, exist_ok=True)

    sample = {
        "svect_foa": np.zeros((2, 4, 4), dtype=np.float32),
        "svect_hoa": np.zeros((2, 4, 4), dtype=np.float32),
    }

    sample_path = room_dir / "room_sim_0000.pkl"
    with sample_path.open("wb") as f:
        pickle.dump(sample, f)

    dataset = SteeringVectorDataset(tmp_path, num_files_per_folder=1, noise_std=0.0)
    foa, hoa = dataset[0]
    assert foa.shape == sample["svect_foa"].shape
    assert hoa.shape == sample["svect_hoa"].shape
