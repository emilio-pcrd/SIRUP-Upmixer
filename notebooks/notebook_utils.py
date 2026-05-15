from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyroomacoustics as pra


@dataclass
class RoomConfig:
    room_dim: Tuple[float, float, float] = (6.0, 5.5, 4.5)
    fs: int = 16000
    rt60: float = 0.3
    snr_db: float = 20.0
    max_order: int | None = 2
    mic_radius: float = 0.042
    mic_count: int = 16


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_room(config: RoomConfig) -> pra.ShoeBox:
    e_absorption, max_order = pra.inverse_sabine(config.rt60, config.room_dim)
    if config.max_order is not None:
        max_order = config.max_order
    room = pra.ShoeBox(
        config.room_dim,
        fs=config.fs,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    return room


def add_microphones(room: pra.ShoeBox, config: RoomConfig) -> np.ndarray:
    mic_center_x = config.room_dim[0] / 2 + random.uniform(-0.5, 0.5)
    mic_center_y = config.room_dim[1] / 2 + random.uniform(-0.5, 0.5)
    mic_center = np.c_[[mic_center_x, mic_center_y, 1.0]]
    sphere_points = pra.doa.GridSphere(config.mic_count)
    mic_positions = mic_center + config.mic_radius * sphere_points.cartesian
    mic_array = pra.MicrophoneArray(mic_positions, config.fs)
    room.add_microphone_array(mic_array)
    return mic_positions


def add_source(room: pra.ShoeBox, signal: np.ndarray, position: Tuple[float, float, float]) -> None:
    room.add_source(position, signal=signal, delay=0)


def simulate_room(room: pra.ShoeBox, snr_db: float) -> np.ndarray:
    room.simulate()
    signals = room.mic_array.signals
    noise = np.random.randn(*signals.shape)
    signal_power = np.mean(signals**2)
    noise = noise * np.sqrt(signal_power / (10 ** (snr_db / 10)))
    return signals + noise


def compute_scm(signals: np.ndarray, nfft: int = 2048, hop: int = 512) -> np.ndarray:
    win = pra.hamming(nfft)
    stft = pra.transform.stft.analysis(signals.T, nfft, hop, win)
    x_ft = np.transpose(stft, (1, 0, 2))
    scm = np.einsum("ftm,ftn->fmn", x_ft, np.conjugate(x_ft)) / x_ft.shape[1]
    return np.transpose(scm, (1, 2, 0))


def compute_steering_vector(scm: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(scm)
    idx = np.argmax(eigvals, axis=0)
    steering = np.stack([eigvecs[:, :, f][:, idx[f]] for f in range(scm.shape[-1])], axis=0)
    return steering


def svect_to_real_imag(svect: np.ndarray) -> np.ndarray:
    return np.stack([svect.real, svect.imag], axis=0)
