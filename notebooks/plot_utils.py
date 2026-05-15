from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_svect_magnitude(svect: np.ndarray, title: str = "Steering vector magnitude") -> None:
    magnitude = np.abs(svect)
    plt.figure(figsize=(6, 3))
    plt.imshow(magnitude.T, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("Frequency bin")
    plt.ylabel("Microphone")
    plt.colorbar()
    plt.tight_layout()


def plot_waveform(signal: np.ndarray, title: str = "Waveform") -> None:
    plt.figure(figsize=(6, 2))
    plt.plot(signal)
    plt.title(title)
    plt.tight_layout()
