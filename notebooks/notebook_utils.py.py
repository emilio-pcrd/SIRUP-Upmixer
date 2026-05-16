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


# ── Frequency utilities ────────────────────────────────────────────────────

def get_freqs(nfft: int = 2048, fs: int = 16000) -> np.ndarray:
    """Return the positive-frequency bins for an STFT of size `nfft`."""
    return np.fft.rfftfreq(nfft, d=1 / fs)


# ── Steering vector helpers ────────────────────────────────────────────────

def svect_from_scm(scm: np.ndarray, k: int = 0) -> np.ndarray:
    """Extract the k-th principal steering vector from a batch of SCMs.

    Args:
        scm: Spatial covariance matrices, shape (M, M, F).
        k:   Index of the eigenvector to return (0 = dominant).

    Returns:
        svect: Complex steering vectors, shape (F, M).
    """
    n_freq = scm.shape[-1]
    svect = np.zeros((n_freq, scm.shape[0]), dtype=complex)
    for f in range(n_freq):
        R = scm[..., f]
        eigvals, eigvecs = np.linalg.eigh(R / np.trace(R))
        idx = np.argsort(eigvals)[::-1][k]
        svect[f] = eigvecs[:, idx]
    return svect


def a_theoric(mic_positions: np.ndarray, theta: float, freq: float,
              c: float = 343.0) -> np.ndarray:
    """Compute the free-field steering vector for a plane wave at angle `theta`.

    Args:
        mic_positions: Array of shape (3, M) — mic positions relative to centre.
        theta:         Azimuth angle in radians.
        freq:          Frequency in Hz.
        c:             Speed of sound (m/s).

    Returns:
        a: Complex steering vector of length M.
    """
    M = mic_positions.shape[-1]
    k = 2 * np.pi * freq / c
    d = np.array([np.cos(theta), np.sin(theta), 0.0])
    a = np.exp(-1j * k * (mic_positions.T @ d))   # (M,)
    return a


# ── Beamforming ────────────────────────────────────────────────────────────

def compute_beampattern(
    a_f: np.ndarray,
    mic_positions: np.ndarray,
    freqs: float,
    c: float,
    n_angles: int = 360,
    plane: str = "xy",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the normalised beampattern for a steering vector `a_f`.

    Args:
        a_f:           Complex beamforming weights, shape (M,).
        mic_positions: Mic positions, shape (3, M).
        freqs:         Frequency in Hz (single bin).
        c:             Speed of sound (m/s).
        n_angles:      Number of scan angles.
        plane:         Scan plane — 'xy', 'xz', or 'yz'.

    Returns:
        angles:      Scan angles in radians, shape (n_angles,).
        beampattern: Normalised response, shape (n_angles,).
    """
    k = 2 * np.pi * freqs / c
    angles = np.linspace(0, 2 * np.pi, n_angles)
    beampattern = np.zeros(n_angles)
    for i, theta in enumerate(angles):
        if plane == "xy":
            u = np.array([np.cos(theta), np.sin(theta), 0.0])
        elif plane == "xz":
            u = np.array([np.cos(theta), 0.0, np.sin(theta)])
        elif plane == "yz":
            u = np.array([0.0, np.cos(theta), np.sin(theta)])
        else:
            raise ValueError(f"plane must be 'xy', 'xz' or 'yz', got '{plane}'.")
        d_f = np.exp(-1j * k * (u @ mic_positions))
        beampattern[i] = np.abs(np.vdot(a_f, d_f))
    beampattern /= beampattern.max() + 1e-12
    return angles, beampattern


def run_beampattern_evaluation(
    data: dict,
    ims_upmixed: "torch.Tensor",
    freq_range: np.ndarray,
    c: float = 343.0,
    nfft: int = 2048,
    fs: int = 16000,
) -> tuple[list, list, list, list, np.ndarray]:
    """Compute frequency-averaged beampatterns for FOA, HOA, VAE-reco, and upmixed.

    Args:
        data:        Loaded scene dict with keys 'mic_positions', 'svect_foa',
                     'svect_hoa', 'doa_radian'.
        ims_upmixed: Upmixed steering vectors from the diffusion model, shape
                     (1, 2, F, M) or (2, F, M).
        freq_range:  Array of frequency bin indices to average over.
        c:           Speed of sound (m/s).
        nfft, fs:    STFT parameters (for `get_freqs`).

    Returns:
        results_foa, results_hoa, results_reco, results_cond_reco: Lists of
            beampatterns (one per frequency bin).
        angles: The common scan angles in radians.
    """
    mic_pos_16 = data["mic_positions"]
    mic_pos_4 = mic_pos_16[:, :4]
    freqs = get_freqs(nfft, fs)

    results_foa, results_hoa, results_reco, results_cond_reco = [], [], [], []

    for freq_bin in freq_range:
        freq_bin = int(freq_bin)
        freq = freqs[freq_bin]

        # FOA ground-truth steering vector
        svect_foa = data["svect_foa"]
        a_f_foa = (svect_foa[0, freq_bin, :4] + 1j * svect_foa[1, freq_bin, :4])
        if hasattr(a_f_foa, "numpy"):
            a_f_foa = a_f_foa.numpy()
        angles, bp_foa = compute_beampattern(a_f_foa, mic_pos_4, freq, c)
        results_foa.append(bp_foa)

        # HOA ground-truth steering vector
        svect_hoa = data["svect_hoa"]
        a_f_hoa = (svect_hoa[0, freq_bin, :] + 1j * svect_hoa[1, freq_bin, :])
        if hasattr(a_f_hoa, "numpy"):
            a_f_hoa = a_f_hoa.numpy()
        _, bp_hoa = compute_beampattern(a_f_hoa, mic_pos_16, freq, c)
        results_hoa.append(bp_hoa)

        # VAE reconstruction
        ims = ims_upmixed.squeeze()
        a_f_reco = ims[0, freq_bin, :] + 1j * ims[1, freq_bin, :]
        if hasattr(a_f_reco, "numpy"):
            a_f_reco = a_f_reco.numpy()
        elif hasattr(a_f_reco, "detach"):
            a_f_reco = a_f_reco.detach().cpu().numpy()
        _, bp_reco = compute_beampattern(a_f_reco, mic_pos_16, freq, c)
        results_reco.append(bp_reco)

        # Diffusion upmixer (conditional)
        a_f_cond = ims[0, freq_bin, :] + 1j * ims[1, freq_bin, :]
        if hasattr(a_f_cond, "detach"):
            a_f_cond = a_f_cond.detach().cpu().numpy()
        _, bp_cond = compute_beampattern(a_f_cond, mic_pos_16, freq, c)
        results_cond_reco.append(bp_cond)

    return results_foa, results_hoa, results_reco, results_cond_reco, angles


def plot_beampatterns_2d(
    results_foa: list,
    results_hoa: list,
    results_reco: list,
    results_cond_reco: list,
    angles: np.ndarray,
    doa_radian: float,
    save_path: str | None = None,
) -> None:
    """Four-panel polar plot comparing FOA / HOA / VAE-reco / Upmixed beampatterns.

    Args:
        results_*:   Lists of beampatterns returned by `run_beampattern_evaluation`.
        angles:      Scan angles in radians.
        doa_radian:  Ground-truth source azimuth in radians.
        save_path:   If provided, save the figure to this path (.pdf recommended).
    """
    import matplotlib.pyplot as plt

    mean_bp = {
        "GT 4-mics":              np.array(results_foa).mean(axis=0),
        "GT 16-mics":             np.array(results_hoa).mean(axis=0),
        "VAE 16-mics":            np.array(results_reco).mean(axis=0),
        "Upmixed 16-mics":        np.array(results_cond_reco).mean(axis=0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 10),
                              subplot_kw={"projection": "polar"})
    for ax, (title, bp) in zip(axes.flat, mean_bp.items()):
        ax.plot(angles, bp ** 2, label=title)
        ax.plot([doa_radian, doa_radian], [0, 1],
                linestyle="--", color="black", label="True angle")
        ax.scatter(doa_radian, 1, color="black", marker="*", s=200, zorder=5)
        ax.set_title(title, pad=12)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Frequency-averaged beampatterns", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()