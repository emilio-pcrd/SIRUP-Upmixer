import numpy as np
import pyroomacoustics as pra
import random
import pickle
import os, sys
import torch

import warnings

from tqdm import tqdm
warnings.filterwarnings("ignore")

sys.path.insert(0, '../')
from utils_beam import select_two_random_flacs, select_one_random_flac, load_sample, compute_steering_vector, svect2complex, complex2svect

# Create directory to save simulations
os.makedirs("mnmf_dataset", exist_ok=True)

# Number of simulations
num_simulations = 1000
fs = 16000

L, hop, win = 2048, 512, pra.hamming(2048)
for idx in tqdm(range(num_simulations)):
    # Load random signals
    signals_random_1_PATH, signals_random_2_PATH = select_two_random_flacs('/...')
    signals_random_1, fs_ = load_sample(signals_random_1_PATH)
    signals_random_2, fs_ = load_sample(signals_random_2_PATH)

    while len(signals_random_1.numpy().squeeze()) < 4*fs:
        signals_random_1_PATH = select_one_random_flac('/...')
        signals_random_1, fs_ = load_sample(signals_random_1_PATH[0])
    while len(signals_random_2.numpy().squeeze()) < 4*fs:
        signals_random_2_PATH = select_one_random_flac('/...')
        signals_random_2, fs_ = load_sample(signals_random_2_PATH[0])

    signals_random_1 = signals_random_1[..., :4*fs].numpy().squeeze()
    signals_random_2 = signals_random_2[..., :4*fs].numpy().squeeze()

    attenuation_db = 20
    attenuation_linear = 10**(-attenuation_db / 20)
    signals_random_2 = signals_random_2 * attenuation_linear

    print(f"Source 1 RMS: {np.sqrt(np.mean(signals_random_1**2)):.4f}")
    print(f"Source 2 RMS: {np.sqrt(np.mean(signals_random_2**2)):.4f}")
    print(f"Attenuation: {attenuation_db} dB ({attenuation_linear:.4f} linear)")

    # Setup
    rs = [1, 1.1, 1.2, 1.3, 1.4, 1.5]

    # Source setup
    doa_deg = random.uniform(0, 359)
    r = np.random.choice(rs)
    r2 = np.random.choice(rs)
    doa_radian = np.deg2rad(doa_deg)

    angle_separation = random.uniform(100, 210)
    doa_deg2 = (doa_deg + angle_separation) % 360
    doa_radian2 = np.deg2rad(doa_deg2)

    # Room settings
    room_dim = [random.uniform(4, 6), random.uniform(4, 6), random.uniform(2.5, 4)]
    # Microphone center with randomness around room center
    mic_center_x = room_dim[0]/2 + random.uniform(-0.3, 0.3)
    mic_center_y = room_dim[1]/2 + random.uniform(-0.3, 0.3)
    mic_center = np.c_[[mic_center_x, mic_center_y, 1]]

    snr = random.uniform(10, 25)
    rt60 = random.uniform(0.2, 0.5)
    absorption, max_order_ = pra.inverse_sabine(rt60, room_dim)

    # Source locations
    source_loc1 = mic_center[:, 0] + np.array([r*np.cos(doa_radian), r*np.sin(doa_radian), 0.4])
    source_loc2 = mic_center[:, 0] + np.array([r2*np.cos(doa_radian2), r2*np.sin(doa_radian2), 0.5])

    # Check if source locations are inside the room
    def is_inside_room(loc, room_dim):
        return all(0 <= loc[i] <= room_dim[i] for i in range(3))

    while not is_inside_room(source_loc1, room_dim):  # or not is_inside_room(source_loc2, room_dim):
        print(f"Warning: One or both sources are outside the room boundaries. Repeating simulation {idx + 1}.")

        # Source setup
        doa_deg = random.uniform(0, 359)
        r = np.random.choice(rs)
        r2 = np.random.choice(rs)
        doa_radian = np.deg2rad(doa_deg)
        angle_separation = random.uniform(0, 360)
        doa_radian2 = np.deg2rad((doa_deg + angle_separation) % 360)

        # Room settings
        room_dim = [random.uniform(4, 6), random.uniform(4, 6), random.uniform(2.5, 4)]
        mic_center_x = room_dim[0]/2 + random.uniform(-0.3, 0.3)
        mic_center_y = room_dim[1]/2 + random.uniform(-0.3, 0.3)
        mic_center = np.c_[[mic_center_x, mic_center_y, 1]]

        # Source locations
        source_loc1 = mic_center[:, 0] + np.array([r*np.cos(doa_radian), r*np.sin(doa_radian), 0.4])
        source_loc2 = mic_center[:, 0] + np.array([r2*np.cos(doa_radian2), r2*np.sin(doa_radian2), 0.4])

    # Microphone array
    radius = 0.042
    sphere_points = pra.doa.GridSphere(16)
    mic_pos_16 = mic_center + radius * sphere_points.cartesian
    mics_16 = pra.MicrophoneArray(mic_pos_16, fs)

    # Create and simulate room
    room_16 = pra.ShoeBox(room_dim, fs=fs, absorption=absorption, max_order=max_order_)
    room_16.add_source(source_loc1, signal=signals_random_1, delay=0)
    room_16.add_source(source_loc2, signal=signals_random_2, delay=2)
    room_16.add_microphone_array(mics_16)

    signals_16 = room_16.simulate(return_premix=True)
    mixture_clean = room_16.mic_array.signals

    # SNR
    signal_power = np.mean(mixture_clean**2)
    noise_power = signal_power / (10**(snr/10))
    noise = np.sqrt(noise_power) * np.random.randn(*mixture_clean.shape)
    mixture = mixture_clean + noise

    ref_mic_idx = 0
    source_1_at_ref_mic = signals_16[0, ref_mic_idx, :]  # Shape: (n_samples,)
    source_2_at_ref_mic = signals_16[1, ref_mic_idx, :]  # Shape: (n_samples,)
    bss_reference_sources = np.vstack([
        source_1_at_ref_mic,  # Source 1 (target)
        source_2_at_ref_mic   # Source 2 (interference)
    ])

    x_stft = pra.transform.stft.analysis(mixture[:, :2*16000].T, L, hop, win)  # (n_frames, freq_bins, n_mics)
    X_FT_16 = torch.tensor(x_stft, dtype=torch.complex64).permute(1, 0, 2)  # (freq_bins, n_frames, n_mics)
    X_FT_16 = X_FT_16
    X_FT_4 = X_FT_16[:, :, :4]
    SCM_16 = torch.einsum("ftm,ftn->fmn", X_FT_16, X_FT_16.conj()) / X_FT_16.shape[1]
    SCM_4 = torch.einsum("ftm,ftn->fmn", X_FT_4, X_FT_4.conj()) / X_FT_4.shape[1]
    SCM_16, SCM_4 = SCM_16.permute(1, 2, 0).cpu().numpy(), SCM_4.permute(1, 2, 0).cpu().numpy()  # (M, M, F)
    # Compute steering vectors from SCM
    a_foa = compute_steering_vector(SCM_4)
    a_hoa = compute_steering_vector(SCM_16)

    # real + imag concat in first channel
    a_foa_data = complex2svect(a_foa.T)  # (2, F, M)
    a_hoa_data = complex2svect(a_hoa.T)  # (2, F, M)

    # pad foa with zeros for channels 5-16
    a_foa_padded = np.zeros((2, 1025, 16))
    a_foa_padded[:, :, :4] = a_foa_data

    # Save data
    data = {
        "mixture": mixture,
        "doa_radian": doa_radian,
        "doa_radian_2": doa_radian2,
        "r1": r,
        "r2": r2,
        "rt60": rt60,
        "snr": snr,
        "room_dim": room_dim,
        "mic_positions": mic_pos_16,
        "fs": fs,
        "signals_16": signals_16,
        "max_order": max_order_,
        "source_1": signals_random_1,
        "source_2": signals_random_2,
        "source_1_at_ref_mic": source_1_at_ref_mic,
        "source_2_at_ref_mic": source_2_at_ref_mic,
        "bss_reference_sources": bss_reference_sources,
        "ref_mic_idx": ref_mic_idx,
        "svect_foa": torch.tensor(a_foa_padded[:, :-1, :]),
        "svect_hoa": torch.tensor(a_hoa_data[:, :-1, :]),
    }

    filename = f"mnmf_dataset/room_sim_{idx:04d}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved simulation {idx + 1}/{num_simulations} to {filename}")
