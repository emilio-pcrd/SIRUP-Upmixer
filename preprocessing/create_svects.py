import numpy as np
import torch
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import random

import os, sys
import yaml
import tqdm
import pickle
import json
import glob
from scipy.io import wavfile

from utils_paper import upmix_steering_vectors, svect2complex, complex2svect, select_data_for_testing, compute_steering_vector

sys.path.insert(0, '../fast_mnmf/')
from utils_beam import compute_beampattern, a_theoric, calculate_metrics
from utils_beam import select_two_random_flacs, load_sample, play_audio

import warnings
warnings.filterwarnings("ignore")

from IPython.display import Audio, display
def play_waveform(waveform, sr=16000, title="Audio"):
    waveform = waveform.detach().cpu() if isinstance(waveform, torch.Tensor) else waveform
    display(Audio(waveform, rate=sr, autoplay=False))

def get_freqs(nfft=2048, fs=16000):
    return np.fft.rfftfreq(nfft, d=1/fs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    # "data_dir": "rt60-0205/",
    "data_dir": "rt60-0507/",
    "nfft": 2048,
    "fs": 16000,
    "hop_size": 512,
    "input_dim": 1024,
    "output_dim": 1,
    "latent_dim": 64,
}


# Process all data files in the directory and add algebraic steering vectors
data_dir = params["data_dir"]
list_filename = os.listdir(data_dir)
print(data_dir)
print(f"Processing {len(list_filename)} files to add estimated steering vectors...")

L, hop, win = 2048, 512, pra.hamming(2048)
for filename in tqdm.tqdm(list_filename):
    file_path = os.path.join(data_dir, filename)

    # Load the data
    with open(file_path, "rb") as f:
        data = pickle.load(f)


    signals = data['mixture']
    print(signals.shape)
    mic_pos_16 = data['mic_positions']
    mic_pos_4 = mic_pos_16[..., :4]

    signals_16 = data['signals_16'][..., :int(2*16000)]  # take only first 2 sec for computation
    mix_wt_noise = signals_16.sum(axis=0)

    x_stft = pra.transform.stft.analysis(signals.T, L, hop, win)  # (n_frames, freq_bins, n_mics)
    X_FT_16 = torch.tensor(x_stft, dtype=torch.complex64).permute(1, 0, 2)  # (freq_bins, n_frames, n_mics)
    X_FT_16 = X_FT_16
    X_FT_4 = X_FT_16[:, :, :4]
    SCM_16 = torch.einsum("ftm,ftn->fmn", X_FT_16, X_FT_16.conj()) / X_FT_16.shape[1]
    SCM_4 = torch.einsum("ftm,ftn->fmn", X_FT_4, X_FT_4.conj()) / X_FT_4.shape[1]
    SCM_16, SCM_4 = SCM_16.permute(1, 2, 0).cpu().numpy(), SCM_4.permute(1, 2, 0).cpu().numpy()  # (M, M, F)

    # compute svects from scm
    a_foa = compute_steering_vector(SCM_4)
    a_hoa = compute_steering_vector(SCM_16)

    # real + imag concat in first channel
    a_foa_data = complex2svect(a_foa.T)  # (2, F, M)
    a_hoa_data = complex2svect(a_hoa.T)  # (2, F, M')

    # pad foa with zeros for channels 5-16
    a_foa_padded = np.zeros((2, 1025, 16))
    a_foa_padded[:, :, :4] = a_foa_data

    data['svect_foa_bis'] = torch.tensor(a_foa_padded[:, :-1, :])
    data['svect_hoa_bis'] = torch.tensor(a_hoa_data[:, :-1, :])

    with open(file_path, "wb") as f:
        pickle.dump(data, f)

print("All files processed and steering vectors added!")
