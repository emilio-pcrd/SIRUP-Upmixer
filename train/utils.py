import os
import pickle
import sys
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# schedule functions
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class LinearNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used in DDPM.
    """

    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        # Mimicking how compvis repo creates schedule
        self.betas = (
                torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        )
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)

    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the nosie predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)

            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0


class CosineNoiseScheduler:
    r"""
    Class for the cosine noise scheduler that is used in DDPM.
    """

    def __init__(self, num_timesteps, s=0.008):
        self.num_timesteps = num_timesteps
        self.s: float = s
        # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(self.betas, 0.0001, 0.9999)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)

    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the noise predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)

            return mean + sigma * z, x0


def generate_mask(image, cols=4):
    """_summary_

    Args:
        image (_type_): _description_
        cols (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
    B, C, H, W = image.shape
    mask = torch.zeros_like(image[:, 1, ...])
    mask[..., :cols] = 1.0
    # resize from 1024, 16 to 128, 128
    mask = mask.view(B, 1, 128, 128)

    # downsample to get latents
    down_rate = 4
    assert down_rate == 4, "wrong shape, please use good down rate"

    mask = F.interpolate(mask, size=(128//down_rate, 128//down_rate), mode='nearest')
    return mask


def get_metrics(scm_tests, ground_truth_scm):
    """compute and returns metrics for evaluation

    Args:
        scm_tests (torch.Tensor): reconstructed steering vectors
        ground_truth_scm (torch.Tensor): ground truth steering vectors

    Returns:
        tuple
    """

    mses, mses_imag, cosine_similarities = [], [], []
    NMSE = []

    def complex_nmse(pred, gt):
        num = torch.sum((pred - gt)**2)
        den = torch.sum(gt**2)
        return num / den

    for x_reco, x_gt in zip(scm_tests, ground_truth_scm):
        x_reco = torch.tensor(x_reco.squeeze(0))  # (2, 1024, 16)
        x_gt = torch.tensor(x_gt.squeeze(0))

        # Real cosine similarity
        cos_sim = F.cosine_similarity(
            x_reco[0].flatten(),  # real part
            x_gt[0].flatten(),
            dim=0
        )
        cosine_similarities.append(cos_sim.item())

        mses.append(F.mse_loss(x_reco[0], x_gt[0]).item())  # real
        mses_imag.append(F.mse_loss(x_reco[1], x_gt[1]).item())  # imag

        # Compute NMSE per frequency bin using both real & imag
        nmse_vals = []
        for f in range(x_reco.shape[1]):  # 1024 bins
            pred = x_reco[:, f, :]  # shape (2, 16)
            gt = x_gt[:, f, :]
            nmse_vals.append(complex_nmse(pred, gt).item())
        NMSE.append(nmse_vals)

    # Average NMSE over all test samples
    average_NMSE = torch.mean(torch.tensor(NMSE), dim=0)
    avgNMSE_dB = 10 * torch.log10(average_NMSE + 1e-10)
    return avgNMSE_dB, np.mean(mses), np.std(mses), np.mean(cosine_similarities), np.std(cosine_similarities)


def plot_nmse(nmse, savefig=True):
    # Plot NMSE in log scale (dB)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(nmse.shape[0]), nmse.numpy(), label="NMSE (dB)")
    plt.xlabel("Frequency Bin")
    plt.ylabel("NMSE (dB)")
    plt.title("Normalized Mean Squared Error (NMSE) in dB")
    plt.semilogx()
    plt.grid(True)
    plt.legend()
    if savefig:
        plt.savefig("/mnt/ssd-samsung/riken_copy/upmixing_ddpm/single_steer_vect/nmse_evol_low.pdf")
    plt.show()


def plot_error(gt, pred, savefig=True):
    error = gt - pred
    print(np.abs(np.mean(error)))
    plt.figure(figsize=(6, 3.5))
    plt.imshow(np.abs(error), aspect='auto', origin='lower', cmap='inferno')
    plt.title("Error Intensities (|Prediction - Ground Truth|)")
    plt.xlabel("Channels")
    plt.ylabel("Frequency Bin")
    plt.colorbar(label="Absolute Error")
    plt.tight_layout()
    if savefig:
        plt.savefig("runs/plots/reconstruction_error.pdf")
    plt.show()



def extract_steering_pairs(input_dir, output_dir):
    """
    Extract (a_16_1, a_4_1) and (a_16_2, a_4_2) pairs from MNMF dataset files and save for training.

    Args:
        input_dir (str): Path to folder with room_sim_xxxx.pkl files
        output_dir (str): Path to output folder for extracted pairs
    """

    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.startswith('room_sim_') and f.endswith('.pkl')]
    files.sort()

    print(f"Processing {len(files)} files...")

    for i, filename in enumerate(files):
        try:
            with open(os.path.join(input_dir, filename), 'rb') as f:
                data = pickle.load(f)

            # Extract steering vectors
            if 'a_16_1' in data and 'a_16_2' in data:
                data['a_16_1'] = torch.tensor(data['a_16_1'][:-1])
                data['a_16_2'] = torch.tensor(data['a_16_2'][:-1])
                data['a_4_1'] = torch.tensor(data['a_4_1'][:-1])
                data['a_4_2'] = torch.tensor(data['a_4_2'][:-1])

                output_file_1 = os.path.join(output_dir, f'pair_1{i:04d}.pkl')
                with open(output_file_1, 'wb') as f1:
                    pickle.dump(data, f1)

                output_file_2 = os.path.join(output_dir, f'pair_2{i:04d}.pkl')
                with open(output_file_2, 'wb') as f2:
                    pickle.dump(data, f2)

            if i % 100 == 0:
                print(f"Processed {i}/{len(files)}")

        except Exception as e:
            print(f"Error with {filename}: {e}")
            continue

    print(f"Done! Saved pairs to {output_dir}")
