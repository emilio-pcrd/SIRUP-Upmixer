import os
import pickle
import yaml
import argparse

import torch
import torch.nn as nn
import random
import numpy as np
import glob

from tqdm import tqdm
# from CODE.upmixing_ddpm.models.vae import VAE, Discriminator
from models.vae import VAE
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from datasets import SVectDatasetMNMF


from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from torch.utils.data import Dataset


device = torch.device('cuda')


# class SVectDataset(Dataset):
#     def __init__(self, data_dir, get_idx=False):
#         super().__init__()
#         self.data_dir = data_dir
#         self.get_idx = get_idx

#     def __len__(self):
#         return len(os.listdir(self.data_dir))

#     def __getitem__(self, idx):
#         data_path = os.path.join(self.data_dir, f"room_sim_{idx:04d}.pkl")
#         with open(data_path, 'rb') as f:
#             data = pickle.load(f)

#         svect_hoa, svect_foa = data['svect_hoa'], data['svect_foa']
#         svect_hoa, svect_foa = svect_hoa, svect_foa
#         if self.get_idx:
#             return svect_foa, svect_hoa, idx

#         return svect_foa, svect_hoa


class SVectDataset(Dataset):
    def __init__(self, base_data_dir, get_idx=False, num_files_per_folder=1500):
        super().__init__()
        self.base_data_dir = base_data_dir
        self.get_idx = get_idx
        self.num_files_per_folder = num_files_per_folder

        # Find all subdirectories in the base directory, excluding system folders
        self.data_folders = []
        ignore_folders = {'__pycache__', '.git', '.vscode', '.idea', 'node_modules', '.DS_Store'}

        for item in os.listdir(base_data_dir):
            # Skip if it's in the ignore list
            if item in ignore_folders:
                continue

            # Skip hidden folders (starting with .)
            if item.startswith('.'):
                continue

            folder_path = os.path.join(base_data_dir, item)
            if os.path.isdir(folder_path):
                # Additional check: make sure the folder contains .pkl files
                pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
                if len(pkl_files) > 0:
                    self.data_folders.append(folder_path)
                else:
                    print(f"Skipping folder {item} - no .pkl files found")

        # Sort folders for consistency
        self.data_folders.sort()

        print(f"Found {len(self.data_folders)} valid data folders:")
        for i, folder in enumerate(self.data_folders):
            folder_name = os.path.basename(folder)
            # Count actual files in the folder for verification
            pkl_count = len([f for f in os.listdir(folder) if f.endswith('.pkl')])
            print(f"  {i+1}. {folder_name} ({pkl_count} .pkl files)")

        if len(self.data_folders) == 0:
            raise ValueError(f"No valid data folders found in {base_data_dir}")

        # Calculate total dataset size
        # We assume each folder has the same number of files (0000 to num_files_per_folder-1)
        self.total_files = len(self.data_folders) * self.num_files_per_folder

        # Create a mapping of dataset index to (folder_idx, file_idx)
        self.index_mapping = []
        for folder_idx in range(len(self.data_folders)):
            for file_idx in range(self.num_files_per_folder):
                self.index_mapping.append((folder_idx, file_idx))

        # Shuffle the mapping to ensure random sampling across folders
        import random
        random.shuffle(self.index_mapping)

    def __len__(self):
        return self.total_files

    def __getitem__(self, idx):
        # Get the folder and file index from the shuffled mapping
        folder_idx, file_idx = self.index_mapping[idx]
        folder_path = self.data_folders[folder_idx]

        data_path = os.path.join(folder_path, f"room_sim_{file_idx:04d}.pkl")

        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            # If file doesn't exist, try a random file from a random folder
            print(f"Warning: File {data_path} not found, selecting random alternative")
            import random
            random_folder_idx = random.randint(0, len(self.data_folders) - 1)
            random_file_idx = random.randint(0, self.num_files_per_folder - 1)
            folder_path = self.data_folders[random_folder_idx]
            data_path = os.path.join(folder_path, f"room_sim_{random_file_idx:04d}.pkl")

            with open(data_path, 'rb') as f:
                data = pickle.load(f)

        random_noise = np.random.randn(*data['svect_hoa'].shape) * 1e-6
        svect_hoa, svect_foa = data['svect_hoa'], data['svect_foa']

        if self.get_idx:
            # Return original idx, folder info, and file info for debugging
            folder_name = os.path.basename(folder_path)
            return svect_foa, svect_hoa, idx

        return svect_foa, svect_hoa



class SteeringVectorFeatureExtractor(nn.Module):
    """
    Multi-scale feature extractor for steering vectors that respects
    spatial (microphone) and frequency structure
    """
    def __init__(self, input_channels=2):
        super().__init__()

        # Frequency-domain convolutions (along freq axis)
        self.freq_conv1 = nn.Conv2d(input_channels, 32, kernel_size=(7, 1), padding=(3, 0))
        self.freq_conv2 = nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0))
        self.freq_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))

        # Spatial-domain convolutions (along microphone axis)
        self.spatial_conv1 = nn.Conv2d(input_channels, 32, kernel_size=(1, 5), padding=(0, 2))
        self.spatial_conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))

        # Combined spatio-frequency convolutions
        self.combined_conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=(1, 1))
        self.combined_conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))

        # Activation
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Extract multi-scale features from steering vectors
        Args:
            x: [B, 2, 1024, 16] - batch of steering vectors
        Returns:
            List of feature maps at different scales
        """
        features = []

        # Frequency-domain features
        freq_f1 = self.act(self.freq_conv1(x))
        freq_f2 = self.act(self.freq_conv2(freq_f1))
        freq_f3 = self.act(self.freq_conv3(freq_f2))

        # Spatial-domain features
        spatial_f1 = self.act(self.spatial_conv1(x))
        spatial_f2 = self.act(self.spatial_conv2(spatial_f1))

        # Combined features
        combined_f1 = self.act(self.combined_conv1(x))
        combined_f2 = self.act(self.combined_conv2(combined_f1))

        # Collect features at different scales
        features = [freq_f1, freq_f2, freq_f3, spatial_f1, spatial_f2, combined_f1, combined_f2]

        return features


class SteeringVectorFeatureMatchingLoss(nn.Module):
    """
    Feature matching loss specifically designed for steering vectors
    """
    def __init__(self, input_channels=2, weights=None):
        super().__init__()
        self.feature_extractor = SteeringVectorFeatureExtractor(input_channels)

        # Default weights for different feature scales
        if weights is None:
            self.weights = [1.0, 1.0, 1.0, 0.8, 0.8, 1.2, 1.2]  # Higher weight for combined features
        else:
            self.weights = weights

        self.criterion = nn.L1Loss()

    def forward(self, generated, target):
        """
        Compute feature matching loss
        Args:
            generated: [B, 2, 1024, 16] - generated steering vectors
            target: [B, 2, 1024, 16] - target steering vectors
        """
        with torch.no_grad():
            target_features = self.feature_extractor(target)

        generated_features = self.feature_extractor(generated)

        total_loss = 0
        for i, (gen_feat, target_feat) in enumerate(zip(generated_features, target_features)):
            feat_loss = self.criterion(gen_feat, target_feat)
            total_loss += self.weights[i] * feat_loss

        return total_loss / len(generated_features)

def spatial_perceptual_loss(pred, target):
    """
    Loss that focuses on spatial relationships
    Args:
        pred: predicted steering vectors (batch, 2, freq_bins, mics) where 0=real, 1=imag
        target: target steering vectors (batch, 2, freq_bins, mics) where 0=real, 1=imag
    """
    # Extract real and imaginary parts
    pred_real, pred_imag = pred[:, 0, ...], pred[:, 1, ...]
    target_real, target_imag = target[:, 0, ...], target[:, 1, ...]

    # Create complex tensors
    pred_complex = torch.complex(pred_real, pred_imag)
    target_complex = torch.complex(target_real, target_imag)

    # Phase difference loss (important for steering vectors)
    pred_phase = torch.angle(pred_complex)
    target_phase = torch.angle(target_complex)
    phase_loss = F.mse_loss(pred_phase, target_phase)

    # Magnitude loss
    pred_mag = torch.abs(pred_complex)
    target_mag = torch.abs(target_complex)
    mag_loss = F.mse_loss(pred_mag, target_mag)

    # Coherence loss (spatial relationship)
    coherence_loss = 1 - F.cosine_similarity(pred.flatten(1), target.flatten(1)).mean()

    return mag_loss + 0.5 * phase_loss + 1 * coherence_loss


def frequency_weighted_loss(pred, target, low_freq_bins=80, weight_factor=3.0):
    """
    Apply frequency-dependent weights to the loss
    Args:
        pred: predicted steering vectors (batch, channels, freq_bins, mics)
        target: target steering vectors (batch, channels, freq_bins, mics)
        low_freq_bins: number of low frequency bins to emphasize
        weight_factor: how much to emphasize low frequencies
    """
    # Get frequency dimension size (should be 1024 in your case)
    freq_dim_size = pred.shape[2]  # frequency is at dimension 2

    # Create frequency weights
    freq_weights = torch.ones(freq_dim_size, device=pred.device)
    freq_weights[:low_freq_bins] = weight_factor  # Emphasize low frequencies

    # Reshape weights for broadcasting: (1, 1, freq_bins, 1)
    freq_weights = freq_weights.view(1, 1, -1, 1)

    # Apply weights to MSE loss
    mse_loss = F.mse_loss(pred, target, reduction='none')
    weighted_loss = mse_loss * freq_weights

    return weighted_loss.mean()



def train(args) -> None:
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Setup loss logging
    config_dir = os.path.dirname(args.config_path)
    loss_log_path = os.path.join(config_dir, 'vae_losses.txt')

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################

    # Create the model and dataset #
    model = VAE(im_channels=dataset_config['input_channels'],
                  model_config=autoencoder_config).to(device)
    load_from_checkpoint = train_config['load_ckpt_vae']
    if load_from_checkpoint:
        print("VAE loaded ckpt")
        checkpoint = torch.load('weights/v2.1/vae_autoencoder_ckpt.pth')
        model.load_state_dict(checkpoint)

    if train_config['train_decoder_only']:
        # Freeze encoder parameters to train only the decoder
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the autoencoder model: {num_params}")

    # Create the dataset
    data_path = os.path.join(dataset_config['im_path'])
    # dataset_train, dataset_val = SVectDatasetMNMF(os.path.join(data_path, 'train')), SVectDatasetMNMF(os.path.join(data_path, 'val'))
    # val_size = len(dataset_val)*0.1
    # dataset_val = torch.utils.data.Subset(dataset_val, list(range(int(val_size))))

    dataset_train = SVectDataset(data_path, get_idx=False)
    # Split the dataset into train and validation sets
    val_ratio = 0.1  # 10% for validation
    total_size = len(dataset_train)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    dataset_train, dataset_val = random_split(dataset_train, [train_size, val_size])

    data_loader = DataLoader(dataset_train, batch_size=train_config['autoencoder_batch_size'], shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=train_config['autoencoder_batch_size'], shuffle=False)

    num_epochs = train_config['autoencoder_epochs']

    # LOSS WEIGHTS WITH SUBSAMPLING
    weights = {
        'reconstruction': 0.8,
        'kl': train_config['kl_weight'],
        'cosine': 1,
        'perceptual_loss': 0.05,
    }
    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss(reduction='mean')
    # cosine Loss can even be BCEWithLogits
    #  feature matching loss
    feature_matching_loss = SteeringVectorFeatureMatchingLoss(
        input_channels=dataset_config['input_channels']
    ).to(device)

    optimizer_g = AdamW(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.9, 0.998))
    scheduler = ExponentialLR(optimizer_g, gamma=0.996)

    disc_step_start = train_config['disc_start']
    step_count = 0

    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    # image_save_steps = train_config['autoencoder_img_save_steps']
    best_val_cosine_similarity = 0
    epoch_losses = []  # Track losses for each epoch

    for epoch_idx in range(num_epochs):
        recon_losses = []
        perceptual_losses = []
        cosine_losses = []
        gen_losses = []
        losses = []
        fw_losses = []

        optimizer_g.zero_grad()

        for _, im in tqdm(data_loader):
            step_count += 1
            im = im.float().to(device)

            # Fetch autoencoders output(reconstructions)
            model_output = model(im)
            output, out_encoder = model_output
            mu, logvar = torch.chunk(out_encoder, 2, dim=1)

            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, im)
            # kl loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()
            # Cosine similarity loss
            cosine_loss = (1 - F.cosine_similarity(output, im, dim=1)).mean()
            # Perceptual loss
            perc_loss = spatial_perceptual_loss(output, im)
            # Combined loss with proper weighting
            g_loss = (
                weights['reconstruction'] * recon_loss +
                weights['kl'] * kl_loss +
                weights['cosine'] * cosine_loss +
                weights['perceptual_loss'] * perc_loss
            )

            # Log all losses for monitoring
            recon_losses.append(recon_loss.item())
            cosine_losses.append(cosine_loss.item())
            perceptual_losses.append(perc_loss.item())

            losses.append(g_loss.item())
            g_loss.backward()
            torch.cuda.empty_cache()

            #####################################

            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        scheduler.step()

        print('Finished epoch: {} | Recon Loss : {:.4f} | Cosine Loss : {:.4f}'
                ' Perceptual Loss : {:.4f}`'.
                format(epoch_idx + 1,
                        np.mean(recon_losses),
                        np.mean(cosine_losses),
                        np.mean(perceptual_losses)))

        # Calculate validation cosine similarity
        val_cosine_similarities = []
        with torch.no_grad():
            for _, val_im in val_loader:
                val_im = val_im.float().to(device)
                val_output, _ = model(val_im)
                val_cosine_similarity = torch.nn.functional.cosine_similarity(
                    val_im.view(val_im.size(0), -1),
                    val_output.view(val_output.size(0), -1),
                    dim=1
                )
                val_cosine_similarities.append(val_cosine_similarity.mean().item())

        avg_val_cosine_similarity = np.mean(val_cosine_similarities)
        print(f"Validation Cosine Similarity: {avg_val_cosine_similarity:.4f}")

        # Track epoch loss
        epoch_loss = np.mean(losses)
        epoch_losses.append(epoch_loss)

        torch.save(model.state_dict(), train_config['vae_autoencoder_ckpt_name'])
        # torch.save(discriminator.state_dict(), train_config['vae_discriminator_ckpt_name'])

    # Save losses to file
    with open(loss_log_path, 'w') as f:
        f.write("epoch\tloss\n")
        for i, loss in enumerate(epoch_losses):
            f.write(f"{i+1}\t{loss:.6f}\n")
    print(f'Losses saved to: {loss_log_path}')
    print('Done Training...')


if __name__ == '__main__':
    print(device)
    parser = argparse.ArgumentParser(description='Arguments for vae training')
    parser.add_argument('--config', dest='config_path',
                        default='single_steer_vect/config.yaml', type=str)
    args = parser.parse_args()
    train(args)
