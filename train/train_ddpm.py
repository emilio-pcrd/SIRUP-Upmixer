import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import torch
import random
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils import LinearNoiseScheduler
from models.denoiser import Unet, EncoderCond
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from models.vae import VAE
from torch.utils.data import random_split
from torch.utils.data import Dataset
import pickle
import glob


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
#         random_noise = np.random.randn(*svect_hoa.shape) * 1e-2
#         svect_foa += random_noise
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

        random_noise = np.random.randn(*data['svect_hoa'].shape) * 1e-2
        svect_hoa, svect_foa = data['svect_hoa'], data['svect_foa']
        svect_foa += random_noise

        if self.get_idx:
            # Return original idx, folder info, and file info for debugging
            folder_name = os.path.basename(folder_path)
            return svect_foa, svect_hoa

        return svect_foa, svect_hoa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from datasets import SVectDatasetMNMF

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)

    # Setup loss logging
    config_dir = os.path.dirname(args.config_path)
    loss_log_path = os.path.join(config_dir, 'ddpm_losses.txt')

    ########################
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################

    # Instanciate dataset and dataloader
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
    data_loader = DataLoader(dataset_train, batch_size=train_config['ldm_batch_size'], shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)

    # Instantiate the unet model
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)

    load_ckpt_ldm = train_config['load_ckpt_ldm']
    if load_ckpt_ldm:
        print("ddpm loaded ckpt")
        model.load_state_dict(torch.load('weights/v0.1/ddpm_ckpt_image_cond.pth',
                                map_location=device))
    model.train()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad_))

    # Instanciate vae because latents have not been saved
    vae = VAE(im_channels=dataset_config['input_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()

    # Load vae if found
    if os.path.exists(train_config['vae_autoencoder_ckpt_name']):
        print('Loaded vae checkpoint')
        print(f'loading vae from ckpt {train_config["vae_autoencoder_ckpt_name"]}')
        vae.load_state_dict(torch.load(train_config['vae_autoencoder_ckpt_name'],
                                        map_location=device))
    else:
        raise Exception('VAE checkpoint not found and use_latents was disabled')

    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = AdamW(model.parameters(), lr=train_config['ldm_lr'], betas=(0.9, 0.999),
                      weight_decay=1e-3)
    scheduler_opt = ExponentialLR(optimizer, gamma=0.999996)
    # inv_gamma = 1_000_000
    # power = 0.5
    # warmup = 0.99
    # def lr_lambda(step):
    #     if step < warmup:
    #         # Linear warmup
    #         return step / warmup
    #     # Inverse LR decay
    #     return (1.0 + step / inv_gamma) ** (-power)
    # scheduler_opt = LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = torch.nn.MSELoss()
    scale_factor = autoencoder_model_config['scale_factor']
    # Load vae and freeze parameters ONLY if latents already not saved
    for param in vae.parameters():
        param.requires_grad = False

    # Run training
    lossess = []
    for epoch_idx in range(num_epochs):
        losses = []
        for data in tqdm(data_loader):
            # load data
            cond_input, im = data
            optimizer.zero_grad()
            im = im.float().to(device)
            cond_input = cond_input.float().to(device)

            # get latents
            with torch.no_grad():
                im, _ = vae.encode(im)
                im = im * scale_factor
                # cond_latent, _ = vae.encode(cond_input)
                # print("cond latent : ", cond_latent.shape)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            if diffusion_config['condition']:
                noise_pred = model(noisy_im, t, cond_input)
            else:
                noise_pred = model(noisy_im, t)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        lossess.append(np.mean(losses))
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        torch.save(model.state_dict(), train_config['ldm_ckpt_name'])
        scheduler_opt.step()

    # Save losses to file in config directory
    with open(loss_log_path, 'w') as f:
        f.write("epoch\tloss\n")
        for i, loss in enumerate(lossess):
            f.write(f"{i+1}\t{loss:.6f}\n")
    print(f'Losses saved to: {loss_log_path}')
    print('Done Training ...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
            default='weights/v2.3/config.yaml', type=str)
    args = parser.parse_args()
    train(args)
