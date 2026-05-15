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
from models.denoiser import Unet
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from models.vae import VAE
from torch.utils.data import random_split
import glob
from datasets.steering_vectors import SteeringVectorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    dataset_train = SteeringVectorDataset(data_path, get_idx=False, noise_std=1e-2)
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
