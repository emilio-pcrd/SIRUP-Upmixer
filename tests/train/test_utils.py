import torch

from train.utils import LinearNoiseScheduler


def test_noise_scheduler_add_noise() -> None:
    scheduler = LinearNoiseScheduler(num_timesteps=10, beta_start=0.1, beta_end=0.2)
    original = torch.zeros(2, 1, 4, 4)
    noise = torch.randn_like(original)
    t = torch.tensor([0, 1])
    out = scheduler.add_noise(original, noise, t)
    assert out.shape == original.shape
