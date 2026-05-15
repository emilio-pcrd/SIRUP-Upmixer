# SIRUP Code Review & Improvement Recommendations
## For ICASSP 2026 Paper Submission

**Overall Grade: 6.5/10**

---

## Executive Summary

This repository contains the implementation of "Diffusion-Based Virtual Upmixing Of First-Order Ambisonics For Highly-Directive Source Localization And Enhancement" for ICASSP 2026. While the core algorithmic contributions are solid, the codebase suffers from organizational issues, incomplete documentation, code duplication, and inconsistent practices that would benefit significantly from refactoring before publication.

---

## Strengths ✅

1. **Clear Technical Architecture**: VAE + Diffusion model pipeline is well-structured conceptually
2. **Comprehensive Model Blocks**: `blocks.py` contains well-designed ResNet-based components
3. **Proper Configuration Management**: YAML-based config system for reproducibility
4. **Good Algorithmic Implementation**: Noise schedulers, time embeddings, and diffusion logic are properly implemented
5. **Audio-Specific Considerations**: Steering vector-specific feature extraction shows domain knowledge

---

## Major Issues ❌

### 1. **Code Duplication (CRITICAL)** 🔴
**Impact**: High | **Effort**: Medium

**Problems:**
- **Duplicate folder structure**: `/models/` and `/train/models/` contain identical files
  - Same classes: `blocks.py`, `denoiser.py`, `vae.py`, `vqvae.py`, `discriminator.py`, `feature_extractor.py`, `lpips.py`
  - This is a maintenance nightmare and source of bugs
- **Duplicate dataset classes**: `SVectDataset` is defined identically in both `train_ddpm.py` and `train_vae.py`
- **Commented-out code blocks**: Large sections in `blocks.py` (old DownBlock/MidBlock implementations) clutter the file

**Recommendations:**
```
RESTRUCTURE:
sirup/
├── models/           # Core model definitions (single source of truth)
│   ├── __init__.py
│   ├── blocks.py
│   ├── denoiser.py
│   ├── vae.py
│   ├── vqvae.py
│   ├── discriminator.py
│   ├── feature_extractor.py
│   └── lpips.py
├── datasets/         # NEW: Data handling
│   ├── __init__.py
│   └── steering_vectors.py
├── train/
│   ├── train_ddpm.py
│   ├── train_vae.py
│   ├── train_vqvae.py
│   └── utils.py
└── [other folders]
```

---

### 2. **Missing Documentation** 🔴
**Impact**: High | **Effort**: High

**Problems:**
- No module/class docstrings except minimal ones
- No README with setup instructions, dependencies, usage examples
- No function documentation for parameters/returns
- Configuration options are undocumented
- Paper title in README is incomplete

**Recommendations:**
Create comprehensive docstrings:
```python
class DownBlock(nn.Module):
    """Downsampling block for diffusion U-Net.

    Implements residual connections with optional attention and cross-attention
    for complex-valued spatial-spectral data (e.g., steering vectors of shape (B, 2, F, M)).

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        t_emb_dim (int, optional): Dimension of time embedding for conditioning
        down_sample (bool): Whether to apply downsampling
        num_heads (int): Number of attention heads
        num_layers (int): Number of residual blocks
        attn (bool): Whether to apply self-attention
        norm_channels (int): Number of groups for GroupNorm
        cross_attn (bool): Whether to apply cross-attention
        context_dim (int, optional): Dimension of context for cross-attention
        freq_dilation (int): Dilation factor along frequency axis

    Input Shape:
        - x: (B, in_channels, F, M)
        - t_emb (optional): (B, t_emb_dim)
        - context (optional): (B, seq_len, context_dim)

    Returns:
        (B, out_channels, F', M') where F', M' depend on down_sample
    """
```

---

### 3. **Inconsistent Import Patterns** 🟡
**Impact**: Medium | **Effort**: Low

**Problems:**
```python
# Commented-out imports scattered throughout
# from blocks import DownBlock, MidBlock
# from CODE.upmixing_ddpm.models.vae import VAE
from models.blocks import DownBlock

# Import duplications
import random  # imported twice in some files
import torch.nn.functional as F  # imported twice in utils.py
```

**Recommendations:**
- Remove all commented import lines
- Use relative imports consistently within package
- Add `from __future__ import annotations` for forward references

---

### 4. **Missing Dependencies File** 🔴
**Impact**: High | **Effort**: Low

**Problems:**
- No `requirements.txt` or `environment.yml`
- No setup.py or pyproject.toml
- Code references undefined modules (e.g., `from datasets import SVectDatasetMNMF`)

**Recommendations:**
Create `requirements.txt`:
```txt
torch>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
tqdm>=4.62.0
PyYAML>=5.4.0
pyroomacoustics>=0.7.0
matplotlib>=3.4.0
# Add audio/acoustic-specific packages
librosa>=0.10.0
soundfile>=0.12.0
```

Create `setup.py`:
```python
from setuptools import setup, find_packages

setup(
    name="sirup",
    version="1.0.0",
    description="Diffusion-based virtual upmixing of first-order ambisonics",
    author="Emilio Picard et al.",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[...],
)
```

---

### 5. **Undefined/Incomplete Code References** 🔴
**Impact**: Critical | **Effort**: Medium

**Problems:**
```python
# train_ddpm.py references undefined classes:
from datasets import SVectDatasetMNMF  # NOT DEFINED ANYWHERE
model = Unet(...)  # UNDEFINED - should this be ResUNetDenoiser?

# train_vae.py:
from datasets import SVectDatasetMNMF  # Same issue

# models/denoiser.py references undefined:
self.im_cond_input_ch  # Never initialized
self.im_cond_output_ch  # Never initialized
self.down_channels  # Never initialized
```

**Recommendations:**
- Implement missing `SVectDatasetMNMF` or replace with `SVectDataset`
- Define all referenced attributes in `__init__`
- Add type hints to catch these at development time

---

### 6. **Incomplete/Broken Code** 🔴
**Impact**: High | **Effort**: Medium

**Problems:**
```python
# models/denoiser.py - ResUNetDenoiser.__init__() is incomplete
# Missing closing braces, undefined attributes used in forward()

# train/models/* - These are duplicates that need removal

# preprocessing/create_dataset.py - References placeholder paths
signals_random_1_PATH, signals_random_2_PATH = select_two_random_flacs('/...')
```

**Recommendations:**
- Complete `ResUNetDenoiser` implementation
- Replace `/...` placeholders with configurable paths
- Add validation that required files exist

---

### 7. **Type Hints & Static Analysis** 🟡
**Impact**: Medium | **Effort**: High

**Problems:**
- Zero type hints across codebase
- No validation of tensor shapes
- Runtime errors can occur without notice

**Example improvements:**
```python
# Before
def forward(self, x, t_emb=None, context=None):
    out = x
    for i in range(self.num_layers):
        ...

# After
def forward(
    self,
    x: torch.Tensor,                    # (B, C, H, W)
    t_emb: Optional[torch.Tensor] = None,  # (B, D)
    context: Optional[torch.Tensor] = None  # (B, S, D)
) -> torch.Tensor:                      # (B, C, H', W')
    """Forward pass with tensor shape validation."""
    assert x.ndim == 4, f"Expected 4D tensor, got {x.ndim}D"
    out = x
    ...
```

---

### 8. **Configuration & Hyperparameter Management** 🟡
**Impact**: Medium | **Effort**: Medium

**Problems:**
- Hardcoded device: `device = torch.device('cuda')`
- Magic numbers scattered: `1e-2`, `0.18215`, `4*fs`
- No config validation
- Train config missing keys referenced in code

**Recommendations:**
```python
# Add validation schema
from dataclasses import dataclass
from typing import List

@dataclass
class DatasetConfig:
    input_channels: int = 2
    w_size: int = 1024
    h_size: int = 16
    name: str = 'steering_vectors'

@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    condition: bool = True

    def __post_init__(self):
        assert 0 < self.beta_start < self.beta_end < 1
```

---

### 9. **Dataset Handling Issues** 🔴
**Impact**: High | **Effort**: Medium

**Problems:**
```python
# Duplicate implementation across files
class SVectDataset:  # Defined in train_ddpm.py
class SVectDataset:  # Also defined in train_vae.py (identical)

# Error handling is weak
try:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    # Falls back to random file - masking data issues
    print(f"Warning: File {data_path} not found, selecting random alternative")

# No validation of data format
```

**Recommendations:**
- Create single `datasets/steering_vectors.py`
- Implement proper data validation
- Log missing files to separate error log
- Add data statistics/summary methods

---

### 10. **Testing & Reproducibility** 🔴
**Impact**: High | **Effort**: High

**Problems:**
- No test suite
- Random seeds set but not consistently
- No reproducibility documentation
- No validation metrics computation

**Recommendations:**
```
tests/
├── __init__.py
├── test_models.py
├── test_data_loading.py
├── test_training_pipeline.py
└── test_inference.py

scripts/
├── download_data.py
├── compute_metrics.py
└── evaluate_model.py
```

---

## Code Quality Issues 🟡

### Minor Issues:

1. **Inconsistent formatting**
   ```python
   # Mixed styles:
   # Style 1:
   self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))

   # Style 2:
   self.encoder_block1 = EncoderBlockRes4B(
       in_channels=input_channels,
       out_channels=32,
   )
   ```
   → Use Black formatter for consistency

2. **Unused imports**
   ```python
   import glob  # imported but never used
   import sys   # imported but rarely used
   ```

3. **Commented code blocks**
   - ~150 lines of old code in `blocks.py`
   - Multiple commented dataset definitions
   → Remove before publication

4. **Magic numbers**
   ```python
   random_noise = np.random.randn(*svect_hoa.shape) * 1e-2  # Why 1e-2?
   scale_factor = 0.18215  # Where does this come from?
   ```
   → Add constants to config with documentation

5. **Inconsistent error handling**
   ```python
   # No consistent exception handling pattern
   # Some code uses try-except, some doesn't
   ```

---

## Organization Recommendations 📁

```
sirup/
├── README.md                    # Project description, setup, usage
├── CONTRIBUTING.md              # For reviewers/collaborators
├── LICENSE
├── requirements.txt
├── setup.py
├── pyproject.toml
├── .gitignore
│
├── sirup/
│   ├── __init__.py
│   │
│   ├── models/                  # Core models (single source of truth)
│   │   ├── __init__.py
│   │   ├── blocks.py
│   │   ├── denoiser.py
│   │   ├── vae.py
│   │   ├── vqvae.py
│   │   ├── discriminator.py
│   │   ├── feature_extractor.py
│   │   └── lpips.py
│   │
│   ├── datasets/                # Data loading (NEW)
│   │   ├── __init__.py
│   │   └── steering_vectors.py
│   │
│   ├── config/                  # Configurations
│   │   └── config.yaml
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   ├── trainer_vae.py
│   │   ├── trainer_ddpm.py
│   │   └── utils.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── create_dataset.py
│   │   └── create_svects.py
│   │
│   └── utils/                   # Utilities (NEW)
│       ├── __init__.py
│       ├── noise_schedulers.py
│       └── metrics.py
│
├── checkpoints/                 # Model checkpoints
│   └── config.yaml
│
├── tests/                       # Unit tests (NEW)
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_datasets.py
│   └── test_training.py
│
├── scripts/                     # Standalone scripts (NEW)
│   ├── download_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
└── docs/                        # Documentation (NEW)
    ├── index.md
    ├── architecture.md
    ├── training.md
    └── api_reference.md
```

---

## Specific Code Improvements 🔧

### 1. Remove Duplication

**Before (duplicated in train_ddpm.py and train_vae.py):**
```python
class SVectDataset(Dataset):
    def __init__(self, base_data_dir, get_idx=False, num_files_per_folder=1500):
        # 100+ lines of identical code
```

**After (datasets/steering_vectors.py):**
```python
from pathlib import Path
from typing import Tuple, Optional

class SteeringVectorDataset(Dataset):
    """Dataset for first-order ambisonics steering vectors."""

    def __init__(self, base_data_dir: Path, ...):
        ...
```

Then import in both training scripts:
```python
from sirup.datasets import SteeringVectorDataset
```

### 2. Fix Undefined References

**Before:**
```python
from datasets import SVectDatasetMNMF  # Undefined!
model = Unet(...)  # Undefined!
```

**After:**
```python
from sirup.datasets import SteeringVectorDataset
from sirup.models.denoiser import ConditionedDiffusionUNet

model = ConditionedDiffusionUNet(
    im_channels=autoencoder_model_config['z_channels'],
    model_config=diffusion_model_config
).to(device)
```

### 3. Add Type Hints & Docstrings

**Before:**
```python
def add_noise(self, original, noise, t):
    r"""Forward method for diffusion"""
    ...
```

**After:**
```python
def add_noise(
    self,
    original: torch.Tensor,  # (B, C, H, W)
    noise: torch.Tensor,     # (B, C, H, W)
    t: torch.Tensor          # (B,)
) -> torch.Tensor:           # (B, C, H, W)
    """Apply noise to original signal according to diffusion schedule.

    Implements: x_t = sqrt(alpha_cum_prod_t) * x_0 + sqrt(1 - alpha_cum_prod_t) * noise

    Args:
        original: Clean signal samples
        noise: Gaussian noise samples
        t: Timestep indices (0 to num_timesteps-1)

    Returns:
        Noised samples at timestep t

    Raises:
        AssertionError: If tensor shapes don't match
    """
```

---

## Documentation Improvements 📚

### 1. Comprehensive README.md

```markdown
# SIRUP: Diffusion-Based Virtual Upmixing of First-Order Ambisonics

## Overview
[Detailed description of the paper and method]

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_checkpoints.py

# Run inference
python scripts/inference.py --input audio.wav --output output.wav
```

## Installation
[Detailed setup instructions]

## Usage
[Usage examples]

## Training
[Training instructions]

## Results
[Quantitative/qualitative results]

## Citation
[BibTeX]
```

### 2. Architecture Documentation (docs/architecture.md)

```markdown
# Architecture Overview

## Pipeline
VAE Encoder → Latent Space → Diffusion Denoiser → VAE Decoder

## Components
1. **VAE**: Encodes steering vectors to latent space
2. **Diffusion Model**: Learns to denoise in latent space with conditioning
3. **Feature Extractor**: Extracts features for conditioning

[Diagrams, detailed descriptions]
```

---

## Priority Action Items 🎯

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 🔴 CRITICAL | Remove duplicate train/models folder | Low | High |
| 🔴 CRITICAL | Fix undefined references (SVectDatasetMNMF, Unet) | High | High |
| 🔴 CRITICAL | Add requirements.txt and setup.py | Low | High |
| 🔴 CRITICAL | Complete ResUNetDenoiser implementation | Medium | High |
| 🟠 HIGH | Consolidate duplicate SVectDataset | Medium | High |
| 🟠 HIGH | Add type hints across codebase | High | Medium |
| 🟠 HIGH | Create comprehensive docstrings | High | Medium |
| 🟠 HIGH | Add test suite | High | Medium |
| 🟡 MEDIUM | Remove commented code blocks | Low | Low |
| 🟡 MEDIUM | Add configuration validation | Medium | Medium |
| 🟡 MEDIUM | Create training/evaluation scripts | Medium | Medium |
| 🟡 MEDIUM | Add LICENSE and CONTRIBUTING.md | Low | Low |

---

## Implementation Timeline ⏱️

**Phase 1: Critical Fixes (1-2 days)**
- Remove train/models duplication
- Fix undefined imports/references
- Add requirements.txt
- Complete broken implementations

**Phase 2: Code Quality (2-3 days)**
- Add type hints to all modules
- Create comprehensive docstrings
- Remove commented code
- Add configuration validation

**Phase 3: Documentation (1-2 days)**
- Write README with setup/usage
- Create architecture docs
- Add API documentation
- Create examples/tutorials

**Phase 4: Testing & Polish (1-2 days)**
- Add unit tests
- Add integration tests
- Performance profiling
- Final review and cleanup

---

## Conclusion

The code implements a sophisticated diffusion-based method with solid technical foundations. However, **organizational and documentation issues prevent it from being publication-ready**. With the recommended refactoring (~5-7 days of work), this codebase can become a high-quality, reproducible research implementation suitable for a top-tier venue like ICASSP.

### Revised Grade After Improvements: **8.5-9/10** ✨

The gap between current (6.5) and potential (8.5-9) is achievable through disciplined cleanup and documentation.

