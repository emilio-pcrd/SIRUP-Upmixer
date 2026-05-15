from setuptools import find_packages, setup

setup(
    name="sirup",
    version="0.1.0",
    description="Diffusion-based virtual upmixing of first-order ambisonics",
    author="Emilio Picard et al.",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "PyYAML>=5.4.0",
        "pyroomacoustics>=0.7.0",
        "matplotlib>=3.4.0",
    ],
)
