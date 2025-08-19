"""Setup script for LightningTune."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text()

setup(
    name="LightningTune",
    version="0.3.0",
    author="LightningTune Contributors",
    description="Config-driven hyperparameter optimization for PyTorch Lightning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=5.4",
        "optuna>=3.0.0",
        "pytorch-lightning>=1.5.0",
    ],
    extras_require={
        "full": [
            "wandb>=0.15.0",
            "plotly>=5.0.0",
            "scikit-optimize>=0.9.0",
            "scipy>=1.7.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-timeout>=2.1.0",
            "pytest-mock>=3.10.0",
            "torchvision>=0.15.0",  # For Fashion-MNIST e2e tests
        ],
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="hyperparameter optimization, pytorch lightning, optuna, machine learning",
    project_urls={
        "Source": "https://github.com/neil-tan/LightningTune",
        "Tracker": "https://github.com/neil-tan/LightningTune/issues",
    },
)