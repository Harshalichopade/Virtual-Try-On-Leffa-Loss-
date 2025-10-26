#!/bin/bash
# Colab Installation Script for Virtual Try-On Leffa Loss
# Run this in a Colab cell with: !bash install_colab.sh

echo "Installing Virtual Try-On Leffa Loss dependencies for Google Colab..."

# First, downgrade numpy to avoid conflicts
pip install numpy==1.26.4 --force-reinstall

# Install core dependencies with specific versions
pip install scipy==1.13.1 --force-reinstall
pip install scikit-learn==1.5.1 --force-reinstall

# Install PyTorch ecosystem
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Install other ML/AI packages
pip install transformers==4.40.0
pip install diffusers==0.27.2
pip install accelerate==0.30.1

# Install computer vision packages
pip install opencv-python==4.9.0.80
pip install scikit-image==0.22.0

# Install remaining dependencies
pip install gradio==4.26.0
pip install omegaconf==2.3.0
pip install einops==0.7.0
pip install fvcore==0.1.5
pip install pycocotools==2.0.7
pip install timm==0.9.16
pip install torchmetrics==1.4.0
pip install safetensors==0.4.3
pip install tokenizers==0.19.1
pip install peft==0.11.1
pip install regex==2024.5.15
pip install cloudpickle==3.0.0
pip install onnxruntime==1.18.0
pip install psutil==5.9.8
pip install imageio==2.34.1
pip install tqdm==4.66.4

echo "Installation completed!"
echo "Please restart your runtime after installation."