# Google Colab Installation Cell
# Copy and paste this into a Colab cell and run it

# Step 1: Downgrade numpy first to avoid conflicts
!pip install numpy==1.26.4 --force-reinstall --quiet

# Step 2: Install core scientific packages
!pip install scipy==1.13.1 --force-reinstall --quiet
!pip install scikit-learn==1.5.1 --force-reinstall --quiet

# Step 3: Install PyTorch with CUDA support
!pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 --quiet

# Step 4: Install Hugging Face and diffusion libraries
!pip install transformers==4.40.0 --quiet
!pip install diffusers==0.27.2 --quiet
!pip install accelerate==0.30.1 --quiet
!pip install tokenizers==0.19.1 --quiet
!pip install safetensors==0.4.3 --quiet

# Step 5: Install computer vision packages
!pip install opencv-python==4.9.0.80 --quiet
!pip install scikit-image==0.22.0 --quiet
!pip install pillow==10.3.0 --quiet

# Step 6: Install Gradio for UI
!pip install gradio==4.26.0 --quiet

# Step 7: Install remaining ML packages
!pip install timm==0.9.16 --quiet
!pip install torchmetrics==1.4.0 --quiet
!pip install omegaconf==2.3.0 --quiet
!pip install einops==0.7.0 --quiet
!pip install fvcore==0.1.5 --quiet

# Step 8: Install utility packages
!pip install pycocotools==2.0.7 --quiet
!pip install peft==0.11.1 --quiet
!pip install regex==2024.5.15 --quiet
!pip install cloudpickle==3.0.0 --quiet
!pip install onnxruntime==1.18.0 --quiet
!pip install imageio==2.34.1 --quiet

print("✅ Installation completed!")
print("⚠️  Please restart your runtime: Runtime -> Restart Runtime")
print("Then run your main code.")