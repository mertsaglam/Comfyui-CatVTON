import sys
import os
import subprocess
from huggingface_hub import snapshot_download

# Determine the paths relative to the ComfyUI directory
COMFYUI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.join(COMFYUI_ROOT, "models")
CUSTOM_NODES_PATH = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(MODELS_DIR, "CatVTON")
HF_REPO_ID = "zhengchong/CatVTON"
VAE_WEIGHTS_PATH = os.path.join(MODELS_DIR, "sd-vae-ft-mse")
VAE_REPO_ID = "stabilityai/sd-vae-ft-mse"
INPAINT_WEIGHTS_PATH = os.path.join(MODELS_DIR, "stable-diffusion-inpainting")
HF_INPAINT_REPO_ID = "runwayml/stable-diffusion-inpainting"

# Ensure required directories exist
os.makedirs(WEIGHTS_PATH, exist_ok=True)
os.makedirs(VAE_WEIGHTS_PATH, exist_ok=True)
os.makedirs(INPAINT_WEIGHTS_PATH, exist_ok=True)

def build_pip_install_cmds(args):
    """
    Builds the pip install command based on the environment.
    """
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        return [sys.executable, '-s', '-m', 'pip', 'install'] + args
    else:
        return [sys.executable, '-m', 'pip', 'install'] + args

def ensure_package():
    """
    Installs packages from requirements.txt in the custom nodes directory.
    """
    requirements_file = os.path.join(CUSTOM_NODES_PATH, 'requirements.txt')
    cmds = build_pip_install_cmds(['-r', requirements_file])
    subprocess.run(cmds, cwd=CUSTOM_NODES_PATH, check=True)

if __name__ == "__main__":
    # Ensure packages are installed
    # Uncomment the next line if you want to ensure requirements are installed
    # ensure_package()
    
    # Download weights from Hugging Face repositories
    snapshot_download(repo_id=HF_REPO_ID, local_dir=WEIGHTS_PATH, local_dir_use_symlinks=False)
    snapshot_download(repo_id=VAE_REPO_ID, local_dir=VAE_WEIGHTS_PATH, local_dir_use_symlinks=False)
    snapshot_download(repo_id=HF_INPAINT_REPO_ID, local_dir=INPAINT_WEIGHTS_PATH, local_dir_use_symlinks=False)
