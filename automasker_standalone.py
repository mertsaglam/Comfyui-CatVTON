"""
ComfyUI AutoMasker Node - Standalone version
Generates clothing masks for virtual try-on applications

This is a simplified version that requires manual model setup.
Place models in: ComfyUI/models/automasker/
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Union, Dict, Optional
import folder_paths

# Add the current directory to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the existing model components
from model.cloth_masker import AutoMasker as AutoMaskerCore
from model.cloth_masker import vis_mask


class AutoMaskerNode:
    """Simplified AutoMasker for ComfyUI integration."""
    
    def __init__(self):
        self.automasker = None
        self.models_loaded = False
    
    def load_models(self, densepose_path: str, schp_path: str, device: str = "cuda"):
        """Load models with error handling."""
        try:
            # Check if paths exist
            if not os.path.exists(densepose_path):
                raise FileNotFoundError(f"DensePose model directory not found: {densepose_path}")
            if not os.path.exists(schp_path):
                raise FileNotFoundError(f"SCHP model directory not found: {schp_path}")
            
            # Check for required files
            densepose_config = os.path.join(densepose_path, "densepose_rcnn_R_50_FPN_s1x.yaml")
            densepose_model = os.path.join(densepose_path, "model_final_162be9.pkl")
            schp_atr = os.path.join(schp_path, "exp-schp-201908301523-atr.pth")
            schp_lip = os.path.join(schp_path, "exp-schp-201908261155-lip.pth")
            
            missing_files = []
            if not os.path.exists(densepose_config):
                missing_files.append(f"DensePose config: {densepose_config}")
            if not os.path.exists(densepose_model):
                missing_files.append(f"DensePose model: {densepose_model}")
            if not os.path.exists(schp_atr):
                missing_files.append(f"SCHP ATR model: {schp_atr}")
            if not os.path.exists(schp_lip):
                missing_files.append(f"SCHP LIP model: {schp_lip}")
            
            if missing_files:
                error_msg = "Missing required model files:\n" + "\n".join(missing_files)
                raise FileNotFoundError(error_msg)
            
            # Initialize the automasker
            self.automasker = AutoMaskerCore(
                densepose_ckpt=densepose_path,
                schp_ckpt=schp_path,
                device=device
            )
            self.models_loaded = True
            
        except Exception as e:
            self.models_loaded = False
            raise RuntimeError(f"Failed to load AutoMasker models: {str(e)}")
    
    def generate_mask(self, image: Image.Image, mask_type: str = "upper") -> Dict:
        """Generate mask for the given image."""
        if not self.models_loaded or self.automasker is None:
            raise RuntimeError("Models not loaded. Please load models first.")
        
        return self.automasker(image, mask_type)


# Global instance to avoid reloading models
_automasker_instance = None


def get_automasker_instance():
    """Get or create global AutoMasker instance."""
    global _automasker_instance
    if _automasker_instance is None:
        _automasker_instance = AutoMaskerNode()
    return _automasker_instance


# ComfyUI Node Classes
class LoadAutoMasker:
    """ComfyUI node to load the AutoMasker model."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reload_models": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "densepose_path": ("STRING", {
                    "default": "automasker/DensePose",
                    "multiline": False
                }),
                "schp_path": ("STRING", {
                    "default": "automasker/SCHP",
                    "multiline": False
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }
    
    RETURN_TYPES = ("AUTOMASKER_MODEL",)
    RETURN_NAMES = ("automasker",)
    FUNCTION = "load_automasker"
    CATEGORY = "AutoMasker"
    
    def load_automasker(self, reload_models, densepose_path="automasker/DensePose", 
                       schp_path="automasker/SCHP", device="cuda"):
        # Get the global instance
        automasker = get_automasker_instance()
        
        # Resolve paths relative to ComfyUI models directory
        models_dir = folder_paths.models_dir
        
        if not os.path.isabs(densepose_path):
            densepose_path = os.path.join(models_dir, densepose_path)
        if not os.path.isabs(schp_path):
            schp_path = os.path.join(models_dir, schp_path)
        
        # Load models if needed
        if reload_models or not automasker.models_loaded:
            print(f"[AutoMasker] Loading models...")
            print(f"[AutoMasker] DensePose path: {densepose_path}")
            print(f"[AutoMasker] SCHP path: {schp_path}")
            
            try:
                automasker.load_models(densepose_path, schp_path, device)
                print("[AutoMasker] Models loaded successfully!")
            except Exception as e:
                print(f"[AutoMasker] Error loading models: {str(e)}")
                raise
        else:
            print("[AutoMasker] Using cached models")
        
        return (automasker,)


class AutoMaskerGenerate:
    """ComfyUI node to generate masks using AutoMasker."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "automasker": ("AUTOMASKER_MODEL",),
                "image": ("IMAGE",),
                "cloth_type": (["upper", "lower", "overall"], {"default": "upper"}),
                "resize_to_standard": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("mask_rgb", "masked_preview", "mask")
    FUNCTION = "generate_mask"
    CATEGORY = "AutoMasker"
    
    def generate_mask(self, automasker, image, cloth_type, resize_to_standard):
        # Convert from ComfyUI tensor format to PIL
        # ComfyUI images are [B, H, W, C] with values in [0, 1]
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            # Extract single image
            single_image = image[i]
            
            # Convert to PIL
            pil_image = Image.fromarray((single_image.cpu().numpy() * 255).astype(np.uint8))
            
            # Optionally resize to standard size
            original_size = pil_image.size
            if resize_to_standard:
                pil_image = pil_image.resize((768, 1024), Image.LANCZOS)
            
            # Generate mask
            try:
                result = automasker.generate_mask(pil_image, cloth_type)
                mask = result['mask']
            except Exception as e:
                print(f"[AutoMasker] Error generating mask: {str(e)}")
                # Create empty mask on error
                mask = Image.new('L', pil_image.size, 0)
            
            # Create masked preview
            masked_preview = vis_mask(pil_image, mask)
            
            # Resize back to original if needed
            if resize_to_standard and original_size != (768, 1024):
                mask = mask.resize(original_size, Image.LANCZOS)
                masked_preview = masked_preview.resize(original_size, Image.LANCZOS)
            
            # Convert mask to tensor
            mask_np = np.array(mask).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(-1)  # Add channel dimension
            
            # Convert mask to RGB format for compatibility
            mask_rgb = mask_tensor.repeat(1, 1, 3)
            
            # Convert preview to tensor
            preview_np = np.array(masked_preview).astype(np.float32) / 255.0
            preview_tensor = torch.from_numpy(preview_np)
            
            results.append({
                'mask': mask_tensor,
                'mask_rgb': mask_rgb,
                'preview': preview_tensor
            })
        
        # Stack results for batch
        mask_batch = torch.stack([r['mask'] for r in results])
        mask_rgb_batch = torch.stack([r['mask_rgb'] for r in results])
        preview_batch = torch.stack([r['preview'] for r in results])
        
        return (mask_rgb_batch, preview_batch, mask_batch)


class AutoMaskerModelInfo:
    """ComfyUI node to display model setup information."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_info"
    CATEGORY = "AutoMasker"
    OUTPUT_NODE = True
    
    def get_info(self):
        models_dir = folder_paths.models_dir
        info = f"""
AutoMasker Model Setup Instructions:

1. Create directory structure in ComfyUI models folder:
   {models_dir}/automasker/
   ├── DensePose/
   │   ├── Base-DensePose-RCNN-FPN.yaml
   │   ├── densepose_rcnn_R_50_FPN_s1x.yaml
   │   └── model_final_162be9.pkl
   └── SCHP/
       ├── exp-schp-201908301523-atr.pth
       └── exp-schp-201908261155-lip.pth

2. Download models:
   - DensePose: https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl
   - Config files: From Detectron2 DensePose configs
   - SCHP models: From SCHP repository

3. The models will be loaded automatically when you use the LoadAutoMasker node.

Current models directory: {models_dir}
        """
        return (info.strip(),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadAutoMasker": LoadAutoMasker,
    "AutoMaskerGenerate": AutoMaskerGenerate,
    "AutoMaskerModelInfo": AutoMaskerModelInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAutoMasker": "Load AutoMasker Models",
    "AutoMaskerGenerate": "Generate Clothing Mask",
    "AutoMaskerModelInfo": "AutoMasker Setup Info",
}


# Model setup instructions
MODEL_SETUP_INSTRUCTIONS = """
# AutoMasker Model Setup

## Required Models

### 1. DensePose Model
- Download from: https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl
- Place in: ComfyUI/models/automasker/DensePose/model_final_162be9.pkl

### 2. DensePose Config Files
- Base config: https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/configs/Base-DensePose-RCNN-FPN.yaml
- Model config: https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml
- Place both in: ComfyUI/models/automasker/DensePose/

### 3. SCHP Models
- ATR model: exp-schp-201908301523-atr.pth
- LIP model: exp-schp-201908261155-lip.pth
- Place in: ComfyUI/models/automasker/SCHP/

## Directory Structure
```
ComfyUI/models/automasker/
├── DensePose/
│   ├── Base-DensePose-RCNN-FPN.yaml
│   ├── densepose_rcnn_R_50_FPN_s1x.yaml
│   └── model_final_162be9.pkl
└── SCHP/
    ├── exp-schp-201908301523-atr.pth
    └── exp-schp-201908261155-lip.pth
```

## Usage in ComfyUI
1. Add "Load AutoMasker Models" node
2. Connect to "Generate Clothing Mask" node
3. Input your image and select cloth type (upper/lower/overall)
4. Get mask output for virtual try-on
"""

if __name__ == "__main__":
    print(MODEL_SETUP_INSTRUCTIONS)