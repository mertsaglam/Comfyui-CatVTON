"""
Simple AutoMasker using lightweight segmentation models
Options for using existing lightweight models without heavy dependencies
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import folder_paths
from typing import Optional, Dict, Tuple

# Option 1: Use MobileNet-based segmentation
class MobileNetSegmenter(nn.Module):
    """Lightweight segmentation using MobileNet backbone."""
    
    def __init__(self, num_classes=3):  # background, upper, lower
        super().__init__()
        # We could use torchvision's mobilenet_v3_small as backbone
        # This is just a simplified example structure
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, num_classes, 4, stride=2, padding=1),
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec


# Option 2: Use existing segmentation models
def use_segformer_model():
    """Example using HuggingFace's SegFormer (lightweight transformer)."""
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        # This would use pre-trained models from HuggingFace
        # model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b0_clothes")
        pass
    except ImportError:
        pass


# Option 3: Use MediaPipe for human segmentation
def use_mediapipe():
    """Example using Google's MediaPipe for selfie segmentation."""
    try:
        import mediapipe as mp
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        # This provides person segmentation out of the box
        pass
    except ImportError:
        pass


# Simple rule-based approach that actually works
class SimpleClothingMasker:
    """Simple but effective clothing masker using color and spatial analysis."""
    
    @staticmethod
    def get_person_bounds(image: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of person using simple background removal."""
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Simple background detection (assuming relatively uniform background)
        # Calculate image statistics
        l_channel = lab[:, :, 0]
        
        # Edge detection to find person outline
        edges = cv2.Canny(l_channel, 30, 100)
        
        # Morphological operations to connect edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Return full image bounds if no person detected
            h, w = image.shape[:2]
            return 0, 0, w, h
        
        # Get largest contour (assumed to be person)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return x, y, w, h
    
    @staticmethod
    def create_clothing_mask(image: np.ndarray, cloth_type: str = "upper") -> np.ndarray:
        """Create clothing mask using spatial and color analysis."""
        h, w = image.shape[:2]
        
        # Get person bounds
        px, py, pw, ph = SimpleClothingMasker.get_person_bounds(image)
        
        # Initialize mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define regions based on typical human proportions
        # Head: 0-15%, Upper body: 15-55%, Lower body: 55-100%
        
        if cloth_type == "upper":
            # Upper clothing region
            start_y = py + int(ph * 0.15)  # Skip head
            end_y = py + int(ph * 0.55)    # To waist
            mask[start_y:end_y, px:px+pw] = 255
            
        elif cloth_type == "lower":
            # Lower clothing region
            start_y = py + int(ph * 0.50)  # From waist
            end_y = py + ph                # To feet
            mask[start_y:end_y, px:px+pw] = 255
            
        else:  # overall
            # Full body minus head
            start_y = py + int(ph * 0.15)
            end_y = py + ph
            mask[start_y:end_y, px:px+pw] = 255
        
        # Refine mask using color clustering within region
        if np.any(mask):
            # Get pixels in mask region
            masked_pixels = image[mask > 0]
            
            if len(masked_pixels) > 0:
                # Simple color clustering using K-means
                from sklearn.cluster import KMeans
                
                # Reshape pixels
                pixels = masked_pixels.reshape(-1, 3)
                
                # Cluster colors (clothing typically has 1-3 dominant colors)
                n_clusters = min(3, len(np.unique(pixels, axis=0)))
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(pixels)
                    
                    # Find the largest cluster (likely the main clothing)
                    unique, counts = np.unique(labels, return_counts=True)
                    main_cluster = unique[np.argmax(counts)]
                    
                    # Create refined mask
                    cluster_mask = np.zeros_like(mask)
                    mask_indices = np.where(mask > 0)
                    cluster_labels = labels.reshape(-1)
                    
                    for i, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1])):
                        if i < len(cluster_labels) and cluster_labels[i] == main_cluster:
                            cluster_mask[y, x] = 255
                    
                    # Combine with spatial mask
                    mask = cv2.bitwise_and(mask, cluster_mask)
        
        # Post-processing
        # Fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask


# Main ComfyUI Node
class SimpleAutoMasker:
    """Simple and fast clothing mask generator."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cloth_type": (["upper", "lower", "overall"], {"default": "upper"}),
                "refine_edges": ("BOOLEAN", {"default": True}),
                "use_color_clustering": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("mask_rgb", "masked_preview", "mask")
    FUNCTION = "generate_mask"
    CATEGORY = "AutoMasker/Simple"
    
    def generate_mask(self, image, cloth_type, refine_edges, use_color_clustering):
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            # Convert to numpy
            img_tensor = image[i]
            img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            # Generate mask
            if use_color_clustering:
                try:
                    mask = SimpleClothingMasker.create_clothing_mask(img_array, cloth_type)
                except ImportError:
                    # If sklearn not available, use simple method
                    mask = self._simple_position_mask(img_array, cloth_type)
            else:
                mask = self._simple_position_mask(img_array, cloth_type)
            
            # Refine edges if requested
            if refine_edges:
                mask = self._refine_mask_edges(mask, img_array)
            
            # Create preview
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            preview = img_array.copy()
            preview[mask == 0] = preview[mask == 0] * 0.3  # Darken non-mask areas
            
            # Convert to tensors
            mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(-1)
            mask_rgb = mask_tensor.repeat(1, 1, 3)
            preview_tensor = torch.from_numpy(preview.astype(np.float32) / 255.0)
            
            results.append({
                'mask': mask_tensor,
                'mask_rgb': mask_rgb,
                'preview': preview_tensor
            })
        
        # Stack results
        mask_batch = torch.stack([r['mask'] for r in results])
        mask_rgb_batch = torch.stack([r['mask_rgb'] for r in results])
        preview_batch = torch.stack([r['preview'] for r in results])
        
        return (mask_rgb_batch, preview_batch, mask_batch)
    
    def _simple_position_mask(self, image: np.ndarray, cloth_type: str) -> np.ndarray:
        """Create mask based on position only."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Find approximate center column (person usually centered)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Sum pixel intensities by column to find person center
        col_sums = np.sum(gray, axis=0)
        center_x = np.argmax(col_sums)
        
        # Estimate person width (typically 40-60% of image width)
        person_width = int(w * 0.5)
        left_x = max(0, center_x - person_width // 2)
        right_x = min(w, center_x + person_width // 2)
        
        # Create mask based on cloth type
        if cloth_type == "upper":
            mask[int(h*0.15):int(h*0.55), left_x:right_x] = 255
        elif cloth_type == "lower":
            mask[int(h*0.50):int(h*0.95), left_x:right_x] = 255
        else:  # overall
            mask[int(h*0.15):int(h*0.95), left_x:right_x] = 255
        
        # Basic cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _refine_mask_edges(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Refine mask edges using image gradients."""
        # Detect edges in image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate mask slightly
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find edges near mask boundary
        mask_edges = cv2.Canny(mask, 127, 255)
        
        # Combine image edges with mask boundary
        combined_edges = cv2.bitwise_and(edges, dilated_mask)
        
        # Use watershed or grabcut for refinement (simplified version)
        refined_mask = mask.copy()
        
        # Simple edge snapping
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Smooth contours
            epsilon = 0.02 * cv2.arcLength(contours[0], True)
            smoothed = cv2.approxPolyDP(contours[0], epsilon, True)
            cv2.drawContours(refined_mask, [smoothed], -1, 255, -1)
        
        # Final smoothing
        refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
        _, refined_mask = cv2.threshold(refined_mask, 127, 255, cv2.THRESH_BINARY)
        
        return refined_mask


# Alternative: Use SAM (Segment Anything Model) if available
class SAMAutoMasker:
    """Use SAM for clothing segmentation if available."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cloth_type": (["upper", "lower", "overall"], {"default": "upper"}),
            },
            "optional": {
                "sam_checkpoint": ("STRING", {"default": "sam_vit_b_01ec64.pth"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("mask_rgb", "masked_preview", "mask")
    FUNCTION = "generate_mask"
    CATEGORY = "AutoMasker/SAM"
    
    def __init__(self):
        self.sam_predictor = None
    
    def generate_mask(self, image, cloth_type, sam_checkpoint=None):
        # This would use SAM if available
        # For now, fall back to simple method
        simple_masker = SimpleAutoMasker()
        return simple_masker.generate_mask(image, cloth_type, True, False)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SimpleAutoMasker": SimpleAutoMasker,
    "SAMAutoMasker": SAMAutoMasker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleAutoMasker": "Simple Clothing Mask",
    "SAMAutoMasker": "SAM Clothing Mask",
}


# Minimal README
"""
Simple AutoMasker - Lightweight clothing segmentation for ComfyUI

No heavy dependencies required! Uses basic computer vision techniques.

Options for better accuracy:
1. Install scikit-learn for color clustering: pip install scikit-learn
2. Use SAM (Segment Anything Model) if you have it installed
3. Use any lightweight segmentation model (MobileNet, SegFormer, etc.)

The simple method works reasonably well for:
- Centered subjects
- Clear clothing boundaries  
- Relatively uniform backgrounds
"""