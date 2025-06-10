"""
Lightweight AutoMasker for ComfyUI
Simple clothing mask generation without heavy dependencies
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
import folder_paths
from typing import Tuple, Optional

class SimpleMaskGenerator:
    """Lightweight mask generator using traditional CV methods."""
    
    @staticmethod
    def detect_person(image: np.ndarray) -> Optional[np.ndarray]:
        """Simple person detection using color and edge detection."""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # Skin detection in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Background detection (assuming relatively uniform background)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find largest contour (likely the person)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create person mask
        person_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(person_mask, [largest_contour], -1, 255, -1)
        
        return person_mask
    
    @staticmethod
    def segment_clothing_simple(image: np.ndarray, person_mask: np.ndarray, 
                               cloth_type: str = "upper") -> np.ndarray:
        """Simple clothing segmentation based on position and color."""
        h, w = person_mask.shape
        
        # Get bounding box of person
        coords = np.column_stack(np.where(person_mask > 0))
        if len(coords) == 0:
            return np.zeros_like(person_mask)
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        person_height = y_max - y_min
        person_width = x_max - x_min
        
        # Create region masks based on cloth type
        region_mask = np.zeros_like(person_mask)
        
        if cloth_type == "upper":
            # Upper body: from neck to waist (roughly top 50-60%)
            upper_start = y_min + int(person_height * 0.15)  # Skip head
            upper_end = y_min + int(person_height * 0.65)
            region_mask[upper_start:upper_end, x_min:x_max] = 255
            
        elif cloth_type == "lower":
            # Lower body: from waist to feet (roughly bottom 40-50%)
            lower_start = y_min + int(person_height * 0.55)
            lower_end = y_max
            region_mask[lower_start:lower_end, x_min:x_max] = 255
            
        else:  # overall
            # Full body minus head
            full_start = y_min + int(person_height * 0.15)
            region_mask[full_start:y_max, x_min:x_max] = 255
        
        # Combine with person mask
        cloth_mask = cv2.bitwise_and(region_mask, person_mask)
        
        # Refine using color clustering
        masked_image = cv2.bitwise_and(image, image, mask=cloth_mask)
        
        # Use GrabCut for refinement (optional, can be removed for speed)
        if np.any(cloth_mask):
            try:
                mask_grab = np.zeros(person_mask.shape, dtype=np.uint8)
                mask_grab[cloth_mask > 0] = cv2.GC_PR_FGD
                mask_grab[person_mask == 0] = cv2.GC_BGD
                
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                cv2.grabCut(image, mask_grab, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
                cloth_mask = np.where((mask_grab == cv2.GC_FGD) | (mask_grab == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            except:
                pass  # Fall back to simple mask if GrabCut fails
        
        # Smooth the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel)
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur for smooth edges
        cloth_mask = cv2.GaussianBlur(cloth_mask, (11, 11), 0)
        _, cloth_mask = cv2.threshold(cloth_mask, 127, 255, cv2.THRESH_BINARY)
        
        return cloth_mask


class UNetMaskGenerator:
    """Optional: Use a lightweight U-Net model for better accuracy."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a pre-trained lightweight segmentation model."""
        # This is a placeholder - you would need to train or find a lightweight model
        # For now, we'll just use the simple method
        pass
    
    def generate_mask(self, image: Image.Image, cloth_type: str = "upper") -> np.ndarray:
        """Generate mask using the model."""
        if self.model is None:
            # Fall back to simple method
            return self.generate_mask_simple(image, cloth_type)
        
        # Model-based generation would go here
        pass
    
    def generate_mask_simple(self, image: Image.Image, cloth_type: str = "upper") -> np.ndarray:
        """Fall back to simple CV-based method."""
        img_array = np.array(image)
        
        # Detect person
        person_mask = SimpleMaskGenerator.detect_person(img_array)
        if person_mask is None:
            return np.zeros((image.height, image.width), dtype=np.uint8)
        
        # Segment clothing
        cloth_mask = SimpleMaskGenerator.segment_clothing_simple(img_array, person_mask, cloth_type)
        
        return cloth_mask


# ComfyUI Nodes
class LightweightAutoMasker:
    """Lightweight clothing mask generator for ComfyUI."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cloth_type": (["upper", "lower", "overall"], {"default": "upper"}),
                "method": (["simple", "grabcut", "edge_based"], {"default": "simple"}),
                "smoothing": ("INT", {"default": 5, "min": 0, "max": 21, "step": 2}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("mask_rgb", "masked_preview", "mask")
    FUNCTION = "generate_mask"
    CATEGORY = "AutoMasker/Lightweight"
    
    def generate_mask(self, image, cloth_type, method, smoothing, threshold):
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            # Convert to PIL
            img_tensor = image[i]
            pil_image = Image.fromarray((img_tensor.cpu().numpy() * 255).astype(np.uint8))
            img_array = np.array(pil_image)
            
            if method == "simple":
                # Basic position-based masking
                h, w = img_array.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # Simple rectangular regions
                if cloth_type == "upper":
                    mask[int(h*0.15):int(h*0.65), :] = 255
                elif cloth_type == "lower":
                    mask[int(h*0.55):h, :] = 255
                else:  # overall
                    mask[int(h*0.15):h, :] = 255
                
                # Apply edge detection to refine
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Find contours within the region
                contours, _ = cv2.findContours(edges & mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    refined_mask = np.zeros_like(mask)
                    cv2.drawContours(refined_mask, contours, -1, 255, -1)
                    mask = refined_mask
            
            elif method == "grabcut":
                # Use GrabCut algorithm
                mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
                
                # Initialize rectangle based on cloth type
                h, w = img_array.shape[:2]
                if cloth_type == "upper":
                    rect = (10, int(h*0.15), w-20, int(h*0.5))
                elif cloth_type == "lower":
                    rect = (10, int(h*0.55), w-20, int(h*0.4))
                else:
                    rect = (10, int(h*0.15), w-20, int(h*0.8))
                
                try:
                    bgd_model = np.zeros((1, 65), np.float64)
                    fgd_model = np.zeros((1, 65), np.float64)
                    cv2.grabCut(img_array, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                    mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
                except:
                    # Fall back to simple method
                    mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 255
            
            else:  # edge_based
                # Edge and color-based detection
                person_mask = SimpleMaskGenerator.detect_person(img_array)
                if person_mask is not None:
                    mask = SimpleMaskGenerator.segment_clothing_simple(img_array, person_mask, cloth_type)
                else:
                    # Fall back to simple rectangular mask
                    h, w = img_array.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    if cloth_type == "upper":
                        mask[int(h*0.15):int(h*0.65), :] = 255
                    elif cloth_type == "lower":
                        mask[int(h*0.55):h, :] = 255
                    else:
                        mask[int(h*0.15):h, :] = 255
            
            # Apply smoothing
            if smoothing > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smoothing, smoothing))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.GaussianBlur(mask, (smoothing, smoothing), 0)
            
            # Apply threshold
            _, mask = cv2.threshold(mask, int(threshold * 255), 255, cv2.THRESH_BINARY)
            
            # Create masked preview
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            masked_preview = cv2.bitwise_and(img_array, mask_3channel)
            
            # Add semi-transparent overlay
            overlay = img_array.copy()
            overlay[mask == 0] = overlay[mask == 0] * 0.3
            masked_preview = cv2.addWeighted(masked_preview, 0.7, overlay, 0.3, 0)
            
            # Convert to tensors
            mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(-1)
            mask_rgb = mask_tensor.repeat(1, 1, 3)
            preview_tensor = torch.from_numpy(masked_preview.astype(np.float32) / 255.0)
            
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


# Node registration
NODE_CLASS_MAPPINGS = {
    "LightweightAutoMasker": LightweightAutoMasker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LightweightAutoMasker": "Clothing Mask (Lightweight)",
}