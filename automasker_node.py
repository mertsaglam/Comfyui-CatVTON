"""
ComfyUI AutoMasker Node - Standalone version
Generates clothing masks for virtual try-on applications
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Union, Dict, List, Tuple
import folder_paths

# ComfyUI imports
from torchvision.transforms.functional import to_pil_image, to_tensor
from diffusers.image_processor import VaeImageProcessor

# Model imports
from collections import OrderedDict
from torchvision import transforms
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image
from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.extractor import create_extractor, CompoundExtractor


# Constants for body part mappings
DENSE_INDEX_MAP = {
    "background": [0],
    "torso": [1, 2],
    "right hand": [3],
    "left hand": [4],
    "right foot": [5],
    "left foot": [6],
    "right thigh": [7, 9],
    "left thigh": [8, 10],
    "right leg": [11, 13],
    "left leg": [12, 14],
    "left big arm": [15, 17],
    "right big arm": [16, 18],
    "left forearm": [19, 21],
    "right forearm": [20, 22],
    "face": [23, 24],
    "thighs": [7, 8, 9, 10],
    "legs": [11, 12, 13, 14],
    "hands": [3, 4],
    "feet": [5, 6],
    "big arms": [15, 16, 17, 18],
    "forearms": [19, 20, 21, 22],
}

ATR_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Sunglasses': 3, 
    'Upper-clothes': 4, 'Skirt': 5, 'Pants': 6, 'Dress': 7,
    'Belt': 8, 'Left-shoe': 9, 'Right-shoe': 10, 'Face': 11, 
    'Left-leg': 12, 'Right-leg': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Bag': 16, 'Scarf': 17
}

LIP_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Glove': 3, 
    'Sunglasses': 4, 'Upper-clothes': 5, 'Dress': 6, 'Coat': 7,
    'Socks': 8, 'Pants': 9, 'Jumpsuits': 10, 'Scarf': 11, 
    'Skirt': 12, 'Face': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Left-leg': 16, 'Right-leg': 17, 'Left-shoe': 18, 'Right-shoe': 19
}

PROTECT_BODY_PARTS = {
    'upper': ['Left-leg', 'Right-leg'],
    'lower': ['Right-arm', 'Left-arm', 'Face'],
    'overall': [],
}

PROTECT_CLOTH_PARTS = {
    'upper': {
        'ATR': ['Skirt', 'Pants'],
        'LIP': ['Skirt', 'Pants']
    },
    'lower': {
        'ATR': ['Upper-clothes'],
        'LIP': ['Upper-clothes', 'Coat']
    },
    'overall': {
        'ATR': [],
        'LIP': []
    },
}

MASK_CLOTH_PARTS = {
    'upper': ['Upper-clothes', 'Coat', 'Dress', 'Jumpsuits'],
    'lower': ['Pants', 'Skirt', 'Dress', 'Jumpsuits'],
    'overall': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits'],
}

MASK_DENSE_PARTS = {
    'upper': ['torso', 'big arms', 'forearms'],
    'lower': ['thighs', 'legs'],
    'overall': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
}


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask."""
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def get_affine_transform(center, scale, rot, output_size):
    """Get affine transformation matrix."""
    def _get_3rd_point(a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)
    
    def _get_dir(src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs
        return src_result
    
    shift = np.array([0, 0], dtype=np.float32)
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]
    
    rot_rad = np.pi * rot / 180
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    
    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])
    
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def transform_logits(logits, center, scale, width, height, input_size):
    """Transform logits back to original image size."""
    trans = get_affine_transform(center, scale, 0, input_size)
    trans_inv = np.linalg.inv(np.vstack([trans, [0, 0, 1]]))[:2]
    
    target_logits = []
    for i in range(logits.shape[2]):
        channel_logits = cv2.warpAffine(
            logits[:, :, i],
            trans_inv,
            (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0)
        )
        target_logits.append(channel_logits)
    
    return np.stack(target_logits, axis=2)


def part_mask_of(part: Union[str, list], parse: np.ndarray, mapping: dict):
    """Create mask for specific body/clothing parts."""
    if isinstance(part, str):
        part = [part]
    mask = np.zeros_like(parse)
    for _ in part:
        if _ not in mapping:
            continue
        if isinstance(mapping[_], list):
            for i in mapping[_]:
                mask += (parse == i)
        else:
            mask += (parse == mapping[_])
    return mask


def hull_mask(mask_area: np.ndarray):
    """Create convex hull mask."""
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros_like(mask_area)
    for c in contours:
        hull = cv2.convexHull(c)
        hull_mask = cv2.fillPoly(np.zeros_like(mask_area), [hull], 255) | hull_mask
    return hull_mask


def vis_mask(image, mask):
    """Visualize mask on image."""
    image = np.array(image).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    mask = mask / 255
    return Image.fromarray((image * (1 - mask)).astype(np.uint8))


class SimpleDensePose:
    """Simplified DensePose wrapper without auto-download."""
    
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.config_path = os.path.join(model_path, 'densepose_rcnn_R_50_FPN_s1x.yaml')
        self.model_path = os.path.join(model_path, 'model_final_162be9.pkl')
        self.min_score = 0.8
        
        # Check if model files exist
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"DensePose config not found at {self.config_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"DensePose model not found at {self.model_path}")
        
        self.cfg = self._setup_config()
        self.predictor = DefaultPredictor(self.cfg)
        self.predictor.model.to(self.device)
    
    def _setup_config(self):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(self.config_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.min_score
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.freeze()
        return cfg
    
    def __call__(self, image: Image.Image, resize=1024) -> Image.Image:
        # Convert PIL to numpy array
        img = np.array(image)[:, :, ::-1]  # RGB to BGR
        h_orig, w_orig = img.shape[:2]
        
        # Resize if needed
        if max(img.shape) > resize:
            scale = resize / max(img.shape)
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        
        # Run prediction
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]
        
        # Extract segmentation
        H, W = img.shape[:2]
        result = np.zeros((H, W), dtype=np.uint8)
        
        if len(outputs) > 0:
            # Get the DensePose result
            boxes = outputs.pred_boxes.tensor.cpu().numpy()
            labels = outputs.pred_densepose.labels.cpu().numpy()
            
            for idx in range(len(outputs)):
                if idx >= len(labels):
                    continue
                    
                x1, y1, x2, y2 = boxes[idx].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                label_map = labels[idx]
                h, w = label_map.shape
                
                # Resize label map to box size
                label_resized = cv2.resize(label_map, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                result[y1:y2, x1:x2] = label_resized
        
        # Resize back to original size
        result = cv2.resize(result, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        return Image.fromarray(result)


class SimpleSCHP:
    """Simplified SCHP wrapper without auto-download."""
    
    def __init__(self, ckpt_path, device="cuda"):
        # Determine dataset type from checkpoint name
        if 'lip' in ckpt_path:
            self.dataset_type = 'lip'
            self.num_classes = 20
        elif 'atr' in ckpt_path:
            self.dataset_type = 'atr'
            self.num_classes = 18
        else:
            raise ValueError("Unknown dataset type in checkpoint path")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"SCHP checkpoint not found at {ckpt_path}")
        
        self.device = device
        self.input_size = [473, 473] if self.dataset_type == 'lip' else [512, 512]
        self.aspect_ratio = self.input_size[1] / self.input_size[0]
        self.palette = get_palette(self.num_classes)
        
        # Initialize model
        from model.SCHP.networks import init_model
        self.model = init_model('resnet101', num_classes=self.num_classes, pretrained=None).to(device)
        self._load_checkpoint(ckpt_path)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
        self.upsample = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)
    
    def _load_checkpoint(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
    
    def _preprocess(self, image: Image.Image):
        img = np.array(image)
        h, w, _ = img.shape
        
        # Calculate person center and scale
        center = np.array([w/2.0, h/2.0], dtype=np.float32)
        scale = np.array([w, h], dtype=np.float32)
        
        # Adjust scale to maintain aspect ratio
        if w > self.aspect_ratio * h:
            h = w / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        
        # Get affine transform
        trans = get_affine_transform(center, scale, 0, self.input_size)
        input_img = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        input_tensor = self.transform(input_img).to(self.device).unsqueeze(0)
        
        meta = {
            'center': center,
            'height': h,
            'width': w,
            'scale': scale,
            'rotation': 0
        }
        return input_tensor, meta
    
    def __call__(self, image: Image.Image) -> Image.Image:
        input_tensor, meta = self._preprocess(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            upsample_output = self.upsample(output[0][-1])
            upsample_output = upsample_output.squeeze(0).permute(1, 2, 0)  # CHW -> HWC
        
        # Transform back to original size
        logits_result = transform_logits(
            upsample_output.cpu().numpy(),
            meta['center'], meta['scale'],
            meta['width'], meta['height'],
            self.input_size
        )
        
        parsing_result = np.argmax(logits_result, axis=2)
        output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
        output_img.putpalette(self.palette)
        
        return output_img


class AutoMasker:
    """Main AutoMasker class for generating cloth-agnostic masks."""
    
    def __init__(self, densepose_ckpt: str, schp_ckpt: str, device: str = 'cuda'):
        """
        Initialize AutoMasker with model paths.
        
        Args:
            densepose_ckpt: Path to DensePose model directory
            schp_ckpt: Path to SCHP model directory
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.device = device
        
        # Initialize models
        self.densepose_processor = SimpleDensePose(densepose_ckpt, device)
        self.schp_processor_atr = SimpleSCHP(
            os.path.join(schp_ckpt, 'exp-schp-201908301523-atr.pth'), 
            device
        )
        self.schp_processor_lip = SimpleSCHP(
            os.path.join(schp_ckpt, 'exp-schp-201908261155-lip.pth'), 
            device
        )
        
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8, 
            do_normalize=False, 
            do_binarize=True, 
            do_convert_grayscale=True
        )
    
    def preprocess_image(self, image: Image.Image) -> Dict:
        """Run all preprocessing models on the image."""
        return {
            'densepose': self.densepose_processor(image, resize=1024),
            'schp_atr': self.schp_processor_atr(image),
            'schp_lip': self.schp_processor_lip(image)
        }
    
    @staticmethod
    def cloth_agnostic_mask(
        densepose_mask: Image.Image,
        schp_lip_mask: Image.Image,
        schp_atr_mask: Image.Image,
        part: str = 'overall'
    ) -> Image.Image:
        """Generate cloth-agnostic mask for specified clothing part."""
        
        assert part in ['upper', 'lower', 'overall'], f"Invalid part: {part}"
        
        w, h = densepose_mask.size
        
        # Calculate kernel sizes
        dilate_kernel = max(w, h) // 250
        dilate_kernel = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
        dilate_kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        
        kernal_size = max(w, h) // 25
        kernal_size = kernal_size if kernal_size % 2 == 1 else kernal_size + 1
        
        # Convert to numpy arrays
        densepose_mask = np.array(densepose_mask)
        schp_lip_mask = np.array(schp_lip_mask)
        schp_atr_mask = np.array(schp_atr_mask)
        
        # Strong Protect Area (Hands, Face, Feet)
        hands_protect_area = part_mask_of(['hands', 'feet'], densepose_mask, DENSE_INDEX_MAP)
        hands_protect_area = cv2.dilate(hands_protect_area, dilate_kernel, iterations=1)
        hands_protect_area = hands_protect_area & \
            (part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_atr_mask, ATR_MAPPING) | \
             part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_lip_mask, LIP_MAPPING))
        
        face_protect_area = part_mask_of('Face', schp_lip_mask, LIP_MAPPING)
        strong_protect_area = hands_protect_area | face_protect_area
        
        # Weak Protect Area
        body_protect_area = part_mask_of(PROTECT_BODY_PARTS[part], schp_lip_mask, LIP_MAPPING) | \
                           part_mask_of(PROTECT_BODY_PARTS[part], schp_atr_mask, ATR_MAPPING)
        
        hair_protect_area = part_mask_of(['Hair'], schp_lip_mask, LIP_MAPPING) | \
                           part_mask_of(['Hair'], schp_atr_mask, ATR_MAPPING)
        
        cloth_protect_area = part_mask_of(PROTECT_CLOTH_PARTS[part]['LIP'], schp_lip_mask, LIP_MAPPING) | \
                            part_mask_of(PROTECT_CLOTH_PARTS[part]['ATR'], schp_atr_mask, ATR_MAPPING)
        
        accessory_parts = ['Hat', 'Glove', 'Sunglasses', 'Bag', 'Left-shoe', 'Right-shoe', 'Scarf', 'Socks']
        accessory_protect_area = part_mask_of(accessory_parts, schp_lip_mask, LIP_MAPPING) | \
                                part_mask_of(accessory_parts, schp_atr_mask, ATR_MAPPING)
        
        weak_protect_area = body_protect_area | cloth_protect_area | hair_protect_area | \
                           strong_protect_area | accessory_protect_area
        
        # Mask Area
        strong_mask_area = part_mask_of(MASK_CLOTH_PARTS[part], schp_lip_mask, LIP_MAPPING) | \
                          part_mask_of(MASK_CLOTH_PARTS[part], schp_atr_mask, ATR_MAPPING)
        
        background_area = part_mask_of(['Background'], schp_lip_mask, LIP_MAPPING) & \
                         part_mask_of(['Background'], schp_atr_mask, ATR_MAPPING)
        
        mask_dense_area = part_mask_of(MASK_DENSE_PARTS[part], densepose_mask, DENSE_INDEX_MAP)
        mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=0.25, fy=0.25, 
                                    interpolation=cv2.INTER_NEAREST)
        mask_dense_area = cv2.dilate(mask_dense_area, dilate_kernel, iterations=2)
        mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=4, fy=4, 
                                    interpolation=cv2.INTER_NEAREST)
        
        # Combine mask areas
        mask_area = (np.ones_like(densepose_mask) & (~weak_protect_area) & (~background_area)) | mask_dense_area
        
        # Apply convex hull
        mask_area = hull_mask(mask_area * 255) // 255
        mask_area = mask_area & (~weak_protect_area)
        
        # Apply Gaussian blur
        mask_area = cv2.GaussianBlur(mask_area * 255, (kernal_size, kernal_size), 0)
        mask_area[mask_area < 25] = 0
        mask_area[mask_area >= 25] = 1
        
        # Final mask processing
        mask_area = (mask_area | strong_mask_area) & (~strong_protect_area)
        mask_area = cv2.dilate(mask_area, dilate_kernel, iterations=1)
        
        return Image.fromarray(mask_area * 255)
    
    def __call__(self, image: Image.Image, mask_type: str = "upper") -> Dict:
        """
        Generate mask for the given image.
        
        Args:
            image: Input PIL Image
            mask_type: Type of mask to generate ('upper', 'lower', 'overall')
            
        Returns:
            Dictionary containing the mask and intermediate results
        """
        assert mask_type in ['upper', 'lower', 'overall'], f"Invalid mask_type: {mask_type}"
        
        # Preprocess image with all models
        preprocess_results = self.preprocess_image(image)
        
        # Generate cloth-agnostic mask
        mask = self.cloth_agnostic_mask(
            preprocess_results['densepose'],
            preprocess_results['schp_lip'],
            preprocess_results['schp_atr'],
            part=mask_type
        )
        
        return {
            'mask': mask,
            'densepose': preprocess_results['densepose'],
            'schp_lip': preprocess_results['schp_lip'],
            'schp_atr': preprocess_results['schp_atr']
        }


# ComfyUI Node Classes
class LoadAutoMasker:
    """ComfyUI node to load the AutoMasker model."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "densepose_model_path": ("STRING", {
                    "default": "models/automasker/DensePose",
                    "multiline": False
                }),
                "schp_model_path": ("STRING", {
                    "default": "models/automasker/SCHP", 
                    "multiline": False
                }),
                "device": (["cuda", "cpu"],),
            }
        }
    
    RETURN_TYPES = ("AUTOMASKER",)
    RETURN_NAMES = ("automasker",)
    FUNCTION = "load_automasker"
    CATEGORY = "AutoMasker"
    
    def load_automasker(self, densepose_model_path, schp_model_path, device):
        # Convert relative paths to absolute paths based on ComfyUI models directory
        models_dir = folder_paths.models_dir
        
        if not os.path.isabs(densepose_model_path):
            densepose_model_path = os.path.join(models_dir, densepose_model_path)
        if not os.path.isabs(schp_model_path):
            schp_model_path = os.path.join(models_dir, schp_model_path)
        
        # Initialize automasker
        automasker = AutoMasker(
            densepose_ckpt=densepose_model_path,
            schp_ckpt=schp_model_path,
            device=device
        )
        
        return (automasker,)


class AutoMaskerGenerate:
    """ComfyUI node to generate masks using AutoMasker."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "automasker": ("AUTOMASKER",),
                "image": ("IMAGE",),
                "cloth_type": (["upper", "lower", "overall"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("mask", "masked_preview")
    FUNCTION = "generate_mask"
    CATEGORY = "AutoMasker"
    
    def generate_mask(self, automasker, image, cloth_type):
        # Convert from ComfyUI tensor format to PIL
        image_tensor = image.squeeze(0).permute(2, 0, 1)
        pil_image = to_pil_image(image_tensor)
        
        # Resize to standard size (768x1024)
        pil_image = pil_image.resize((768, 1024), Image.LANCZOS)
        
        # Generate mask
        result = automasker(pil_image, cloth_type)
        mask = result['mask']
        
        # Create masked preview
        masked_preview = vis_mask(pil_image, mask)
        
        # Convert back to ComfyUI tensor format
        mask_tensor = to_tensor(mask).permute(1, 2, 0).repeat(1, 1, 3).unsqueeze(0)
        preview_tensor = to_tensor(masked_preview).permute(1, 2, 0).unsqueeze(0)
        
        return (mask_tensor, preview_tensor)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadAutoMasker": LoadAutoMasker,
    "AutoMaskerGenerate": AutoMaskerGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAutoMasker": "Load AutoMasker",
    "AutoMaskerGenerate": "Generate Mask (AutoMasker)",
}