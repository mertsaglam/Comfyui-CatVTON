# AutoMasker Node for ComfyUI

A standalone clothing mask generator for virtual try-on applications in ComfyUI.

## Features

- Generates accurate clothing masks for upper body, lower body, or full outfit
- Uses DensePose for body part detection
- Uses SCHP (Self-Correction Human Parsing) for semantic segmentation
- No automatic model downloads - full control over model management
- Optimized for ComfyUI integration

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r automasker_requirements.txt
   ```

2. **Install the node:**
   - Copy `automasker_standalone.py` to your `ComfyUI/custom_nodes/` directory
   - Or create a symbolic link to keep it in the original location

## Model Setup

### Required Models

1. **DensePose Model (R-50-FPN)**
   - Download: [model_final_162be9.pkl](https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl) (~256 MB)
   - Config files:
     - [Base-DensePose-RCNN-FPN.yaml](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/configs/Base-DensePose-RCNN-FPN.yaml)
     - [densepose_rcnn_R_50_FPN_s1x.yaml](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml)

2. **SCHP Models**
   - ATR model: `exp-schp-201908301523-atr.pth` (~250 MB)
   - LIP model: `exp-schp-201908261155-lip.pth` (~250 MB)
   - Download from the original SCHP repository or model zoo

### Directory Structure

Create the following structure in your ComfyUI models directory:

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

1. **Load AutoMasker Models**
   - Add the "Load AutoMasker Models" node
   - Set paths if different from defaults
   - Choose device (cuda/cpu)

2. **Generate Clothing Mask**
   - Connect the automasker output to "Generate Clothing Mask" node
   - Input your image
   - Select cloth type:
     - `upper`: Upper body clothing (shirts, jackets, etc.)
     - `lower`: Lower body clothing (pants, skirts, etc.)
     - `overall`: Full outfit

3. **Outputs**
   - `mask_rgb`: RGB mask for compatibility
   - `masked_preview`: Visual preview with mask applied
   - `mask`: Single-channel mask for further processing

## Workflow Example

```
[Load Image] -> [Generate Clothing Mask] -> [mask] -> [Your Virtual Try-On Pipeline]
                          ^
                          |
              [Load AutoMasker Models]
```

## Tips

- The models are cached after first load to avoid reloading
- Use "reload_models" option to force reload if needed
- Standard processing size is 768x1024 for best results
- Disable "resize_to_standard" to process at original resolution

## Troubleshooting

1. **Models not found**: Check that all model files are in the correct directories
2. **CUDA out of memory**: Try using CPU mode or reduce image size
3. **Import errors**: Ensure all dependencies are installed, especially detectron2

## License

This node uses models from:
- DensePose (Detectron2) - Apache 2.0 License
- SCHP - MIT License