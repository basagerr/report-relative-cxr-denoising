# Third-Party Notices

This repository includes research software developed for the
accompanying BMVC 2026 workshop paper.

## Adapted source code

### Pytorch-UNet
Source: https://github.com/milesial/Pytorch-UNet
License: GNU General Public License v3.0

The U-Net implementation in src/model.py adapts components
from Pytorch-UNet by milesial and contributors.

The original authors retain their respective rights.
The modifications made for this research project are
documented in src/model.py.

## External software and models

The following are external dependencies or models:

- PyTorch: https://github.com/pytorch/pytorch
- Torchvision: https://github.com/pytorch/vision
- TorchXRayVision: https://github.com/mlmed/torchxrayvision
- BioViL-T: https://github.com/microsoft/hi-ml
- RAD-DINO: https://huggingface.co/microsoft/rad-dino
- LPIPS: https://github.com/richzhang/PerceptualSimilarity
- pytorch-msssim: https://github.com/VainF/pytorch-msssim

These third-party projects retain their own licenses.
Users should consult the applicable upstream licenses and
model-weight terms. No third-party model weights are
redistributed in this repository.

## Restricted datasets

The experiments use MIMIC-CXR and MIMIC-CXR-JPG from PhysioNet.

No patient images, reports, restricted dataset files, or
trained checkpoints are redistributed in this repository.

Researchers must obtain their own authorized access and
follow the applicable PhysioNet data use agreements.
