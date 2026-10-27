# Beyond Reference Fidelity: Report-Relative Evaluation of Self-Supervised Chest-Radiograph Denoising

Code for the paper accepted at the MFM4Health workshop, BMVC 2026.

Self-supervised denoisers are usually evaluated with PSNR, SSIM and LPIPS, which compare
the output with a reference image only. This work measures whether a denoised chest
radiograph stays compatible with its paired radiology report, using the change in
BioViL-T image–report similarity (Δcos), and adds a one-sided report-relative constraint
and a dense feature term to Noise2Void training.

## Installation

Tested with Python 3.11 and PyTorch on an NVIDIA GH200. Install PyTorch for your
platform first, then:

```bash
pip install -r requirements.txt
```

## Data

The experiments use [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) and
[MIMIC-CXR](https://physionet.org/content/mimic-cxr/) reports, which require credentialed
access through PhysioNet. No data, data-derived files or trained checkpoints are included
in this repository.

## Usage

Run the tests (synthetic data only):

```bash
python -m pytest -q
```

Evaluate one trained model on the held-out test set (requires the data, the holdout
manifests and the checkpoints):

```bash
PYTHONPATH=. python scripts/run_one_seed.py --method proposed --seed 42 \
  --publication-dir <publication-dir> --final-data-dir <data-dir> --output-dir <output-dir>
```

`--method` is one of `proposed`, `n2v_baseline`, `noise2noise`, `supervised`; seeds are
`7`, `42`, `123`. The output directory must be outside the repository.

Summarise the per-image results into the holdout tables of the paper:

```bash
PYTHONPATH=. python scripts/make_paper_tables.py --results-dir <output-dir>
```

## Repository structure

```
src/       model, noise and masking, losses, BioViL-T wrapper, metrics, training, evaluation
scripts/   evaluation, tables and statistics
tests/     unit tests and a GPU smoke test
```

The selected configuration of the paper is given in `src/published.py`.

## Citation

```bibtex
@inproceedings{basager2026beyond,
  title     = {Beyond Reference Fidelity: Report-Relative Evaluation of Self-Supervised Chest-Radiograph Denoising},
  author    = {Basager, Raghed and Lawson, Daniel and Ke, Rihuan},
  booktitle = {BMVC 2026 Workshop on Multimodal Foundation Models for Healthcare (MFM4Health)},
  year      = {2026}
}
```

## License

GPL-3.0; see [LICENSE](LICENSE). The U-Net in `src/model.py` is adapted from
[Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) (GPL-3.0).

For third-party software, model and dataset notices, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgements

We thank the authors of MIMIC-CXR, BioViL-T (hi-ml-multimodal), TorchXRayVision,
RAD-DINO, LPIPS and Pytorch-UNet.
