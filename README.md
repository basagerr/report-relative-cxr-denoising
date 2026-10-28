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

The experiments use [MIMIC-CXR-JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
images and [MIMIC-CXR v2.0.0](https://physionet.org/content/mimic-cxr/2.0.0/)
free-text radiology reports. Both require credentialed PhysioNet
access and compliance with the applicable data use agreements.

No patient images, reports, patient-level manifests, evaluation
outputs or trained checkpoints are distributed in this repository.

### Preparing the private data directory

Download the JPG images, `mimic-cxr-2.0.0-split.csv.gz` and
`mimic-cxr-2.0.0-metadata.csv.gz` from MIMIC-CXR-JPG, and the
corresponding report text files from MIMIC-CXR.

Preserve this directory structure outside the Git repository:

```text
<data-dir>/files/pXX/p<subject_id>/s<study_id>/<dicom_id>.jpg
<data-dir>/files/pXX/p<subject_id>/s<study_id>.txt
```

These are placeholders, not actual patient or study identifiers.
`pXX` is the patient-prefix directory. The report belongs to a
study and may correspond to multiple images. The metadata index
provides `ViewPosition` for identifying AP/PA images.

### Loading development data

`src/dataset.py` defines `MIMICCXRDataset`, which accepts a
preconstructed pandas DataFrame with one row per image and these
columns:

- `subject_id`: patient identifier
- `study_id`: study identifier
- `dicom_id`: image identifier
- `path`: local JPG image path
- `text`: full report text for the corresponding study

For an already prepared private development table, run from
the repository root:

```python
import pandas as pd
from src.config import Config
from src.dataset import MIMICCXRDataset

training = pd.read_csv("<private-training-table>.csv")
train_dataset = MIMICCXRDataset(training, Config(), split="train")
```

Use a separate patient-disjoint validation table with
`split="val"`. The loader reads the JPG image and returns the
report text. The BioViL-T wrapper applies `src/report_utils.py`,
preferring Impression, then Findings, with a 100-word limit.

The development cohort and patient-disjoint splitting protocol
are described in Section 4.1 of the paper. The original notebook
workflow used to construct these tables is not included here.

### Locked holdout evaluation

`scripts/run_one_seed.py` uses `src/final_data.py` to read
two private CSV manifests:

- `final_holdout_eligible_manifest.csv` requires `subject_id`,
  `study_id`, `dicom_id`, `relative_path`, `report_sha256`.
- `final_holdout_candidate_manifest.csv` requires `subject_id`,
  `study_id`, `dicom_id`, `report_relative_path`, `ViewPosition`,
  `split`.

These are the columns consumed by the loader, rather than a
complete specification of the original manifest files.

The manifests are joined on (`subject_id`, `study_id`, `dicom_id`).
The image and report paths are resolved under `<data-dir>`;
absolute paths and paths escaping that directory are rejected.

The locked evaluation expects 204 images from 204 distinct
patients, selected from the official test split with AP/PA views
and disjoint from development.

The loader verifies the SHA-256 of each report, computes image
SHA-256 values and verifies the combined manifest content
against `content_hash` in `final_holdout_eligible_meta.json`.

### Private evaluation inputs

The `--publication-dir` argument in the Usage command expects:

```text
<publication-dir>/
  final_holdout_eligible_manifest.csv
  final_holdout_candidate_manifest.csv
  final_holdout_eligible_meta.json
  publication_sweep/
    publication_freeze.json
    references/
      ref_noise2noise_seed<seed>/
        completed.json
        checkpoints/best.pt
      ref_supervised_seed<seed>/
        completed.json
        checkpoints/best.pt
```

The private freeze record also identifies the proposed and
N2V-baseline checkpoint paths. Evaluation seeds are 7, 42
and 123.

### Reproducibility scope

The repository contains model components, data loaders,
evaluation code and tests. It does not include the original
notebook workflow used to construct the development and
locked-holdout manifests, the reference-model training driver,
the private freeze records or the trained checkpoints.

Therefore, the public code alone is not sufficient to reproduce
the exact published experiments from raw PhysioNet downloads.

Keep credentialed data, patient-level manifests, checkpoints
and evaluation outputs outside the Git repository.

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

The authors acknowledge the use of resources provided by the Isambard-AI National AI
Research Resource (AIRR). Isambard-AI is operated by the University of Bristol and is
funded by the UK Government's Department for Business, Innovation, Science and Trade
(DBIST) via UK Research and Innovation; and the Science and Technology Facilities
Council [ST/AIRR/I-A-I/1023].

S. McIntosh-Smith, S. R. Alam and C. Woods. Isambard-AI: a leadership class
supercomputer optimised specifically for Artificial Intelligence. arXiv:2410.11199, 2024.
https://arxiv.org/abs/2410.11199

R. Basager is supported by a PhD scholarship from Imam Abdulrahman Bin Faisal University, Saudi Arabia.

We thank the authors of MIMIC-CXR, BioViL-T (hi-ml-multimodal), TorchXRayVision,
RAD-DINO, LPIPS and Pytorch-UNet.
