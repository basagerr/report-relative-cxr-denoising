from typing import Any, Dict
import numpy as np
from .noise import add_iid_noise, stable_int_seed, make_blind_spot_mask

from typing import Optional
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .config import Config

"""Original MIMICCXRDataset from the publication notebook.
"""

class MIMICCXRDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, config: Config, split: str = "train",
                 add_synthetic_noise: bool = True, fixed_noise_type: Optional[str] = None,
                 fixed_sigma: Optional[float] = None):
        self.data_df=data_df.reset_index(drop=True); self.config=config; self.split=split
        self.add_synthetic_noise=add_synthetic_noise
        self.fixed_noise_type=fixed_noise_type; self.fixed_sigma=fixed_sigma
        self._bad_files=set()
        self.transform=transforms.Compose([transforms.Resize((config.image_size,config.image_size)), transforms.ToTensor()])
        self.augment=transforms.Compose([transforms.RandomAffine(degrees=5, translate=(0.05,0.05), scale=(0.95,1.05))]) if split=="train" else None
        print(f"Dataset initialized: {len(self.data_df)} images ({split}; noise={fixed_noise_type or 'mixture'})")
    def __len__(self): return len(self.data_df)
    def _load_image(self, path):
        if path in self._bad_files:
            raise RuntimeError("Previously marked unreadable publication image")
        try:
            with Image.open(path) as image:
                return self.transform(image.convert("L"))
        except Exception as exc:
            self._bad_files.add(path)
            raise RuntimeError(f"Unreadable publication image: {type(exc).__name__}") from exc



def _mimic_getitem(self, idx: int) -> Dict[str, Any]:
    row=self.data_df.iloc[idx]
    clean_image=self._load_image(row["path"])
    if self.augment is not None and self.split=="train": clean_image=self.augment(clean_image)

    if self.add_synthetic_noise:
        if self.split=="train":
            noise_type=self.fixed_noise_type or str(np.random.choice(self.config.noise_types))
            sigma=float(self.fixed_sigma if self.fixed_sigma is not None else np.random.uniform(self.config.noise_sigma_min,self.config.noise_sigma_max))
            noisy_image=add_iid_noise(clean_image,noise_type,sigma,clip=self.config.clip_noisy_before_model)
        else:
            noise_type=self.fixed_noise_type or self.config.noise_types[stable_int_seed(str(row.dicom_id)+"_type") % len(self.config.noise_types)]
            sigma=float(self.fixed_sigma if self.fixed_sigma is not None else np.random.RandomState(stable_int_seed(str(row.dicom_id)+"_sigma")).uniform(self.config.noise_sigma_min,self.config.noise_sigma_max))
            seed=stable_int_seed(f"{row.dicom_id}:{noise_type}:{sigma:.8f}")
            noisy_image=add_iid_noise(clean_image,noise_type,sigma,deterministic_seed=seed,clip=self.config.clip_noisy_before_model)
    else:
        noise_type="none"; sigma=0.0; noisy_image=clean_image.clone()

    mask_seed=None if self.split=="train" else stable_int_seed(str(row.dicom_id)+"_mask")
    blind_spot_mask=make_blind_spot_mask(clean_image.shape,self.config.blind_spot_ratio,mask_seed,self.config.mask_exclusion_radius)
    report=row.get("text",""); report="" if pd.isna(report) else str(report).strip()
    return {"clean_image":clean_image,"noisy_image":noisy_image,"blind_spot_mask":blind_spot_mask,
            "report_text":report,"subject_id":int(row.get("subject_id",0)),"study_id":int(row.get("study_id",0)),
            "dicom_id":str(row.get("dicom_id","")),"noise_type":noise_type,"noise_sigma":sigma}


MIMICCXRDataset.__getitem__ = _mimic_getitem
