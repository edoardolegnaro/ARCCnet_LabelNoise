"""
Dataset utilities for Hale classification using existing ARCCNet utilities.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import arccnet_labelnoise.config as config
from arccnet.models import preprocessing_common as pp_common
from arccnet.models.cutouts.hale.dataset import get_fold_data as get_fold_data
from arccnet.models.cutouts.hale.dataset import (
    get_transforms,
    normalize_continuum,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(row, data_type="magnetogram", *, data_folder=None, dataset_folder=None):
    """Load the magnetogram or continuum FITS cutout of one observation as np.ndarray.

    Same as ``arccnet.models.cutouts.hale.dataset.load_image`` in ARCCnet v0.4.0, with the
    data folders passed as arguments. They default to the folders in this package's config.
    """
    data_folder = config.DATA_FOLDER if data_folder is None else data_folder
    dataset_folder = config.DATASET_FOLDER if dataset_folder is None else dataset_folder
    magnetogram_path = pp_common.resolve_preferred_cutout_fits_path(
        row,
        data_folder=data_folder,
        dataset_folder=dataset_folder,
    )
    if magnetogram_path is None:
        raise RuntimeError(f"No cutout FITS path available for row index {getattr(row, 'name', 'unknown')}")

    if data_type == "magnetogram":
        fits_file_path = magnetogram_path
    elif data_type == "continuum":
        mag_name = Path(magnetogram_path).name
        continuum_names = [
            mag_name.replace("_mag_", "_cont_"),
            mag_name.replace(".magnetogram.", ".continuum."),
        ]
        continuum_names = list(dict.fromkeys(continuum_names))

        fits_file_path = None
        for continuum_name in continuum_names:
            candidate = Path(magnetogram_path).with_name(continuum_name)
            if candidate.exists():
                fits_file_path = candidate
                break
            resolved = pp_common.resolve_cutout_fits_path(
                continuum_name,
                data_folder=data_folder,
                dataset_folder=dataset_folder,
            )
            if resolved is not None:
                fits_file_path = resolved
                break

        if fits_file_path is None:
            raise RuntimeError(f"Failed to resolve continuum FITS for magnetogram {magnetogram_path}")
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    try:
        return pp_common.load_fits_hdu_data(fits_file_path, hdu_index=1, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to load FITS file {fits_file_path}: {exc}") from exc


def resize_complete_fov_cutout(
    image: torch.Tensor,
    target_height: int,
    target_width: int,
    require_complete_fov: bool = True,
    resize_mode: str = "keep-ratio",
) -> torch.Tensor:
    """Resize a complete 2:1 cutout to the model input size."""
    if image.ndim != 3:
        raise ValueError(f"Expected cutout tensor with shape (C, H, W), got {tuple(image.shape)}")
    if resize_mode not in config.RESIZE_MODES:
        raise ValueError(f"Unsupported resize mode {resize_mode!r}; expected one of {config.RESIZE_MODES}.")

    source_height, source_width = image.shape[-2:]
    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    if resize_mode == "keep-ratio":
        reference_ratio = target_ratio
    else:
        expected_height, expected_width = config.EXPECTED_HMI_CUTOUT_SHAPE
        reference_ratio = expected_width / expected_height
    if require_complete_fov and not np.isclose(source_ratio, reference_ratio, rtol=0.0, atol=1e-6):
        raise ValueError(
            "Incomplete or geometrically incompatible cutout: "
            f"source shape {(source_height, source_width)} has aspect ratio {source_ratio:.6f}, "
            f"expected {reference_ratio:.6f}."
        )

    if resize_mode == "pad":
        scale = min(target_height / source_height, target_width / source_width)
        fit_height = max(1, int(round(source_height * scale)))
        fit_width = max(1, int(round(source_width * scale)))
        resized = transforms.functional.resize(image, [fit_height, fit_width], antialias=True)
        pad_top = (target_height - fit_height) // 2
        pad_left = (target_width - fit_width) // 2
        resized = torch.nn.functional.pad(
            resized,
            (pad_left, target_width - fit_width - pad_left, pad_top, target_height - fit_height - pad_top),
            value=0.0,
        )
    else:
        resized = transforms.functional.resize(image, [target_height, target_width], antialias=True)
    if resized.shape[-2:] != (target_height, target_width):
        raise RuntimeError(
            f"Cutout resize produced {tuple(resized.shape[-2:])}; expected {(target_height, target_width)}."
        )
    return resized


class HaleDataset(Dataset):
    """Dataset for Hale classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_type: str = "magnetogram",
        divisor: float = None,
        target_height: int = None,
        target_width: int = None,
        is_training: bool = False,
    ):
        self.df = df.copy()
        self.data_type = data_type
        self.divisor = divisor or config.IMAGE_DIVISOR
        self.target_height = target_height or config.IMAGE_TARGET_HEIGHT
        self.target_width = target_width or config.IMAGE_TARGET_WIDTH
        # Stored on the instance: DataLoader workers are spawned and re-import config defaults.
        self.data_folder = config.DATA_FOLDER
        self.dataset_folder = config.DATASET_FOLDER
        self.resize_mode = getattr(config, "RESIZE_MODE", "keep-ratio")
        self.is_training = is_training
        self.transform = get_transforms(is_training=is_training)
        if "model_labels" not in self.df.columns:
            raise ValueError("DataFrame must contain 'model_labels' column with contiguous indices starting from 0")
        self.return_delta_label = bool(float(getattr(config, "DELTA_AUX_WEIGHT", 0.0)) > 0.0)
        if self.return_delta_label and "delta_labels" not in self.df.columns:
            raise ValueError("Delta auxiliary training requires a 'delta_labels' column")
        unique_labels = sorted(self.df["model_labels"].unique())
        if unique_labels != list(range(len(unique_labels))):
            raise ValueError(
                f"model_labels must be contiguous starting from 0. Got: {unique_labels}, Expected: {list(range(len(unique_labels)))}"
            )
        if not hasattr(HaleDataset, "_logged_labels"):
            logger.info(f"Dataset using model_labels: {[int(label) for label in unique_labels]}")
            if "grouped_labels" in self.df.columns:
                grouped_mapping = self.df.groupby("model_labels")["grouped_labels"].first().sort_index().to_dict()
                logger.info("Original grouped_labels mapping: %s", grouped_mapping)
            HaleDataset._logged_labels = True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.data_type == "magnetogram":
            image = torch.from_numpy(
                np.nan_to_num(
                    load_image(row, "magnetogram", data_folder=self.data_folder, dataset_folder=self.dataset_folder),
                    nan=0.0,
                )
            ).unsqueeze(0)
            image = torch.clamp(image / self.divisor, config.HARDTANH_MIN_VAL, config.HARDTANH_MAX_VAL)
        elif self.data_type == "continuum":
            image = load_image(row, "continuum", data_folder=self.data_folder, dataset_folder=self.dataset_folder)
            image = normalize_continuum(image)
            image = torch.from_numpy(image).unsqueeze(0)
        elif self.data_type == "both":
            mag = torch.from_numpy(
                np.nan_to_num(
                    load_image(row, "magnetogram", data_folder=self.data_folder, dataset_folder=self.dataset_folder),
                    nan=0.0,
                )
            )
            mag = torch.clamp(mag / self.divisor, config.HARDTANH_MIN_VAL, config.HARDTANH_MAX_VAL)
            cont = load_image(row, "continuum", data_folder=self.data_folder, dataset_folder=self.dataset_folder)
            cont = normalize_continuum(cont)
            stacked = torch.stack([mag, torch.from_numpy(cont)], dim=0)
            image = stacked
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")
        image = resize_complete_fov_cutout(
            image,
            self.target_height,
            self.target_width,
            require_complete_fov=config.REQUIRE_COMPLETE_FOV,
            resize_mode=self.resize_mode,
        )
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row["model_labels"], dtype=torch.long)
        if self.return_delta_label:
            return image, label, torch.tensor(row["delta_labels"], dtype=torch.float32)
        return image, label
