"""Prepare complete-field, NOAA-grouped cutouts."""

import numpy as np
import pandas as pd

from arccnet.models import dataset_utils as ut_d
from arccnet.models import preprocessing_common as pp_common
from arccnet.models.cutouts.hale.data_preparation import create_and_validate_cv_folds

from . import config


def _matches_expected_cutout_shape(value: object, expected_shape: tuple[int, int]) -> bool:
    """Return whether a stored ``(height, width)`` value matches the complete field of view."""
    try:
        dimensions = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    if dimensions.size < 2 or not np.all(np.isfinite(dimensions[:2])):
        return False
    return bool(np.allclose(dimensions[:2], expected_shape, rtol=0.0, atol=0.5))


def complete_fov_mask(
    df: pd.DataFrame,
    hmi_shape: tuple[int, int] = config.EXPECTED_HMI_CUTOUT_SHAPE,
    mdi_shape: tuple[int, int] = config.EXPECTED_MDI_CUTOUT_SHAPE,
) -> pd.Series:
    """Identify rows whose preferred-instrument cutout contains the complete requested field of view."""
    required_columns = {"dim_image_cutout_hmi", "dim_image_cutout_mdi"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise KeyError(f"Complete-FOV filtering requires columns: {missing_columns}")

    hmi_available = pp_common.availability_mask(df, "path_image_cutout_hmi")
    mdi_available = pp_common.availability_mask(df, "path_image_cutout_mdi")
    hmi_complete = df["dim_image_cutout_hmi"].map(lambda value: _matches_expected_cutout_shape(value, hmi_shape))
    mdi_complete = df["dim_image_cutout_mdi"].map(lambda value: _matches_expected_cutout_shape(value, mdi_shape))

    # Match the image loader's HMI-first policy. MDI is considered only when HMI is unavailable.
    return (hmi_available & hmi_complete) | (~hmi_available & mdi_available & mdi_complete)


def prepare_dataset(save_path):
    raw, _, _ = ut_d.make_dataframe(config.DATA_FOLDER, config.DATASET_FOLDER, config.DF_FILE_NAME)
    clean = ut_d.cleanup_df(raw)
    clean, _ = pp_common.filter_cutouts_by_nan_threshold(
        clean, nan_threshold=0.05, data_folder=config.DATA_FOLDER, dataset_folder=config.DATASET_FOLDER, hdu_index=1
    )
    _, mapped = ut_d.undersample_group_filter(
        clean, label_mapping=config.label_mapping, long_limit_deg=config.LONG_LIMIT_DEG, undersample=False
    )
    complete = mapped.loc[complete_fov_mask(mapped)].copy()
    result = create_and_validate_cv_folds(complete, n_splits=config.N_FOLDS, random_state=config.RANDOM_STATE)
    result.to_parquet(save_path, index=False)
    return result
