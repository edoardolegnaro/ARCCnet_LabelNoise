# ARCCnet_LabelNoise

Code and derived tables for Legnaro et al., "Deep-Learning Classification and
Label-Noise Analysis of Solar Active-Region Cutouts". The images and labels are
the ARCAFF:CCD catalog (doi:10.5281/zenodo.17865447).

The package builds on [ARCCnet](https://github.com/ARCAFF/ARCCnet) v0.4.0 and uses
its data utilities and Hale classification modules unchanged.

## Installation

```sh
git clone https://github.com/edoardolegnaro/ARCCnet_LabelNoise.git
cd ARCCnet_LabelNoise
pip install -e ".[tests]"
pytest
```

## Data

Download ARCAFF:CCD and set the paths to it:

```sh
export ARCAFF_DATA_FOLDER=/path/to/data          # folder that contains the dataset folder
export ARCAFF_DATASET_FOLDER=arccnet-ar-classification-v20251016
export ARCAFF_DF_FILE_NAME=region_classification.parq
```

## Tables

- `data/folds.csv`: the 26,015 observations (4,817 NOAA regions). Each row gives the
  cutout file used by the image loader (HMI when available, otherwise MDI), the
  catalog and reduced class, and the validation and test fold (1–8).
- `data/predictions.csv`: held-out predictions for every observation. Columns cover
  Swin-Tiny on both modalities (class and probabilities), whether the observation
  is near a label transition (a different catalog class for the same region within
  two days), and the mean-probability committee of the ten best configurations.
  `review_candidate` marks the 2,103 observations on which all ten members predict
  the same non-catalog class, and `prediction_on_adjacent_day` marks candidates whose
  predicted class is cataloged for the same region on the preceding or following day.

## Training

```sh
python -m arccnet_labelnoise.paper_training prepare
python -m arccnet_labelnoise.paper_training train-fold \
    --run-root runs/benchmark --model-name swin_tiny_patch4_window7_224.ms_in1k \
    --data-type both --fold 1
python -m arccnet_labelnoise.paper_training collect --run-root runs/benchmark
```

`prepare` rebuilds the fold assignments in `data/folds.csv`. Other options cover the
loss ablation (`--loss-policy`), the input geometry (`--input-height`,
`--input-width`, `--resize-mode`), single-instrument training (`--instrument`) and
the δ head (`--delta-aux-weight`).

## Paper sections

| Paper | Code |
|---|---|
| Sections 4.1–4.2, Table 5 | `arccnet_labelnoise.paper_training` |
| Section 4.3 | `arccnet_labelnoise.conditional_delta_comparison` |
| Section 4.4, Appendix B | `arccnet_labelnoise.simplex_tuning` |
| Section 5 (per-observation diagnostics) | `arccnet_labelnoise.label_noise_analysis` |
| Section 5, Table 4, Appendix C | `analysis/adjacent_labels.py` |
| Section 4.1 (input geometry, seeds), Section 5 (committee) | `analysis/ensemble_and_geometry.py` |
| Section 4.1 (single-instrument transfer) | `analysis/instrument_transfer.py` |
| Section 4.1 (paired 2010 HMI/MDI views) | `arccnet_labelnoise.label_noise_analysis --paired-mdi`, `analysis/paired_views.py`, `analysis/magnetic_scale.py` |
| Section 5 (projection, crowding, temporal averaging) | `analysis/observing_context.py` |

## License

BSD 3-Clause, as ARCCnet (see `LICENSE.rst`).
