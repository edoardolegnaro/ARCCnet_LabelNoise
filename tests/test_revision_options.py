"""Tests for the input-policy and instrument options added for the ApJ revision."""

import pytest
import torch

from arccnet_labelnoise import config, paper_training
from arccnet_labelnoise.dataset import resize_complete_fov_cutout


def _cutout(height=400, width=800):
    torch.manual_seed(0)
    return torch.randn(2, height, width)


def test_keep_ratio_is_the_default_and_preserves_the_paper_shape():
    resized = resize_complete_fov_cutout(_cutout(), 112, 224)
    assert resized.shape == (2, 112, 224)
    assert torch.equal(resized, resize_complete_fov_cutout(_cutout(), 112, 224, resize_mode="keep-ratio"))


def test_keep_ratio_rejects_a_square_target():
    with pytest.raises(ValueError, match="aspect ratio"):
        resize_complete_fov_cutout(_cutout(), 224, 224)


def test_stretch_resizes_to_a_square_target():
    assert resize_complete_fov_cutout(_cutout(), 224, 224, resize_mode="stretch").shape == (2, 224, 224)


def test_pad_centres_the_aspect_preserving_resize_and_zero_fills():
    image = _cutout(100, 200)
    padded = resize_complete_fov_cutout(image, 224, 224, resize_mode="pad")
    assert padded.shape == (2, 224, 224)
    assert torch.allclose(padded[:, 56:168, :], resize_complete_fov_cutout(image, 112, 224))
    assert torch.count_nonzero(padded[:, :56, :]) == 0
    assert torch.count_nonzero(padded[:, 168:, :]) == 0


@pytest.mark.parametrize("mode", ["stretch", "pad"])
def test_non_default_modes_still_reject_truncated_cutouts(mode):
    with pytest.raises(ValueError, match="Incomplete"):
        resize_complete_fov_cutout(_cutout(300, 800), 224, 224, resize_mode=mode)


def test_unknown_resize_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported resize mode"):
        resize_complete_fov_cutout(_cutout(), 112, 224, resize_mode="crop")


def test_cli_defaults_reproduce_the_paper_policy():
    args = paper_training.build_parser().parse_args(
        ["train-fold", "--run-root", "/tmp/run", "--data-type", "both", "--fold", "1"]
    )
    assert (args.input_height, args.input_width) == (112, 224)
    assert args.resize_mode == "keep-ratio"
    assert args.instrument == "all"


def test_apply_hale_config_updates_the_model_input_size(monkeypatch):
    # apply_hale_config mutates the shared config module; restore every setting afterwards.
    for name in [n for n in vars(config) if n.isupper()] + ["classes", "class_names", "label_mapping"]:
        monkeypatch.setattr(config, name, getattr(config, name))
    args = paper_training.build_parser().parse_args(
        [
            "train-fold",
            "--run-root",
            "/tmp/run",
            "--data-type",
            "both",
            "--fold",
            "1",
            "--input-height",
            "224",
            "--input-width",
            "224",
            "--resize-mode",
            "pad",
            "--instrument",
            "mdi",
        ]
    )
    paper_training.apply_hale_config(args, data_type="both")
    assert config.IMAGE_TARGET_SIZE == (224, 224)
    assert config.RESIZE_MODE == "pad"
    assert config.INSTRUMENT_FILTER == "mdi"
