"""
PyTorch Lightning model for Hale classification.
"""

import heapq
import logging
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics import Accuracy, F1Score

import arccnet_labelnoise.config as config
from arccnet_labelnoise.metrics import (
    compute_binary_classification_metric_report,
    compute_classification_metric_report,
)
from arccnet.models.train_utils import replace_activations

logger = logging.getLogger(__name__)

try:
    import timm
except ImportError:  # pragma: no cover - timm is part of the models extra, but keep imports robust.
    timm = None

FUSION_3D_SUFFIX = "_3d_fusion"
LATE_FUSION_SUFFIX = "_late_fusion"
FUSION_3D_CHANNELS = 8
LEGACY_TRANSFORMER_INPUT_SIZE = (224, 224)

TORCHVISION_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
    "swin_t": (models.swin_t, models.Swin_T_Weights.DEFAULT),
}
TRANSFORMER_BACKBONES = {"vit_b_16", "swin_t"}
TIMM_TRANSFORMER_BACKBONES = (
    "deit_tiny_patch16_224.fb_in1k",
    "deit_small_patch16_224.fb_in1k",
    "deit3_small_patch16_224.fb_in1k",
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "swin_tiny_patch4_window7_224.ms_in1k",
)
ALL_TRANSFORMER_BACKBONES = TRANSFORMER_BACKBONES | set(TIMM_TRANSFORMER_BACKBONES)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_input_size(input_size: tuple[int, int] | list[int]) -> tuple[int, int]:
    """Validate and normalize a model input ``(height, width)``."""
    if len(input_size) != 2:
        raise ValueError(f"Input size must contain height and width, got {input_size}.")
    height, width = (int(value) for value in input_size)
    if height <= 0 or width <= 0:
        raise ValueError(f"Input dimensions must be positive, got {(height, width)}.")
    return height, width


def is_3d_fusion_model(model_name: str) -> bool:
    """Return whether a model name requests the 3D modality-fusion stem."""
    return str(model_name).endswith(FUSION_3D_SUFFIX)


def is_late_fusion_model(model_name: str) -> bool:
    """Return whether a model name requests two independent modality branches."""
    return str(model_name).endswith(LATE_FUSION_SUFFIX)


def requires_both_modalities(model_name: str) -> bool:
    """Return whether a model architecture can only run on paired magnetogram/continuum input."""
    return is_3d_fusion_model(model_name) or is_late_fusion_model(model_name)


def base_model_name(model_name: str) -> str:
    """Return the 2D backbone name for a possibly suffixed model name."""
    model_name = str(model_name)
    if is_3d_fusion_model(model_name):
        return model_name.removesuffix(FUSION_3D_SUFFIX)
    if is_late_fusion_model(model_name):
        return model_name.removesuffix(LATE_FUSION_SUFFIX)
    return model_name


def supported_model_names() -> list[str]:
    """Return all supported Hale model names."""
    base_names = list(TORCHVISION_BACKBONES)
    return (
        base_names
        + [f"{name}{FUSION_3D_SUFFIX}" for name in base_names if name not in TRANSFORMER_BACKBONES]
        + [f"{name}{LATE_FUSION_SUFFIX}" for name in base_names if name not in TRANSFORMER_BACKBONES]
        + list(TIMM_TRANSFORMER_BACKBONES)
    )


def _adapt_conv_weight(original_weight: torch.Tensor, input_channels: int) -> torch.Tensor:
    """Adapt pretrained RGB first-layer weights to the requested input channel count."""
    if original_weight.shape[1] == input_channels:
        return original_weight
    if input_channels == 1:
        return original_weight.mean(dim=1, keepdim=True)
    if input_channels == 2 and original_weight.shape[1] >= 2:
        return original_weight[:, :2, :, :]

    adapted = original_weight.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1)
    return adapted * (original_weight.shape[1] / input_channels)


def _replace_first_conv2d(module: nn.Module, input_channels: int) -> None:
    """Replace the first Conv2d in a torchvision model with an input-channel adapted version."""
    for child_name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            new_conv = nn.Conv2d(
                input_channels,
                child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            with torch.no_grad():
                new_conv.weight.copy_(_adapt_conv_weight(child.weight.data, input_channels))
                if child.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(child.bias.data)
            setattr(module, child_name, new_conv)
            return
        try:
            _replace_first_conv2d(child, input_channels)
            return
        except RuntimeError:
            continue
    raise RuntimeError("No Conv2d layer found to adapt input channels.")


def _replace_last_linear(
    module: nn.Module, replacement: nn.Module | None = None, num_classes: int | None = None
) -> int:
    """Replace the last Linear layer in a torchvision model and return its input dimension."""
    for child_name, child in reversed(list(module.named_children())):
        if isinstance(child, nn.Linear):
            if replacement is None:
                if num_classes is None:
                    raise ValueError("num_classes is required when no replacement module is provided.")
                replacement = nn.Sequential(nn.Dropout(config.DROPOUT_RATE), nn.Linear(child.in_features, num_classes))
            setattr(
                module,
                child_name,
                replacement,
            )
            return child.in_features
        try:
            return _replace_last_linear(child, replacement=replacement, num_classes=num_classes)
        except RuntimeError:
            continue
    raise RuntimeError("No Linear classifier layer found to replace.")


def _process_rectangular_vit_input(self, x: torch.Tensor) -> torch.Tensor:
    """Patchify a rectangular input for a torchvision VisionTransformer."""
    input_height, input_width = self._arccnet_input_size
    patch_size = int(self.patch_size)
    batch_size, _, height, width = x.shape
    torch._assert(height == input_height, f"Wrong image height! Expected {input_height} but got {height}!")
    torch._assert(width == input_width, f"Wrong image width! Expected {input_width} but got {width}!")
    x = self.conv_proj(x)
    x = x.reshape(batch_size, self.hidden_dim, (height // patch_size) * (width // patch_size))
    return x.permute(0, 2, 1)


def _adapt_torchvision_vit_input(backbone: nn.Module, input_size: tuple[int, int]) -> nn.Module:
    """Resize a pretrained torchvision ViT positional grid for a rectangular input."""
    input_height, input_width = _normalize_input_size(input_size)
    patch_size = int(backbone.patch_size)
    if input_height % patch_size or input_width % patch_size:
        raise ValueError(f"ViT input {(input_height, input_width)} must be divisible by patch size {patch_size}.")
    if (input_height, input_width) == LEGACY_TRANSFORMER_INPUT_SIZE:
        return backbone

    position_embedding = backbone.encoder.pos_embedding.detach()
    class_embedding = position_embedding[:, :1]
    patch_embedding = position_embedding[:, 1:]
    old_grid_size = math.isqrt(patch_embedding.shape[1])
    if old_grid_size * old_grid_size != patch_embedding.shape[1]:
        raise ValueError("Torchvision ViT pretrained positional embedding is not a square patch grid.")

    hidden_dim = patch_embedding.shape[-1]
    patch_embedding = patch_embedding.reshape(1, old_grid_size, old_grid_size, hidden_dim).permute(0, 3, 1, 2)
    patch_embedding = nn.functional.interpolate(
        patch_embedding,
        size=(input_height // patch_size, input_width // patch_size),
        mode="bicubic",
        align_corners=True,
    )
    patch_embedding = patch_embedding.permute(0, 2, 3, 1).reshape(1, -1, hidden_dim)
    backbone.encoder.pos_embedding = nn.Parameter(torch.cat([class_embedding, patch_embedding], dim=1))

    backbone._arccnet_input_size = (input_height, input_width)
    backbone._process_input = _process_rectangular_vit_input.__get__(backbone, type(backbone))
    return backbone


def _load_torchvision_backbone(model_name: str, input_size: tuple[int, int]) -> nn.Module:
    """Load a torchvision backbone and fall back to random weights if pretrained weights are unavailable."""
    if model_name not in TORCHVISION_BACKBONES:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {supported_model_names()}")
    model_fn, pretrained_weights = TORCHVISION_BACKBONES[model_name]
    try:
        backbone = model_fn(weights=pretrained_weights)
    except Exception as exc:
        logger.warning(
            "Could not load pretrained weights for %s (%s). Falling back to random initialization.",
            model_name,
            exc,
        )
        backbone = model_fn(weights=None)
    if model_name == "vit_b_16":
        backbone = _adapt_torchvision_vit_input(backbone, input_size)
    return backbone


def _load_timm_backbone(
    model_name: str,
    input_channels: int,
    num_classes: int,
    input_size: tuple[int, int],
) -> nn.Module:
    """Load a timm classifier backbone with pretrained weights when available."""
    if timm is None:
        raise ImportError(
            f"Model {model_name} requires timm. Install the models extra with `python -m pip install -e '.[models]'`."
        )
    try:
        return timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            in_chans=input_channels,
            img_size=input_size,
        )
    except Exception as exc:
        logger.warning(
            "Could not load pretrained timm weights for %s (%s). Falling back to random initialization.",
            model_name,
            exc,
        )
        return timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=input_channels,
            img_size=input_size,
        )


def _load_timm_feature_backbone(
    model_name: str,
    input_channels: int,
    input_size: tuple[int, int],
) -> tuple[nn.Module, int]:
    """Load a timm backbone that returns penultimate features."""
    if timm is None:
        raise ImportError(
            f"Model {model_name} requires timm. Install the models extra with `python -m pip install -e '.[models]'`."
        )
    try:
        backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            in_chans=input_channels,
            img_size=input_size,
        )
    except Exception as exc:
        logger.warning(
            "Could not load pretrained timm feature weights for %s (%s). Falling back to random initialization.",
            model_name,
            exc,
        )
        backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            in_chans=input_channels,
            img_size=input_size,
        )
    feature_dim = int(getattr(backbone, "num_features", 0))
    if feature_dim <= 0:
        with torch.no_grad():
            probe = torch.zeros(1, input_channels, *input_size)
            feature_dim = int(backbone(probe).shape[-1])
    return backbone, feature_dim


class ModalityFusion3DStem(nn.Module):
    """Fuse magnetogram and continuum inputs with 3D convolutions before a 2D backbone."""

    def __init__(self, out_channels: int = FUSION_3D_CHANNELS):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(1, out_channels, kernel_size=(2, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(config.LEAKY_RELU_NEGATIVE_SLOPE, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(config.LEAKY_RELU_NEGATIVE_SLOPE, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("3D fusion expects stacked magnetogram/continuum input with shape (N, 2, H, W).")
        x = self.layers(x.unsqueeze(1))
        return x.squeeze(2)


class TransformerInputResize(nn.Module):
    """Adapt transformer channels and enforce one shared spatial input policy."""

    def __init__(
        self,
        backbone: nn.Module,
        input_channels: int,
        input_size: tuple[int, int],
        resize_legacy_input: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.input_size = _normalize_input_size(input_size)
        self.resize_legacy_input = bool(resize_legacy_input)
        self.rgb_adapter = self._build_rgb_adapter(input_channels)
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @staticmethod
    def _build_rgb_adapter(input_channels: int) -> nn.Module:
        if not getattr(config, "TRANSFORMER_RGB_ADAPTER", False):
            return nn.Identity()
        adapter = nn.Conv2d(input_channels, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            adapter.weight.zero_()
            if input_channels == 1:
                adapter.weight[:, 0, 0, 0] = 1.0
            elif input_channels == 2:
                adapter.weight[0, 0, 0, 0] = 1.0
                adapter.weight[1, 1, 0, 0] = 1.0
                adapter.weight[2, :, 0, 0] = 0.5
            else:
                for channel in range(min(input_channels, 3)):
                    adapter.weight[channel, channel, 0, 0] = 1.0
        return adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.input_size:
            if not self.resize_legacy_input:
                raise ValueError(f"Transformer expected input {self.input_size}, received {tuple(x.shape[-2:])}.")
            x = nn.functional.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        x = self.rgb_adapter(x)
        if getattr(config, "TRANSFORMER_IMAGENET_NORMALIZE", False):
            x = (x - self.imagenet_mean) / self.imagenet_std
        return self.backbone(x)


def _build_classification_backbone(
    model_name: str,
    input_channels: int,
    num_classes: int,
    input_size: tuple[int, int] | None = None,
) -> nn.Module:
    """Build a single-stream classifier backbone with adapted channels and classifier."""
    resolved_input_size = _normalize_input_size(input_size or LEGACY_TRANSFORMER_INPUT_SIZE)
    resize_legacy_input = input_size is None
    if model_name in TIMM_TRANSFORMER_BACKBONES:
        timm_input_channels = 3 if getattr(config, "TRANSFORMER_RGB_ADAPTER", False) else input_channels
        backbone = _load_timm_backbone(model_name, timm_input_channels, num_classes, resolved_input_size)
        return TransformerInputResize(
            backbone,
            input_channels,
            resolved_input_size,
            resize_legacy_input=resize_legacy_input,
        )

    backbone = _load_torchvision_backbone(model_name, resolved_input_size)
    replace_activations(backbone, nn.ReLU, nn.LeakyReLU, negative_slope=config.LEAKY_RELU_NEGATIVE_SLOPE)
    if model_name in ALL_TRANSFORMER_BACKBONES and getattr(config, "TRANSFORMER_RGB_ADAPTER", False):
        _replace_first_conv2d(backbone, 3)
    else:
        _replace_first_conv2d(backbone, input_channels)
    _replace_last_linear(backbone, num_classes=num_classes)
    if model_name in ALL_TRANSFORMER_BACKBONES:
        return TransformerInputResize(
            backbone,
            input_channels,
            resolved_input_size,
            resize_legacy_input=resize_legacy_input,
        )
    return backbone


def _build_feature_backbone(
    model_name: str,
    input_channels: int = 1,
    input_size: tuple[int, int] | None = None,
) -> tuple[nn.Module, int]:
    """Build a backbone that returns penultimate features for late fusion."""
    resolved_input_size = _normalize_input_size(input_size or LEGACY_TRANSFORMER_INPUT_SIZE)
    resize_legacy_input = input_size is None
    if model_name in TIMM_TRANSFORMER_BACKBONES:
        timm_input_channels = 3 if getattr(config, "TRANSFORMER_RGB_ADAPTER", False) else input_channels
        backbone, feature_dim = _load_timm_feature_backbone(model_name, timm_input_channels, resolved_input_size)
        return (
            TransformerInputResize(
                backbone,
                input_channels,
                resolved_input_size,
                resize_legacy_input=resize_legacy_input,
            ),
            feature_dim,
        )

    backbone = _load_torchvision_backbone(model_name, resolved_input_size)
    replace_activations(backbone, nn.ReLU, nn.LeakyReLU, negative_slope=config.LEAKY_RELU_NEGATIVE_SLOPE)
    if model_name in ALL_TRANSFORMER_BACKBONES and getattr(config, "TRANSFORMER_RGB_ADAPTER", False):
        _replace_first_conv2d(backbone, 3)
    else:
        _replace_first_conv2d(backbone, input_channels)
    feature_dim = _replace_last_linear(backbone, replacement=nn.Identity())
    if model_name in ALL_TRANSFORMER_BACKBONES:
        backbone = TransformerInputResize(
            backbone,
            input_channels,
            resolved_input_size,
            resize_legacy_input=resize_legacy_input,
        )
    return backbone, feature_dim


class FeatureHeadClassifier(nn.Module):
    """Classifier with a shared feature extractor and optional delta auxiliary head."""

    def __init__(self, feature_backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.feature_backbone = feature_backbone
        self.classifier = nn.Sequential(nn.Dropout(config.DROPOUT_RATE), nn.Linear(feature_dim, num_classes))
        self.delta_head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return self.classifier(features), self.delta_head(features).squeeze(1)


class LateFusionClassifier(nn.Module):
    """Fuse magnetogram and continuum features from separate pretrained backbones."""

    def __init__(self, backbone_name: str, num_classes: int, input_size: tuple[int, int] | None = None):
        super().__init__()
        self.magnetogram_branch, mag_dim = _build_feature_backbone(
            backbone_name,
            input_channels=1,
            input_size=input_size,
        )
        self.continuum_branch, cont_dim = _build_feature_backbone(
            backbone_name,
            input_channels=1,
            input_size=input_size,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(mag_dim + cont_dim),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(mag_dim + cont_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("Late-fusion models require stacked magnetogram/continuum input with shape (N, 2, H, W).")
        mag_features = self.magnetogram_branch(x[:, 0:1])
        cont_features = self.continuum_branch(x[:, 1:2])
        return self.classifier(torch.cat([mag_features, cont_features], dim=1))


class HaleLightningModel(pl.LightningModule):
    """Simple Lightning model for Hale classification (Alpha, Beta, Beta-Gamma)."""

    def __init__(
        self,
        num_classes: int = 3,
        learning_rate: float = 1e-3,
        model_name: str = "resnet18",
        class_weights: torch.Tensor = None,
        loss_policy: str = "balanced_ce",
        focal_loss_gamma: float = 2.0,
        delta_aux_weight: float = 0.0,
        delta_pos_weight: torch.Tensor = None,
        delta_aux_mask_alpha: bool = False,
        alpha_class_index: int | None = None,
        data_type: str = None,
        input_size: tuple[int, int] | list[int] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.data_type = data_type or config.DATA_TYPE
        self.model_name = model_name
        self.input_size = _normalize_input_size(input_size) if input_size is not None else None
        self.uses_3d_fusion = is_3d_fusion_model(model_name)
        self.uses_late_fusion = is_late_fusion_model(model_name)
        self.delta_aux_weight = float(delta_aux_weight)
        self.use_delta_aux = self.delta_aux_weight > 0.0
        self.delta_aux_mask_alpha = bool(delta_aux_mask_alpha)
        self.alpha_class_index = alpha_class_index

        if self.use_delta_aux and self.delta_aux_mask_alpha and self.alpha_class_index is None:
            raise ValueError("Delta auxiliary Alpha masking requires alpha_class_index.")

        if requires_both_modalities(model_name) and self.data_type != "both":
            raise ValueError("Fusion models require data_type='both'.")

        backbone_name = base_model_name(model_name)
        if self.use_delta_aux and self.uses_late_fusion:
            raise ValueError("Delta auxiliary head is not implemented for late-fusion models.")
        if self.uses_late_fusion:
            self.modality_fusion = None
            self.backbone = LateFusionClassifier(backbone_name, num_classes, input_size=self.input_size)
        elif self.use_delta_aux:
            input_channels = FUSION_3D_CHANNELS if self.uses_3d_fusion else (2 if self.data_type == "both" else 1)
            self.modality_fusion = ModalityFusion3DStem(input_channels) if self.uses_3d_fusion else None
            feature_backbone, feature_dim = _build_feature_backbone(
                backbone_name,
                input_channels,
                input_size=self.input_size,
            )
            self.backbone = FeatureHeadClassifier(feature_backbone, feature_dim, num_classes)
        else:
            input_channels = FUSION_3D_CHANNELS if self.uses_3d_fusion else (2 if self.data_type == "both" else 1)
            self.modality_fusion = ModalityFusion3DStem(input_channels) if self.uses_3d_fusion else None
            self.backbone = _build_classification_backbone(
                backbone_name,
                input_channels,
                num_classes,
                input_size=self.input_size,
            )

        self.loss_policy = loss_policy
        self.focal_loss_gamma = focal_loss_gamma
        self.class_weights = class_weights

        if loss_policy in {"unweighted_ce", "balanced_ce", "sqrt_balanced_ce"}:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_policy == "focal_loss":
            self.criterion = None
        else:
            raise ValueError(f"Unsupported Hale loss policy: {loss_policy}")

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_balanced_accuracy = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.val_balanced_accuracy = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.test_balanced_accuracy = Accuracy(task="multiclass", num_classes=num_classes, average="macro")

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.delta_criterion = nn.BCEWithLogitsLoss(pos_weight=delta_pos_weight) if self.use_delta_aux else None
        if self.use_delta_aux:
            self.train_delta_accuracy = Accuracy(task="binary")
            self.val_delta_accuracy = Accuracy(task="binary")
            self.test_delta_accuracy = Accuracy(task="binary")
            self.train_delta_f1 = F1Score(task="binary")
            self.val_delta_f1 = F1Score(task="binary")
            self.test_delta_f1 = F1Score(task="binary")

        self.test_predictions = []
        self.test_targets = []
        self.test_logits = []
        self.test_delta_predictions = []
        self.test_delta_targets = []
        self.test_delta_logits = []
        self.test_delta_main_targets = []
        self.test_delta_all_targets = []
        self.test_delta_all_logits = []
        self.test_misclassified_samples = []
        self._test_misclassified_counter = 0

    def forward(self, x):
        if self.input_size is not None and x.shape[-2:] != self.input_size:
            raise ValueError(f"Model expected input {self.input_size}, received {tuple(x.shape[-2:])}.")
        if self.modality_fusion is not None:
            x = self.modality_fusion(x)
        return self.backbone(x)

    def _split_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if len(batch) == 3:
            images, labels, delta_labels = batch
            return images, labels, delta_labels.float()
        images, labels = batch
        return images, labels, None

    def _forward_logits(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self(images)
        if self.use_delta_aux:
            logits, delta_logits = outputs
            return logits, delta_logits
        return outputs, None

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the configured classification loss."""
        if self.loss_policy != "focal_loss":
            return self.criterion(logits, labels)

        ce_loss = nn.functional.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        if self.class_weights is not None:
            ce_loss = ce_loss * self.class_weights.to(logits.device)[labels]
        return ((1.0 - pt) ** self.focal_loss_gamma * ce_loss).mean()

    def _compute_total_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        delta_logits: torch.Tensor | None,
        delta_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        class_loss = self._compute_loss(logits, labels)
        if not self.use_delta_aux:
            return class_loss, None
        if delta_logits is None or delta_labels is None:
            raise ValueError("Delta auxiliary training requires delta logits and labels.")
        eligible_delta_logits, eligible_delta_labels = self._select_delta_samples(
            delta_logits,
            delta_labels,
            labels,
        )
        if eligible_delta_labels.numel() == 0:
            delta_loss = delta_logits.sum() * 0.0
        else:
            delta_loss = self.delta_criterion(
                eligible_delta_logits,
                eligible_delta_labels.to(eligible_delta_logits.device),
            )
        return class_loss + self.delta_aux_weight * delta_loss, delta_loss

    def _select_delta_samples(
        self,
        delta_logits: torch.Tensor,
        delta_labels: torch.Tensor,
        main_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select samples that are physically eligible for the auxiliary delta task."""
        eligible_mask = self._delta_eligibility_mask(main_labels, delta_logits.device)
        return delta_logits[eligible_mask], delta_labels.to(delta_logits.device)[eligible_mask]

    def _delta_eligibility_mask(self, main_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Return the ground-truth eligibility mask for the auxiliary delta task."""
        labels = main_labels.to(device)
        if not self.delta_aux_mask_alpha:
            return torch.ones_like(labels, dtype=torch.bool)
        return labels != self.alpha_class_index

    def _update_delta_metrics(
        self,
        accuracy_metric,
        f1_metric,
        delta_logits: torch.Tensor,
        delta_labels: torch.Tensor,
        main_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update auxiliary metrics on eligible samples and return their predictions and targets."""
        eligible_logits, eligible_labels = self._select_delta_samples(delta_logits, delta_labels, main_labels)
        delta_preds = (torch.sigmoid(eligible_logits) >= 0.5).long()
        eligible_targets = eligible_labels.long()
        if eligible_targets.numel() > 0:
            accuracy_metric.update(delta_preds, eligible_targets)
            f1_metric.update(delta_preds, eligible_targets)
        return delta_preds, eligible_targets

    def training_step(self, batch, batch_idx):
        images, labels, delta_labels = self._split_batch(batch)
        logits, delta_logits = self._forward_logits(images)
        loss, delta_loss = self._compute_total_loss(logits, labels, delta_logits, delta_labels)

        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, labels)
        self.train_balanced_accuracy.update(preds, labels)
        self.train_f1.update(preds, labels)
        if self.use_delta_aux:
            self._update_delta_metrics(
                self.train_delta_accuracy,
                self.train_delta_f1,
                delta_logits,
                delta_labels,
                labels,
            )

        batch_size = labels.size(0)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if delta_loss is not None:
            self.log("train_delta_loss", delta_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def on_train_epoch_end(self):
        """Log training epoch summary."""
        train_acc = float(self.train_accuracy.compute().detach().cpu())
        train_balanced_acc = float(self.train_balanced_accuracy.compute().detach().cpu())
        train_f1 = float(self.train_f1.compute().detach().cpu())
        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_balanced_acc", train_balanced_acc)
        self.log("train_f1", train_f1)
        if self.use_delta_aux:
            train_delta_acc = float(self.train_delta_accuracy.compute().detach().cpu())
            train_delta_f1 = float(self.train_delta_f1.compute().detach().cpu())
            self.log("train_delta_acc", train_delta_acc)
            self.log("train_delta_f1", train_delta_f1)
            self.train_delta_accuracy.reset()
            self.train_delta_f1.reset()
        print(
            f"Training Epoch {self.current_epoch} completed - "
            f"Acc: {train_acc:.4f}, Balanced Acc: {train_balanced_acc:.4f}, F1: {train_f1:.4f}"
        )
        self.train_accuracy.reset()
        self.train_balanced_accuracy.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        images, labels, delta_labels = self._split_batch(batch)
        logits, delta_logits = self._forward_logits(images)
        loss, delta_loss = self._compute_total_loss(logits, labels, delta_logits, delta_labels)

        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, labels)
        self.val_balanced_accuracy.update(preds, labels)
        self.val_f1.update(preds, labels)
        if self.use_delta_aux:
            self._update_delta_metrics(
                self.val_delta_accuracy,
                self.val_delta_f1,
                delta_logits,
                delta_labels,
                labels,
            )

        batch_size = labels.size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if delta_loss is not None:
            self.log("val_delta_loss", delta_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def on_validation_epoch_end(self):
        """Log validation epoch summary."""
        val_acc = float(self.val_accuracy.compute().detach().cpu())
        val_balanced_acc = float(self.val_balanced_accuracy.compute().detach().cpu())
        val_f1 = float(self.val_f1.compute().detach().cpu())
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_balanced_acc", val_balanced_acc)
        self.log("val_f1", val_f1)
        if self.use_delta_aux:
            val_delta_acc = float(self.val_delta_accuracy.compute().detach().cpu())
            val_delta_f1 = float(self.val_delta_f1.compute().detach().cpu())
            self.log("val_delta_acc", val_delta_acc)
            self.log("val_delta_f1", val_delta_f1)
            self.val_delta_accuracy.reset()
            self.val_delta_f1.reset()
        print(
            f"Validation Epoch {self.current_epoch} completed - "
            f"Acc: {val_acc:.4f}, Balanced Acc: {val_balanced_acc:.4f}, F1: {val_f1:.4f}"
        )
        self.val_accuracy.reset()
        self.val_balanced_accuracy.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        images, labels, delta_labels = self._split_batch(batch)
        logits, delta_logits = self._forward_logits(images)
        loss, delta_loss = self._compute_total_loss(logits, labels, delta_logits, delta_labels)

        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, labels)
        self.test_balanced_accuracy.update(preds, labels)
        self.test_f1.update(preds, labels)
        if self.use_delta_aux:
            self.test_delta_all_targets.extend(delta_labels.long().detach().cpu().numpy())
            self.test_delta_all_logits.append(delta_logits.detach().cpu())
            delta_preds, eligible_delta_targets = self._update_delta_metrics(
                self.test_delta_accuracy,
                self.test_delta_f1,
                delta_logits,
                delta_labels,
                labels,
            )
            self.test_delta_predictions.extend(delta_preds.detach().cpu().numpy())
            self.test_delta_targets.extend(eligible_delta_targets.detach().cpu().numpy())
            eligible_delta_logits, _ = self._select_delta_samples(delta_logits, delta_labels, labels)
            self.test_delta_logits.append(eligible_delta_logits.detach().cpu())
            eligible_mask = self._delta_eligibility_mask(labels, delta_logits.device)
            self.test_delta_main_targets.extend(labels.to(delta_logits.device)[eligible_mask].detach().cpu().numpy())

        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(labels.cpu().numpy())
        self.test_logits.append(logits.detach().cpu())
        self._update_top_misclassified_samples(images, labels, preds, logits, batch_idx)

        batch_size = labels.size(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        if delta_loss is not None:
            self.log("test_delta_loss", delta_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        return {"test_loss": loss, "preds": preds, "targets": labels}

    def on_test_epoch_end(self) -> None:
        """Reset aggregated test metrics after epoch end."""
        test_acc = float(self.test_accuracy.compute().detach().cpu())
        test_balanced_acc = float(self.test_balanced_accuracy.compute().detach().cpu())
        test_f1 = float(self.test_f1.compute().detach().cpu())
        self.log("test_acc", test_acc)
        self.log("test_balanced_acc", test_balanced_acc)
        self.log("test_f1", test_f1)
        if self.use_delta_aux:
            test_delta_acc = float(self.test_delta_accuracy.compute().detach().cpu())
            test_delta_f1 = float(self.test_delta_f1.compute().detach().cpu())
            self.log("test_delta_acc", test_delta_acc)
            self.log("test_delta_f1", test_delta_f1)

        metric_report = self.get_classification_metric_report(config.class_names)
        if metric_report is not None:
            self.log("test_macro_f1", metric_report["macro_f1"])
            self.log("test_weighted_f1", metric_report["weighted_f1"])
            self.log("test_macro_precision", metric_report["macro_precision"])
            self.log("test_macro_recall", metric_report["macro_recall"])

        self.test_accuracy.reset()
        self.test_balanced_accuracy.reset()
        self.test_f1.reset()
        if self.use_delta_aux:
            self.test_delta_accuracy.reset()
            self.test_delta_f1.reset()

    def get_delta_metric_report(self) -> dict | None:
        """Compute eligible and within-base delta metrics plus a label-only control."""
        if not self.test_delta_predictions or not self.test_delta_targets:
            return None
        delta_scores = torch.sigmoid(torch.cat(self.test_delta_logits, dim=0)).numpy()
        delta_targets = np.asarray(self.test_delta_targets, dtype=int)
        delta_predictions = np.asarray(self.test_delta_predictions, dtype=int)
        main_targets = np.asarray(self.test_delta_main_targets, dtype=int)
        if not (len(delta_targets) == len(delta_scores) == len(main_targets)):
            raise ValueError("Collected eligible delta outputs and main targets are misaligned.")

        report = compute_binary_classification_metric_report(
            delta_targets,
            delta_predictions,
            delta_scores,
            class_names=("No-Delta", "Delta"),
        )
        class_names = list(getattr(config, "class_names", []))
        by_base_class = {}
        for class_index, class_name in enumerate(class_names):
            if str(class_name).lower() == "alpha":
                continue
            class_mask = main_targets == class_index
            if not class_mask.any():
                continue
            by_base_class[class_name] = compute_binary_classification_metric_report(
                delta_targets[class_mask],
                delta_predictions[class_mask],
                delta_scores[class_mask],
                class_names=("No-Delta", "Delta"),
            )
        report["by_base_class"] = by_base_class

        if "Beta-Gamma" in class_names:
            beta_gamma_index = class_names.index("Beta-Gamma")
            base_only_predictions = (main_targets == beta_gamma_index).astype(int)
            report["base_class_only_control"] = compute_binary_classification_metric_report(
                delta_targets,
                base_only_predictions,
                base_only_predictions.astype(float),
                class_names=("No-Delta", "Delta"),
            )
        return report

    def get_delta_prediction_data(self) -> dict[str, np.ndarray] | None:
        """Return aligned per-sample outputs for conditional and routed evaluation."""
        if not self.test_delta_all_targets or not self.test_delta_all_logits:
            return None
        main_targets = np.asarray(self.test_targets, dtype=int)
        main_predictions = np.asarray(self.test_predictions, dtype=int)
        delta_targets = np.asarray(self.test_delta_all_targets, dtype=int)
        delta_scores = torch.sigmoid(torch.cat(self.test_delta_all_logits, dim=0)).numpy()
        delta_predictions = (delta_scores >= 0.5).astype(int)
        if not (len(main_targets) == len(main_predictions) == len(delta_targets) == len(delta_scores)):
            raise ValueError("Collected main and delta test outputs are misaligned.")

        class_names = list(getattr(config, "class_names", []))
        if len(class_names) != self.num_classes:
            raise ValueError("Configured class names do not match the main classifier outputs.")
        base_target_names = np.asarray([class_names[index] for index in main_targets], dtype=object)
        base_prediction_names = np.asarray([class_names[index] for index in main_predictions], dtype=object)
        alpha_index = class_names.index("Alpha") if "Alpha" in class_names else -1
        eligible = main_targets != alpha_index

        def routed_labels(base_names: np.ndarray, delta_values: np.ndarray) -> np.ndarray:
            return np.asarray(
                [
                    base_name if base_name == "Alpha" or not bool(delta_value) else f"{base_name}-Delta"
                    for base_name, delta_value in zip(base_names, delta_values, strict=True)
                ],
                dtype=object,
            )

        return {
            "base_targets": main_targets,
            "base_predictions": main_predictions,
            "base_target_names": base_target_names,
            "base_prediction_names": base_prediction_names,
            "delta_targets": delta_targets,
            "delta_scores": delta_scores,
            "delta_predictions": delta_predictions,
            "eligible_ground_truth": eligible,
            "routed_true_labels": routed_labels(base_target_names, delta_targets),
            "routed_predicted_labels": routed_labels(base_prediction_names, delta_predictions),
        }

    def get_delta_routed_metric_report(self) -> dict | None:
        """Evaluate the complete label emitted after predicted-base routing."""
        prediction_data = self.get_delta_prediction_data()
        if prediction_data is None:
            return None
        taxonomy_names = ["Alpha", "Beta", "Beta-Delta", "Beta-Gamma", "Beta-Gamma-Delta"]
        label_to_index = {name: index for index, name in enumerate(taxonomy_names)}
        true_indices = [label_to_index[label] for label in prediction_data["routed_true_labels"]]
        predicted_indices = [label_to_index[label] for label in prediction_data["routed_predicted_labels"]]
        return compute_classification_metric_report(
            true_indices,
            predicted_indices,
            labels=list(range(len(taxonomy_names))),
            class_names=taxonomy_names,
        )

    def get_classification_metric_report(self, class_names=None) -> dict | None:
        """Compute paper-facing metrics from collected test predictions."""
        if not self.test_predictions or not self.test_targets:
            return None

        if class_names is None or len(class_names) != self.num_classes:
            class_names = [f"Class_{i}" for i in range(self.num_classes)]

        return compute_classification_metric_report(
            self.test_targets,
            self.test_predictions,
            labels=list(range(self.num_classes)),
            class_names=class_names,
        )

    def get_confusion_matrix_and_classification_report(self, class_names=None):
        """Compute confusion matrix and classification report from collected test predictions."""
        if not self.test_predictions or not self.test_targets:
            return None, None

        if class_names is None or len(class_names) != self.num_classes:
            class_names = [f"Class_{i}" for i in range(self.num_classes)]

        metric_report = self.get_classification_metric_report(class_names)
        if metric_report is None:
            return None, None

        cm = np.array(metric_report["confusion_matrix"])
        class_report = metric_report["classification_report"]

        return cm, class_report

    def _update_top_misclassified_samples(self, images, labels, preds, logits, batch_idx: int) -> None:
        """Track top-confidence misclassifications during the existing test pass using a heap for efficiency."""
        misclassified_mask = preds != labels
        if not misclassified_mask.any():
            return

        probabilities = torch.softmax(logits.detach(), dim=1)
        wrong_confidences = probabilities[misclassified_mask].max(dim=1)[0]
        misclassified_images = images[misclassified_mask].detach().cpu()
        misclassified_true = labels[misclassified_mask].detach().cpu()
        misclassified_pred = preds[misclassified_mask].detach().cpu()

        max_samples = max(1, int(getattr(config, "MISCLASSIFIED_SAMPLES_TO_LOG", 10)))

        for i in range(len(misclassified_images)):
            sample = {
                "image": misclassified_images[i],
                "true_label": misclassified_true[i].item(),
                "pred_label": misclassified_pred[i].item(),
                "confidence": wrong_confidences[i].detach().cpu().item(),
                "batch_idx": batch_idx,
                "sample_idx": i,
            }
            heap_item = (sample["confidence"], self._test_misclassified_counter, sample)
            self._test_misclassified_counter += 1

            # Use heap to maintain only top-K samples without sorting every batch
            if len(self.test_misclassified_samples) < max_samples:
                heapq.heappush(self.test_misclassified_samples, heap_item)
            elif sample["confidence"] > self.test_misclassified_samples[0][0]:
                heapq.heapreplace(self.test_misclassified_samples, heap_item)

    def get_top_misclassified_samples(self, num_samples: int = 10) -> list[dict]:
        """Return cached misclassified samples sorted by confidence descending."""
        if not self.test_misclassified_samples:
            return []

        samples = [sample for _, _, sample in self.test_misclassified_samples]
        samples.sort(key=lambda x: x["confidence"], reverse=True)
        return samples[:num_samples]

    def reset_test_collections(self):
        """Reset the test predictions and targets collections."""
        self.test_predictions = []
        self.test_targets = []
        self.test_logits = []
        self.test_delta_predictions = []
        self.test_delta_targets = []
        self.test_delta_logits = []
        self.test_delta_main_targets = []
        self.test_delta_all_targets = []
        self.test_delta_all_logits = []
        self.test_misclassified_samples = []
        self._test_misclassified_counter = 0

    def configure_optimizers(self):
        optimizer_name = str(getattr(config, "OPTIMIZER", "adam")).lower()
        optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
        optimizer = optimizer_cls(self.parameters(), lr=self.learning_rate, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.LR_SCHEDULER_MODE,
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": config.LR_SCHEDULER_MONITOR,
            },
        }
