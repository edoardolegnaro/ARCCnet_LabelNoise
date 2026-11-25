import os

from arccnet_labelnoise.labels import CLASS_NAMES_DICT, LABEL_MAPPING_DICT


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Classes & Data Type
classes = "a-b-bg"  # Set with --classes; mappings in labels.py
class_names = CLASS_NAMES_DICT.get(classes, ["Unknown"])
label_mapping = LABEL_MAPPING_DICT[classes]
NUM_CLASSES = len(class_names)
DATA_TYPE = "both"  # Options: "magnetogram", "continuum", "both"

MODEL_NAME = "swin_tiny_patch4_window7_224.ms_in1k"
LEAKY_RELU_NEGATIVE_SLOPE = 0.01
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1e-4
OPTIMIZER = os.getenv("ARCCNET_OPTIMIZER", "adamw")
LOSS_POLICY = "sqrt_balanced_ce"
FOCAL_LOSS_GAMMA = 2.0
DELTA_AUX_WEIGHT = float(os.getenv("ARCCNET_DELTA_AUX_WEIGHT", "0.0"))
DELTA_AUX_MASK_ALPHA = _env_flag("ARCCNET_DELTA_AUX_MASK_ALPHA", default=False)
DELTA_POS_WEIGHT_POWER = float(os.getenv("ARCCNET_DELTA_POS_WEIGHT_POWER", "1.0"))
GRADIENT_CLIP_VAL = float(os.getenv("ARCCNET_GRADIENT_CLIP_VAL", "1.0"))
AUTO_LR_FIND = _env_flag("ARCCNET_AUTO_LR_FIND", default=True)
LR_FIND_MIN_LR = float(os.getenv("ARCCNET_LR_FIND_MIN_LR", "1e-7"))
LR_FIND_MAX_LR = float(os.getenv("ARCCNET_LR_FIND_MAX_LR", "1e-3"))
LR_FIND_NUM_TRAINING = int(os.getenv("ARCCNET_LR_FIND_NUM_TRAINING", "100"))
TRANSFORMER_RGB_ADAPTER = _env_flag("ARCCNET_TRANSFORMER_RGB_ADAPTER", default=True)
TRANSFORMER_IMAGENET_NORMALIZE = _env_flag("ARCCNET_TRANSFORMER_IMAGENET_NORMALIZE", default=True)

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_EPOCHS = 120
N_FOLDS = 8
RANDOM_STATE = 42

# Hardware
ACCELERATOR = "gpu"  # Options: "gpu", "cpu", "auto"
DEVICES = 1
NUM_WORKERS = 16

# Image Preprocessing
IMAGE_TARGET_HEIGHT = 112
IMAGE_TARGET_WIDTH = 224
IMAGE_TARGET_SIZE = (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH)
EXPECTED_HMI_CUTOUT_SHAPE = (400, 800)
EXPECTED_MDI_CUTOUT_SHAPE = (100, 200)
REQUIRE_COMPLETE_FOV = True
SPATIAL_POLICY_TAG = "complete-fov-112x224"
# Model-input resize policy. "keep-ratio" (default, used in the paper) resizes the
# 2:1 cutout to a 2:1 target; "stretch" resizes to any target and changes the
# geometric aspect ratio; "pad" keeps the aspect ratio, fits the cutout inside
# the target, and zero-pads the remainder.
RESIZE_MODE = "keep-ratio"
RESIZE_MODES = ("keep-ratio", "stretch", "pad")
# Restrict training and evaluation to observations whose preferred cutout is
# HMI or MDI. Fold assignments are not changed. "all" is the paper default.
INSTRUMENT_FILTER = "all"
INSTRUMENT_FILTERS = ("all", "hmi", "mdi")
IMAGE_DIVISOR = 800.0
HARDTANH_MIN_VAL = -1.0
HARDTANH_MAX_VAL = 1.0

# Data Augmentation (only applied to training data)
USE_AUGMENTATION = True
ROTATION_DEGREES = 20
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
PERSPECTIVE_DISTORTION_SCALE = 0.05
PERSPECTIVE_PROB = 0.25
AFFINE_TRANSLATE = (0.03, 0.03)
AFFINE_SCALE = (0.98, 1.03)
AFFINE_SHEAR = 3

EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MONITOR = "val_f1"
EARLY_STOPPING_MODE = "max"
CHECKPOINT_MONITOR = "val_f1"
CHECKPOINT_MODE = "max"
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MODE = "max"
LR_SCHEDULER_MONITOR = "val_f1"

DATALOADER_PREFETCH_FACTOR = 2
DATALOADER_PIN_MEMORY = True
DATALOADER_PERSISTENT_WORKERS = True
DATALOADER_MULTIPROCESSING_CONTEXT = "spawn"

LOG_EVERY_N_STEPS = 10
MISCLASSIFIED_SAMPLES_TO_LOG = 10
ENABLE_MODEL_SUMMARY = False
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
AUDIT_OUTPUT_DIR = os.path.join(LOG_DIR, "audit")
SAVE_DATASET_AUDIT = True
PROJECT_NAME = f"arcaff-v20250805-{classes}"
ENABLE_COMET = _env_flag("ARCCNET_ENABLE_COMET", default=False)
ENABLE_TENSORBOARD = True
ENABLE_CSV = True
TENSORBOARD_LOG_HYPERPARAMS = True

# Dataset Paths & Processing
DATA_FOLDER = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
DATASET_FOLDER = os.getenv("ARCAFF_DATASET_FOLDER", "arccnet-ar-classification-v20251016")
DF_FILE_NAME = os.getenv("ARCAFF_DF_FILE_NAME", "region_classification.parq")
DATASET_CACHE_TAG = DATASET_FOLDER.strip(os.sep).replace(os.sep, "_").replace("/", "_")
PROCESSED_DATASET_FILENAME = (
    f"processed_dataset_{DATASET_CACHE_TAG}_{classes}_{SPATIAL_POLICY_TAG}_{N_FOLDS}-splits_rs-{RANDOM_STATE}.parquet"
)
LONG_LIMIT_DEG = 65
UNDERSAMPLE = False
