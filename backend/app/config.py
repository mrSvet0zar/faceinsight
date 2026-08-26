"""Central configuration: paths, task definitions, dataset constants.

Ethical constraint (CLAUDE.md #2): the UTKFace ethnicity label is deliberately
absent from every constant and loader in this project. It is parsed away at
filename level and never stored, trained on, or exposed.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

FER2013_DIR = DATA_DIR / "fer2013"
UTKFACE_DIR = DATA_DIR / "utkface"
CELEBA_DIR = DATA_DIR / "celeba"

CHECKPOINTS_DIR = BACKEND_DIR / "app" / "models" / "checkpoints"

# ---------------------------------------------------------------------------
# Image / model input
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224  # aligned face crop fed to the shared backbone

# ImageNet normalization (backbone is ImageNet-pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Emotion task (FER2013 — 7 classes)
# ---------------------------------------------------------------------------
# Order matches the FER2013 folder names (Kaggle msambare/fer2013).
EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]
# French display labels for the API (cf. CLAUDE.md response format)
EMOTION_LABELS_FR = {
    "angry": "colère",
    "disgust": "dégoût",
    "fear": "peur",
    "happy": "joie",
    "neutral": "neutre",
    "sad": "tristesse",
    "surprise": "surprise",
}

# ---------------------------------------------------------------------------
# Age / gender task (UTKFace)
# ---------------------------------------------------------------------------
# UTKFace filename format: [age]_[gender]_[race]_[date&time].jpg
# gender: 0 = male, 1 = female. The third field is NEVER read (constraint #2).
AGE_MIN = 0
AGE_MAX = 116  # max age present in UTKFace
GENDER_CLASSES = ["male", "female"]
GENDER_LABELS_FR = {"male": "homme", "female": "femme"}

# ---------------------------------------------------------------------------
# CelebA multi-label tasks
# ---------------------------------------------------------------------------
FACIAL_HAIR_ATTRS = [
    "No_Beard",
    "Goatee",
    "Mustache",
    "5_o_Clock_Shadow",
    "Sideburns",
]
HAIR_ATTRS = [
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Gray_Hair",
    "Bald",
    "Bangs",
    "Straight_Hair",
    "Wavy_Hair",
]
CELEBA_ATTRS = FACIAL_HAIR_ATTRS + HAIR_ATTRS

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
# Strict per-dataset split, no leakage between them (cf. CLAUDE.md).
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 42  # deterministic splits so train/val/test never shift between runs
