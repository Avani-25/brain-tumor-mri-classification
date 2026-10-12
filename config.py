"""Central configuration shared by all notebooks and the Streamlit app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = ROOT / "data" / "Tumour"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

# Output paths
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
MODELS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Data settings
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]  # alphabetical = Keras label order
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png"}