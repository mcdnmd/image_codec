from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
WEIGHTS_DIR = ROOT_DIR / "weights"

BATCH_SIZE = 64
EPOCHS = 3000
LEARNING_RATE = 0.003

VAL_EPOCH_EACH_STEP = 300


WANDB_PROJECT_NAME = "image_codec"
