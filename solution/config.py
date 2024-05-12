from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent
DATASET_DIR = ROOT_DIR / "dataset"

BATCH_SIZE = 8
EPOCHS = 3000
LEARNING_RATE = 0.02

VAL_EPOCH_EACH_STEP = 300
