from shared.config.core_paths import RAW_DATA_DIR

SPLIT_DATA_DIR = RAW_DATA_DIR / "train_validation_test"

# --- Data Splitting Config ---
RAW_CSV_PATH = RAW_DATA_DIR / "flights.csv"
TRAIN_SET_SIZE = 0.70
VAL_SET_SIZE = 0.15
# remaining is validation set size
