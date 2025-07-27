from shared.config.core_paths import GOLD_DATA_DIR, LOGS_DIR, PROJECT_ROOT

# --- Gold Data Flow ---
GOLD_PROCESSED_DIR = GOLD_DATA_DIR / "processed"
GOLD_QUARANTINE_DIR = GOLD_DATA_DIR / "quarantine"

# --- Gold Logging ---
GOLD_PIPELINE_LOGS_PATH = LOGS_DIR / "gold_pipeline.log"

# --- Gold Pipeline Config ---
GOLD_DROP_COLS = ["travel_code", "user_code", "date", "day_of_year", "week_of_year"]
TARGET_COLUMN = "price"
IMPUTER_STRATEGY = {
    "median": ["price", "time", "distance"],
    "mode": ["agency", "flight_type"],
    "constant": {"from_location": "Unknown", "to_location": "Unknown"},
}

MODELS_DIR = PROJECT_ROOT / "models"

# Path to save the new imputer object

SIMPLE_IMPUTER_PATH = MODELS_DIR / "simple_imputer.json"

# Path to save the new encoder object
CATEGORICAL_ENCODER_PATH = MODELS_DIR / "categorical_encoder.joblib"

# Configuration dictionary for the encoder.
# Customize this to match the categorical columns in your dataset.
ENCODING_CONFIG = {
    # Columns to convert into 0s and 1s.
    "onehot_cols": ["from_location", "to_location", "agency"],
    # Columns to convert into ordered numbers (e.g., 0, 1, 2).
    "ordinal_cols": ["flight_type"],
    # The specific order for the ordinal columns. This is required!
    "ordinal_mapping": {"flight_type": ["economic", "premium", "firstClass"]},
}
