from shared.config.core_paths import GOLD_DATA_DIR, LOGS_DIR, PROJECT_ROOT

# --- Gold Data Flow ---
GOLD_PROCESSED_DIR = GOLD_DATA_DIR / "processed"
GOLD_QUARANTINE_DIR = GOLD_DATA_DIR / "quarantine"

# --- Gold Logging ---
GOLD_PIPELINE_LOGS_PATH = LOGS_DIR / "gold_pipeline.log"

# --- Gold Pipeline Config ---
GOLD_DROP_COLS = ["travel_code", "user_code"]
TARGET_COLUMN = "price"
IMPUTER_STRATEGY = {
    "median": ["price", "time", "distance"],
    "mode": ["agency", "flight_type"],
    "constant": {"from_location": "Unknown", "to_location": "Unknown"},
}
SIMPLE_IMPUTER_PATH = PROJECT_ROOT / "models" / "simple_imputer.json"
