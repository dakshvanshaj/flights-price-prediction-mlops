from shared.config.core_paths import GOLD_DATA_DIR, LOGS_DIR, MODELS_DIR

# =============================================================================
# --- GENERAL FILE PATHS & LOGGING ---
# =============================================================================
GOLD_PROCESSED_DIR = GOLD_DATA_DIR / "processed"
GOLD_QUARANTINE_DIR = GOLD_DATA_DIR / "quarantine"
GOLD_PIPELINE_LOGS_PATH = LOGS_DIR / "gold_pipeline.log"


# =============================================================================
# --- STAGE 2: DATA CLEANING ---
# =============================================================================
# List of columns to remove at the start of the gold pipeline
GOLD_DROP_COLS = ["travel_code", "user_code", "date", "day_of_year", "week_of_year"]

# The target variable for the model
TARGET_COLUMN = "price"


# =============================================================================
# --- STAGE 3: IMPUTATION ---
# =============================================================================
# Path to save the fitted imputer object
SIMPLE_IMPUTER_PATH = MODELS_DIR / "simple_imputer.json"

# Strategy for filling missing values in specified columns
IMPUTER_STRATEGY = {
    "median": ["price", "time", "distance"],
    "mode": ["agency", "flight_type"],
    "constant": {"from_location": "Unknown", "to_location": "Unknown"},
}


# =============================================================================
# --- STAGE 4: FEATURE ENGINEERING ---
# =============================================================================
# Configuration for creating cyclical features (e.g., month_sin, month_cos)
CYCLICAL_FEATURES_MAP = {
    "month": 12,
    "day_of_week": 7,
    "day": 31,
}

# Configuration for creating new features by combining categorical columns
INTERACTION_FEATURES_CONFIG = {
    "route": ["from_location", "to_location"],
    "agency_flight_type": ["agency", "flight_type"],
    "route_agency": ["from_location", "to_location", "agency"],
}


# =============================================================================
# --- STAGE 5: RARE CATEGORY GROUPING ---
# =============================================================================
# Path to save the fitted grouper object
RARE_CATEGORY_GROUPER_PATH = MODELS_DIR / "rare_category_grouper.joblib"

# List of high-cardinality columns to apply the grouping strategy to
HIGH_CARDINALITY_COLS = ["route", "route_agency"]

# Threshold for grouping. Categories appearing in less than this fraction
# of the data will be grouped into a single 'Other' category.
CARDINALITY_THRESHOLD = 0.01


# =============================================================================
# --- STAGE 6: ENCODING ---
# =============================================================================
# Path to save the fitted encoder object
CATEGORICAL_ENCODER_PATH = MODELS_DIR / "categorical_encoder.joblib"

# Configuration for the final encoding of all categorical features
ENCODING_CONFIG = {
    # Columns to be one-hot encoded
    "onehot_cols": [
        "from_location",
        "to_location",
        "agency",
        "route",
        "agency_flight_type",
        "route_agency",
    ],
    # Columns to be ordinally encoded (order matters)
    "ordinal_cols": ["flight_type"],
    # The specific order for the ordinal columns
    "ordinal_mapping": {"flight_type": ["economic", "premium", "firstClass"]},
}

# =============================================================================
# --- STAGE 7: OUTLIER HANDLING ---
# =============================================================================
OUTLIER_DETECTION_STRATEGY = "isolation_forest"
OUTLIER_HANDLING_STRATEGY = "trim"
OUTLIER_HANDLER_COLUMNS = ["price", "time", "distance"]
OUTLIER_HANDLER_PATH = MODELS_DIR / "outlier_handler.joblib"
ISO_FOREST_CONTAMINATION = 0.01


# =============================================================================
# --- STAGE 7: Power Transformations ---
# =============================================================================
POWER_TRANSFORMER_STRATEGY = "yeo-johnson"
POWER_TRANSFORMER_PATH = MODELS_DIR / "power_transformer.joblib"
POWER_TRANSFORMER_COLUMNS = ["price", "time", "distance"]

# =============================================================================
# --- STAGE 8: Scaling ---
# =============================================================================
SCALER_COLUMNS = ["price", "time", "distance"]
SCALER_PATH = MODELS_DIR / "scaler.joblib"
SCALER_STRATEGY = "standard"
