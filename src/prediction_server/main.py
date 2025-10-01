"""
Main FastAPI application for the flight price prediction server.

This module sets up and runs the FastAPI application, including:
- Loading the ML model and preprocessing artifacts on startup.
- Defining the API endpoints for health checks and predictions.
- Handling prediction requests, including data validation, preprocessing,
  prediction, and postprocessing.
"""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from prediction_server.config_app import MODEL_FLAVOR, MODEL_NAME, MODEL_VERSION_ALIAS
from prediction_server.model_loader import (
    load_preprocessing_artifacts,
    load_production_model,
)
from prediction_server.predict import (
    postprocessing_for_target,
    preprocessing_for_prediction,
)
from prediction_server.schemas import InputSchema, OutputSchema
from shared.config import config_logging, config_training
from shared.utils import setup_logging_from_yaml

# --- Application Setup ---

# Configure logging from a YAML file
setup_logging_from_yaml(
    log_path=config_logging.MAIN_FAST_API_LOGS_PATH,
    default_yaml_path=config_logging.LOGGING_YAML,
    default_level=logging.INFO,
)
logger = logging.getLogger(__name__)

# A dictionary to cache the loaded model and preprocessors in memory
ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan events.
    Loads the ML model and artifacts on startup and clears them on shutdown.
    """
    logger.info("Application startup: Loading ML model and artifacts...")
    ml_artifacts["preprocessors"] = load_preprocessing_artifacts()
    ml_artifacts["model"] = load_production_model(
        model_name=MODEL_NAME,
        model_version_alias=MODEL_VERSION_ALIAS,
        model_flavor=MODEL_FLAVOR,
    )
    logger.info("Successfully loaded and cached ML artifacts.")
    yield
    logger.info("Application shutdown: Clearing ML artifacts...")
    ml_artifacts.clear()


app = FastAPI(
    title="Flight Price Prediction API",
    description="An API to predict flight prices using a pre-trained LightGBM model.",
    version="1.5",
    lifespan=lifespan,
)


# --- API Endpoints ---


@app.get("/", tags=["Monitoring"])
async def health_check():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Welcome to the Flight Price Prediction API"}


@app.post("/prediction", response_model=OutputSchema, tags=["Prediction"])
async def predict(request_body: InputSchema):
    """
    Receives flight data, preprocesses it, and returns a price prediction.
    """
    try:
        logger.info(f"Received prediction request: {request_body.model_dump()}")

        # Retrieve the cached model and preprocessors
        preprocessors = ml_artifacts["preprocessors"]
        model = ml_artifacts["model"]

        # Convert the Pydantic model to a DataFrame for processing
        input_df = pd.DataFrame([request_body.model_dump()])

        # Apply the same preprocessing steps used in training
        preprocessed_df = preprocessing_for_prediction(input_df, preprocessors)

        # Get the model's prediction (which is on a transformed scale)
        predicted_price_scaled = model.predict(preprocessed_df)[0]

        # Create a DataFrame to hold the scaled prediction for postprocessing
        prediction_df = pd.DataFrame(
            {config_training.TARGET_COLUMN: [predicted_price_scaled]}
        )

        # Inverse transform the prediction to get the final, interpretable price
        final_prediction_df = postprocessing_for_target(prediction_df, preprocessors)
        final_price = final_prediction_df[config_training.TARGET_COLUMN].iloc[0]

        logger.info(f"Prediction successful. Final price: {final_price:.2f}")
        return OutputSchema(predicted_price=round(final_price, 2))

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        # Return a generic but informative error response
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the request.",
        )
