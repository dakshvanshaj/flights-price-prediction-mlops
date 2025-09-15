import pandas as pd
from fastapi import FastAPI
from prediction_server.schemas import InputSchema, OutputSchema
from prediction_server.model_loader import (
    load_preprocessing_artifacts,
    load_production_model,
)
from prediction_server.predict import preprocessing_for_prediction
from prediction_server.config_app import MODEL_NAME, MODEL_VERSION_ALIAS, MODEL_FLAVOR
import logging

logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/")
def index():
    return {"greeting": "Welcome to the Flights Price Prediction API"}


# prediction endpoint


@app.post("/prediction", response_model=OutputSchema)
def predict(input: InputSchema):
    input_df = pd.DataFrame([input.model_dump()])

    preprocessed_df = preprocessing_for_prediction(
        input_df, load_preprocessing_artifacts()
    )

    model = load_production_model(
        model_name=MODEL_NAME,
        model_version_alias=MODEL_VERSION_ALIAS,
        model_flavor=MODEL_FLAVOR,
    )

    prediction = model.predict(preprocessed_df)[0]
    prediction = round(prediction, 2)

    return OutputSchema(predicted_price=prediction)
