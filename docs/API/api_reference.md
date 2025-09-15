# API Reference

This document provides a detailed reference for the FastAPI prediction server.

-   **Source Code:** `src/prediction_server/`

## 1. Overview

The prediction server is a high-performance API built with FastAPI that serves the champion LightGBM model. Its primary purpose is to provide real-time flight price predictions based on input features.

The server is designed with production best practices in mind:
-   **Startup & Shutdown Logic**: It loads the model and all necessary preprocessing artifacts into memory on startup to ensure low-latency predictions and gracefully releases them on shutdown.
-   **Data Validation**: It uses Pydantic schemas to enforce a strict contract for all incoming requests and outgoing responses, preventing invalid data from causing errors.
-   **Centralized Configuration**: Key settings, like the production model name, are managed in a dedicated configuration module.

## 2. Running the API Server

To run the server locally, navigate to the project's root directory and use `uvicorn`:

If you have installed the project using pyproject.toml then

```bash
uvicorn prediction_server.main:app --reload 
```

Once running, the interactive API documentation (provided by Swagger UI) will be available at:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 3. API Endpoints

### Health Check

-   **Endpoint:** `GET /`
-   **Description:** A simple health check endpoint to confirm that the API is running and responsive.
-   **Response:**
    ```json
    {
      "status": "ok",
      "message": "Welcome to the Flight Price Prediction API"
    }
    ```

### Prediction

-   **Endpoint:** `POST /prediction`
-   **Description:** The core endpoint that takes flight details, preprocesses them, runs the prediction, and returns the final estimated price.

#### Request Body

The request body must be a JSON object conforming to the following structure. The API will automatically validate the data types and constraints (e.g., `time` and `distance` must be greater than 0).

```json
{
  "from_location": "Recife (PE)",
  "to_location": "Florianopolis (SC)",
  "flight_type": "firstClass",
  "time": 1.76,
  "distance": 676.53,
  "agency": "FlyingDrops",
  "date": "2019-09-26"
}
```

#### Response Body

A successful request will return a JSON object with the predicted price.

```json
{
  "predicted_price": 1434.35
}
```

## 4. Configuration

The server's behavior is controlled by the `src/prediction_server/config_app.py` module. This file specifies which model to load from the MLflow Model Registry.

-   `MODEL_NAME`: The registered name of the model (e.g., `LGBMR_Champion`).
-   `MODEL_VERSION_ALIAS`: The alias of the model version to use (e.g., `production`).
-   `MODEL_FLAVOR`: The MLflow flavor required to load the model (e.g., `lightgbm`).

This setup allows for easy updates to the production model without changing the API's source code.
