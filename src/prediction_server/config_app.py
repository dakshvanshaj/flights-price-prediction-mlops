"""
Application-specific configuration for the prediction server.

This module centralizes key settings for the FastAPI application, such as the
name, version alias, and MLflow flavor of the production model to be served.
"""
# ---------------------------------------------------------------------------- #
#                        Production Model Configuration                        #
# ---------------------------------------------------------------------------- #

MODEL_NAME = "LGBMR_Champion"
MODEL_VERSION_ALIAS = "production"
MODEL_FLAVOR = "lightgbm"
