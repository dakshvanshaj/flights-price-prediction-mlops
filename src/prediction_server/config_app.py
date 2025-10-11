"""
Application-specific configuration for the prediction server.

This module centralizes key settings for the FastAPI application, such as the
name, version alias, and MLflow flavor of the production model to be served.
"""
# ---------------------------------------------------------------------------- #
#                        Production Model Configuration                        #
# ---------------------------------------------------------------------------- #

MODEL_NAME = "LGBMR_Champion"
MODEL_VERSION_ALIAS = "champion"
MODEL_FLAVOR = "lightgbm"
MODEL_VERSION = 2
RUN_ID = "abc55ca591aa41218427f3db3f3471bb"
