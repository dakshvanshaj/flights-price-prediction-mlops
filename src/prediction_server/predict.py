import pandas as pd
import logging

logger = logging.getLogger(__name__)


def preprocessing_model_input(data: dict):
    df = pd.DataFrame(data)
