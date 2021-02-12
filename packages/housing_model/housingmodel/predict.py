import numpy as np
import pandas as pd
import json
from housingmodel.config import config
from housingmodel.processing.data_management import load_pipeline
from housingmodel.processing.validation import validate_inputs
from housingmodel import __version__ as _version

import logging
import typing as t


_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """

    #makes sure that we have a pandas dataframe
    def get_dataframe(input_data):
        if type(input_data) is pd.core.frame.DataFrame:
            return input_data
        if type(input_data) is list:
            return pd.DataFrame(input_data)

    input_data = get_dataframe(input_data)
    
    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=data)

    prediction = _price_pipe.predict(data[config.FEATURES])
    prediction = _price_pipe.predict(validated_data[config.FEATURES])

    output = np.exp(prediction)

    results = {"predictions": output, "version": _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {validated_data} "
        f"Inputs: {data} "
        f"Predictions: {results}"
    )

    return results
