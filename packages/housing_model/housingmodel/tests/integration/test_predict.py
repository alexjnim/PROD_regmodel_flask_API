import math
import pandas as pd

from housingmodel.predict import make_prediction
from housingmodel.processing.data_management import load_dataset
from housingmodel.config import config


def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name=config.TESTING_DATA_FILE)
    single_test_input = test_data[0:1]

    # When
    expect = make_prediction(input_data=single_test_input)

    # Then
    assert expect is not None
    assert isinstance(expect.get("predictions")[0], float)
    assert math.ceil(expect.get("predictions")[0]) == 454019


def test_make_multiple_predictions():
    # Given
    test_data = load_dataset(file_name="raw/housing.csv")
    original_data_length = len(test_data)
    multiple_test_input = test_data

    # When
    expect = make_prediction(input_data=multiple_test_input)

    # Then
    assert expect is not None
    assert len(expect.get("predictions")) == 20640

    # We expect some rows to be filtered out
    # assert len(expect.get("predictions")) != original_data_length
