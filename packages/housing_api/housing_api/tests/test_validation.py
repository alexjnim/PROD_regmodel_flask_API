import json
import pytest
import numpy as np
from housingmodel.config import config as model_config
from housingmodel.processing.data_management import load_dataset


def test_prediction_endpoint_validation_200(flask_test_client):
    # Given
    # Load the test data from the housingmodel package.
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    test_data = load_dataset(file_name=model_config.TESTING_DATA_FILE)
    post_json = test_data.to_json(orient="records")

    # When
    response = flask_test_client.post(
        "/v1/predict/regression", json=json.loads(post_json)
    )

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)

    # Check correct number of errors removed
    assert (
        len(response_json.get("predictions")) + len(response_json.get("errors"))
        == 20640
    )


# parameterizationa allows us to try many combinations of data
# within the same test, see the pytest docs for details:
# https://docs.pytest.org/en/latest/parametrize.html
@pytest.mark.parametrize(
    "field, field_value, index, expected_error",
    (
        (
            "longitude",
            "hello",  # expected float
            0,
            {"0": {"longitude": ["Not a valid number."]}},
        ),
        (
            "ocean_proximity",
            12.0,  # expected string
            45,
            {"45": {"ocean_proximity": ["Not a valid string."]}},
        ),
        (
            "households",
            np.nan,  # nan not allowed
            34,  # expected string
            {"34": {"households": ["Field may not be null."]}},
        ),
        (
            "total_bedrooms",
            "",  # expected float
            2,
            {"2": {"total_bedrooms": ["Not a valid number."]}},
        ),
    ),
)
def test_prediction_validation(
    field, field_value, index, expected_error, flask_test_client
):
    # Given
    # In this test, inputs are changed to incorrect values to check the validation.
    test_data = load_dataset(file_name=model_config.TESTING_DATA_FILE)
    test_data.loc[index, field] = field_value
    post_json = test_data.to_json(orient="records")

    # When
    response = flask_test_client.post(
        "/v1/predict/regression", json=json.loads(post_json)
    )

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json["errors"] == expected_error