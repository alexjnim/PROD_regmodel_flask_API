import json
import pandas as pd
from housingmodel.predict import make_prediction
from housingmodel.config import config
from housingmodel.processing.data_management import load_dataset
from housing_api.resources.validation import validate_inputs

# test_data = load_dataset(file_name=config.TESTING_DATA_FILE)
test_data = pd.read_csv("housing_api/data/test_for_validating.csv")
json_data = test_data[0:3].to_json(orient="records")


json_data = [
    {
        "longitude": -122.23,
        "latitude": 37.89,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 32.0,
        "households": 126.0,
        "median_income": 8.3252,
        "median_house_value": 400.0,
        "ocean_proximity": "NEAR BAY",
    },
    {
        "longitude": -123.22,
        "latitude": 37.86,
        "housing_median_age": 21.0,
        "total_rooms": 7099.0,
        "total_bedrooms": 1106.0,
        "population": 2401.0,
        "households": 1138.0,
        "median_income": 8.3014,
        "median_house_value": 21.02,
        "ocean_proximity": "NEAR BAY",
    },
    {
        "longitude": -122.24,
        "latitude": 37.85,
        "housing_median_age": 52.0,
        "total_rooms": 1467.0,
        "total_bedrooms": 190.0,
        "population": "ssdds",
        "households": 177.0,
        "median_income": 7.2574,
        "median_house_value": 352100.0,
        "ocean_proximity": 23,
    },
]

# print(json_data)

# data_list = json.loads(json_data)
data_list = json_data


def get_dataframe(input_data):
    if type(input_data) is pd.core.frame.DataFrame:
        return input_data
    if type(input_data) is list:
        return pd.DataFrame(input_data)


validated_input, err = validate_inputs(json_data)


data_df = get_dataframe(data_list)

results = make_prediction(input_data=data_list)

print(validated_input)

print(err)
