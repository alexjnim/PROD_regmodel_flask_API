import pathlib
import housingmodel.config

import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(housingmodel.config.__file__).resolve().parent.parent
DATASET_DIR = PACKAGE_ROOT / "data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "models"


# data
TRAINING_DATA_FILE = "raw/housing.csv"
TESTING_DATA_FILE = "raw/housing.csv"
# variables

TARGET = "median_house_value"

FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "median_house_value",
    "ocean_proximity",
]

DROP_NULLS_FROM_FEATURES = [
    "total_bedrooms",
]

# this variable is to calculate the temporal variable,
# can be dropped afterwards
DROP_FEATURES = None

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "median_income",
]

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = [
    "ocean_proximity",
]

# TEMPORAL_VARS = "YearRemodAdd"

# these are for the column1_divide_by_column2 transformer
NUMERATORS = [
    "total_rooms",
    "total_bedrooms",
    "population",
]
DENOMINATOR = [
    "households",
]

# variables to log transform
NUMERICALS_LOG_VARS = [
    "households",
    "housing_median_age",
    "median_income",
    "total_bedrooms",
    "total_rooms",
]

# categorical variables to encode
CATEGORICAL_VARS = [
    "ocean_proximity",
]

NUMERICAL_NA_NOT_ALLOWED = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CATEGORICAL_VARS if feature not in CATEGORICAL_VARS_WITH_NA
]

PIPELINE_NAME = "regression"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05
