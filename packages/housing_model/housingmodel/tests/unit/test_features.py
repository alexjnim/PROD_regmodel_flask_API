import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from packages.housing_model.housingmodel.processing.features import (
    AddFeature_column1_divide_by_column2,
)


def test_AddFeature_column1_divide_by_column2_should_create_new_columns():
    df = pd.DataFrame(
        {"column1": [12, 20, 23, 25], "column2": [2, 4, 1, 5], "column3": [1, 2, 1, 5]}
    )
    output = pd.DataFrame(
        {
            "column1": [12, 20, 23, 25],
            "column2": [2, 4, 1, 5],
            "column3": [1, 2, 1, 5],
            "column1_per_column3": [12.0, 10.0, 23.0, 5.0],
            "column2_per_column3": [2.0, 2.0, 1.0, 1.0],
        }
    )
    expected = AddFeature_column1_divide_by_column2(
        column1=["column1", "column2"], column2=["column3"]
    ).fit_transform(X=df)
    assert_frame_equal(output, expected)