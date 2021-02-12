import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from housingmodel.processing.preprocessors import DropUnecessaryFeatures


def test_DropUnecessaryFeatures_should_drop_column():
    df = pd.DataFrame({"column1": [1, 2, 2, np.nan], "column2": [12, 12, np.nan, 1]})
    output = pd.DataFrame({"column2": [12, 12, np.nan, 1]})
    expected = DropUnecessaryFeatures(variables_to_drop=["column1"]).fit_transform(X=df)
    assert_frame_equal(output, expected)