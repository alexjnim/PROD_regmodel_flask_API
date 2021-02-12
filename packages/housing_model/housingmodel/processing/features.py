import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from housingmodel.processing.errors import InvalidModelInputError
import pandas as pd

pd.set_option("display.max_columns", None)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Logarithm transformer."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X):
        X = X.copy()

        # check that the values are non-negative for log transform
        if not (X[self.variables] > 0).all().all():
            vars_ = self.variables[(X[self.variables] <= 0).any()]
            raise InvalidModelInputError(
                f"Variables contain zero or negative values, "
                f"can't apply log for vars: {vars_}"
            )

        for feature in self.variables:
            X[feature] = np.log(X[feature])

        return X


class AddFeature_column1_divide_by_column2(BaseEstimator, TransformerMixin):
    def __init__(self, column1=None, column2=None):
        if not isinstance(column1, list):
            self.column1 = [column1]
        else:
            self.column1 = column1
        self.column2 = column2

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.column1:
            new_feature_name = feature + "_per_" + self.column2[0]
            X[new_feature_name] = X[feature] / X[self.column2[0]]

        # gets rid of any nulls and fills with the mode
        X = X.replace([np.inf, -np.inf], np.nan)
        null_columns = X.columns[X.isna().any()].tolist()
        for feature in null_columns:
            X[feature] = X[feature].fillna(X[feature].mode()[0])

        return X