from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from housingmodel.processing import preprocessors as pp
from housingmodel.processing import features
from housingmodel.config import config

import logging


_logger = logging.getLogger(__name__)


price_pipe = Pipeline(
    [
        (
            "categorical_imputer",
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA),
        ),
        (
            "numerical_inputer",
            pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA),
        ),
        # (
        #     "temporal_variable",
        #     pp.TemporalVariableEstimator(
        #         variables=config.TEMPORAL_VARS, reference_variable=config.DROP_FEATURES
        #     ),
        # ),
        (
            "rare_label_encoder",
            pp.RareLabelCategoricalEncoder(tol=0.01, variables=config.CATEGORICAL_VARS),
        ),
        (
            "categorical_encoder",
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS),
        ),
        (
            "log_transformer",
            features.LogTransformer(variables=config.NUMERICALS_LOG_VARS),
        ),
        (
            "column1_divide_by_column2",
            features.AddFeature_column1_divide_by_column2(
                column1=config.NUMERATORS,
                column2=config.DENOMINATOR,
            ),
        ),
        (
            "drop_features",
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES),
        ),
        ("scaler", MinMaxScaler()),
        # ("Linear_model", Lasso(alpha=0.005, random_state=0)),
        (
            "Random_Forest_Reg",
            RandomForestRegressor(random_state=42, max_features=8, n_estimators=30),
        ),
    ]
)
