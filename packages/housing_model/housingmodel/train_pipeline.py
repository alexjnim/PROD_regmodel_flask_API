import numpy as np
from sklearn.model_selection import train_test_split

import housingmodel.pipeline as pipeline
from housingmodel.processing.data_management import load_dataset
from housingmodel.config import config
from housingmodel import __version__ as _version
from housingmodel.processing.data_management import load_dataset, save_pipeline


import logging

_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    )

    # # transform the target
    y_train = np.log(y_train)

    pipeline.price_pipe.fit(X_train[config.FEATURES], y_train)

    # _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.price_pipe)

    y_pred = pipeline.price_pipe.predict(X_test)

    from sklearn.metrics import mean_squared_error

    print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))


if __name__ == "__main__":
    run_training()
