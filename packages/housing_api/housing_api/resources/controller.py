from flask import Blueprint, request, jsonify
from housingmodel.predict import make_prediction
from housingmodel import __version__ as _version
import os
from werkzeug.utils import secure_filename

from housing_api.config.config import get_logger, UPLOAD_FOLDER
from housing_api.resources.validation import validate_inputs, allowed_file
from housing_api import __version__ as api_version

_logger = get_logger(logger_name=__name__)

import numpy as np

prediction_app = Blueprint("prediction_app", __name__)


@prediction_app.route("/health", methods=["GET"])
def health():
    if request.method == "GET":
        _logger.info("health status OK")
        return "ok"


@prediction_app.route("/version", methods=["GET"])
def version():
    if request.method == "GET":
        return jsonify({"model_version": _version, "api_version": api_version})


@prediction_app.route("/v1/predict/regression", methods=["POST"])
def predict():
    if request.method == "POST":
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        _logger.debug(f"Inputs: {json_data}")

        # Step 2: Validate the input using marshmallow schema
        input_data, errors = validate_inputs(input_data=json_data)

        # Step 3: Model prediction
        result = make_prediction(input_data=input_data)
        _logger.debug(f"Outputs: {result}")

        # Step 4: Convert numpy ndarray to list
        predictions = result.get("predictions").tolist()
        version = result.get("version")

        # Step 5: Return the response as JSON
        return jsonify(
            {"predictions": predictions, "version": version, "errors": errors}
        )
