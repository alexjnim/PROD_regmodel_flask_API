import typing as t

from marshmallow import Schema, fields
from marshmallow import ValidationError
from housing_api.config import config


class InvalidInputError(Exception):
    """Invalid model input."""


SYNTAX_ERROR_FIELD_MAP = {
    "1stFlrSF": "FirstFlrSF",
    "2ndFlrSF": "SecondFlrSF",
    "3SsnPorch": "ThreeSsnPortch",
}

# Create a schema by defining a class with variables mapping attribute names to Field objects.
class HouseDataRequestSchema(Schema):
    longitude = fields.Float()
    latitude = fields.Float()
    housing_median_age = fields.Float(allow_none=True)
    total_rooms = fields.Float(allow_none=True)
    total_bedrooms = fields.Float(allow_none=True)
    population = fields.Float(allow_none=True)
    households = fields.Float()
    median_income = fields.Float(allow_none=True)
    median_house_value = fields.Float()
    ocean_proximity = fields.Str(allow_none=True)


def _filter_error_rows(errors: dict, validated_input: t.List[dict]) -> t.List[dict]:
    """Remove input data rows with errors."""

    indexes = errors.keys()
    # delete them in reverse order so that you
    # don't throw off the subsequent indexes.
    for index in sorted(indexes, reverse=True):
        del validated_input[index]

    return validated_input


def validate_inputs(input_data):
    """Check prediction inputs against schema."""

    # set many=True to allow passing in a list
    schema = HouseDataRequestSchema(many=True)

    # this will use Marshmallow to validate the data against the schema and  raises a ValidationError error when invalid data are passed in.
    errors = []
    try:
        # .load validates and deserializes an input dictionary to an application-level data pdstructure
        schema.load(input_data)
    except ValidationError as exc:
        errors = (
            exc.messages
        )  # this will give a dictionary of validation errors, e.g. {"email": ['"foo" is not a valid email address.']}

    if errors:
        validated_input = _filter_error_rows(errors=errors, validated_input=input_data)
    else:
        validated_input = input_data
    validated_input = input_data
    return validated_input, errors


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )
