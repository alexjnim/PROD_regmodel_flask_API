import pytest

from housing_api.app import create_app
from housing_api.config.config import TestingConfig


@pytest.fixture
def app():
    app = create_app(config_object=TestingConfig)

    with app.app_context():
        yield app


# pass the app fixture into flask_test_client fixture to create flask test_client
@pytest.fixture
def flask_test_client(app):
    with app.test_client() as test_client:
        yield test_client
