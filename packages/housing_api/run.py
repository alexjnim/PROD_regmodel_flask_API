from housing_api.app import create_app
from housing_api.config.config import DevelopmentConfig, ProductionConfig


application = create_app(config_object=ProductionConfig)

if __name__ == "__main__":
    application.run()