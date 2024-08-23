from dotenv import load_dotenv
from pyprojroot import here
import yaml
import os

load_dotenv()


class LoadConfig:
    def __init__(self) -> None:
        with open(here("app/configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Load data directory
        self.inference_folder = app_config["directories"]["inference_folder"]

        # Flask endpoint
        self.fastapi_endpoint = app_config["serve"]["fastapi_endpoint"]

        # Creat directory
        self.create_directory(self.inference_folder)

    @staticmethod
    def create_directory(directory: str):
        """
        Creates directory if it doesn't exist
        :param directory: inference folder
        """
        os.makedirs(directory, exist_ok=True)
