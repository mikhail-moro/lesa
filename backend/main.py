import typing

if typing.TYPE_CHECKING:
    from flask import Flask

import atexit
import json
import os

from server import remove_server_temps, Server
from models import remove_models_temps, Analyzer


MAIN_PATH = __file__[:-8]  # убираем \main.py


def create_flask_app() -> 'Flask':
    if "CFG_PATH" in os.environ:
        config_path = os.environ["CFG_PATH"]
    else:
        config_path = "./config.json"

    with open(os.path.join(MAIN_PATH, config_path), 'r') as cfg_file:
        config = json.load(cfg_file)

    # TODO добавить проверку аргументов

    if config["weights_destination"] == "local":
        params = config["local_weights"]

        models_kwargs = {
            "weights_dir_path": os.path.join(MAIN_PATH, params["weights_dir_path"])
        }
    elif config["weights_destination"] == "remote":
        params = config["remote_weights"]

        models_kwargs = {
            "weights_dir_id": params["weights_dir_id"],
            "google_drive_credentials_path": os.path.join(MAIN_PATH, params["google_drive_credentials_path"])
        }
    else:
        models_kwargs = {}

    analyzer = Analyzer(
        selected_models=config["models"],
        weights_destination=config["weights_destination"],
        **models_kwargs
    )

    flask_app = Server(
        import_name=__name__,
        analyzer=analyzer,
        logs_file_path=os.path.join(MAIN_PATH, config["logs_path"])
    )

    # удаление временных файлов при завершении скрипта
    atexit.register(lambda: [func() for func in [remove_server_temps, remove_models_temps]])

    return flask_app


app = create_flask_app()

if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
    # "models": ["U-Net", "Unet++", "ResNet-DeepLabV3+", "EfficientNet-DeepLabV3+"],
