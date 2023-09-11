import os.path


_PACKAGE_PATH = os.path.join(__file__)[:-12]  # убираем \__init__.py

TMP_WEIGHTS_DIR_PATH = os.path.join(_PACKAGE_PATH, "tmp")


from .server import run_server
from .utils import remove_server_temps, Logger
