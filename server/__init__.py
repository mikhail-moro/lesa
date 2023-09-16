import os.path


_PACKAGE_PATH = os.path.join(__file__)[:-12]  # убираем \__init__.py

TMP_DIR_PATH = os.path.join(_PACKAGE_PATH, "tmp")

STATIC_DIR_PATH = os.path.join(_PACKAGE_PATH, "static")
TEMPLATE_DIR_PATH = os.path.join(_PACKAGE_PATH, "templates")


from .server import Server
from .utils import remove_server_temps, Logger
from .tile_processing import TilesDownloader
