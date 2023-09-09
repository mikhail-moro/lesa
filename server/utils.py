import os

from . import TMP_WEIGHTS_DIR_PATH


def remove_server_temps():
    for file in os.listdir(TMP_WEIGHTS_DIR_PATH):
        os.remove(os.path.join(TMP_WEIGHTS_DIR_PATH, file))
