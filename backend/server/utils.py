import os
import datetime

from typing import Literal
from . import TMP_DIR_PATH


class Logger:
    _logs_file_path = None

    _formatted_string = """
{log_type} log {datetime}:
    Request: 
        {request}
    Traceback:
        {traceback}

    """

    def __init__(self, logs_file_path: str):
        self.logs_file_path = logs_file_path

        if not os.path.isfile(logs_file_path):
            open(logs_file_path, 'w').close()

    def log(self, log_type: Literal['Exception', 'Warning'] | str, request, traceback):
        with open(self.logs_file_path, 'a') as logs_file:
            logs_file.write(
                self._formatted_string.format(
                    log_type=log_type,
                    datetime=str(datetime.datetime.now()),
                    request=request,
                    traceback=traceback
                )
            )


def remove_server_temps():
    for file in os.listdir(TMP_DIR_PATH):
        os.remove(os.path.join(TMP_DIR_PATH, file))
