import os.path


_PACKAGE_PATH = os.path.join(__file__)[:-12]  # убираем \__init__.py

TMP_WEIGHTS_DIR_PATH = os.path.join(_PACKAGE_PATH, "tmp")


from .utils import remove_models_temps
from .models import Analyzer
