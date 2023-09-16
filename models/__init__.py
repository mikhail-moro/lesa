import os.path


_PACKAGE_PATH = os.path.join(__file__)[:-12]  # убираем \__init__.py

WEIGHTS_DIR_PATH = os.path.join(_PACKAGE_PATH, "weights")
TMP_WEIGHTS_DIR_PATH = os.path.join(_PACKAGE_PATH, "tmp")

BENCHMARK_IMAGES_DIR_PATH = os.path.join(_PACKAGE_PATH, "benchmark_data", "tiles")
BENCHMARK_MASKS_DIR_PATH = os.path.join(_PACKAGE_PATH, "benchmark_data", "masks")


from .utils import remove_models_temps
from .models import Analyzer, AnalyzeModel
