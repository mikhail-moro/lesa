import unittest
import warnings
import json
import os

import tensorflow as tf

from backend.server import Server
from backend.models import Analyzer


TESTS_PATH = __file__[:-9]  # убираем \tests.py
MOCK_DIR_PATH = os.path.join(TESTS_PATH, "tests_data", "mock_data")
TEST_TILE_PATH = os.path.join(TESTS_PATH, "tests_data", "test_tile.jpg")
TESTS_LOGS_PATH = os.path.join(TESTS_PATH, "tests_data", "tests_logs.txt")
REQUESTS_DIR_PATH = os.path.join(TESTS_PATH, "tests_data", "json_requests")


class AnalyzerMock(Analyzer):
    """
    Mock-класс симулирующий работу моделей, выдавая готовые тензоры вместо полноценной загрузки модели и анализа
    """
    def __init__(self):
        super().__init__([])

        unet_out = tf.saved_model.load(os.path.join(MOCK_DIR_PATH, "unet_out"))
        unet_plus_plus_out = tf.saved_model.load(os.path.join(MOCK_DIR_PATH, "unet_plus_plus_out"))
        deeplab_v3_plus_out = tf.saved_model.load(os.path.join(MOCK_DIR_PATH, "unet_plus_plus_out"))

        self._analyzers['U-Net'] = lambda _: unet_out
        self._analyzers['Unet++'] = lambda _: unet_plus_plus_out
        self._analyzers['EfficientNet-DeepLabV3+'] = lambda _: deeplab_v3_plus_out


class ServerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        analyzer = AnalyzerMock()

        app = Server(
            import_name=__name__,
            analyzer=analyzer,
            logs_file_path=TESTS_LOGS_PATH
        )

        # В ходе тестов по неизвестной причине могут возникать предупреждения ResourceWarning, на результаты тестов это
        # не влияет
        warnings.filterwarnings("ignore", category=ResourceWarning)

        app.config['TESTING'] = True
        app.config['CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_correct_request(self):
        with open(os.path.join(REQUESTS_DIR_PATH, "correct_request.json")) as request:
            response = self.client.post(
                "/analyze",
                json=json.load(request),
                content_type='application/json'
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['success'], True)
        self.assertEqual(response.json['message'], None)

    def test_incorrect_coords_request(self):
        with open(os.path.join(REQUESTS_DIR_PATH, "incorrect_coords_request.json")) as request:
            response = self.client.post(
                "/analyze",
                json=json.load(request),
                content_type='application/json'
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['success'], False)
        self.assertEqual(response.json['message'], "Client Error: неправильные координаты в запросе, попробуйте выделить область ещё раз")

    def test_incorrect_model_request(self):
        with open(os.path.join(REQUESTS_DIR_PATH, "incorrect_model_request.json")) as request:
            response = self.client.post(
                "/analyze",
                json=json.load(request),
                content_type='application/json'
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['success'], False)
        self.assertEqual(response.json['message'], "Client Error: данная модель недоступна")


class ModelsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tile = tf.io.read_file(TEST_TILE_PATH)
        tile = tf.image.decode_jpeg(tile)
        tile = tf.keras.layers.Rescaling(1./255.)(tile)
        tile = tf.reshape(tile, (1, 256, 256, 3))

        self.input_tensor = tile
        self.analyzer = Analyzer()

    def test_unet(self):
        model = self.analyzer['U-Net']
        out_tensor = model(self.input_tensor)

        self.assertEqual(tuple(out_tensor.shape), (1, 256, 256, 1))

    def test_unet_plus_plus(self):
        model = self.analyzer['Unet++']
        out_tensor = model(self.input_tensor)

        self.assertEqual(tuple(out_tensor.shape), (1, 256, 256, 1))

    def test_resnet_deeplab_v3_plus(self):
        model = self.analyzer['ResNet-DeepLabV3+']
        out_tensor = model(self.input_tensor)

        self.assertEqual(tuple(out_tensor.shape), (1, 256, 256, 1))

    def test_effnet_deeplab_v3_plus(self):
        model = self.analyzer['EfficientNet-DeepLabV3+']
        out_tensor = model(self.input_tensor)

        self.assertEqual(tuple(out_tensor.shape), (1, 256, 256, 1))


if __name__ == "__main__":
    unittest.main()
