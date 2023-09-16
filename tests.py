import os
import unittest
import warnings


TESTS_PATH = os.path.join(__file__)[:-9]  # убираем \tests.py
MOCK_DIR_PATH = os.path.join(TESTS_PATH, "tests_data", "mock_data")
TESTS_LOGS_PATH = os.path.join(TESTS_PATH, "tests_data", "tests_logs.txt")
REQUESTS_DIR_PATH = os.path.join(TESTS_PATH, "tests_data", "json_requests")


import json
import tensorflow as tf
from server import Server
from models import Analyzer


class AnalyzerMock(Analyzer):
    """
    Mock-класс симулирующий работу моделей, выдавая готовые тензоры вместо полноценной загрузки модели и анализа
    """
    def __init__(self):
        super().__init__([])

        unet_out = tf.saved_model.load(os.path.join(MOCK_DIR_PATH, "unet_out"))
        unet_plus_plus_out = tf.saved_model.load(os.path.join(MOCK_DIR_PATH, "unet_plus_plus_out"))
        deeplab_v3_plus_out = tf.saved_model.load(os.path.join(MOCK_DIR_PATH, "unet_plus_plus_out"))

        self._analyzers['unet'] = lambda _: unet_out
        self._analyzers['unet_plus_plus'] = lambda _: unet_plus_plus_out
        self._analyzers['deeplab_v3_plus'] = lambda _: deeplab_v3_plus_out


analyzer = AnalyzerMock()

app = Server(
    import_name=__name__,
    analyzer=analyzer,
    logs_file_path=TESTS_LOGS_PATH,
    tiles_download_max_replies=5
)


class ServerTestCase(unittest.TestCase):
    def setUp(self) -> None:
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


# TODO тесты для модуля models

if __name__ == "__main__":
    unittest.main()
