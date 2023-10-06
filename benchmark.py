import math
import os.path
import typing

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

from backend.models import Analyzer


MAIN_PATH = __file__[:-13]  # убираем \main.py


layers = ["zoom_16", "zoom_17", "zoom_18"]
rescaling = tf.keras.layers.Rescaling(1. / 255.)


def _load_image(image_path: str, mask=False):
    img = tf.image.decode_jpeg(
        tf.io.read_file(image_path)
    )

    if mask:
        return img.numpy()[:, :, 0].reshape((256, 256, 1))
    else:
        return img.numpy()


class Benchmark:
    _metrics_computed = False
    _images_data_init = False
    _images_data = {}

    def __init__(
        self,
        test_metrics: tf.keras.metrics.Metric | typing.Iterable[tf.keras.metrics.Metric],
        images_path: str,
        masks_path: str,
        save_for_visualising: bool = False
    ):
        """
        Класс позволяющий тестировать модель на заранее размеченных изображениях разного масштаба
        :param test_metrics: метрики тестов
        :param save_for_visualising: сохранять ли результаты тестов в виде изображений для последующей визуализации (их можно будет получить с помощью метода get_image_results)
        """
        if not isinstance(test_metrics, (tuple, list)):
            self.metrics = [test_metrics]
        else:
            self.metrics = test_metrics

        self.save_for_visualising = save_for_visualising
        self.images_paths = images_path
        self.masks_paths = masks_path

    def compute(self, test_model: tf.keras.models.Model) -> dict:
        """
        Возвращает результаты тестов по каждой метрике в виде словаря следующего вида:

        {
            "zoom_16_metric1": float,

            "zoom_17_metric1": float,

            "zoom_18_metric1": float,

            "mean_metric1": float,

            "zoom_16_metric2": float,

            ...
        }

        *zoom_16, zoom_17, zoom_18 - разные масштабы, mean - среднее для всех масштабов*

        :param test_model: тестируемая модель
        """
        result = {}

        for metric in self.metrics:
            result[f"mean_{metric.name}"] = 0.

        for zoom in layers:
            images_paths = [os.path.join(self.images_paths, zoom, img) for img in os.listdir(os.path.join(self.images_paths, zoom))]
            masks_paths = [os.path.join(self.masks_paths, zoom, msk) for msk in os.listdir(os.path.join(self.masks_paths, zoom))]

            images_batch = rescaling([_load_image(p) for p in images_paths])
            masks_batch = rescaling([_load_image(p, mask=True) for p in masks_paths])

            predicted_masks_batch = test_model.predict(
                images_batch,
                verbose=None
            )

            if self.save_for_visualising:
                if not self._images_data_init:
                    self._images_data["original_images"] = tf.cast(images_batch * 255, tf.uint8)
                    self._images_data["true_masks"] = masks_batch
                    self._images_data["pred_masks"] = predicted_masks_batch
                    self._images_data_init = True
                else:
                    self._images_data["original_images"] = tf.concat([self._images_data["original_images"], tf.cast(images_batch * 255, tf.uint8)], 0)
                    self._images_data["true_masks"] = tf.concat([self._images_data["true_masks"], masks_batch], 0)
                    self._images_data["pred_masks"] = tf.concat([self._images_data["pred_masks"], predicted_masks_batch], 0)

            for metric in self.metrics:
                metric.reset_state()
                metric.update_state(masks_batch, predicted_masks_batch)

                layer_value = metric.result()

                result[f"{zoom}_{metric.name}"] = layer_value.numpy()
                result[f"mean_{metric.name}"] += layer_value.numpy()

        for metric in self.metrics:
            result[f"mean_{metric.name}"] = result[f"mean_{metric.name}"] / 3

        self._metrics_computed = True

        return result

    def get_images_data(self) -> dict | None:
        """
        Возвращает изображения полученные во время последнего теста в виде словаря следующего вида:

        {
            "original_images": tf.Tensor: shape=(None, 256, 256, 3), dtype=uint8, - оригинальные изображения

            "true_masks": tf.Tensor: shape=(None, 256, 256, 1), dtype=float32, - размеченные в ручную маски

            "pred_masks": tf.Tensor: shape=(None, 256, 256, 1), dtype=float32 - полученные в результате работы модели маски
        }

        *Для использования метода необходимо хоть раз применить метод Benchmark.compute()*
        """
        if self._metrics_computed and self.save_for_visualising:
            return self._images_data
        else:
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-models', '-model', action='append', help='Модели для анализа, на данный момент доступны ' + str(Analyzer.get_registered_models_names()) + '. Пример ввода: -model U-Net -model Unet++.')
    parser.add_argument('-metrics', '-metric', action='append', help='Метрики для оценки моделей, полный список метрик доступен в документации Tensotflow: https://www.tensorflow.org/api_docs/python/tf/keras/metrics. Пример ввода: -metric BinaryIoU -metric BinaryCrossentropy.')
    parser.add_argument('--benchmark_data_path', '--benchmark_data_path', default='benchmark_data', help='Путь до директории с данными для бенчмаркинга.')
    parser.add_argument('--weights-destination', '--weights-destination', default='remote', help="Расположение весов для моделей. 'local' - искать веса в локальной директории, 'remote' - брать веса из удаленной Google Drive директории.")
    parser.add_argument('--weights_dir', '--weights_dir_path', default='12yzyc54hrUGNYQLBVfejX38OdQBSnQLk', help="Путь до локальной директории с .h5 файлами весов если выбрано weights-destination - 'local', id удаленной Google Drive директории с .h5 файлами весов если - 'remote'.")
    parser.add_argument('--google_drive_credentials_path', '--google_drive_credentials_path', default='./backend/google-api-credits.json', help="Путь до файла с Google Drive credentials с api key для доступа к удаленной директории если weights-destination - 'remote'.")
    args = parser.parse_args()

    if args.models:
        models_names = [''.join(i) for i in args.models]
    else:
        models_names = Analyzer.get_registered_models_names()

    if args.metrics:
        metrics = [''.join(i) for i in args.metrics]
    else:
        metrics = ['BinaryIoU']

    if args.weights_destination == 'local':
        models_kwargs = {
            "weights_dir_path": os.path.join(MAIN_PATH, args.weights_dir)
        }
    elif args.weights_destination == 'remote':
        models_kwargs = {
            "weights_dir_id": args.weights_dir,
            "google_drive_credentials_path": os.path.join(MAIN_PATH, args.google_drive_credentials_path)
        }
    else:
        models_kwargs = {}

    analyzer = Analyzer(
        selected_models=models_names,
        weights_destination=args.weights_destination,
        **models_kwargs
    )

    models = [analyzer[i] for i in models_names]
    metrics = [tf.keras.metrics.get(i) for i in metrics]

    benchmark = Benchmark(
        test_metrics=metrics,
        images_path=os.path.join(args.benchmark_data_path, 'images'),
        masks_path=os.path.join(args.benchmark_data_path, 'masks')
    )

    width = 0.75 / len(models)
    xticks = None

    for i, model in enumerate(models):
        results = benchmark.compute(test_model=model.get_tf_model())

        if xticks is None:
            xticks = list(results.keys())

        plt.bar(np.arange(len(results)) + width*i, list(results.values()), width=width, label=model._client_name)

    plt.style.use('ggplot')
    plt.xticks(np.arange(len(xticks)) + (len(models)/2 - 0.5)*width, xticks, rotation=10)
    plt.title('Models comparison')
    plt.legend()
    plt.show()
