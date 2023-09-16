import argparse
import os.path
import tensorflow

from . import BENCHMARK_IMAGES_DIR_PATH, BENCHMARK_MASKS_DIR_PATH


unet_archs = [
    'unet',
    'u-net'
]

unet_plus_plus_archs = [
    'unet++',
    'u-net++',
    'unet-plus-plus',
    'u-net-plus-plus',
    'unet_plus_plus',
    'u-net_plus-plus'
]

deeplab_v3_plus_archs = [
    'deeplab',
    'deeplabv3+',
    'deeplab_v3+',
    'deeplab-v3+',
    'deeplab_v3_plus',
    'deeplab-v3-plus'
]

models_archs = [
    *unet_archs,
    *unet_plus_plus_archs,
    *deeplab_v3_plus_archs
]

layers = ["zoom_16", "zoom_17", "zoom_18"]
rescaling = tensorflow.keras.layers.Rescaling(1. / 255.)


def _load_image(image_path: str, mask=False):
    img = tensorflow.image.decode_jpeg(
        tensorflow.io.read_file(image_path)
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
        test_metrics: tensorflow.keras.metrics.Metric | list[tensorflow.keras.metrics.Metric] | tuple[tensorflow.keras.metrics.Metric] = tensorflow.metrics.BinaryIoU(),
        save_for_visualising: bool = False,
        images_path: str = None,
        masks_path: str = None
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

    def compute(self, test_model: tensorflow.keras.models.Model) -> dict:
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
                    self._images_data["original_images"] = tensorflow.cast(images_batch * 255, tensorflow.uint8)
                    self._images_data["true_masks"] = masks_batch
                    self._images_data["pred_masks"] = predicted_masks_batch
                    self._images_data_init = True
                else:
                    self._images_data["original_images"] = tensorflow.concat([self._images_data["original_images"], tensorflow.cast(images_batch * 255, tensorflow.uint8)], 0)
                    self._images_data["true_masks"] = tensorflow.concat([self._images_data["true_masks"], masks_batch], 0)
                    self._images_data["pred_masks"] = tensorflow.concat([self._images_data["pred_masks"], predicted_masks_batch], 0)

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
    parser.add_argument("model_arch", type=str)
    parser.add_argument("weights_path", type=str)

    args = parser.parse_args()
    arch = str(args.model_arch).lower()
    weights_path = args.weights_path

    # добавить проверки аргументов

    benchmark = Benchmark()

    if arch in unet_archs:
        from lesa.models.models import build_unet

        # try
        print(benchmark.compute(build_unet(weights_path=weights_path)))
    elif arch in unet_plus_plus_archs:
        from lesa.models.models import build_unet_plus_plus

        print(benchmark.compute(build_unet_plus_plus(weights_path=weights_path)))
    elif arch in deeplab_v3_plus_archs:
        from lesa.models.models import build_deeplab_v3_plus

        print(benchmark.compute(build_deeplab_v3_plus(weights_path=weights_path)))
    else:
        raise ValueError()  #
