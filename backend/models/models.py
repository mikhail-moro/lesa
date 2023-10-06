import typing
import abc
import os

import tensorflow as tf

from .utils import get_local_weights_path, get_remote_weights_path
from .layers import DSPP, DecoderBlock, EncoderBlock, ConvBlock


INPUT_SHAPE = (256, 256, 3)


def register_model(model_client_name: str):
    """
    Только модели к которым был применен данный декоратор будут доступны для web-клиента.

    :param model_client_name: название модели на стороне клиента (приходит в json-файле при запросе и отображается
    в меню выбора модели для анализа)
    """
    def _registration(target):
        target._client_name = model_client_name
        target._registered = True

        return target

    return _registration


class AnalyzeModel(abc.ABC):
    """
    Абстрактный класс-обертка для использования tensorflow.keras моделей.

    В случае простой инициализации через метод *__init__(args, kwargs)* создаст модель без обученных весов, для
    инициализации обученной модели используйте класс-методы: *from_local_weights* и *from_remote_weights*.

    - from_local_weights: загрузка весов из локального .h5 файла
    - from_remote_weights: загрузка весов из удаленного файла с помощью Google Drive Api
    """
    _tf_model: tf.keras.models.Model = None
    _client_name: str = None
    _registered: bool = False

    @abc.abstractmethod
    def build_model(self) -> tf.keras.models.Model:
        """
        Метод инициализирующий модель, должен возвращять экземпляр класса tf.keras.models.Model

        Важно: *при автоматических загрузке и сохранении весов учитывается не поле _client_name, а _tf_model.name которое
        задается при инициализации tf.keras.models.Model(args, kwargs, name=your_model_name), поэтому этот параметр
        должен быть задан.*
        """
        raise NotImplementedError()

    def __init__(self):
        self._tf_model = self.build_model()

    @classmethod
    def from_local_weights(
        cls,
        weights_dir_path: str,
        weights_file: typing.Literal['auto'] | str = 'auto'
    ):
        """
        :param weights_file: название .h5 файла с весами или 'auto' (будет использован последный созданный для этой модели файл весов)
        :param weights_dir_path: абсолютный путь к локальной директории с сохраненными весами
        """
        obj = cls()

        if weights_file == 'auto':
            weights_path = get_local_weights_path(obj._tf_model.name, weights_dir_path)
        else:
            weights_path = os.path.join(weights_dir_path, weights_file)

        obj._tf_model.load_weights(weights_path)

        return obj

    @classmethod
    def from_remote_weights(
        cls,
        weights_dir_id: str,
        google_drive_credentials_path: str,
        weights_file: typing.Literal['auto'] | str = 'auto'
    ):
        """
        :param weights_file: название .h5 файла с весами или 'auto' (будет использован последный созданный для этой модели файл весов)
        :param weights_dir_id: id к google drive директории с сохраненными весами
        :param google_drive_credentials_path: путь к json-файлу с Api-Key для доступа к удаленной Google Drive директории
        """
        obj = cls()

        if weights_file == 'auto':
            weights_path = get_remote_weights_path(obj._tf_model.name, weights_dir_id, google_drive_credentials_path)
        else:
            weights_path = get_remote_weights_path(weights_file, weights_dir_id, google_drive_credentials_path)

        obj._tf_model.load_weights(weights_path)
        os.remove(weights_path)  # удаляем временный файл весов после загрузки

        return obj

    def __call__(self, input_tensor: tf.Tensor) -> tf.Tensor:
        return self._tf_model.predict(input_tensor, verbose=None)

    def get_tf_model(self) -> tf.keras.models.Model:
        return self._tf_model


@register_model('U-Net')
class Unet(AnalyzeModel):
    def build_model(self):
        model_input = tf.keras.layers.Input(INPUT_SHAPE)

        conv_1, pool_1 = EncoderBlock(16)(model_input)  # 256x256 -> 128x128
        conv_2, pool_2 = EncoderBlock(32)(pool_1)  # 128x128 -> 64x64
        conv_3, pool_3 = EncoderBlock(64)(pool_2)  # 64x64 -> 32x32
        conv_4, pool_4 = EncoderBlock(128)(pool_3)  # 32x32 -> 16x16

        conv_5, _ = EncoderBlock(256, use_pooling=False)(pool_4)

        conv_6 = DecoderBlock(128, concatenate_with=[conv_4])(conv_5)  # 16x16 -> 32x32
        conv_7 = DecoderBlock(64, concatenate_with=[conv_3])(conv_6)  # 32x32 -> 64x64
        conv_8 = DecoderBlock(32, concatenate_with=[conv_2])(conv_7)  # 64x64 -> 128x128
        conv_9 = DecoderBlock(16, concatenate_with=[conv_1])(conv_8)  # 128x128 -> 256x256

        model_output = tf.keras.layers.Conv2D(1, (1, 1))(conv_9)
        model_output = tf.keras.layers.Activation('sigmoid')(model_output)

        return tf.keras.models.Model(inputs=[model_input], outputs=[model_output], name='Unet_orig')


@register_model('Unet++')
class UnetPlusPlus(AnalyzeModel):
    def build_model(self):
        model_input = tf.keras.layers.Input(INPUT_SHAPE)

        conv_0_0, pool_1 = EncoderBlock(16)(model_input)  # 256x256 -> 128x128

        conv_1_0, pool_2 = EncoderBlock(32)(pool_1)  # 128x128 -> 64x64
        conv_0_1 = DecoderBlock(16, concatenate_with=[conv_0_0])(conv_1_0)

        conv_2_0, pool_3 = EncoderBlock(64)(pool_2)  # 64x64 -> 32x32
        conv_1_1 = DecoderBlock(32, concatenate_with=[conv_1_0])(conv_2_0)
        conv_0_2 = DecoderBlock(16, concatenate_with=[conv_0_0, conv_0_1])(conv_1_1)

        conv_3_0, pool_4 = EncoderBlock(128)(pool_3)  # 32x32 -> 16x16
        conv_2_1 = DecoderBlock(64, concatenate_with=[conv_2_0])(conv_3_0)
        conv_1_2 = DecoderBlock(32, concatenate_with=[conv_1_0, conv_1_1])(conv_2_1)
        conv_0_3 = DecoderBlock(16, concatenate_with=[conv_0_0, conv_0_1, conv_0_2])(conv_1_2)

        conv_4_0, _ = EncoderBlock(256, use_pooling=False)(pool_4)

        conv_3_1 = DecoderBlock(128, concatenate_with=[conv_3_0])(conv_4_0)  # 16x16 -> 32x32
        conv_2_2 = DecoderBlock(64, concatenate_with=[conv_2_0, conv_2_1])(conv_3_1)  # 32x32 -> 64x64
        conv_1_3 = DecoderBlock(32, concatenate_with=[conv_1_0, conv_1_1, conv_1_2])(conv_2_2)  # 64x64 -> 128x128
        conv_0_4 = DecoderBlock(16, concatenate_with=[conv_0_0, conv_0_1, conv_0_2, conv_0_3])(conv_1_3)  # 128x128 -> 256x256

        model_output = tf.keras.layers.Conv2D(1, (1, 1))(conv_0_4)
        model_output = tf.keras.layers.Activation('sigmoid')(model_output)

        return tf.keras.models.Model(inputs=[model_input], outputs=[model_output], name='Unet_plus_plus')


@register_model('ResNet-DeepLabV3+')
class ResnetDeeplabV3plus(AnalyzeModel):
    def build_model(self):
        model_input = tf.keras.layers.Input(INPUT_SHAPE)

        resnet50 = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=model_input,
            input_shape=INPUT_SHAPE
        )

        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = DSPP()(x)

        input_a = tf.keras.layers.UpSampling2D(
            size=(INPUT_SHAPE[0] // 4 // x.shape[1], INPUT_SHAPE[1] // 4 // x.shape[2]),
            interpolation="bilinear"
        )(x)

        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = ConvBlock(num_filters=128, kernel_size=1)(input_b)

        x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
        x = ConvBlock()(x)
        x = ConvBlock()(x)
        x = tf.keras.layers.UpSampling2D(
            size=(INPUT_SHAPE[0] // x.shape[1], INPUT_SHAPE[1] // x.shape[2]),
            interpolation="bilinear"
        )(x)

        model_output = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), padding="same")(x)
        model_output = tf.keras.layers.Activation('sigmoid')(model_output)

        return tf.keras.models.Model(inputs=[model_input], outputs=[model_output], name='ResNet50_DeepLabV3_plus')


@register_model('EfficientNet-DeepLabV3+')
class EffnetDeeplabV3plus(AnalyzeModel):
    def build_model(self):
        model_input = tf.keras.layers.Input(INPUT_SHAPE)

        effnet_v2s = tf.keras.applications.EfficientNetV2S(
            weights="imagenet",
            include_top=False,
            input_tensor=model_input,
            input_shape=INPUT_SHAPE
        )

        x = effnet_v2s.get_layer("block6a_expand_activation").output
        x = DSPP()(x)

        input_a = tf.keras.layers.UpSampling2D(
            size=(INPUT_SHAPE[0] // 4 // x.shape[1], INPUT_SHAPE[1] // 4 // x.shape[2]),
            interpolation="bilinear"
        )(x)

        input_b = effnet_v2s.get_layer("block2b_expand_activation").output
        input_b = ConvBlock(num_filters=128, kernel_size=1)(input_b)

        x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
        x = ConvBlock()(x)
        x = ConvBlock()(x)
        x = tf.keras.layers.UpSampling2D(
            size=(INPUT_SHAPE[0] // x.shape[1], INPUT_SHAPE[1] // x.shape[2]),
            interpolation="bilinear"
        )(x)

        model_output = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), padding="same")(x)
        model_output = tf.keras.layers.Activation('sigmoid')(model_output)

        return tf.keras.models.Model(inputs=[model_input], outputs=[model_output], name='EfficientNetV2S_DeepLabV3_plus')


class Analyzer:
    """
    Класс реализующий инициализацию моделей для анализа картографических изображений и предоставляющий доступ к ним.

    Доступ к моделям возможен с помощью индекс оператора через название модели: *model = analyzer['model_name']*

    :param selected_models: список моделей которые будут инициализированны, None - использовать все доступные модели
    :param weights_destination: расположение файлов с весами: 'local' - локальная директория, 'remote' - удаленная GoogleDrive директория, None - не загружать веса
    :param models_kwargs: параметры которые будут использованны при инициализации каждой модели, *подробнее смотреть в models.models.AnalyzeModel*
    """
    _analyzers: dict[str, 'AnalyzeModel'] = {}

    def __init__(
        self,
        selected_models: list[str] = None,
        weights_destination: typing.Literal['local', 'remote'] = None,
        **models_kwargs
    ):
        all_models = self.get_registered_models_names()

        if selected_models is None:
            selected_models = all_models
        else:
            selected_models = [sel_model for sel_model in selected_models if sel_model in all_models]

        for model in selected_models:
            model = [i for i in AnalyzeModel.__subclasses__() if i._client_name == model][0]

            print(f"Инициализация {model._client_name}...")

            if weights_destination == 'local':
                self._analyzers[model._client_name] = model.from_local_weights(**models_kwargs)
            elif weights_destination == 'remote':
                self._analyzers[model._client_name] = model.from_remote_weights(**models_kwargs)
            else:
                self._analyzers[model._client_name] = model()

    def __getitem__(self, item) -> AnalyzeModel:
        if item in self._analyzers:
            return self._analyzers[item]
        else:
            raise ValueError("Client Error: данная модель недоступна")

    @staticmethod
    def get_registered_models_names():
        """
        Возвращает названия всех моделей к которым был применен декоратор register_model
        """
        return [model._client_name for model in AnalyzeModel.__subclasses__() if model._registered and model._client_name]

    def get_availible_models_names(self):
        """
        Возвращает названия всех моделей доступных для данного экземпляра класса
        """
        return list(self._analyzers.keys())
