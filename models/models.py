import abc
import os
import sys

import tensorflow as tf

from abc import ABC
from typing import Literal


# Данный скрипт может быть запущен как из файлов main.py и tests.py (стандартным способом), так и отдельно импортирован,
# например для обучения в Google Colab, в таком случае использовать относительные импорты не получиться
SCRIPT_RUN_SEPARATE = sys.path[0] == __file__[:-10]

if not SCRIPT_RUN_SEPARATE:
    from .utils import get_local_weights_path, get_remote_weights_path


class EncoderBlock:
    def __init__(
        self,
        num_filters,
        use_pooling=True,
        dropout=0.1
    ):
        self.use_pooling = use_pooling

        self.conv_1 = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False
        )
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.activation_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False
        )
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.ReLU()

        if use_pooling:
            self.pooling = tf.keras.layers.MaxPooling2D((2, 2))

        self.dropout = tf.keras.layers.Dropout(dropout)

    def __call__(self, inputs):
        conv = self.conv_1(inputs)
        conv = self.batch_norm_1(conv)
        conv = self.activation_1(conv)

        conv = self.conv_2(conv)
        conv = self.batch_norm_2(conv)
        conv = self.activation_2(conv)

        if self.use_pooling:
            pool = self.pooling(conv)
            pool = self.dropout(pool)
        else:
            pool = None

        return conv, pool


class DecoderBlock:
    def __init__(
        self,
        num_filters,
        concatenate_with=None,
        dropout=0.1
    ):
        self.concatenate_with = concatenate_with if concatenate_with else []

        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
            num_filters,
            (3, 3),
            padding='same',
            strides=(2, 2),
            kernel_initializer='he_normal',
            use_bias=False
        )
        self.weights_concat = tf.keras.layers.Concatenate()
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.conv_1 = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False
        )
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.activation_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False
        )
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.ReLU()

    def __call__(self, inputs):
        unfl = self.conv_transpose(inputs)
        unfl = self.weights_concat([unfl, *self.concatenate_with])
        unfl = self.dropout(unfl)

        conv = self.conv_1(unfl)
        conv = self.batch_norm_1(conv)
        conv = self.activation_1(conv)

        conv = self.conv_2(conv)
        conv = self.batch_norm_2(conv)
        conv = self.activation_2(conv)

        return conv


class DeepLabConvBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            num_filters=256,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False
    ):
        super().__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            self.num_filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            use_bias=self.use_bias,
            padding='same',
            kernel_initializer=tf.keras.initializers.HeNormal(),
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        conv = self.conv(inputs)
        conv = self.batch_norm(conv)
        conv = self.activation(conv)

        return conv


class DoubleDeepLabConvBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            num_filters=128,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
            dropout=0.1
    ):
        super().__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.dropout_value = dropout

    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2D(
            self.num_filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            use_bias=self.use_bias,
            padding='same',
            kernel_initializer=tf.keras.initializers.HeNormal(),
        )
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.activation_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(
            self.num_filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            use_bias=self.use_bias,
            padding='same',
            kernel_initializer=tf.keras.initializers.HeNormal(),
        )
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.ReLU()

        self.dropout = tf.keras.layers.Dropout(self.dropout_value)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        conv = self.conv_1(inputs)
        conv = self.batch_norm_1(conv)
        conv = self.activation_1(conv)

        conv = self.conv_2(conv)
        conv = self.batch_norm_2(conv)
        conv = self.activation_2(conv)

        conv = self.dropout(conv)

        return conv


class DSPP(tf.keras.layers.Layer):
    _not_build_up_sampling = True

    def __init__(self, mode: str = 'original', **kwargs):
        super().__init__(**kwargs)

        if mode == "double":
            self.conv_block = DoubleDeepLabConvBlock
        else:
            self.conv_block = DeepLabConvBlock

    def _build_up_sampling(self, conv):
        self.up_sampling = tf.keras.layers.UpSampling2D(
            size=(self._build_input_shape[-3] // conv.shape[1], self._build_input_shape[-2] // conv.shape[2]),
            interpolation="bilinear"
        )

        self._not_build_up_sampling = False

    def build(self, input_shape):
        self.av_pooling = tf.keras.layers.AveragePooling2D(pool_size=(input_shape[-3], input_shape[-2]))
        self.start_conv = self.conv_block(kernel_size=1, use_bias=True)
        self.up_sampling = None  # будет инициализирован когда будет известен output_shape self.start_conv

        self.conv_1 = self.conv_block(kernel_size=1, dilation_rate=1)
        self.conv_6 = self.conv_block(kernel_size=3, dilation_rate=6)
        self.conv_12 = self.conv_block(kernel_size=3, dilation_rate=12)
        self.conv_18 = self.conv_block(kernel_size=3, dilation_rate=18)

        self.weights_concat = tf.keras.layers.Concatenate(axis=-1)
        self.final_conv = self.conv_block(kernel_size=1)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.av_pooling(inputs)
        x = self.start_conv(x)

        if self._not_build_up_sampling:
            self._build_up_sampling(x)

        out_pool = self.up_sampling(x)

        out_1 = self.conv_1(inputs)
        out_6 = self.conv_6(inputs)
        out_12 = self.conv_12(inputs)
        out_18 = self.conv_18(inputs)

        x = self.weights_concat([out_pool, out_1, out_6, out_12, out_18])
        output = self.final_conv(x)

        return output


_registered_analyzers = {}


class AnalyzeModel(ABC):
    """
    Абстрактный класс-обертка для использования tensorflow.keras моделей

    Веса для модели можно загрузить из удаленной или локальной директории:
        - Локальная загрузка:
            - weights_file: название .h5-файла или 'auto',
            - weights_dir: абсолютный путь к директории с файлами весов,
            - weights_destination: 'local',
            - google_drive_credits_path: None.
        - Удаленная загрузка:
            - weights_file: название .h5-файла или 'auto',
            - weights_dir: id Google Drive директории,
            - weights_destination: 'remote',
            - google_drive_credits_path: путь к json-файлу для доступа к удаленной Google Drive директории.
        - Пустая модель для обучения с нуля (*можно просто не задавать аргументы*):
            - weights_file: None,
            - weights_dir: None,
            - weights_destination: None,
            - google_drive_credits_path: None.

    *ключевое слово 'auto' доступно только при запуске из main.py или tests.py*

    :param input_shape: размер входного изображения
    :param weights_file: название файла с весами или 'auto' (будет использован последный созданный для этой модели файл весов)
    :param weights_dir: абсолютный путь/id к директории с сохраненными весами
    :param weights_destination: 'local' - использовать веса из локальной директории, 'remote' - использовать веса из удаленной Google Drive директории
    :param google_drive_credentials_path: путь к json-файлу для доступа к удаленной Google Drive директории
    """
    tf_model: tf.keras.models.Model = None
    client_name: str = None

    @abc.abstractmethod
    def build_model(self, input_shape) -> tf.keras.models.Model:
        """
        Метод инициализирующий модель, должен возвращять экземпляр класса tf.keras.models.Model

        Важно: *при автоматических загрузке и сохранении весов учитывается не поле client_name, а tf_model.name которое
        задается при инициализации tf.keras.models.Model(args, kwargs, name=your_model_name), поэтому этот параметр
        должен быть задан.*
        """
        raise NotImplementedError()

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (256, 256, 3),
        weights_file: Literal['auto'] | str = None,
        weights_dir: str = None,
        weights_destination: Literal['local', 'remote'] = None,
        google_drive_credentials_path: str = None
    ):
        self.tf_model = self.build_model(input_shape)

        if not (SCRIPT_RUN_SEPARATE or all((weights_file, weights_dir, weights_destination)) is None):
            if weights_file == "auto":
                if weights_destination == "local":
                    weights_path = get_local_weights_path(self.tf_model.name, weights_dir)

                    self.tf_model.load_weights(weights_path)

                if weights_destination == "remote":
                    weights_path = get_remote_weights_path(self.tf_model.name, weights_dir, google_drive_credentials_path)

                    self.tf_model.load_weights(weights_path)
                    os.remove(weights_path)  # удаляем временный файл весов после загрузки
            else:
                self.tf_model.load_weights(os.path.join(weights_dir, weights_file))

    def __call__(self, input_tensor: tf.Tensor):
        return self.tf_model.predict(input_tensor, verbose=None)

    def get_tf_model(self):
        return self.tf_model

    @staticmethod
    def register_model(model_client_name: str):
        """
        Только модели к которым был применен данный декоратор будут доступны для web-клиента.

        :param model_client_name: название модели на стороне клиента (приходит в json-файле при запросе и отображается
        в меню выбора модели для анализа)
        """
        def _registration(target):
            target.client_name = model_client_name
            _registered_analyzers[model_client_name] = target
            return target

        return _registration


@AnalyzeModel.register_model('U-Net')
class Unet(AnalyzeModel):
    def build_model(self, input_shape):
        model_input = tf.keras.layers.Input(input_shape)

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


@AnalyzeModel.register_model('Unet++')
class UnetPlusPlus(AnalyzeModel):
    def build_model(self, input_shape):
        model_input = tf.keras.layers.Input(input_shape)

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


@AnalyzeModel.register_model('DeepLabV3+')
class DeeplabV3plus(AnalyzeModel):
    """
    :param mode: режим сверстки:
        'original' - стандартный сверточный блок Conv + BN + ReLU,
        'double' - двойной сверточный блок [Conv + BN + ReLU]x2.
        *В случае двойной сверстки используется такое же количество фильтров, просто они распределенны на 2 блока*
    """
    mode = None

    def __init__(self, mode: Literal['original', 'double'] = 'original', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mode = mode

    def build_model(self, input_shape):
        if self.mode == 'double':
            conv_block = DoubleDeepLabConvBlock
        else:
            conv_block = DeepLabConvBlock

        model_input = tf.keras.layers.Input(input_shape)

        resnet50 = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=model_input,
            input_shape=input_shape
        )

        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = DSPP(mode=self.mode)(x)

        input_a = tf.keras.layers.UpSampling2D(
            size=(input_shape[0] // 4 // x.shape[1], input_shape[1] // 4 // x.shape[2]),
            interpolation="bilinear"
        )(x)

        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = conv_block(num_filters=128, kernel_size=1)(input_b)

        x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
        x = conv_block()(x)
        x = conv_block()(x)
        x = tf.keras.layers.UpSampling2D(
            size=(input_shape[0] // x.shape[1], input_shape[1] // x.shape[2]),
            interpolation="bilinear"
        )(x)

        model_output = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), padding="same")(x)
        model_output = tf.keras.layers.Activation('sigmoid')(model_output)

        return tf.keras.models.Model(inputs=[model_input], outputs=[model_output], name='DeepLabV3_plus')


# @AnalyzeModel.register_model('ResNet U-net')
class ResNetBackboneUnet(AnalyzeModel):
    def build_model(self, input_shape) -> tf.keras.models.Model:
        model_input = tf.keras.layers.Input(input_shape)
        backbone = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, input_tensor=model_input)

        features_layers = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']

        conv_1 = backbone.get_layer(features_layers[0]).output  # 256x256 -> 128xx128
        conv_2 = backbone.get_layer(features_layers[1]).output  # 128x128 -> 64x64
        conv_3 = backbone.get_layer(features_layers[2]).output  # 64x64 -> 32x32
        conv_4 = backbone.get_layer(features_layers[3]).output  # 32x32 -> 16x16

        conv_5 = DecoderBlock(256)(backbone.output)  # 16x16 -> 8x8 -> 16x16

        conv_6 = DecoderBlock(128, concatenate_with=[conv_4])(conv_5)  # 16x16 -> 32x32
        conv_7 = DecoderBlock(64, concatenate_with=[conv_3])(conv_6)  # 32x32 -> 64x64
        conv_8 = DecoderBlock(32, concatenate_with=[conv_2])(conv_7)  # 64x64 -> 128x128
        conv_9 = DecoderBlock(16, concatenate_with=[conv_1])(conv_8)  # 128x128 -> 256x256

        model_output = tf.keras.layers.Conv2D(1, (1, 1))(conv_9)
        model_output = tf.keras.layers.Activation('sigmoid')(model_output)

        return tf.keras.models.Model(inputs=[model_input], outputs=[model_output], name='Unet_orig_resnet')


class Analyzer:
    """
    Класс реализующий инициализацию моделей для анализа картографических изображений и предоставляющий доступ к ним.

    Доступ к моделям возможен с помощью индекс оператора через название модели: *analyzer = manager['model_name']*

    :param selected_models: инициализируемые модели
    :param analyzers_kwargs: параметры которые будут использованны при инициализации каждой модели, *подробнее смотреть в models.models.AnalyzeModel*
    """
    _analyzers = None

    def __init__(self, selected_models: list[str], **models_kwargs):
        self._analyzers = {}

        for model_name, model in _registered_analyzers.items():
            if model_name in selected_models:
                print(f"Инициализация {model_name}...")

                self._analyzers[model_name] = model(**models_kwargs)

    def __getitem__(self, item):
        if item in self._analyzers:
            return self._analyzers[item]
        else:
            raise ValueError("Client Error: данная модель недоступна")

    def get_models_names(self):
        return list(self._analyzers.keys())
