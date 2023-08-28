import os

import tensorflow as tf
from lesa.models.utils import get_local_weights_path, get_remote_weights_path


class EncoderBlock:
    def __init__(
        self,
        num_filters,
        use_pooling=True
    ):

        self.use_pooling = use_pooling

        self.conv_1 = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False
        )
        self.activation_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False
        )
        self.activation_2 = tf.keras.layers.ReLU()

        self.batch_norm = tf.keras.layers.BatchNormalization()

        if use_pooling:
            self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
            self.dropout = tf.keras.layers.Dropout(0.1)

    def __call__(self, inputs):
        conv = self.conv_1(inputs)
        conv = self.activation_1(conv)

        conv = self.conv_2(conv)
        conv = self.activation_2(conv)

        if self.use_pooling:
            pool = self.pooling(conv)
            pool = self.dropout(pool)
        else:
            pool = None

        return conv, pool


class DecoderBlock:
    def __init__(self, num_filters, concatenate_with=None):
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
        self.dropout = tf.keras.layers.Dropout(0.1)

        self.conv_1 = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False
        )
        self.activation_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False
        )
        self.activation_2 = tf.keras.layers.ReLU()

        self.batch_norm = tf.keras.layers.BatchNormalization()

    def __call__(self, inputs):
        unfl = self.conv_transpose(inputs)
        unfl = self.weights_concat([unfl, *self.concatenate_with])
        unfl = self.dropout(unfl)

        conv = self.conv_1(unfl)
        conv = self.activation_1(conv)

        conv = self.conv_2(conv)
        conv = self.activation_2(conv)

        conv = self.batch_norm(conv)

        return conv


class UnetEncoderBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            num_filters,
            use_pooling=True
    ):
        super().__init__()

        self.num_filters = num_filters
        self.use_pooling = use_pooling

    # def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2D(
            self.num_filters,
            (3, 3),
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False
        )
        self.activation_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(
            self.num_filters,
            (3, 3),
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False
        )
        self.activation_2 = tf.keras.layers.ReLU()

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(0.1)

        # super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        conv = self.conv_1(inputs)
        conv = self.activation_1(conv)

        conv = self.conv_2(conv)
        conv = self.activation_2(conv)

        if self.use_pooling:
            pool = self.pooling(conv)
            pool = self.dropout(pool)
        else:
            pool = None

        return conv, pool


class UnetDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, concatenate_with=None):
        super().__init__()

        self.num_filters = num_filters
        self.concatenate_with = concatenate_with if concatenate_with else []

    # def build(self, input_shape):
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
            self.num_filters,
            (3, 3),
            padding='same',
            strides=(2, 2),
            kernel_initializer='he_normal',
            use_bias=False
        )
        self.weights_concat = tf.keras.layers.Concatenate()
        self.dropout = tf.keras.layers.Dropout(0.1)

        self.conv_1 = tf.keras.layers.Conv2D(
            self.num_filters,
            (3, 3),
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False
        )
        self.activation_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(
            self.num_filters,
            (3, 3),
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False
        )
        self.activation_2 = tf.keras.layers.ReLU()

        self.batch_norm = tf.keras.layers.BatchNormalization()

        # super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        unfl = self.conv_transpose(inputs)
        unfl = self.weights_concat([unfl, *self.concatenate_with])
        unfl = self.dropout(unfl)

        conv = self.conv_1(unfl)
        conv = self.activation_1(conv)

        conv = self.conv_2(conv)
        conv = self.activation_2(conv)

        conv = self.batch_norm(conv)

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


class DSPP(tf.keras.layers.Layer):
    _not_build_up_sampling = True

    def _build_up_sampling(self, conv):
        self.up_sampling = tf.keras.layers.UpSampling2D(
            size=(
                self._build_input_shape[-3] // conv.shape[1], self._build_input_shape[-2] // conv.shape[2]),
            interpolation="bilinear"
        )

        self._not_build_up_sampling = False

    def build(self, input_shape):
        self.av_pooling = tf.keras.layers.AveragePooling2D(pool_size=(input_shape[-3], input_shape[-2]))
        self.start_conv = DeepLabConvBlock(kernel_size=1, use_bias=True)
        self.up_sampling = None  # будет инициализирован когда будет известен output_shape self.start_conv

        self.conv_1 = DeepLabConvBlock(kernel_size=1, dilation_rate=1)
        self.conv_6 = DeepLabConvBlock(kernel_size=3, dilation_rate=6)
        self.conv_12 = DeepLabConvBlock(kernel_size=3, dilation_rate=12)
        self.conv_18 = DeepLabConvBlock(kernel_size=3, dilation_rate=18)

        self.weights_concat = tf.keras.layers.Concatenate(axis=-1)
        self.final_conv = DeepLabConvBlock(kernel_size=1)

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


def build_unet(
        input_shape: tuple[int, int, int] = (256, 256, 3),
        weights_path: str = None
) -> tf.keras.models.Model:
    """
    Возвращает не скомпилированную модель с архитектурой U-Net

    Можно написать путь к весам в ручную или использовать ключевые слова:
        - 'auto_local' - автоматически найти путь к последним локально сохраненным весам,
        - 'auto_remote' - автоматически найти путь к последним удаленно сохраненным весам,
        - None - создать пустую модель (для обучения с нуля)

    :param input_shape: размер входного изображения
    :param weights_path: путь к сохраненным весам модели
    """
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

    model = tf.keras.models.Model(inputs=[model_input], outputs=[model_output], name='Unet')

    if weights_path:
        if weights_path == "auto_local":
            weights_path = get_local_weights_path(model.name)

            model.load_weights(weights_path)
        elif weights_path == "auto_remote":
            weights_path = get_remote_weights_path(model.name)

            model.load_weights(weights_path)
            os.remove(weights_path)
        else:
            model.load_weights(weights_path)

    return model


def build_unet_plus_plus(
        input_shape: tuple[int, int, int] = (256, 256, 3),
        weights_path: str = None
) -> tf.keras.models.Model:
    """
    Возвращает не скомпилированную модель с архитектурой U-Net++

    Можно написать путь к весам в ручную или использовать ключевые слова:
        - 'auto_local' - автоматически найти путь к последним локально сохраненным весам,
        - 'auto_remote' - автоматически найти путь к последним удаленно сохраненным весам,
        - None - создать пустую модель (для обучения с нуля)

    :param input_shape: размер входного изображения
    :param weights_path: путь к сохраненным весам модели
    """
    model_input = tf.keras.layers.Input(input_shape)

    conv_0_0, pool_1 = UnetEncoderBlock(16)(model_input)  # 256x256 -> 128x128

    conv_1_0, pool_2 = UnetEncoderBlock(32)(pool_1)  # 128x128 -> 64x64
    conv_0_1 = UnetDecoderBlock(16, concatenate_with=[conv_0_0])(conv_1_0)

    conv_2_0, pool_3 = UnetEncoderBlock(64)(pool_2)  # 64x64 -> 32x32
    conv_1_1 = UnetDecoderBlock(32, concatenate_with=[conv_1_0])(conv_2_0)
    conv_0_2 = UnetDecoderBlock(16, concatenate_with=[conv_0_0, conv_0_1])(conv_1_1)

    conv_3_0, pool_4 = UnetEncoderBlock(128)(pool_3)  # 32x32 -> 16x16
    conv_2_1 = UnetDecoderBlock(64, concatenate_with=[conv_2_0])(conv_3_0)
    conv_1_2 = UnetDecoderBlock(32, concatenate_with=[conv_1_0, conv_1_1])(conv_2_1)
    conv_0_3 = UnetDecoderBlock(16, concatenate_with=[conv_0_0, conv_0_1, conv_0_2])(conv_1_2)

    conv_4_0, _ = UnetEncoderBlock(256, use_pooling=False)(pool_4)

    conv_3_1 = UnetDecoderBlock(128, concatenate_with=[conv_3_0])(conv_4_0)  # 16x16 -> 32x32
    conv_2_2 = UnetDecoderBlock(64, concatenate_with=[conv_2_0, conv_2_1])(conv_3_1)  # 32x32 -> 64x64
    conv_1_3 = UnetDecoderBlock(32, concatenate_with=[conv_1_0, conv_1_1, conv_1_2])(conv_2_2)  # 64x64 -> 128x128
    conv_0_4 = UnetDecoderBlock(16, concatenate_with=[conv_0_0, conv_0_1, conv_0_2, conv_0_3])(
        conv_1_3)  # 128x128 -> 256x256

    model_output = tf.keras.layers.Conv2D(1, (1, 1))(conv_0_4)
    model_output = tf.keras.layers.Activation('sigmoid')(model_output)

    model = tf.keras.models.Model(inputs=[model_input], outputs=[model_output], name='Unet_plus_plus')

    if weights_path:
        if weights_path == "auto_local":
            weights_path = get_local_weights_path(model.name)

            model.load_weights(weights_path)
        elif weights_path == "auto_remote":
            weights_path = get_remote_weights_path(model.name)

            model.load_weights(weights_path)
            os.remove(weights_path)
        else:
            model.load_weights(weights_path)

    return model


def build_deeplab_v3_plus(
        input_shape: tuple[int, int, int] = (256, 256, 3),
        weights_path: str = None
) -> tf.keras.models.Model:
    """
    Возвращает не скомпилированную модель с архитектурой DeepLabV3+

    *Данная реализация DeepLabV3+ взята из публичного репозитория:*

    https://github.com/keras-team/keras-io/blob/master/examples/vision/ipynb/deeplabv3_plus.ipynb

    Можно написать путь к весам в ручную или использовать ключевые слова:
        - 'auto_local' - автоматически найти путь к последним локально сохраненным весам,
        - 'auto_remote' - автоматически найти путь к последним удаленно сохраненным весам,
        - None - создать пустую модель (для обучения с нуля)

    :param input_shape: размер входного изображения
    :param weights_path: путь к сохраненным весам модели
    """
    model_input = tf.keras.layers.Input(input_shape)

    resnet50 = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=model_input,
        input_shape=input_shape
    )

    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DSPP()(x)

    input_a = tf.keras.layers.UpSampling2D(
        size=(input_shape[0] // 4 // x.shape[1], input_shape[1] // 4 // x.shape[2]),
        interpolation="bilinear"
    )(x)

    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = DeepLabConvBlock(num_filters=48, kernel_size=1)(input_b)

    x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
    x = DeepLabConvBlock()(x)
    x = DeepLabConvBlock()(x)
    x = tf.keras.layers.UpSampling2D(
        size=(input_shape[0] // x.shape[1], input_shape[1] // x.shape[2]),
        interpolation="bilinear"
    )(x)

    model_output = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), padding="same")(x)
    model_output = tf.keras.layers.Activation('sigmoid')(model_output)

    model = tf.keras.models.Model(inputs=[model_input], outputs=[model_output], name='DeepLabV3_plus')

    if weights_path:
        if weights_path == "auto_local":
            weights_path = get_local_weights_path(model.name)

            model.load_weights(weights_path)
        elif weights_path == "auto_remote":
            weights_path = get_remote_weights_path(model.name)

            model.load_weights(weights_path)
            os.remove(weights_path)
        else:
            model.load_weights(weights_path)

    return model
