import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_filters: int = 256,
        kernel_size: int | tuple[int, int] = 3,
        dilation_rate: int = 1,
        dropout: float = .0,
        use_bias: bool = False,
    ):
        super().__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout = dropout
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

        if self.dropout > 0.:
            self.use_dropout = True
            self.drop = tf.keras.layers.Dropout(self.dropout)
        else:
            self.use_dropout = False

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        conv = self.conv(inputs)
        conv = self.batch_norm(conv)
        conv = self.activation(conv)

        if self.use_dropout:
            conv = self.drop(conv)

        return conv


class EncoderBlock:
    def __init__(
        self,
        num_filters,
        use_pooling=True,
        dropout=0.1
    ):
        self.use_pooling = use_pooling

        self.conv_1 = ConvBlock(
            num_filters=num_filters,
            kernel_size=(3, 3),
            use_bias=False
        )

        self.conv_2 = ConvBlock(
            num_filters=num_filters,
            kernel_size=(3, 3),
            use_bias=False
        )

        if use_pooling:
            self.pooling = tf.keras.layers.MaxPooling2D((2, 2))

        self.dropout = tf.keras.layers.Dropout(dropout)

    def __call__(self, inputs):
        conv = self.conv_1(inputs)
        conv = self.conv_2(conv)

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
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.activation_1 = tf.keras.layers.ReLU()

        self.weights_concat = tf.keras.layers.Concatenate()

        self.conv = ConvBlock(
            num_filters=num_filters,
            kernel_size=(3, 3),
            use_bias=False
        )

        self.dropout = tf.keras.layers.Dropout(dropout)

    def __call__(self, inputs):
        conv = self.conv_transpose(inputs)
        conv = self.batch_norm_1(conv)
        conv = self.activation_1(conv)
        conv = self.weights_concat([conv, *self.concatenate_with])
        conv = self.conv(conv)
        conv = self.dropout(conv)

        return conv


class DSPP(tf.keras.layers.Layer):
    _not_build_up_sampling = True

    def _build_up_sampling(self, conv):
        self.up_sampling = tf.keras.layers.UpSampling2D(
            size=(self._build_input_shape[-3] // conv.shape[1], self._build_input_shape[-2] // conv.shape[2]),
            interpolation="bilinear"
        )

        self._not_build_up_sampling = False

    def build(self, input_shape):
        self.av_pooling = tf.keras.layers.AveragePooling2D(pool_size=(input_shape[-3], input_shape[-2]))
        self.start_conv = ConvBlock(kernel_size=1, use_bias=True)
        self.up_sampling = None  # будет инициализирован когда будет известен output_shape self.start_conv

        self.conv_1 = ConvBlock(kernel_size=1, dilation_rate=1)
        self.conv_6 = ConvBlock(kernel_size=3, dilation_rate=6)
        self.conv_12 = ConvBlock(kernel_size=3, dilation_rate=12)
        self.conv_18 = ConvBlock(kernel_size=3, dilation_rate=18)

        self.weights_concat = tf.keras.layers.Concatenate(axis=-1)
        self.final_conv = ConvBlock(kernel_size=1, dropout=0.1)

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
