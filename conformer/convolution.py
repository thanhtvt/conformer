import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    def __init__(self,
                 name: str = "GLU",
                 **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        mat1, mat2 = tf.split(inputs, 2, axis=-1)
        mat2 = tf.nn.sigmoid(mat2)

        return tf.math.multiply(mat1, mat2)


class ConvolutionModule(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int,
                 expansion_factor: int = 2,
                 kernel_size: int = 3,
                 dropout_rate: float = 0.4,
                 name: str = "ConvolutionModule",
                 **kwargs):
        super(ConvolutionModule, self).__init__(name=name, **kwargs)

        self.conv_module = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(filters=filters * expansion_factor,      # Pointwise Conv
                                   kernel_size=1),
            GLU(),
            tf.keras.layers.Conv1D(filters=filters,                         # 1D Depthwise Conv
                                   kernel_size=kernel_size,
                                   padding='same',
                                   groups=filters),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.silu),
            tf.keras.layers.Conv1D(filters=filters,                         # Pointwise Conv
                                   kernel_size=1),
            tf.keras.layers.Dropout(rate=dropout_rate)
        ])
        self.add = tf.keras.layers.Add()

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv_module(inputs, training=training)
        outputs = self.add([inputs, outputs])
        
        return outputs
