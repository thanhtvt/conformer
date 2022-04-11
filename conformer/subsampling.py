import tensorflow as tf
from typing import Optional


L2 = tf.keras.regularizers.l2(1e-6)


class ConvSubsampling(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        dropout_rate: float = 0.0,
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = L2,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = L2,
        name: str = "ConvSubsampling",
        **kwargs
    ):
        super(ConvSubsampling, self).__init__(name=name, **kwargs)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            padding='same',
            name=f'{name}_conv1'
        )
        self.do1 = tf.keras.layers.Dropout(rate=dropout_rate, name=f'{name}_do1')
        self.relu1 = tf.keras.layers.ReLU(name=f'{name}_relu1')
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            padding='same',
            name=f'{name}_conv2'
        )
        self.do2 = tf.keras.layers.Dropout(rate=dropout_rate, name=f'{name}_do2')
        self.relu2 = tf.keras.layers.ReLU(name=f'{name}_relu2')
        
    def summary(self):
        pass
    
    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv1(inputs, training=training)
        outputs = self.do1(outputs, training=training)
        outputs = self.relu1(outputs)
        outputs = self.conv2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.relu2(outputs)

        return outputs
