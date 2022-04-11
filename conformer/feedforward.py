import tensorflow as tf
from typing import Optional


L2 = tf.keras.regularizers.l2(1e-6)


class FFModule(tf.keras.layers.Layer):
    def __init__(
        self,
        ffn_dim: int,
        dropout_rate: float = 0.4,
        expansion_factor: int = 4,
        output_reduction_factor: int = 0.5,
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = L2,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = L2,
        name: str = "FFModule",
        **kwargs
    ):
        super(FFModule, self).__init__(name=name, **kwargs)
        self.output_reduction_factor = output_reduction_factor

        self.ln = tf.keras.layers.LayerNormalization(
            beta_regularizer=bias_regularizer,
            gamma_regularizer=kernel_regularizer,
            name=f'{name}_ln'
        )
        self.fc1 = tf.keras.layers.Dense(
            units=ffn_dim * expansion_factor,
            name=f'{name}_fc1',
        )
        self.swish = tf.keras.layers.Activation(tf.nn.silu, name=f'{name}_swish')
        self.do1 = tf.keras.layers.Dropout(rate=dropout_rate, name=f'{name}_do1')
        self.fc2 = tf.keras.layers.Dense(
            units=ffn_dim,
            name=f'{name}_fc2',
        )
        self.do2 = tf.keras.layers.Dropout(rate=dropout_rate, name=f'{name}_do2')
        self.add = tf.keras.layers.Add(name=f'{name}_add')

    def summary(self):
        pass

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        outputs = self.fc1(outputs)
        outputs = self.swish(outputs)
        outputs = self.do1(outputs, training=training)
        outputs = self.fc2(outputs)
        outputs = self.do2(outputs, training=training)
        outputs = self.add([inputs, outputs * self.output_reduction_factor])
        
        return outputs
