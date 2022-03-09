import tensorflow as tf


class FeedForwardModule(tf.keras.layers.Layer):
    def __init__(self,
                 ffn_dim: int,
                 dropout_rate: float = 0.4,
                 expansion_factor: int = 4,
                 output_reduction_factor: int = 0.5,
                 name: str = "FeedForwardModule",
                 **kwargs):
        super(FeedForwardModule, self).__init__(name=name, **kwargs)
        self.output_reduction_factor = output_reduction_factor

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(ffn_dim * expansion_factor),
            tf.keras.layers.Activation(tf.nn.silu),     # Swish activation with beta=1
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(ffn_dim),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()


    def call(self, inputs, training=False, **kwargs):
        outputs = self.ffn(inputs, training=training)
        outputs = self.add([inputs, outputs * self.output_reduction_factor])
        
        return outputs
