import tensorflow as tf
from typing import Optional, List

from conformer.subsampling import ConvSubsampling
from conformer.convolution import ConvolutionModule
from conformer.feedforward import FeedForwardModule
from conformer.attention import MultiHeadedSelfAttention, MHSAModule


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 num_blocks: int = 1,
                 encoder_dim: int = 512,
                 num_heads: int = 8,
                 dropout_rate: float = 0.4,
                 name: str = "ConformerBlock",
                 **kwargs):
        super(ConformerBlock, self).__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.ff_module = FeedForwardModule(encoder_dim)
        self.attention = MultiHeadedSelfAttention(num_heads=num_heads, d_model=encoder_dim, dropout_rate=dropout_rate)
        # self.attention = MHSAModule(head_size=encoder_dim, 
        #                             num_heads=num_heads, 
        #                             d_model=encoder_dim, 
        #                             dropout_rate=dropout_rate)
        self.conv = ConvolutionModule(encoder_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs: tf.Tensor, training=False, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        for _ in range(self.num_blocks):
            x = self.ff_module(inputs, training=training)
            x = self.attention(x, training=training, mask=mask)
            x = self.conv(x, training=training)
            x = self.ff_module(x, training=training)
            x = self.layer_norm(x, training=training)

        return x


class Conformer(tf.keras.Model):
    def __init__(self,
                 num_conv_filters: List[int],
                 num_blocks: int = 1,
                 encoder_dim: int = 512,
                 num_heads: int = 8,
                 dropout_rate: float = 0.4,
                 num_classes:int = 10,
                 include_top: bool = True,
                 name: str = "Conformer",
                 **kwargs):
        super(Conformer, self).__init__(name=name, **kwargs)
        self.include_top = include_top
        self.conv_subsampling = ConvSubsampling(filters=num_conv_filters, dropout_rate=dropout_rate)
        self.linear = tf.keras.layers.Dense(encoder_dim)
        self.out_linear = tf.keras.layers.Dense(num_classes)
        self.relu = tf.keras.layers.Activation(tf.nn.relu)
        self.log_softmax = tf.keras.layers.Activation(tf.nn.log_softmax)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.conformer_block = ConformerBlock(num_blocks=num_blocks, encoder_dim=encoder_dim, num_heads=num_heads, dropout_rate=dropout_rate)

    def call(self, inputs: tf.Tensor, training=False, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        x = self.conv_subsampling(inputs, training=training)
        x = self.linear(x, training=training)
        x = self.relu(x, training=training)
        x = self.dropout(x, training=training)
        x = self.conformer_block(x, training=training, mask=mask)

        if self.include_top:
            x = self.out_linear(x, training=training)
            x = self.log_softmax(x, training=training)

        return x
