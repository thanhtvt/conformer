import tensorflow as tf
from typing import List


class ConvSubsampling(tf.keras.layers.Layer):
    def __init__(self,
                 filters: List[int],
                 kernel_size: List[int] = [3, 3],
                 num_blocks: int = 1,
                 num_layers_per_block: int = 2,
                 dropout_rate: float = 0.0,
                 name: str = "ConvSubsampling",
                 **kwargs):
        
        super(ConvSubsampling, self).__init__(name=name, **kwargs)
        
        self.conv_blocks = tf.keras.Sequential()
        for i in range(num_blocks):
            convs = tf.keras.Sequential()
            for _ in range(num_layers_per_block):
                conv = tf.keras.layers.Conv2D(filters=filters[i],
                                              kernel_size=kernel_size[i],
                                              padding='same')
                dropout = tf.keras.layers.Dropout(rate=dropout_rate)
                relu = tf.keras.layers.ReLU()

                convs.add(conv)
                convs.add(dropout)
                convs.add(relu)
            
            self.conv_blocks.add(convs)
    
    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv_blocks(inputs, training=training)
        return outputs
