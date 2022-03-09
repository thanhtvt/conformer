import tensorflow as tf
from conformer import Conformer

batch_size, seq_len, dim = 3, 15, 512

inputs = tf.random.uniform((batch_size, seq_len, dim),
                            minval=-40,
                            maxval=40)



model = Conformer(num_conv_filters=[512, 512], num_blocks=1, encoder_dim=512, num_heads=8, dropout_rate=0.4, num_classes=10, include_top=True)
inputs = tf.expand_dims(inputs, axis=1)
outputs = model(inputs)
outputs = tf.squeeze(outputs, axis=1)
