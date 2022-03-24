import tensorflow as tf
from conformer import Conformer

batch_size, seq_len, input_dim = 3, 15, 256

model = Conformer(
    num_conv_filters=[512, 512], 
    num_blocks=1, 
    encoder_dim=512, 
    num_heads=8, 
    dropout_rate=0.4, 
    num_classes=10, 
    include_top=True
)

# Get sample input
inputs = tf.random.uniform((batch_size, seq_len, input_dim),
                            minval=-40,
                            maxval=40)

# Convert to 4-dimensional tensor to fit Conv2D
inputs = tf.expand_dims(inputs, axis=1)  

# Get output
outputs = model(inputs)     # [batch_size, 1, seq_len, num_class]
outputs = tf.squeeze(outputs, axis=1)