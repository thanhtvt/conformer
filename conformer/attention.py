import tensorflow as tf
from typing import Optional


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Implements the sinusoidal positional encoding function
    Based on https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
    """
    def __init__(self,
                 d_model: int = 512,
                 name: str = "PositionalEncoding",
                 **kwargs):
        self.d_model = d_model
        super(PositionalEncoding, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        d_model = input_shape[-1]
        assert d_model == self.d_model, f"d_model must be equal to the last dimension of the input, which is {self.d_model}"

    @staticmethod
    def encode(max_len, d_model):
        pe = tf.zeros([max_len, d_model])
        position = tf.expand_dims(tf.range(0, max_len), axis=1)
        position = tf.cast(position, dtype=tf.float32)
        div_term = tf.math.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / float(d_model)))
        
        # Have to set up this way cause Tensorflow not allow assigning to EagerTensor
        pe = tf.Variable(pe)
        pe[:, 0::2].assign(tf.math.sin(position * div_term))
        pe[:, 1::2].assign(tf.math.cos(position * div_term))
        pe = tf.convert_to_tensor(pe)
        pe = tf.expand_dims(pe, axis=0)

        return pe

    def call(self, inputs, **kwargs):
        max_len, d_model = tf.shape(inputs)[-2], tf.shape(inputs)[-1]
        pe = self.encode(max_len, d_model)
        # outputs = tf.math.add(inputs, pe)

        return pe


class RelativeMHA(tf.keras.layers.Layer):
    """
    Multi-head Attention with Relative Positional Embedding
    Based on https://github.com/sooftware/conformer/blob/main/conformer/attention.py
    """
    def __init__(self,
                 num_heads: int = 8,
                 d_model: int = 512,
                 dropout_rate: float = 0.4,
                 name: str = "RelativeMHA",
                 **kwargs):
        super(RelativeMHA, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.query_linear = tf.keras.layers.Dense(d_model)
        self.key_linear = tf.keras.layers.Dense(d_model)
        self.value_linear = tf.keras.layers.Dense(d_model)
        self.pos_linear = tf.keras.layers.Dense(d_model)
        self.out_linear = tf.keras.layers.Dense(d_model)

        self.u_bias = tf.Variable(tf.keras.initializers.HeUniform()([self.num_heads, self.d_head]))
        self.v_bias = tf.Variable(tf.keras.initializers.HeUniform()([self.num_heads, self.d_head]))


    def build(self, input_shape):
        d_model = input_shape[-1]
        assert d_model == self.d_model, f"d_model must be equal to the last dimension of the input, which is {self.d_model}"
        assert d_model % self.num_heads == 0, f"num_heads must be divisible by {d_model}"
    
    def call(self, 
             query: tf.Tensor,
             key: tf.Tensor,
             value: tf.Tensor,
             pos_embedding: tf.Tensor,
             training=False,
             attention_mask: Optional[tf.Tensor] = None) -> tf.Tensor:

        batch_size, seq_len = tf.shape(query)[0], tf.shape(query)[2]
        query = tf.reshape(self.query_linear(query, training=training), [batch_size, -1, self.num_heads, self.d_head])
        key = tf.transpose(tf.reshape(self.key_linear(key, training=training), [batch_size, -1, self.num_heads, self.d_head]), perm=[0, 2, 1, 3])
        value = tf.transpose(tf.reshape(self.value_linear(value, training=training), [batch_size, -1, self.num_heads, self.d_head]), perm=[0, 2, 1, 3])
        pos_embedding = tf.reshape(self.pos_linear(pos_embedding, training=training), [batch_size, -1, self.num_heads, self.d_head])

        content_score = tf.linalg.matmul(tf.transpose(query + self.u_bias, perm=[0, 2, 1, 3]), tf.transpose(key, perm=[0, 1, 3, 2]))
        pos_score = tf.linalg.matmul(tf.transpose(query + self.v_bias, perm=[0, 2, 1, 3]), tf.transpose(pos_embedding, perm=[0, 2, 3, 1]))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / tf.math.sqrt(float(self.d_model))

        if attention_mask is not None:
            attention_mask = tf.expand_dims(attention_mask, axis=1)
            score = tf.where(attention_mask, tf.fill(tf.shape(score), -1e9), score)

        attn = tf.nn.softmax(score, axis=-1)
        attn = self.dropout(attn, training=training)
        context = tf.transpose(tf.linalg.matmul(attn, value), perm=[0, 2, 1, 3])
        context = self.out_linear(tf.reshape(context, [batch_size, -1, seq_len, self.d_model]), training=training)

        return context

    @staticmethod
    def _relative_shift(pos_score: tf.Tensor) -> tf.Tensor:
        batch_size, num_heads, seq_len1, seq_len2 = tf.shape(pos_score)
        zeros = tf.zeros([batch_size, num_heads, seq_len1, 1])
        padded_pos_score = tf.concat([zeros, pos_score], axis=-1)

        padded_pos_score = tf.reshape(padded_pos_score, [batch_size, num_heads, seq_len2 + 1, seq_len1])
        pos_score = tf.reshape(padded_pos_score[:, :, 1:], tf.shape(pos_score))

        return pos_score


class MultiHeadedSelfAttention(tf.keras.layers.Layer):
    """
    Multi headed self-attention module using relative positional encoding
    """
    def __init__(self,
                 num_heads: int = 8,
                 d_model: int = 512,
                 dropout_rate: float = 0.4,
                 name: str = "MultiHeadedSelfAttention",
                 **kwargs):
        super(MultiHeadedSelfAttention, self).__init__(name=name, **kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.positional_encoding = PositionalEncoding(d_model)
        self.attention = RelativeMHA(num_heads=num_heads, d_model=d_model, dropout_rate=dropout_rate)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs: tf.Tensor, training=False, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = tf.concat([pos_embedding for _ in range(batch_size)], axis=0)
        
        outputs = self.layer_norm(inputs, training=training)
        outputs = self.attention(outputs, outputs, outputs, pos_embedding, training=training, attention_mask=mask)
        outputs = self.dropout(outputs, training=training)  

        return outputs


class MHSAModule(tf.keras.layers.Layer):
    """
    Multi headed self-attention module using regular positional encoding
    """
    def __init__(self,
                 head_size: int,
                 num_heads: int = 8,
                 d_model: int = 512,
                 dropout_rate: float = 0.4,
                 name: str = "MHSAModule",
                 **kwargs):
        super(MHSAModule, self).__init__(name=name, **kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.positional_encoding = PositionalEncoding(d_model)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                            key_dim=head_size)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()

    def call(self, inputs: tf.Tensor, training=False, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = tf.concat([pos_embedding for _ in range(batch_size)], axis=0)
        pos_embedding = tf.cast(pos_embedding, dtype=inputs.dtype)

        outputs = self.layer_norm(inputs, training=training)
        outputs = self.add([outputs, pos_embedding])
        outputs = self.attention(outputs, outputs, outputs, attention_mask=mask, training=training)
        outputs = self.dropout(outputs, training=training)
        outputs = self.add([inputs, outputs])
        
        return outputs
