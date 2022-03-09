import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.layers.experimental.preprocessing import PreprocessingLayer


class SpecAugment(PreprocessingLayer):
    def __init__(self, 
                 freq_mask_prob: float = 0.5,
                 freq_mask_param: float = 10,
                 time_mask_prob: float = 0.5,
                 time_mask_param: float = 10):
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_param = freq_mask_param
        self.time_mask_prob = time_mask_prob
        self.time_mask_param = time_mask_param
    
    def call(self, features):
        prob = tf.random.uniform([])
        augmented = tfio.audio.freq_mask(features, param=self.freq_mask_param)
        features = tf.cond(prob >= self.freq_mask_prob,
                           lambda: augmented,
                           lambda: features)
        
        prob = tf.random.uniform([])
        augmented = tfio.audio.time_mask(features, param=self.time_mask_param)
        features = tf.cond(prob >= self.time_mask_prob,
                           lambda: augmented,
                           lambda: features)

        return features
