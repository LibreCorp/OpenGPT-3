import torch
import torch.nn as nn
from typing import Dict

class TokenEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim, std=0.02, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.std = std

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=[self.vocab_size, self.embed_dim],
            initializer=tf.random_normal_initializer(stddev=self.std),
            trainable=True
        )

    def call(self, x, transposed=False):
        if transposed:
            # for final-logits projection: [..., embd] → [..., vocab]
            x_flat = tf.reshape(x, [-1, self.embed_dim])
            return tf.matmul(x_flat, self.weight, transpose_b=True)
        else:
            # standard token lookup: [..., seq] → [..., seq, embd]
            return tf.nn.embedding_lookup(self.weight, tf.cast(x, tf.int32))


class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, embed_dim, std=0.01, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.std = std

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=[self.max_len, self.embed_dim],
            initializer=tf.random_normal_initializer(stddev=self.std),
            trainable=True
        )

    def call(self, x, offset=0):
        """
        x: tensor of shape (..., seq_len)  (only for shape)
        offset: how many positions to skip for past cache
        """
        seq_len = tf.shape(x)[-1]
        pos = tf.range(offset, offset + seq_len, dtype=tf.int32)
        pos = tf.reshape(pos, (1,) * (x.shape.ndims - 1) + (seq_len,))
        pos = tf.tile(pos, tf.shape(x) // tf.shape(pos))
        return tf.nn.embedding_lookup(self.weight, pos)
