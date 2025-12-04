import tensorflow as tf

def hardmax(logits):
    axis = -1
    max_vals = tf.reduce_max(logits, axis=axis, keepdims=True)
    return tf.cast(tf.equal(logits, max_vals), logits.dtype)
