import tensorflow as tf

tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='tanh'),
  tf.keras.layers.Dense(128, activation='tanh'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])