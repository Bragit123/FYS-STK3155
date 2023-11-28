import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
tf.keras.datasets.mnist.load_data(path="mnist.npz")
