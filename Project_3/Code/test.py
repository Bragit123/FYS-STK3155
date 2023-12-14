
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.preprocessing import minmax_scale, LabelBinarizer
import matplotlib.pyplot as plt

(X_train, t_train), (X_test, t_test) = datasets.mnist.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0

print(np.shape(X_train))

model = models.Sequential()
model.add(layers.Conv2D(28, (3, 3), activation="sigmoid", input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(56, (3, 3), activation="sigmoid"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(56, (3, 3), activation="sigmoid"))

model.add(layers.Flatten())
model.add(layers.Dense(56, activation='sigmoid'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, t_train, epochs=10, 
                    validation_data=(X_test, t_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig("cnn_accs.pdf")

test_loss, test_acc = model.evaluate(X_test, t_test, verbose=2)
print(test_acc)