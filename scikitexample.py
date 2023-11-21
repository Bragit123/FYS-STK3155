# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# ensure the same random numbers appear every time
np.random.seed(0)


# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target


# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))


# choose some random images to display
#indices = np.arange(n_inputs)
#random_indices = np.random.choice(indices, size=5)

#for i, image in enumerate(digits.images[random_indices]):
#    plt.subplot(1, 5, i+1)


plt.axis('off')
plt.imshow(digits.images[3], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("Label: %d" % digits.target[3])
plt.show()
