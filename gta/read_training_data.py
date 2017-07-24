# Read training data

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
data = np.load('training_data.npy')

plt.imshow(data[1][0],cmap='gray')
plt.show()
