import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('roses.jpg')  #Opens image to be processed
img = np.array(img) #Converting into a numpy array
original_shape=np.shape(img)
print("Original size =",original_shape)
X = np.reshape(img, [-1, 3]) #Flattening image
print("Flattened size =",np.shape(X))
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)  #Finding bandwidth chosen for the kernel
print("%d" % bandwidth)

ms = MeanShift(bandwidth=60, bin_seeding=True)
ms.fit(X) #Running meanshift on the image to segment it

labels = ms.labels_
labels_unique = np.unique(labels)  #Finding unique labels to avoid redundancy
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

segmented_image = np.reshape(labels, original_shape[:2])  #Final segmented image, taking only size

#Plotting both images
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.axis('off')
plt.show()
