import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Load the dataset using this command
digits = datasets.load_digits()
#print(digits)
# To know about the dataset we can use descr
#print(digits.DESCR)
# To know about targets 
print(digits.target)
# We will plot first a random image plt.gray will make it in the shades of black and white 
plt.gray()
plt.matshow(digits.images[100])
plt.show()
# To check if it is 4 
print(digits.target[100])
# To visualize more than one image

# Figure size (width, height)

fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images

for i in range(64):

    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position

    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    # Display an image at the i-th position

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value

    ax.text(0, 7, str(digits.target[i]))

plt.show()

# Now we will cluster our 10 digits using Kmeans
# random state can be any number this just ensures that model is built in the same way every time 
model = KMeans(n_clusters = 10, random_state = 42)
model.fit(digits.data)
fig = plt.figure(figsize = (8,3))
fig.suptitle("Cluster center images", fontsize = 14, fontweight = 'bold')
for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

new_samples = np.array([
[0.00,0.00,1.53,2.67,0.00,0.00,0.00,0.00,0.00,0.00,4.57,6.86,0.00,0.00,0.00,0.00,0.00,0.00,4.57,6.86,0.00,0.00,0.00,0.00,0.00,0.00,4.20,7.40,0.00,0.00,0.00,0.00,0.00,0.00,2.82,7.62,1.37,0.00,0.00,0.00,0.00,0.00,2.06,7.62,2.29,0.00,0.00,0.00,0.00,0.00,0.61,6.41,1.98,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.46,0.53,0.00,0.00,0.00,0.00,0.00,6.18,7.63,7.40,1.60,0.00,0.00,0.00,0.00,3.05,3.89,7.62,3.05,0.00,0.00,0.00,0.00,0.00,0.76,7.62,3.05,0.00,0.00,0.00,0.00,0.00,2.06,7.62,2.67,0.00,0.00,0.00,0.00,1.75,7.09,7.62,5.19,1.07,0.00,0.00,0.00,3.81,7.62,7.24,6.33,1.68,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.38,2.29,2.29,0.38,0.00,0.00,0.00,0.00,3.51,7.62,7.62,5.19,0.00,0.00,0.00,0.00,0.46,1.75,5.95,5.95,0.00,0.00,0.00,0.00,4.73,6.63,7.62,5.11,0.00,0.00,0.00,0.00,3.35,4.66,6.56,7.55,0.00,0.00,0.00,5.42,4.12,0.08,4.42,7.55,0.00,0.00,0.00,4.88,7.62,7.62,7.62,5.03,0.00,0.00,0.00,0.00,2.29,3.05,2.90,0.38,0.00],
[0.00,0.00,1.37,1.68,0.00,0.54,0.00,0.00,0.00,0.00,5.34,6.10,1.14,7.55,1.83,0.00,0.00,0.00,5.34,6.10,2.52,7.62,2.29,0.00,0.00,0.00,5.34,6.79,5.26,7.62,2.82,0.00,0.00,0.00,2.13,7.32,7.63,7.62,3.28,0.00,0.00,0.00,0.00,0.38,0.69,7.63,3.81,0.00,0.00,0.00,0.00,0.00,0.00,7.62,3.81,0.00,0.00,0.00,0.00,0.00,0.00,2.29,0.76,0.00]
]

)
new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
