import pickle
from sklearn import datasets
import matplotlib.pyplot as plt


data, labels = datasets.make_circles(n_samples=25000, noise=0.03, factor=0.3)
plt.plot([x[0] for x in data], [x[1] for x in data], 'ro')
pickle.dump(data, open('data.pickle', 'wb'))
