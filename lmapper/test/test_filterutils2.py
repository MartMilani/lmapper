from filterutils import my_distance
import numpy as np
from scipy.spatial.distance import cdist
from lmapper.datasets import wisconsin_breast_cancer

if __name__ == '__main__':
    xs, _ = wisconsin_breast_cancer()
    xs = xs.astype('double')
    new_data = np.empty(xs.shape)
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            new_data[i, j] = xs[i, j]
    dm = cdist(new_data, new_data)
    my_dm = np.zeros((np.alen(dm), np.alen(dm))).astype('double')
    my_distance(new_data, my_dm, 4, "euclidean")
    print(dm)
    print(my_dm)
