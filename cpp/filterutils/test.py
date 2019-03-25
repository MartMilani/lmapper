from filterutils import my_distance
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv('/Users/martinomilani/Documents/III_semester/PACS/project/wisconsinBreastCancer/data.csv')
    xs = data[data.columns[2:-1]].copy().values
    xs = xs.astype('double')
    new_data = np.empty(xs.shape)
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            new_data[i, j] = xs[i, j]
    xs[0]
    new_data[0]
    [(x0-x1)**2 for x0, x1 in zip(xs[0], xs[1])]
    new_data.shape
    dm = cdist(new_data, new_data)
    my_dm = np.zeros((np.alen(dm), np.alen(dm))).astype('double')
    my_dm.shape
    my_distance(new_data, my_dm, 4, "euclidean")
    print(dm)
    print(my_dm)
