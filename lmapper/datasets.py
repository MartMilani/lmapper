import numpy as np
import gzip
import pandas as pd


def cat():
    filename = '../datasets/cat-reference.csv.gz'
    with gzip.open(filename, 'r') as inputfile:
        cat = np.loadtxt(inputfile, delimiter=',', dtype=np.float)
    return cat


def wisconsin_breast_cancer():
    data = pd.read_csv('../datasets/wisconsinbreastcancer.csv')
    x = data[data.columns[2:-1]].values
    y = data[data.columns[1]].values
    y_ = np.asarray([0 if x == 'M' else 1 for x in y])
    return x, y_


def synthetic_dataset():
    from numpy import genfromtxt
    x = genfromtxt('../datasets/synthetic.csv',
                   delimiter=',')
    # preprocessing of data
    # eliminating the first row of nans
    x = x[1:]
    # separating features and labels
    y = np.asarray([row[3] for row in x])
    x = np.asarray([row[0:3] for row in x])
    return x, y
