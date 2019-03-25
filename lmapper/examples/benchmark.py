"""Author: maritno milani
m.milani@l2f.ch

Test routine for the mapper module
"""
import numpy as np
import kmapper as km
import sklearn
import time
import lmapper as lm
from lmapper.filter import Projection
from lmapper.cover import KeplerCover
from lmapper.cluster import Linkage
from lmapper.cutoff import FirstGap


def test(x):
    """Basic usage"""
    # instantiate a Mapper object
    start = time.time()
    filter = Projection(ax=2)
    cover = KeplerCover(nintervals=25,
                        overlap=0.4)
    cluster = Linkage(method='single',
                      metric='euclidean',
                      cutoff=FirstGap(0.05))
    mapper = lm.Mapper(data=x,
                       filter=filter,
                       cover=cover,
                       cluster=cluster)
    mapper.fit(skeleton_only=True).plot
    print('Martino mapper: {0:.4f} sec'.format(time.time()-start))

    start = time.time()
    mapper = km.KeplerMapper(verbose=2)
    projected_data = mapper.fit_transform(x, projection=[2])
    graph = mapper.map(
        projected_data,
        x,
        nr_cubes=25,
        overlap_perc=0.4,
        clusterer=sklearn.cluster.AgglomerativeClustering(linkage='single'))

    print('Kepler mapper: {0:.4f} sec'.format(time.time()-start))

    return 0


def main():
    from numpy import genfromtxt
    x = genfromtxt('../datasets/synthetic.csv', delimiter=',')
    x = x[1:]  # eliminating the first row of nans
    x = np.asarray([row[0:3] for row in x])
    test(x)
    return 0


if __name__ == '__main__':
    main()
