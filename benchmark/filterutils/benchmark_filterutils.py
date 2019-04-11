"""Author: Martino Milani
martino.milani94@gmail.com

Test routine for the lmapper module
"""
import numpy as np
import kmapper as km
import sklearn
import time
import lmapper as lm
from lmapper.filter import Eccentricity, Projection
from lmapper.cover import KeplerCover, BalancedCover
from lmapper.cluster import Linkage
import predmap as mapp
from lmapper.cutoff import FirstGap


def test(x):
    """Basic usage"""
    # instantiate a Mapper object
    start = time.time()
    filter = Eccentricity(nthreads=1)
    cover = KeplerCover(nintervals=25,
                        overlap=0.4)
    cluster = Linkage(method='single',
                      metric='euclidean',
                      cutoff=FirstGap(0.05))
    mapper = lm.Mapper(data=x,
                       filter=filter,
                       cover=cover,
                       cluster=cluster)
    mapper.fit(skeleton_only=True)
    print('1 thread: {0:.4f} sec'.format(time.time()-start))

    start = time.time()
    filter = Eccentricity(nthreads=8)
    cover = KeplerCover(nintervals=25,
                        overlap=0.4)
    cluster = Linkage(method='single',
                      metric='euclidean',
                      cutoff=FirstGap(0.05))
    mapper = lm.Mapper(data=x,
                       filter=filter,
                       cover=cover,
                       cluster=cluster)
    mapper.fit(skeleton_only=True)
    print('16 threads: {0:.4f} sec'.format(time.time()-start))

    return 0


def main():
    from numpy import genfromtxt
    x = genfromtxt('/Users/Mart/Documents/POLIMI/IV_anno/II_semestre/PACS/project/lmapper/lmapper/datasets/synthetic.csv', delimiter=',')
    x = x[1:]  # eliminating the first row of nans
    x = np.asarray([row[0:3] for row in x])
    test(x)
    return 0


if __name__ == '__main__':
    main()
