"""Author: Martino Milani
martino.milani94@gmail.com

Test routine for the lmapper module
"""
import time
import lmapper as lm
from lmapper.filter import Eccentricity
from lmapper.cover import KeplerCover
from lmapper.cluster import Linkage
from lmapper.cutoff import FirstGap
from lmapper.datasets import importer_synthetic


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
    x, _ = importer_synthetic()
    test(x)
    return 0


if __name__ == '__main__':
    main()
