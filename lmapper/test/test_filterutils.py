import numpy as np
import time
from lmapper.filter import Eccentricity


def test2(data, nthreads):
    e = Eccentricity(exponent=10, nthreads=nthreads, metric="euclidean")
    e(data)
    e(data)
    e(data)
    e(data)


if __name__ == '__main__':

    # to avoid the following error:
    #
    # >>> OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib
    # >>> already initialized
    #
    # we need to add the following two lines:
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    np.random.seed(seed=0)
    N = 10000
    data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=N)

    start = time.time()
    test2(data, 1)
    print('1 thread: it took {0:0.10f} seconds'.format((time.time() - start)/4.))

    start = time.time()
    test2(data, 2)
    print('2 threads: it took {0:0.10f} seconds'.format((time.time() - start)/4.))

    start = time.time()
    test2(data, 4)
    print('4 threads: it took {0:0.10f} seconds'.format((time.time() - start)/4.))
