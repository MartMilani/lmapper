import filterutils as ff
import numpy as np
import time
from lmapper.filter import Eccentricity


def test(data, nthreads):
    p = ff.Projection(0, nthreads)
    p(data)
    p(data)
    p(data)
    p(data)


def test2(data, nthreads):
    p = ff.Eccentricity(nthreads, 1, "euclidean")
    p(data)
    p(data)
    p(data)
    p(data)


if __name__ == '__main__':

    np.random.seed(seed=0)
    data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=10000)

    start = time.time()
    test(data, 1)
    print('1 thread: it took {0:0.10f} seconds'.format((time.time() - start)/4.))

    start = time.time()
    test(data, 4)
    print('4 threads: it took {0:0.10f} seconds'.format((time.time() - start)/4.))

    start = time.time()
    test2(data, 1)
    print('1 thread: it took {0:0.10f} seconds'.format((time.time() - start)/4.))

    start = time.time()
    test2(data, 2)
    print('4 threads: it took {0:0.10f} seconds'.format((time.time() - start)/4.))

    start = time.time()
    test2(data, 4)
    print('4 threads: it took {0:0.10f} seconds'.format((time.time() - start)/4.))

    f = Eccentricity(exponent=1, metric="correlation")

    start = time.time()
    ecc = f(data)
    print('python class with cpp functions: it took {0:0.10f} seconds'.format(time.time() - start))

    p = ff.Eccentricity(4, 1, "correlation")
    start = time.time()
    ecccpp = p(data)
    print('cpp class: it took {0:0.10f} seconds'.format(time.time() - start))
    print(ecc)
    print(ecccpp)
