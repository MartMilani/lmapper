"""Author: maritno milani
m.milani@l2f.ch

Test routine for the mapper module
"""

import numpy as np
import lmapper as lm
from lmapper.filter import Eccentricity
from lmapper.cover import KeplerCover
from lmapper.cluster import Linkage
from lmapper.cutoff import FirstGap
from lmapper.datasets import cat
# to avoid the following error:
#
# >>> OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib
# >>> already initialized
#
# we need to add the following two lines:
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def test1(x):
    """Basic usage"""

    # instantiate a Mapper object
    """lmapper example usage"""
    filter = Eccentricity(exponent=2, metric="euclidean")
    cover = KeplerCover(nintervals=30,
                        overlap=0.4)
    cluster = Linkage(method='single',
                      metric='euclidean',
                      cutoff=FirstGap(0.05))
    mapper = lm.Mapper(data=x,
                       filter=filter,
                       cover=cover,
                       cluster=cluster)
    mapper.fit(skeleton_only=True)

    """change of clustering algorithm"""
    newcluster = Linkage(method='single',
                         metric='euclidean',
                         cutoff=FirstGap(0.1))
    mapper.set_params(cluster=newcluster)
    mapper.fit()
    mapper.plot()

    """change of clustering algorithm"""
    cluster = Linkage(method='single',
                      metric='euclidean',
                      cutoff=FirstGap(0.2))
    mapper.set_params(cluster=cluster)
    mapper.fit()

    """change of filter using a string argument"""
    mapper.set_params(filter='Projection')
    mapper.fit()

    """change of all parameters using a string argument"""
    mapper.set_params(filter='Projection',
                      cover='UniformCover',
                      cluster='Linkage')
    mapper.fit()
    return mapper


def test2(mapper):
    """Personalize filter values directly with a numpy.ndarray"""
    # passing personalized filter values
    values = np.random.rand(mapper.data.shape[0])
    mapper.set_params(filter=values)
    mapper.fit()

    """passing a personalized filter function"""
    def ProjectionOnFirstAxis(data, verbose=True):
        if verbose:
            print("Projecting data on first axis")
        return data[:, 0]

    mapper.set_params(filter=ProjectionOnFirstAxis)
    mapper.fit()
    return mapper


def main():
    x = cat()
    print(x.shape)
    m = test1(x)
    test2(m)
    print("\n\nTEST SUCCESSFUL")
    return 0


if __name__ == '__main__':
    main()
