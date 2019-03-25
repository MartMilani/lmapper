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

"""import sys
sys.path.append('/Users/martinomilani/Documents/lmapper/cpp')
from fastfilter import Eccentricity"""


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

    cluster = Linkage(method='single',
                      metric='euclidean',
                      cutoff=FirstGap(0.1))
    mapper.set_params(cluster=cluster)
    mapper.fit()

    mapper.set_params(filter='Projection')
    mapper.fit()

    mapper.set_params(filter='Projection',
                      cover='UniformCover',
                      cluster='Linkage')
    mapper.fit()

    mapper.set_params(filter='Projection')
    mapper.fit()

    return mapper


def test2(mapper):
    """Personalize filter values"""

    # passing personalized filter values
    values = np.random.rand(mapper.data.shape[0])
    mapper.set_params(filter=values)
    mapper.fit()

    # passing a callback
    def ProjectionOnFirstAxis(data, verbose):
        if verbose:
            print("Projecting data on first axis")
        return data[:, 0]

    mapper.set_params(filter=ProjectionOnFirstAxis)
    mapper.fit()
    return mapper


def main():
    import gzip
    import numpy as np
    filename = '/Users/martinomilani/Documents/III_semester/PACS/mapper/exampleshapes/cat-reference.csv.gz'
    with gzip.open(filename, 'r') as inputfile:
        x = np.loadtxt(inputfile, delimiter=',', dtype=np.float)
    print(x.shape)
    m = test1(x)
    test2(m)
    print("\n\nTEST SUCCESSFUL")
    return 0


if __name__ == '__main__':
    main()
