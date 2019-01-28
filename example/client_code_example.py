"""Author: maritno milani
m.milani@l2f.ch

Test routine for the mapper module
"""
import sys
sys.path.append('/Users/martinomilani/Documents/III_semester/PACS/project/pymapper')
import numpy as np
import amapper as mp
from filter import Eccentricity
from cover import UniformCover
from cluster import Linkage
from cutoff import FirstGap


def test1(x):
    """Basic usage"""

    # instantiate a Mapper object
    filter = Eccentricity(exponent=2)
    cover = UniformCover(nintervals=30,
                         overlap=0.4)
    cluster = Linkage(method='single',
                      metric='euclidean',
                      cutoff=FirstGap(0.05))
    mapper = mp.Mapper(data=x,
                       filter=filter,
                       cover=cover,
                       cluster=cluster)
    mapper.fit(skeleton_only=False).plot(node_labels=False)

    cluster = Linkage(method='single',
                      metric='euclidean',
                      cutoff=FirstGap(0.1))
    mapper.set_params(cluster=cluster)
    mapper.fit().plot()

    mapper.set_params(filter='Projection')
    mapper.fit().plot()

    mapper.set_params(filter='Projection',
                      cover='UniformCover',
                      cluster='Linkage')
    mapper.fit().plot()

    mapper.set_params(filter='Projection')
    mapper.fit().plot(node_labels=False)

    return mapper


def test2(mapper):
    """Reset parameters on the way"""

    # modifying the desired parameter

    mapper.set_params(filter='gaussian')
    mapper.fit(minimum_dimension_nodes=10, minimum_ovelap_points=10)
    mapper.plot()
    return mapper


def test3(mapper):
    """Personalize clustering algorithm"""

    from scipy.cluster.hierarchy import linkage
    mapper.set(cluster=linkage,
               cluster_method='single')
    mapper.fit()
    mapper.plot()
    return mapper


def test4(mapper):
    """Personalize filter values"""

    # passing personalized filter values
    values = np.random.rand(size=mapper.x.shape[0])
    mapper.set(filter=values)
    mapper.fit()
    mapper.plot()

    # passing a callback
    def f(x):
        return x[0]

    mapper.set(filter=f)
    mapper.fit()
    mapper.plot()
    return mapper


def test5(mapper):
    """Parallel computing test"""

    from multiprocessing import cpu_count
    mapper.set(cluster='average linkage')
    mapper.fit(nthreads=cpu_count())


def main():
    import gzip
    import numpy as np
    filename = '/Users/martinomilani/Documents/mapper/exampleshapes/cat-reference.csv.gz'
    with gzip.open(filename, 'r') as inputfile:
        x = np.loadtxt(inputfile, delimiter=',', dtype=np.float)
    print(x.shape)
    test1(x)
    return 0


if __name__ == '__main__':
    main()
