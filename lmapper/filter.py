# -*- coding: utf-8 -*-
"""This module implements some filter functions. A base class is implemented
which every new filter that will be implemented has to inherit from.

"""
import sys
import numpy as np
from scipy.spatial.distance import cdist
sys.path.append('/Users/martinomilani/Documents/lmapper/cpp/filterutils')
try:
    from filterutils import eccentricity
except ImportError:
    sys.stderr.write('Warning: Could not load the C++ module '
                     '“filterutils”.\nThe Python implementation of the eccentricity is '
                     'used instead, but it will be slower.\n')

    def eccentricity(dm, ecc, exp, nthread):
        """Python implementation of the eccentricity function.

        Args:
            dm (np.ndarray(dtype='float')): distance matrix
            ecc (np.ndarray(dtype='float')): empty np.ndarray to be filled
            exp (int): exponent of the eccentricity
            nthread (int): useless for this python implementation

        Note:
            it's fundamental that dm and ecc are arrays of floats!
        """
        N = np.alen(dm)
        if exp == -1:  # -1 here stands for np.inf
            ecc[:] = np.amax(dm, axis=1)
        elif exp == 1.:
                ecc[:] = np.sum(dm, axis=1)/float(N)
        else:
            dsum = np.sum(np.power(dm, exp), axis=1)
            ecc[:] = np.power(dsum/float(N), 1./exp)

try:
    from filterutils import my_distance
except ImportError:
    sys.stderr.write('Warning: Could not load the module '
                     '“fastdistance”.\nThe scipy.distance.cdist implementation is '
                     'used instead.\n')


class Filter():
    """Base class impementing the interface for a filter.

    Methods:
        __call__(data)
        factory(filtertype)

    """

    def __call__(self, x):
        """
        Args:
            x (np.ndarray): has to be a two-dimensional np.ndarray
        """
        raise NotImplementedError()

    @staticmethod
    def factory(filtertype):
        """This method is called by the ShapeGraph.fit() method to create
        the correct filter type

        Args:
            filtertype (str): has to be exacltly the name of the filter class
                we want to instantiate

        Returns:
            Filter: a Filter object

        """

        assert filtertype in [filter.__name__ for filter in Filter.__subclasses__()]
        return globals()[filtertype]()
        assert 0, "Bad filter creation: " + filtertype


class Projection(Filter):
    """Filter implementing the projection on one axis

    Args:
        ax (int): ax to project on.
    """
    def __init__(self, ax=0):
        self.ax = ax

    def __call__(self, data):

        return np.array([row[self.ax] for row in data])

    def for_assignment_only(self, x, data):
        """Method used by the companion package PredMap
        """
        return x[self.ax]


class Eccentricity(Filter):
    """Consider the full (N×N)-matrix of pairwise distances. The eccentricity of the
    i-th data point is the Minkowski norm of the i-th row with the respective exponent.
    """
    def __init__(self, exponent=1., metric='euclidean'):
        self.exponent = exponent
        self.metric = metric

    def __call__(self, data, nthread=4):
        """Just a wrapper of a call to my_distance and a call to eccentricity().
        """
        # defining the parameters
        data_ = data.astype('double')
        N = np.alen(data)
        # instantiating the distance matrix and the return value
        ecc = np.zeros(N, dtype='double')
        dm = np.zeros((N, N), dtype='double')

        # computing the distance matrix
        try:
            my_distance(data_, dm, nthread, self.metric)
        except NameError:
            dm = cdist(data_, data_, metric=self.metric)

        # computing eccentricity
        if self.exponent in (np.inf, 'Inf', 'inf'):
            eccentricity(dm, ecc, -1, nthread)
        elif self.exponent == 1.:
            eccentricity(dm, ecc, 1, nthread)
        else:
            eccentricity(dm, ecc, int(self.exponent), 4)
        return ecc

    def for_assignment_only(self, x, data):
        """Method used by the companion package PredMap.
        Serial (x has to be a one dimensional np.ndarray representing a single datapoint)

        TODO:
            refactor the code for prediction in order to vectorize it
        """
        N = np.alen(data)
        x = x.reshape((1, len(x)))
        # calculating the distance vector dm
        dm = cdist(x, data, metric=self.metric)
        if self.exponent in (np.inf, 'Inf', 'inf'):
            ecc = dm.max()
        elif self.exponent == 1.:
            ecc = dm.sum()/float(N+1)
        else:
            dsum = np.power(dm, self.exponent).sum()
            ecc = np.power(dsum/float(N+1), 1./self.exponent)
        return ecc


_all_filters_ = [f.__name__ for f in Filter.__subclasses__()]


if __name__ == '__main__':
    def test():
        """Test routine for the filter module

        Returns:
            (int):

        Raises:
            AssertionError: if the filter does not behave as predicted
        """

        import pickle
        x = pickle.load(open('/Users/martinomilani/Documents/III_semester/PACS/shapegraph/data.pickle', 'rb'))
        f = Filter.factory('Projection')
        filter_values = f(x)
        assert (filter_values == np.array([row[0] for row in x])).all()
        print('OK')

        def eq(a, b):
            for aa, bb in zip(a, b):
                if not np.asarray([x < y+1e-12 and x > y-1e-12 for x, y in zip(aa, bb)]).all():
                        return 0
            return 1

        from numpy import genfromtxt
        x = genfromtxt('/Users/martinomilani/Documents/III_semester/PACS/project/synthetic_dataset/synthetic.csv',
                       delimiter=',')
        x = x[1:]  # eliminating the first row of nans
        x = np.asarray([row[0:3] for row in x])
        x = x[0:1000]
        N = np.alen(x)
        dm = np.zeros((N, N), dtype='double')
        my_distance(x, dm, 4, 'euclidean')
        dm_check = cdist(x, x, 'euclidean')
        assert eq(dm, dm_check)
        dm = np.zeros((N, N), dtype='double')
        my_distance(x, dm, 4, 'correlation')
        dm_check = cdist(x, x, 'correlation')
        assert eq(dm, dm_check)
        return 0
    test()
