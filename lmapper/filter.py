# -*- coding: utf-8 -*-
"""This module implements some filter functions. A base class is implemented
which every new filter that will be implemented has to inherit from.

"""
import sys
import numpy as np
from scipy.spatial.distance import cdist
# --------------------------------------------
# trying to import the filterutils module.
# in case of failure, define the corresponding
# functions in Python
# --------------------------------------------
sys.path.append(__file__[:-10]+'/cpp/filterutils')
print(__file__[:-10]+'/cpp/filterutils')
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
                     '“filterutils”.\nThe scipy.distance.cdist implementation is '
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
            refactor the code for prediction in order to vectorize it. Or maybe
            delegate this method to the predmap package.
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
