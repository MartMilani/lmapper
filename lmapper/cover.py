# -*- coding: utf-8 -*-
"""Module implementing all the classes necessary to handle the cover and pullback cover.

A Cover object is designed as an iterable container of Fiber objects. Each Fiber is a
container of the data points contained in the pre-image through a Filter of one open set
defined on the Image(Filter(data)).

A cover object will be passed (and modified) as an argument to the __call__
method of a Cluster object that is in charge of creating the Nodes inside each Fiber.

Notes:
    check for existing iterators that could avoid defining the ListIterator class
"""

import numpy as np


class ListIterator():
    """A simple list iterator, necessary for making the Cover object an iterable
    on the list of Fibers.

    Notes:
        Check existing implementation! I'm sure there's one available

    """

    def __init__(self, lst):
        self.lst = lst
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos == len(self.lst):
            raise StopIteration()
        item = self.lst[self.pos]
        self.pos += 1
        return item


class Cover(object):
    """Abstract class implementing the interface for any cover to be implemented.
    Any cover implemented has to inherit from this class.

    Attributes:
        _fibers (list): list of Fibers, to be filled by the fit() method
        intersecting_dict (dict): {fiber_index: [fiber_index]}
            Each value represent the list of the fibers that overlap, and thus could
            potentially have nodes that share points, leading to edges and simplices
            in the final Complex.
            In other words, nodes belonging to fibers that do not overlap cannot be
            connected by edges in the final complex.
            This information is used by the Complex.fit() method to avoid useless checks
            for intersections between nodes.

    """
    def __init__(self):
        """Any Cover must have a list of Fibers and a dictionary
        """

        self._fibers = []
        self.intersecting_dict = {}

    def fit(self, filter_values, data, verbose):
        raise NotImplementedError()

    def __iter__(self):
        return ListIterator(self._fibers)

    def __len__(self):
        return len(self._fibers)

    def get_params(self):
        raise NotImplementedError()

    @staticmethod
    def factory(covertype):
        assert covertype in [cover.__name__ for cover in Cover.__subclasses__()]
        return globals()[covertype]()
        assert 0, "Bad cover creation: " + covertype

    @staticmethod
    def find_intersecting_dict(list_of_as, list_of_bs):
        """
        Args:
            list_of_as (list): list of the lower bounds of each interval
            list_of_bs (list): list of the higher bounds of each interval
        Returns:
            interseting_dict (dict): dictionary of type
                {int: [int]}
                containing informations of which fibers intersect in order to
                avoid looking for trivially empty intersections in the complex.fit()
                method afterwards.

        Warning: this implementation does not look at fiber indices!
            This makes the implementation simpler, but this could be prone to bugs
            when going multi-dimensional, where this implementation will not be coherent
            any more.
        """
        intersecting_dict = {}
        for i, [a, b] in enumerate(zip(list_of_as, list_of_bs)):
            intersecting_fibers = []
            for j, [_a, _b] in enumerate(zip(list_of_as, list_of_bs)):
                if j != i:
                    # checking for overlaps between the two intervals
                    if (_b < b or _b > a) or (_a < b or _a > a) or (_a < a and _b > b):
                        intersecting_fibers.append(j)
            intersecting_dict[i] = intersecting_fibers
        return intersecting_dict


class OverlapCover(Cover):

    def __init__(self, nintervals=10, overlap=0.4):
        super().__init__()
        assert overlap < 1 and overlap > 0, "Bad parameter: overlap has to be in (0, 1)"
        self._nintervals = nintervals
        self._overlap = overlap
        self._max_filter = None
        self._min_filter = None

    @property
    def nintervals(self):
        return self._nintervals

    @nintervals.setter
    def nintervals(self, value):
        """Setter for the _nintervals. Implements checks on the value, that has to
        be a positive integer
        """

        if not isinstance(value, int):
            raise ValueError("The number of intervals must be a positive integer")
        if value <= 0:
            raise ValueError("The number of intervals must be a positive integer")
        self._nintervals = value

    @property
    def overlap(self):
        return self._overlap

    @overlap.setter
    def overlap(self, value):
        """Setter for the _overlap. Implements checks on the value, that has to
        be a float in (0,1)
        """
        if not isinstance(value, float):
            raise ValueError("The overlap must be a positive float value")
        if value <= 0. or value >= 1.:
            raise ValueError("The overlap has to be strictly greater than 0.0 and "
                             "strictly less than 1.0")
        self.overlap = value

    def get_params(self):
        return {"nintervals": self._nintervals, "overlap": self._overlap}

    def fit(self, filter_values, data, verbose):
        raise NotImplementedError()

    def find_entries(self, list_of_as, list_of_bs, filter_values, data, verbose):
        """Function that allocates the Fiber objects
        Args:
            list_of_as (list): list of the lower bounds of each interval
            list_of_bs (list): list of the higher bounds of each interval
            filter_values (np.ndarray): array of the computed filter values
                ordered in the same way of data
            data (np.ndarray): two-dimensional array of the data
        Returns:
            fibers (list): list of Fibers objects newly instanciated
            interseting_dict (dict): dictionary of type
                {int: [int]}
                containing informations of which fibers intersect in order to
                avoid looking for trivially empty intersections in the complex.fit()
                method afterwards. The fibers are here identified by their position index
                in the list Cover._fibers.
        """
        N = len(list_of_as)
        naive_pointlabels = np.arange(0, len(filter_values))
        fibers = []
        for i, [a, b] in enumerate(zip(list_of_as, list_of_bs)):
            mask = [x >= a and x <= b for x in filter_values]
            corresponding_pointlabels = naive_pointlabels[mask]
            corresponding_filtervalues = filter_values[mask]
            # creating the corresponding fiber
            points = data[corresponding_pointlabels]
            f = Fiber(corresponding_pointlabels, a, b, i, corresponding_filtervalues, points)
            fibers.append(f)
            # updating the intersecting dictionary
            intersecting_dict = Cover.find_intersecting_dict(list_of_as, list_of_bs)

            if verbose >= 1:
                print("Interval {0:3d}/{1:3d}, I = ({2:0.3f}, {3:0.3f}), found {4:5d} points".format(
                    i+1, N, a, b, len(corresponding_pointlabels)))

        return fibers, intersecting_dict


class UniformCover(OverlapCover):
    """Implementation of a Uniform cover. A Uniform cover is made by
    overlapping intervals of equal length.

    Notes:
        So far the implementation is only for scalar filters. To be extended with
        multidimensional filters! The attributes _max_filter and _min_filter must
        be transformed in list or tuples, and the fit() method must be rewritten.

    Attributes:
        _nintervals (int): the number of intervals of the one dimensional filter image.
        _overlap (float): percentage of overlap between the intervals
        _max_filter (int): the maximum filter value
        _min_filter (int): the minimum filter value

    """
    def __init__(self, nintervals=10, overlap=0.4):
        super().__init__(nintervals, overlap)

    def fit(self, filter_values, data, verbose):
        """Creates the list of Fibers.

        Args:
            filter_values (np.ndarray): one dimensional array (Nx1)
            data (np.ndarray): two dimensional array (NxM)

        Notes:
            the current implementation creates a copy of filter values and of data points
            and gives ownership of these copies to the Fibers, that have their own copy
            of the corresponding points and filter values.
        """
        self._max_filter = np.max(filter_values)
        self._min_filter = np.min(filter_values)

        # renaming just for compactness of the code
        L = self._max_filter - self._min_filter
        N = float(self._nintervals)
        p = float(self._overlap)

        # enlarging a bit the image to avoid that numerical approximations could exclude
        # the extremal values when calculating list_of_as, list_of_bs.
        safe_min = self._min_filter-L*1e-8
        safe_max = self._max_filter+L*1e-8
        L = safe_max - safe_min

        # real algorithm starts here by finding list_of_as, list_of_bs
        length = L/(N-(N-1)*p)  # length of each interval (a,b)
        list_of_as = np.arange(safe_min, safe_max, length*(1-p))[:-1]
        list_of_bs = list_of_as + length

        self._fibers, self.intersecting_dict = self.find_entries(
            list_of_as, list_of_bs, filter_values, data, verbose)


class BalancedCover(OverlapCover):
    """Implementation of a balanced cover. A balanced cover is made by intervals
    containing the same number of filter values.

    Notes:
        So far the implementation is only for scalar filters. To be extended with
        multidimensional filters! The attributes _max_filter and _min_filter must
        be transformed in list or tuples, and the fit() method must be rewritten.

    Attributes:
        _nintervals (int): the number of intervals of the one dimensional filter image.
        _overlap (float): percentage of overlap between the intervals
        _max_filter (int): the maximum filter value
        _min_filter (int): the minimum filter value

    """
    def __init__(self, nintervals=10, overlap=0.4):
        super().__init__(nintervals, overlap)

    def fit(self, filter_values, data, verbose):
        """Creates the list of Fibers.

        Args:
            filter_values (np.ndarray): one dimensional array (Nx1)
            data (np.ndarray): two dimensional array (NxM)

        Notes:
            the current implementation creates a copy of filter values and of data points
            and gives ownership of these copies to the Fibers, that have their own copy
            of the corresponding points and filter values.
        """
        self._max_filter = np.max(filter_values)
        self._min_filter = np.min(filter_values)

        # renaming just for compactness of the code
        N = self._nintervals
        p = self._overlap

        # enlarging a bit the image to avoid for numerical approximations to exclude
        # the extremal values
        ordered_labels = np.argsort(filter_values)
        L = len(filter_values)

        # real algorithm starts here
        length = L/(N-(N-1)*p)  # length of each interval (a,b)
        list_of_as = np.arange(0, L, length*(1-p))[:-1]
        list_of_bs = list_of_as + length
        list_of_bs[-1] = L  # just to avoid numerical errors

        for i, [a, b] in enumerate(zip(list_of_as, list_of_bs)):
            mask = [x >= a and x <= b for x in range(L)]
            corresponding_pointlabels = ordered_labels[mask]
            corresponding_filtervalues = filter_values[corresponding_pointlabels]

            # creating the corresponding fiber
            points = data[corresponding_pointlabels]
            f = Fiber(corresponding_pointlabels, a, b, i, corresponding_filtervalues, points)
            self._fibers.append(f)

            # updating the intersecting dictionary
            self.intersecting_dict = Cover.find_intersecting_dict(list_of_as, list_of_bs)

            if verbose >= 1:
                print("Interval {0:3d}/{1:3d}, I = ({2:0.3f}, {3:0.3f}), found {4:2d} points".format(
                    i+1, N, a, b, len(corresponding_pointlabels)))


class KeplerCover(OverlapCover):
    """Implementation of the cover implemented by Kepler Mapper to be able to confront
    results and performances.

    Attributes:
        _nintervals (int): the number of intervals of the one dimensional filter image.
        _overlap (float): percentage of overlap between the intervals
        _max_filter (int): the maximum filter value
        _min_filter (int): the minimum filter value

    """
    def __init__(self, nintervals=10, overlap=0.4):
        super().__init__(nintervals, overlap)

    def fit(self, filter_values, data, verbose):
        """Creates the list of Fibers.

        Args:
            filter_values (np.ndarray): one dimensional array (Nx1)
            data (np.ndarray): two dimensional array (NxM)

        Notes:
            the current implementation creates a copy of filter values and of data points
            and gives ownership of these copies to the Fibers, that have their own copy
            of the corresponding points and filter values.
        """
        self._max_filter = np.max(filter_values)
        self._min_filter = np.min(filter_values)
        list_of_as = []
        list_of_bs = []

        # renaming just for compactness of the code
        p = self._overlap
        L = self._max_filter - self._min_filter
        N = self._nintervals

        # here we find the intervals and the corresponding entries
        chunk = L/N
        for i in range(N):
            a = self._min_filter + i*(chunk)
            b = a + chunk*(1+p)
            list_of_as.append(a)
            list_of_bs.append(b)
        self._fibers, self.intersecting_dict = self.find_entries(
            list_of_as, list_of_bs, filter_values, data, verbose)


class Fiber():
    """data structure for a Fiber

    Attributes:
        _fiber_index (int): unique identifier of a Fiber
        _filter_minima (int): minimum value of the filter values of one Fiber
        _filter_maxima (int): maximum value of the filter values of one Fiber
        _pointlabels (np.ndarray): one dimensional nparray, holding the indices
            of the points of the Fiber. These indices can be used in the original
            data attribute of the Mapper object to retrieve the points of the Fiber.
        _points (np.ndarray): two-dimensional copy of the data points of the Fiber
        _filtervalues (np.ndarray): one dimensional filter values of the points
        _clusterinfo (ClusterInfo): struct containing info on the clustering algorithm
            It's created by applying a Cluster call on the fiber.
        _nodes (list): list of Nodes. It's created by applying a Cluster call on the fiber.
    """

    def __init__(self, point_labels, a, b, i, filtervalues, points):
        self._fiber_index = i
        self._filter_minima = a
        self._filter_maxima = b
        self._pointlabels = point_labels
        self._points = points
        self._filtervalues = filtervalues
        self._clusterinfo = None
        self._nodes = []

    def __iter__(self):
        return ListIterator(self._nodes)


_all_covers_ = [c.__name__ for c in Cover.__subclasses__()]


if __name__ == '__main__':
    def test():
        import pickle
        from filter import Filter
        import matplotlib.pyplot as plt

        data = pickle.load(open('/Users/martinomilani/Documents/III_semester/PACS/shapegraph/data.pickle', 'rb'))
        plt.plot([x[0] for x in data], [x[1] for x in data], 'ro')
        plt.show()

        f = Filter.factory('Projection')
        cover = Cover.factory('UniformCover')
        filter_values = f(data)
        cover.fit(filter_values, data)
        for fiber in cover:
            print('\n new fiber')
            print(fiber._fiber_index)
            print(fiber._filter_minima)
            print(fiber._filter_maxima)
            print(fiber._pointlabels)
            print(len(fiber._points))
            plt.plot([x[0] for x in fiber._points], [x[1] for x in fiber._points], 'ro')
            plt.show()

        print('OK')
    test()
