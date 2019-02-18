# -*- coding: utf-8 -*-
"""This module implements the Mapper class, the key class that implements the
Mapper algorithm.

Example:
    >>> import lmapper as lm
    >>> mapper = lm.Mapper(data=x,
    >>>                    filter='Projection',
    >>>                    cluster='Linkage',
    >>>                    cover='BalancedCover)
    >>> mapper.fit()
    >>> mapper.plot()

The aim of this module is to provide an easy, well designed API for the client code
to be easily readable and coherent with the pipeline of the Mapper algorithm.
"""


import numpy as np
# keeping track of everything that has been implemented
# copying this information inside the class provides a bit more of order
# we import them at the beginning of the file in order to ease future
# development and extensions.
from lmapper.cover import Cover, _all_covers_
from lmapper.filter import Filter, _all_filters_
from lmapper.cluster import Cluster, _all_clusters_
from lmapper.complex import Complex


class Mapper():
    """Class responsible for the implementation of the Mapper algorithm.

    Basic example:
        >>> import lmapper as lm
        >>> from lm.filter import Projection
        >>> from lm.cover import UniformCover
        >>> from lm.cluster import Linkage
        >>> from lm.cutoff import FirstGap
        >>>
        >>>
        >>> filter = Projection(ax=0)
        >>> cover = UniformCover(nintervals=15,
        >>>                     overlap=0.4)
        >>> cluster = Linkage(method='single',
        >>>                   cutoff=FirstGap(0.05)
        >>> mapper = lm.Mapper(data=x,
        >>>                    filter=filter,
        >>>                    cover=cover,
        >>>                    cluster=cluster)
        >>> mapper.fit().plot()

    Attributes:
        data (np.ndarray): n-dimensional array containing the coordinates of
            the data points
        filter (Filter): filter function
        filter_values (np.ndarray): filter values
        cover (Cover): cover object implementing the particular chosen open cover
            on the image of the filter function
        cluster (Cluster): cluster object implementing the clustering algorithm
            that will create the nodes of the complex starting from each Fiber contained
            in the cover
        complex (Complex): to be used only after having called the fit() method
        _nodes (dict): created by the fit() method
        _fit_flag (int): assumes values in {0, 1, 2, 3, 4} and kepps track of the last
            pipeline step that was performed in case of callinf the method fit()
            after a change of parameters through the method set_params()
    """

    def __init__(self,
                 data,
                 filter='Projection',
                 cover='UniformCover',
                 cluster='Linkage'):
        """Accepts as a preferred input Cover, Filter, and Cluster objects.
        As an alternative and for a more basic and simple usage, accepts strings that
        must match exactly the classes defined in the modules filter.py, cover.py,
        cluster.py. It can accept also a custom numpy.ndarray as filter, representing
        the filter values calculated by the user.

        Args:
            data (numpy.ndarray): two dimensional array
            filter (mapper.Filter/numpy.ndarray/str): either a Filter object, either a
                numpy.ndarray of the filter values of the same length of data, either a
                string that must match exactly one of the classes defined in the module
                filter.py
            cover (mapper.Cover/str): either a Cover object, either a
                string that must match exactly one of the classes defined in the module
                cover.py
            cluster (mapper.Cluster/str): either a Cluster object, either a
                string that must match exactly one of the classes defined in the module
                cluster.py
        """

        # base objects needed for the algorithm
        assert isinstance(filter, (str, Filter, np.ndarray)), 'Wrong filter format'
        if isinstance(filter, str):
            assert filter in _all_filters_
            self.filter = Filter.factory(filter)
            self.filter_values = None
        elif isinstance(filter, Filter):
            self.filter = filter
            self.filter_values = None
        else:
            assert(filter.alen() == data.alen())
            self.filter_values = filter
            self.filter = None

        assert isinstance(cover, (str, Cover)), 'Wrong cover format'
        if isinstance(cover, str):
            assert cover in _all_covers_
            self.cover = Cover.factory(cover)
        else:
            self.cover = cover

        assert isinstance(cluster, (str, Cluster)), 'Wrong cluster format'
        if isinstance(cluster, str):
            assert cluster in _all_clusters_
            self.cluster = Cluster.factory(cluster)
        else:
            self.cluster = cluster

        self.data = data
        self.complex = Complex()
        self._fit_flag = 0
        self._nodes = {}

    def set_params(self, **kwargs):
        """Generic function that updates the following attributes:

        * data
        * filter
        * cover
        * cluster

        and sets the _fit_flag attribute to the corresponding value for a subsequent
        call to the fit() method. fit() will thus be able not to recompute values
        that do not have to be recomputed
        """

        # check the input format
        data = kwargs.get('data')
        filter = kwargs.get('filter')
        cover = kwargs.get('cover')
        cluster = kwargs.get('cluster')

        if data:
            try:
                assert isinstance(data, np.ndarray)
                assert len(data.shape) == 2  # checking it is actually a matrix
            except AssertionError:
                print('parameter format not accepted. '
                      'Expecting a two-dimensional np.array as self.data')
            self.data = data
            self._fit_flag = min([self._fit_flag, 0])

        if filter:
            assert isinstance(filter, (str, Filter)), 'Wrong filter format'
            if isinstance(filter, str):
                assert filter in _all_filters_
                self.filter = Filter.factory(filter)
            else:
                self.filter = filter
            self._fit_flag = min([self._fit_flag, 0])

        if cover:
            assert isinstance(cover, (str, Cover)), 'Wrong cover format'
            if isinstance(cover, str):
                assert cover in _all_covers_
                self.cover = Cover.factory(cover)
            else:
                self.cover = cover
            self._fit_flag = min([self._fit_flag, 1])

        if cluster:
            assert isinstance(cluster, (str, Cluster)), 'Wrong cluster format'
            if isinstance(cluster, str):
                assert cluster in _all_clusters_
                self.cluster = Cluster.factory(cluster)
            else:
                self.cluster = cluster
            self._fit_flag = min([self._fit_flag, 2])

        return self

    def fit(self, skeleton_only=True, verbose=1):
        """Here the mapper algorithm is implemented. This implementation is
        really high level, and tries to delegate the real computations to the
        methods of the filter, cover, cluster objects in order to ease
        further improvements (parallelization, better algorithms, C++ implementation)

        Depending on the state of the _fit_flag attribute (modified by a previous call
        to the set_params() method) it starts the Mapper pipeline of
            * computing the filter values
            * computing the cover
            * clustering the elements of the pullback cover to create the nodes
            * finding intersections between the nodes to create the nerve complex
        only where needed.

        Args:
            skeleton_only (bool): if True, it does not look for simplices of dimension
                higher than 1.
            verbose (int): verbose attribute, can assume values in {0, 1, 2}
        """
        if self._fit_flag == 0:

            # compute filter values
            self.filter_values = self.filter(self.data, verbose)

            # updating the _fit_flag signaling that this step was done successfully
            self._fit_flag = 1

        if self._fit_flag == 1:

            # cleaning the cover object in case the call to fit() happens after
            # a call to set_params()
            self.cover.__init__(**self.cover.get_params())

            # computing the fibers of the cover object
            self.cover.fit(self.filter_values, self.data, verbose)

            # updating the _fit_flag signaling that this step was done successfully
            self._fit_flag = 2

        if self._fit_flag == 2:

            # cleaning the cluster object in case the call to fit() happens after
            # a call to set_params()
            self.cluster.__init__(**self.cluster.get_params())

            # compute the nodes for every Fibers
            for fiber in self.cover:
                self.cluster(fiber, verbose)

        # updating the _fit_flag signaling that this step was done successfully
        self._fit_flag = 3

        if self._fit_flag == 3:

            # cleaning the cluster object in case the call to fit() happens after
            # a call to set_params()
            self.complex.__init__()

            # finding intersection between nodes
            self.complex.fit(self.cover, skeleton_only, verbose)

            # finishing the correct instantiation of the nodes.
            # TODO: think about a cleaner implementation, this process of complex.fit()
            # instantiating the nodes and the next 3 lines finishing the setting of the
            # _id attribute is cumbersome and prone to future bugs
            for fiber in self.cover:
                for node in fiber:
                    self._nodes[node._id] = node
            self._fit_flag = 4

        # next assert just check that the interaction between fit() and set_params()
        # went well.
        # TODO: test what happens if set_params() is called twice in a row?
        assert self._fit_flag == 4, ("Something went wrong in fitting the complex. "
                                     "_fit_flag should be equal to 4 at the end of "
                                     "this function")
        return self

    def plot(self, node_labels=False, edge_labels=False, node_color=None, pos=None):
        """Just a wrapper of Complex.plot()
        """
        self.complex.plot(node_labels=node_labels,
                          edge_labels=edge_labels,
                          node_color=node_color,
                          _pos=pos,
                          nodes=self._nodes)
        return self
