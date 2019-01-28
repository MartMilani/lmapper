"""Author: Martino Milani

implementation of Node, Simplex and Complex classes
"""

import numpy as np
from itertools import combinations
import networkx as nx
import scipy.special
import matplotlib.pyplot as plt


class Edge():
    """Class used in the Complex.fit() method. It is used to seamlessly implement
    a undirected edge thanks to the overloading of the __eq__() method

    Note:
        This class could lead to unnecessary overheads. Maybe in a second time
        it could be eliminated when a better implementation of the Complex.fit()
        will be available.

    Attributes:
        _edge (np.ndarray): list of two ints (Node._ids)

    """

    _edge = []

    def __init__(self, first, second, intersection):
        if first < second:
            self._edge = np.asarray([first, second])
        else:
            self._edge = np.asarray([second, first])
        self._intersection = intersection

    def __eq__(self, other):
        if isinstance(other, Edge):
            if self._edge[0] == other._edge[0] and self._edge[1] == other._edge[1]:
                return True
        return False

    def __getitem__(self, key):
        return self._edge[key]


class Node():
    """class responsible for storing the information of one node of the
    final complex, result of the Cluster call on a Fiber.

    Attributes:
        _id (int): unique id of the node inside the Fiber it belongs to.
            It is assigned by the Complex.fit() method, by

            >>> id = range(1,alot)
            >>> for fiber in cover:
            >>>     for node in fiber:
            >>>         node._id = next(id)

        _labels (np.ndarray): labels of the data points contained in the node
        _attribute (int): attribute that will decide the color of the plot
        _fiber_index (int): index of the Fiber that owns this node
        _neighbours (list): list of id of the neighbours. to be filled by the
            Complex.

    """

    def __init__(self, labels, attribute, fiber_index):
        # checking format of the inputs
        try:
            assert isinstance(labels, np.ndarray)
        except AssertionError:
            # to take care of sinlgetons.
            # if a one-dimensional nparray is passed as argument to a function,
            # it is parsed automatically to a single np.int64
            if isinstance(labels, np.int64):
                labels = np.asarray([labels])
            else:
                print("------- WARNING! -------\n"
                      "Unknown format for the parameter 'labels' in function Node.__init__().\n"
                      "Undefined behaviour.")

        self._id = None
        self._labels = labels
        self._attribute = attribute
        self._fiber_index = fiber_index
        self._neighbours = []

    def sort_neighbours(self):
        self._neighbours.sort(key=lambda x: x._id)

    @property
    def size(self):
        return len(self._labels)


class Complex():
    """Class responsible for storing all the simplices, edges, nodes of the
    Mapper graph. It also implements method for drawing.

    Attributes:
        _simplices (set): set of Simplices. A Simplex is just a tuple of int, where
            each int is the id of a node.
        _facets (set): set of maximal Simplices
        _graph (nx.Graph): NetworkX graph containing only the 1d simplices to draw the
            skeleton of the simplex easily using _graph.draw(). Each node inside this
            NetworkX graph is identified by the Node._id attribute.
        _node_color (list): list of floats that is created by the fit() method,
            that just stores a copy of the Node._attribute ordered by Node._ids
        _node_size (list): list of floats that is created by the fit() method,
            that just stores the number of datapoints per node ordered by Node._ids
        _intersection_dict (dict): {(u, v, ...): [13, 15, 16, ...]} where (u, v, ...) is a tuple
            of node ids identifying a simplex and the value is a list of point labels
            belonging to the intersection common to the whole simplex
        _weights (dict): {(u, v, ...): 15} where (u, v, ...) is a tuple
            of node ids identifying a simplex and the value is the number of points
            belonging to the intersection common to the whole simplex
        _dimension (int): dimension of the complex
        _max_dim (int): maximal dimension of the complex. Simplices with dimension higher
            than self._max_dim will not be created. This attribute is needed as for
            self._max_dim >=12 the computational cost is often unbearable.
    """
    def __init__(self):
        self._simplices = set()
        self._facets = set()
        self._graph = nx.Graph()
        self._node_color = []
        self._node_size = []
        self._intersection_dict = {}
        self._weights = {}
        self._dimension = 0
        self._max_dim = 10

    def fit(self, cover, skeleton_only=True, verbose=1, max_dim=10):
        """this is the method that populates the attributes of the class.

        Args:
            cover (Cover): a cover that has been fitted and that have already been
                clustered

        Notes:
            implements the following submethods:
                _find_neighbours
                _complete_edges
                _find_higher_dimensonal_simplices
                    _is_complete
                    _find_intersection
                _rename_nodes

        """

        def _find_neighbours(node, cover, fiber):
            """Finds the neigbours of a node, fills its _neighbours attribute,
            sorts it by _id and returns the edges (sorted as well by _id).

            Args:
                node (Node): node we want to find the neighbours of
                cover (Cover): Cover object needed to have the intersecting fibers
                fiber (Fiber): Fiber that the node node belongs to

            Returns:
                edges (list): list of Egdes from the node to its neighbours

            """
            edges = []

            overlapping_fibers_indices = cover.intersecting_dict[fiber._fiber_index]
            for overlapping_fiber_index in overlapping_fibers_indices:
                overlapping_fiber = cover._fibers[overlapping_fiber_index]
                for mth_node in overlapping_fiber._nodes:
                    intersection = np.intersect1d(node._labels, mth_node._labels, assume_unique=True)
                    if len(intersection):
                        node._neighbours.append(mth_node)
                        edge = Edge(node._id, mth_node._id, intersection)
                        edges.append(edge)
            node.sort_neighbours()
            return edges

        def _complete_edges(_node):
            """Finds the edges between the neighbours of a node

            Args:
                node (Node): node

            Returns:
                new_edges (list): list of Edges

            """

            new_edges = []
            for n, m in combinations(_node._neighbours, 2):
                intersection = np.intersect1d(m._labels, n._labels, assume_unique=True)
                if len(intersection):
                    new_edges.append(Edge(n._id, m._id, intersection))
            return new_edges

        def _find_higher_dim_simplices(_node, edges):
            """finds higher (dim >= 2) dimensional simpleces from the set of edges
            between a node and all of its neighbours

            Args:
                node (Node): central node
                edges  (list): list of Edges between the central node and all of its
                    neighbours

            Returns:
                new_simplices (list): list of lists, where a simplex is represented by an
                    ordered list of node._ids
                temp_intersection_dict (dict): {(u, v, ...): [13, 15, 16, ...]} where (u, v, ...) is a tuple
                    of node ids identifying a simplex and the value is a list of point labels
                    belonging to the intersection common to the whole simplex
                temp_weights (dict): {(u, v, ...): 15} where (u, v, ...) is a tuple
                    of node ids identifying a simplex and the value is the number of points
                    belonging to the intersection common to the whole simplex

            """

            def _is_complete(_combination, all_edges):
                my_edges = [e for e in all_edges if e[0] in _combination and e[1] in _combination]
                if len(my_edges) == scipy.special.binom(len(_combination), 2):
                    return True
                return False

            def _check_intersection(__combination):
                n = __combination[0]
                m = __combination[1]
                inter = np.intersect1d(n._labels, m._labels, assume_unique=True)
                if len(__combination) >= 2:
                    for node in __combination[2:]:
                        inter = np.intersect1d(inter, node._labels, assume_unique=True)
                        if not len(inter):
                            return False, []
                    return True, inter
                return len(inter) >= 0, inter

            temp_intersection_dict = {}
            temp_weights = {}
            new_simplices = [e._edge for e in edges]
            vertices = [_node]+_node._neighbours
            vertices.sort(key=lambda x: x._id)
            for e in edges:
                temp_intersection_dict[tuple(e)] = e._intersection
                temp_weights[tuple(e)] = len(e._intersection)
            for i in range(3, min(self._max_dim, len(vertices))):
                for nodes in combinations(vertices, i):
                    nodes_ids = [c._id for c in nodes]
                    if (nodes_ids not in new_simplices):
                        if _is_complete(nodes, edges):
                            not_empty, intersection = _check_intersection(nodes)
                            if not_empty:
                                simplex = [node._id for node in nodes]
                                simplex.sort()
                                new_simplices.append(simplex)
                                temp_intersection_dict[tuple(simplex)] = intersection
                                temp_weights[tuple(simplex)] = len(intersection)
            return new_simplices, temp_intersection_dict, temp_weights

        def _rename_nodes(_cover):
            """renaming all the node ids in order to make them unique

            """
            cont = 0
            for fiber in _cover:
                for node in fiber._nodes:
                    node._id = cont
                    cont += 1

        # -------- fit() -------- #
        self._max_dim = max_dim

        if verbose >= 1:
            print("Maximum dimension of the complex limited to {}".format(self._max_dim))

        _rename_nodes(cover)
        for fiber in cover:
            if verbose >= 1:
                print("Processing fiber {}".format(fiber._fiber_index), end='\r')
            for node in fiber._nodes:
                edges = _find_neighbours(node, cover, fiber)
                # each edge is recomputed twice!
                # TODO: optimize it
                if not skeleton_only:
                    if edges:
                        edges += _complete_edges(node)
                        new_simplices, temp_intersection_dict, temp_weights = _find_higher_dim_simplices(node, edges)
                        for simplex in new_simplices:
                            self.add_simplex(simplex)
                            self.add_facet(simplex)  # such simplex is for sure a facet.
                            if len(simplex)-1 > self._dimension:
                                self._dimension = len(simplex)-1
                        # adding informations in order to have a nice visualization
                        self._node_color.append(node._attribute)
                        self._node_size.append(node.size)
                        self._intersection_dict.update(temp_intersection_dict)
                        self._weights.update(temp_weights)
                    else:
                        self.add_simplex([node._id])
                        self._node_color.append(node._attribute)
                        self._node_size.append(node.size)

                if skeleton_only:
                    if edges:
                        temp_intersection_dict = {}
                        temp_weights = {}
                        for e in edges:
                            simplex = tuple(e._edge)
                            temp_intersection_dict[simplex] = e._intersection
                            temp_weights[simplex] = len(e._intersection)
                            self.add_simplex(simplex)
                            self.add_facet(simplex)  # such simplex is for sure a facet.
                            if len(simplex)-1 > self._dimension:
                                self._dimension = len(simplex)-1
                        # adding informations in order to have a nice visualization
                        self._node_color.append(node._attribute)
                        self._node_size.append(node.size)
                        self._intersection_dict.update(temp_intersection_dict)
                        self._weights.update(temp_weights)
                    else:
                        self.add_simplex([node._id])
                        self._node_color.append(node._attribute)
                        self._node_size.append(node.size)
        if verbose >= 1:
            print("Successfully fittted the complex")

    def add_simplex(self, simplex):
        """This function sorts, creates all the lower dimensional simplices
        and add the input simplex to the _simpleces attribute of the class Complex.
        Furthermore, it updates the NetworkX graph with the 1-skeleton of the simplicial
        complex.

        Args:
            simplex (list): list of (int) representing node._ids
            intersection (list): list of point_labels in the intersection of the simplex

        """

        dimension = len(simplex)-1
        if dimension == 0:
            self._graph.add_node(simplex[0])
            return
        for i in range(dimension+1):
            for c in combinations(simplex, i+1):  # c is a tuple
                self._simplices.add(c)
                # adding to the NetworkX graph the edges
                if i == 1:
                    self._graph.add_edge(*c)

    def add_facet(self, facet):
        """
        Args:
            facet (list)

        """
        self._facets.add(tuple(facet))

    def plot(self, node_labels=True, edge_labels=False, node_color=None, _pos=None, nodes=None):
        """Plotting the graph
        Args:
            node_labels (bool): True if we want to plot the node labels
            edge_labels (bool): True if we want to plot the edge labels
            node_color (list): list of floats ordered in the same way you traverse the
                nodes in the cover

                Example:

                node_color = []
                for fiber in cover:
                    for node in fiber:
                        node_color.append(something(node))

        """
        pos = nx.spring_layout(self._graph, iterations=800)

        M = max(self._node_size)
        normalized_node_size = [x/M*400 for x in self._node_size]
        if _pos == 'attribute':
            for nodeid in nodes:
                node = nodes[nodeid]
                pos[nodeid][0] = node._attribute
        if not node_color:
            nx.draw(self._graph, pos=pos,
                    node_color=self._node_color, node_size=normalized_node_size)
        else:
            nx.draw(self._graph, pos=pos,
                    node_color=node_color, node_size=normalized_node_size)
        if edge_labels:
            nx.draw_networkx_edge_labels(self._graph, pos=pos,
                                         edge_labels=self._weights)
        if node_labels:
            nx.draw_networkx_labels(self._graph, pos, font_size=8, font_weight='bold', font_color='k')
        plt.show()


if __name__ == '__main__':
    import pickle
    from filter import Filter
    from cover import Cover
    from cluster import Cluster

    def test():
        data = pickle.load(open('/Users/martinomilani/Documents/III_semester/PACS/pymapper/data.pickle', 'rb'))
        f = Filter.factory('Projection')
        cover = Cover.factory('UniformCover')
        filter_values = f(data)
        cover.fit(filter_values, data)
        cluster = Cluster.factory('Linkage')

        for fiber in cover:
            cluster(fiber)
            print(len(fiber._nodes))
            print('OK\n')

        complex = Complex()

        print(cover.intersecting_dict)

        complex.fit(cover)
        complex.plot()
        """testing _rename_nodes
        complex._rename_nodes(cover)
        for fiber in cover:
            for node in fiber:
                print(node._id)

        testing _find_neighbours()
        fiber = cover._fibers[2]
        first_node = fiber._nodes[2]
        plt.plot([x[0] for x in fiber._points], [x[1] for x in fiber._points], 'ro')
        plt.plot([x[0] for x in data[first_node._labels]], [x[1] for x in data[first_node._labels]], 'bo')
        plt.show()
        edges = complex._find_neighbours(first_node, cover, fiber)
        print('node id:', first_node._id)
        print('# of edges:', len(edges))
        for i in range(len(edges)):
            print(edges[i][0], edges[i][1])
        print('# of neighbours: ', len(first_node._neighbours))
        for i in range(len(first_node._neighbours)):
            print(first_node._neighbours[i]._id)

        testing _complete_edges
        edges += complex._complete_edges(first_node)
        print('# of edges:', len(edges))
        for i in range(len(edges)):
            print(edges[i][0], edges[i][1])

        testing _find_higher_dim_simpleces
        new_simplices = complex._find_higher_dim_simplices(first_node, edges)
        print('# of edges:', len(new_simplices))
        for i in range(len(new_simplices)):
            print(new_simplices[i][0], new_simplices[i][1])"""

    test()
