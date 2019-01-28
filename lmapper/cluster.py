"""Author: Martino Milani"""

from scipy.cluster.hierarchy import linkage
from lmapper.complex import Node
from scipy.spatial.distance import pdist
from lmapper.cutoff import FirstGap
import numpy as np


class Cluster():
    """Abstract class implementing the interface for a Clustering algorithm and a factory
    """

    def __call__(self, fiber, verbose):
        raise NotImplementedError()

    def get_params(self):
        raise NotImplementedError()

    def set_params(self):
        raise NotImplementedError()

    @staticmethod
    def factory(type):
        assert type in [cluster.__name__ for cluster in Cluster.__subclasses__()]
        if type == "Linkage":
            return Linkage()
        assert 0, "Bad cluster creation: " + type


class ClusterInfo():
    """Class to implement a data structure to be stored in each Fiber object
    """
    def __init__(self, method, cutoff, metric, Z, eps, num_clust):
        self.method = method
        self.cutoff = cutoff
        self.metric = metric
        self.Z = Z
        self.eps = eps
        self.num_clust = num_clust

    def __repr__(self):
        return ('method: {}\n'
                'metric: {}\n'
                'eps: {}\n'
                'number of cluster: {}'.format(
                    self.method,
                    self.metric,
                    self.eps,
                    self.num_clust))


class Linkage(Cluster):
    """
    Attributes:
        _method (str): method (single, complete, etc. etc.)

    Notes:
        it modifies the Fiber by adding a ClusterInfo, and creating the
        Nodes in the Fiber._nodes. Nodes are created with an _id attribute initialized
        to None. This attribute is changed by the call to the complex.fit(cover) method,
        that has access to the cover and modifies the node ids.
    """

    def __init__(self, method='single', metric='euclidean', cutoff=FirstGap(0.1)):
        self._method = method
        self._cutoff = cutoff
        self._metric = metric

    def __call__(self, fiber, verbose=1):
        # cleaning the fiber in case the mapper object was already fitted
        # and this call is after a call to Mapper.set_params() method
        fiber._nodes = []
        fiber._clusterinfo = None

        # shortcut
        if not len(fiber._points):
            return 0
        if len(fiber._points) == 1:
            fiber._nodes.append(Node(fiber._pointlabels[0], fiber._filtervalues[0], fiber._fiber_index))
            return 0

        # performing the clustering step
        if self._cutoff.__name__ == "FirstGap":
            compressed_distances = pdist(fiber._points, metric=self._metric)
            R = np.max(compressed_distances)
            Z = linkage(compressed_distances, method=self._method)
            clusters, eps, num_clust = self._cutoff.num_clusters(Z, R)

            if verbose >= 1:
                print("Fiber {0:3d}\nR = {1:0.3f}, eps={2:0.3f}\nfound {3:3d} clusters\n".format(
                    fiber._fiber_index, R, eps, num_clust))

        else:
            assert 0, "cutoff not implemented yet!"

        # instantiating the Node objects
        for i in range(len(set(clusters))):
            submask = [c == i for c in clusters]  # this submask identify the original
            # point labels starting from the sublist of point labels contained in the
            # fiber. This is necessary to maintain a coherent notion of "labels"
            labels = fiber._pointlabels[submask]
            attribute = np.median(fiber._filtervalues[submask])
            fiber._nodes.append(Node(labels, attribute, fiber._fiber_index))

        fiber._clusterinfo = ClusterInfo(self._method,
                                         self._cutoff,
                                         self._metric,
                                         Z,
                                         eps,
                                         num_clust)
        return fiber

    def get_params(self):
        dict = {"method": self._method,
                "cutoff": self._cutoff,
                "metric": self._metric}
        return dict

    def set_params(self, method):
        self._method = method


_all_clusters_ = [f.__name__ for f in Cluster.__subclasses__()]


if __name__ == '__main__':
    import pickle
    from filter import Filter
    from cover import Cover
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt

    def test():
        data = pickle.load(open('/Users/martinomilani/Documents/III_semester/PACS/shapegraph/data.pickle', 'rb'))
        f = Filter.factory('Projection')
        cover = Cover.factory('BalancedCover')
        filter_values = f(data)
        cover.fit(filter_values, data)
        cluster = Cluster.factory('Linkage')

        for fiber in cover:
            plt.plot([x[0] for x in fiber._points], [x[1] for x in fiber._points], 'ro')
            plt.show()
            cluster(fiber)
            print(len(fiber._nodes))

            print('\nOK')

    test()
