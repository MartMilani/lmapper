import numpy as np


def fcluster(Z, num_clust):
    '''Taken from Daniel Mullner's Python Mapper.

    Generate cluster assignments from the dendrogram Z. The parameter
    num_clust specifies the exact number of clusters. (The method in SciPy
    does not always produce the exact number of clusters, if several heights
    in the dendrogram are equal, or if singleton clustering is requested.)

    This method starts labeling clusters at 0 while the SciPy indices
    are 1-based.'''
    assert isinstance(num_clust, (int, np.integer))
    N = np.alen(Z) + 1
    assert 0 < num_clust <= N

    if num_clust == 1:  # shortcut
        return np.zeros(N, dtype=np.int)

    Z = Z[:N - num_clust, :2].astype(np.int)

    # Union-find data structure
    parent = np.empty(2 * N - num_clust, dtype=np.int)
    parent.fill(-1)

    for i, (a, b) in enumerate(Z):
        parent[a] = parent[b] = N + i

    clust = np.empty(N, dtype=np.int)
    for i in range(N):
        idx = i
        if (parent[idx] != -1):  # a → b
            p = idx
            idx = parent[idx]
            if (parent[idx] != -1):  # a → b → c
                while True:
                    idx = parent[idx]
                    if parent[idx] == -1:
                        break
                while True:
                    parent[p], p = idx, parent[p]
                    if parent[p] == idx:
                        break
        clust[i] = idx

    # clust contains cluster assignments, but not necessarily numbered
    # 0...num_clust-1. Relabel the clusters.
    idx = np.unique(clust)
    idx2 = np.empty_like(parent)
    idx2[idx] = np.arange(idx.size)

    return idx2[clust]


class Cutoff():
    @staticmethod
    def check_input(heights, diam):
        '''Input checking'''
        assert isinstance(heights, np.ndarray)
        assert heights.ndim == 1
        assert heights.dtype == np.float
        assert np.min(heights) >= 0
        assert isinstance(diam, (float, np.floating))
        if diam < np.max(heights):
            import pdb
            pdb.set_trace()


class FirstGap(Cutoff):
    '''
    Look for the first gap of size or bigger in the
    heights of the clustering tree.
    Args:
        gap (float): gap size
    '''

    def __init__(self, gap):
        self.__name__ = "FirstGap"
        assert isinstance(gap, (float, np.floating))
        assert 0 < gap < 1
        self.gap = gap

    def __str__(self):
        return 'First gap of relative width {0}'.format(self.gap)

    def num_clusters(self, Z, diam):
        '''
        Args:
            heights (nump.ndarray(n, dtype=float)): vector of heights at which the
            dendogram collapsed clusters
            diam (float): The diameter of the data set, ie the maximal pairwise
            distance between points.

        Returns:
            (int): number of clusters
        '''
        heights = Z[:, 2]
        self.check_input(heights, diam)

        # differences between subsequent elements (and heights[0] at the
        # beginning)
        diff = np.ediff1d(heights, to_begin=heights[0])
        gap_idx, = np.nonzero(diff >= self.gap * diam)
        if gap_idx.size:
            num_clust = heights.size + 1 - gap_idx[0]
        else:
            # no big enough gap -> one single cluster
            num_clust = 1
        clusters = fcluster(Z, num_clust)
        if num_clust > 1:
            eps = (Z[np.alen(Z)-num_clust, 2] + Z[np.alen(Z)-num_clust + 1, 2])/2
        else:
            eps = Z[-1, 2]
        return clusters, eps, num_clust
