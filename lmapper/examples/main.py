"""Author: maritno milani
m.milani@l2f.ch

Test routine for the mapper module
"""
import numpy as np
import lmapper as lm
from lmapper.filter import Eccentricity
from lmapper.cover import BalancedCover
from lmapper.cluster import Linkage
import predmap as mapp
from lmapper.cutoff import FirstGap


def test(xs, y):
    """Basic usage"""
    # instantiate a Mapper object
    x = np.empty(xs.shape)
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            x[i, j] = xs[i, j]
    print(x.shape)

    filter = Eccentricity(exponent=2, metric='correlation')
    cover = BalancedCover(nintervals=4,
                          overlap=0.49)
    cluster = Linkage(method='average',
                      metric='correlation',
                      cutoff=FirstGap(0.01))
    mapper = lm.Mapper(data=x,
                       filter=filter,
                       cover=cover,
                       cluster=cluster)
    mapper.fit(skeleton_only=False).plot()
    print("dimension = ", mapper.complex._dimension)

    predictor = mapp.BinaryClassifier(mapper=mapper,
                                      response_values=y,
                                      _lambda=0.015,
                                      a=0.5,
                                      beta=1)
    predictor.fit().plot_majority_votes()
    # --------------------------
    # predictor.plot_majority_votes()
    return predictor.leave_one_out(x)


def main():
    import numpy as np
    import pandas as pd
    data = pd.read_csv('../datasets/data.csv')
    x = data[data.columns[2:-1]].values
    y = data[data.columns[1]].values
    y_ = np.asarray([0 if x == 'M' else 1 for x in y])
    pred = test(x, y_)
    print('ACCURACY: ', np.mean(pred == y_))

    return 0


if __name__ == '__main__':
    main()
