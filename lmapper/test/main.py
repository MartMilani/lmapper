import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

from lmapper.cutoff import FirstGap
import predmap as mapp
from lmapper.cluster import Linkage
from lmapper.cover import UniformCover
from lmapper.filter import Projection
import lmapper as lm
import numpy as np



x = [0, 0, 0, 0, 0, 10, 10, 10, 20, 20, 30, 30, 30, 40, 40, 50, 50, 50, 60, 60]
y = [0, 1, 2, 3, 4, 0, 1, 3, 1, 1.5, 1, 2, 3, 1.5, 2, 0.5, 1, 2, 1, 2]
plt.plot(x, y, 'bo')
plt.show()

data = np.asarray([[x0, y0] for x0, y0 in zip(x, y)])
data
filter = Projection(ax=1)
cover = UniformCover(nintervals=4,
                     overlap=0.2)
cluster = Linkage(method='single',
                  metric='euclidean',
                  cutoff=FirstGap(0.1))
mapper = lm.Mapper(data=data,
                   filter=filter,
                   cover=cover,
                   cluster=cluster)
mapper.fit().plot()
for fiber in mapper.cover:
    print(len(fiber._nodes))
    print(fiber._clusterinfo)
