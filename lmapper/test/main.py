import pdb
import sys
sys.path.append(
    '/Users/martinomilani/Documents/III_semester/PACS/project/shapegraph')
sys.path.append(
    '/Users/martinomilani/Documents/III_semester/PACS/project/predictive_shapegraph')
from cutoff import FirstGap
import mapperpredictor as mapp
from cluster import Linkage
from cover import UniformCover, BalancedCover
from filter import Eccentricity, Projection
import shapegraph as sg
import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt


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
shapegraph = sg.ShapeGraph(data=data,
                           filter=filter,
                           cover=cover,
                           cluster=cluster)
shapegraph.fit().plot(_pos='attribute')
for fiber in shapegraph.cover:
    print(len(fiber._nodes))
    print(fiber._clusterinfo)
