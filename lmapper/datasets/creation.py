from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math
from scipy.stats import bernoulli
from scipy.spatial.distance import cdist
import pandas as pd


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

r = 1
N = 400
phi = 2 * math.pi * np.random.random(N)
costheta = 2 * np.random.random(N) - 1
u = np.random.random(N)

theta = np.arccos(costheta)

# Scatter graph
X = r * np.cos(phi)
Y = r * np.sin(phi)
Z = np.random.uniform(3, 4.5, N)


phi = 2 * math.pi * np.random.random(N)
costheta = 2 * np.random.random(N) - 1
u = np.random.random(N)

theta = np.arccos(costheta)
X1 = r * np.cos(phi)
Y1 = r * np.sin(phi)
Z1 = np.random.uniform(-4.5, -3, N)

r = 3
n = 4000
phi = 2 * math.pi * np.random.random(n)
costheta = 2 * np.random.random(n) - 1
u = np.random.random(n)
theta = np.arccos(costheta)
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


c1 = np.column_stack((X, Y, Z))
c2 = np.column_stack((X1, Y1, Z1))
c = np.column_stack((x, y, z))

print(np.where(c[:, 0] > 1)[0])

ro = 1.5
w = 0.1

c = np.delete(c, np.where(c[:, 0]**2 + c[:, 2]**2 < ro)[0], axis=0)

c1 = np.delete(c1, np.where(c1[:, 0]**2 < w), axis=0)
c2 = np.delete(c2, np.where(c2[:, 0]**2 < w), axis=0)

v = np.concatenate((c, c1, c2))

print(v.shape)
print(c[:, 0])

centersxy = np.array([0, 0])
centersz = np.array([-4, 4])
centers = np.column_stack((centersxy, centersxy, centersz))


metricpar = {'metric': 'euclidean'}

outcome = np.zeros(len(v))

for i in np.arange(len(outcome)):

    if v[i, 2] > 0:

        p = 1 / (cdist(v[i:i + 1], centers[1:2], **metricpar)[0][0]**(1 / 1.1))
        #p = math.sqrt(p)

        outcome[i] = bernoulli.rvs(p, size=1)

    else:
        p = 1 / (cdist(v[i:i + 1], centers[0:1], **metricpar)[0][0]**(1 / 1.1))
        #p = math.sqrt(p)

        outcome[i] = bernoulli.rvs(p, size=1)


print(np.mean(outcome[v[:, 2]**2 <= 9]))
print(np.mean(outcome[v[:, 2] < -3]))
print(np.mean(outcome[v[:, 2] > 3]))

print(type(v))
'''
ax.scatter(c[:,0],c[:,1],c[:,2], c= outcome[v[:,2]**2 <= 9])
ax.scatter(c2[:,0],c2[:,1],c2[:,2], c = outcome[v[:,2]<-3])
ax.scatter(c1[:,0], c1[:,1], c1[:,2], c = outcome[v[:,2]>3])

plt.show()
'''

disturb = np.random.normal(0, scale=0.08, size=(len(v), 3))

turb = v + disturb

ax.scatter(turb[:, 0], turb[:, 1], turb[:, 2], c=outcome)

plt.show()

v = turb

v = np.column_stack((v, outcome))
v = pd.DataFrame(v, columns=('x', 'y', 'z', 'outcome'))
print(v.shape)

v.to_csv('synthetic.csv', index=False)
