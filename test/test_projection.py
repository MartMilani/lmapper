from unittest import TestCase
from lmapper.filter import Projection
import numpy as np


class TestJoke(TestCase):
    def test_proj(self):
        x = np.random.multivariate_normal(mean=[0, 0, 0], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size=100)
        f = Projection()
        filter_values = f(x)
        self.assertTrue((filter_values == np.array([row[0] for row in x])).all())
