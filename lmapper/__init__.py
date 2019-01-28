
__version__ = '0.1'
__date__ = 'January 28, 2019'

import sys
if sys.hexversion < 0x03000000:
    raise ImportError('Mapper requires at least Python version 3.0.')
del sys


from lmapper._mapper import *

from lmapper import filter
from lmapper import cover
from lmapper import cutoff
from lmapper import cluster
