Lmapper
-------

This package implements the Mapper algorithm.

Example:

    >>> import lmapper as lm
    >>> mapper = lm.Mapper(data=x,
    >>>                    filter='Projection',
    >>>                    cluster='Linkage',
    >>>                    cover='BalancedCover)
    >>> mapper.fit()
    >>> mapper.plot()
