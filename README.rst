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


How to install on Mac OS High Sierra 10.13.6
--------------------------------------------

The filterutils package uses OpenMP: to install it, run

$ brew install openmp

This installation procedure is meant to be used as indicated in https://iscinumpy.gitlab.io/post/omp-on-high-sierra/

Once installed openmp as above, run the command

$ cd <path_to_this_directory>
$ pip install .

this command will read the setup.py file in this directory and install both lmapper,
and filterutils

Check your installation
-----------------------

Run the following tests:

$ python lmapper/test/test_filterutils.py
$ python lmapper/test/test1.py

If no exeption is raised, the installation should be successful
