from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='lmapper',
      version='0.1',
      description=('This is a Python implementation of the TDA Mapper algorithm'
                   ' for visualization of high-dimensional data'),
      long_description='',
      url='http://github.com/MartMilani/lmapper',
      author='Martino Milani',
      author_email='matrino.milani94@gmail.com',
      license='MIT',
      packages=['lmapper'],
      install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'networkx',
        'matplotlib',
        'pybind11'
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
