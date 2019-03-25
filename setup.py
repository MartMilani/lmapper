from distutils.core import setup, Extension
import os


os.environ["CXX"] = "clang++"
os.environ["CC"] = "clang++"

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
      tests_require=['nose'],
      ext_modules=[Extension('filterutils', ['cpp/filterutils/filterutils.cpp'],
                             extra_compile_args=["-O3 -Wall -shared -std=c++11  -I/Users/Mart/anaconda3/envs/pdm/include -mmacosx-version-min=10.9 -m64 -fPIC -Ipybind11/include -I/Users/Mart/anaconda3/envs/pdm/include/python3.7m -undefined dynamic_lookup -Xpreprocessor -fopenmp -lomp `python3 -m pybind11 --includes`"])])
