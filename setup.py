from distutils.core import setup, Extension
import os


os.environ["CXX"] = "clang++"
os.environ["CC"] = "clang++"


def readme():
    with open('README.rst') as f:
        return f.read()


compiler_args = ['-std=c++11',
                 '-stdlib=libc++',
                 '-mmacosx-version-min=10.7',
                 '-m64',
                 '-fPIC',
                 '-Xpreprocessor',
                 '-fopenmp'
                 ]

linker_args = ['-Xpreprocessor',
               '-fopenmp',
               '-lomp']

setup(name='lmapper',
      version='0.1',
      description=('This is a Python implementation of the TDA Mapper algorithm'
                   ' for visualization of high-dimensional data'),
      long_description='',
      url='http://github.com/MartMilani/lmapper',
      author='Martino Milani',
      author_email='martino.milani94@gmail.com',
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
                             include_dirs=['pybind11/include'],
                             language='c++11',
                             extra_compile_args=compiler_args,
                             extra_link_args=linker_args
                             )],
      package_dir={'lmapper': 'lmapper'},
      package_data={'lmapper': ['datasets/*.csv', 'test/*.py', 'example/*.py']})
