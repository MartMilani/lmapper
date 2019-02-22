from distutils.core import setup, Extension


cpp_args = ['-std=c++11', '-std=libc++', '-mmacosx-version-min=10.7', '-Xpreprocessor', '-fopenmp', '-lomp']

ext_modules = [
    Extension(
        'pi',
        ['code.cpp'],
        include_dirs=['pybind11/include'],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='pi',
    version='0.0.1',
    author='Cliburn Chan',
    author_email='cliburn.chan@duke.edu',
    description='Example',
    ext_modules=ext_modules,
)
