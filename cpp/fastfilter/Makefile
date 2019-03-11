example: fastfilter.cpp
	clang++ -O3 -Wall -shared -std=c++11  -g -I/Users/Mart/anaconda3/envs/pdm/include -mmacosx-version-min=10.9 -m64 -fPIC -Ipybind11/include -I/Users/Mart/anaconda3/envs/pdm/include/python3.7m -undefined dynamic_lookup -Xpreprocessor -fopenmp -lomp `python3 -m pybind11 --includes` fastfilter.cpp -o fastfilter`python3-config --extension-suffix`
