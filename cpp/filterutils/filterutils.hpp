#ifndef filterutils_h
#define filterutils_h

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void my_distance(py::array_t<double> data, py::array_t<double> dm, int nthreads, std::string metric);
void eccentricity(py::array_t<double> dm, py::array_t<double> ecc, double exponent, int nthreads);

#endif
