#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
namespace py = pybind11;
#include "filterutils.hpp"


// binding code
PYBIND11_MODULE(filterutils, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("eccentricity", [](py::array_t<double> dm, py::array_t<double> ecc, int exponent, int nthreads) {
        /* Release GIL before calling into C++ code, since we are not creating or destroying
        any Python object we won't mess up with the reference count and thus we are safe from
        data races */
        py::gil_scoped_release release;
        return eccentricity(dm, ecc, exponent, nthreads);
    });
    m.def("my_distance", [](py::array_t<double> data, py::array_t<double> dm, int nthreads, std::string metric) {
        /* Release GIL before calling into C++ code, since we are not creating or destroying
        any Python object we won't mess up with the reference count and thus we are safe from
        data races */
        py::gil_scoped_release release;
        return my_distance(data, dm, nthreads, metric);
      });
}
