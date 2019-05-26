//
//  wrap.cpp
//  
//
//  Created by Martino Milani on 26.05.19.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "Filter.hpp"
#include "PyFilter.hpp"
#include "Projection.hpp"
#include "Eccentricity.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fastfilter, m) {
    py::class_<Filter, PyFilter> filter(m, "Filter");
    filter
    .def(py::init<int>())
    .def("__call__", &Filter::operator() )
    .def("factory", &Filter::factory);
    py::class_<Projection>(m, "Projection", filter)
    .def(py::init<int, int>())
    .def("__call__", &Projection::operator() );
    py::class_<Eccentricity>(m, "Eccentricity", filter)
    .def(py::init<int, int, std::string>())
    .def("__call__", &Eccentricity::operator() );
}
