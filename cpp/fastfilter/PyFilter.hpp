//
//  PyFilter.hpp
//  
//
//  Created by Martino Milani
//
#ifndef PyFilter_h
#define PyFilter_h

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "Filter.hpp"

namespace py = pybind11;

class PyFilter : public Filter {
    // Filter is not constructible, and we clearly (?) require some kind of “trampoline”
    // that redirects virtual calls back to Python, as explained in more details in
    // https://pybind11.readthedocs.io/en/stable/advanced/classes.html
public:
    // Inherits the constructor
    using Filter::Filter;
    
    // Trampoline
    py::array_t<double> operator() (py::array_t<double>& data) override {
        PYBIND11_OVERLOAD_PURE_NAME(
                                    py::array_t<double>,    // Return type
                                    Filter,                 // Parent class
                                    "__call__",             // Name of function in Python
                                    "operator()",           // Name of function in C++
                                    data                    // Argument(s)
                                    );
    }
};


#endif
