//
//  Filter.hpp
//  
//
//  Created by Martino Milani on 26.05.19.
//

#ifndef Filter_h
#define Filter_h

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

class Filter{
    // Abstract base class from which all filters have to inherit to obey the API.
    // Furthermore, it implements a simple static method as factory, in order to be easily
    // ported to Python
public:
    Filter(int n): nthreads(n){};
    virtual ~Filter() = default;
    virtual py::array_t<double> operator()(py::array_t<double>& data) = 0;
    static std::unique_ptr<Filter> factory(std::string filter_name);
protected:
    int nthreads;
};

#endif /* Filter_h */
