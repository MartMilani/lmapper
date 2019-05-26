//
//  Projection.hpp
//  
//
//  Created by Martino Milani
//

#ifndef Projection_h
#define Projection_h

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "Filter.hpp"

namespace py = pybind11;

class Projection: public Filter{
    // Filter implementing the projection on one axis
public:
    Projection(int ax, int nthreads) : Filter(nthreads), axis(ax){};
    py::array_t<double> operator() (py::array_t<double>& data) override;
private:
    int axis;
};

#endif /* Projection_h */
