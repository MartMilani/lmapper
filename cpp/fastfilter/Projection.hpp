//
//  Projection.hpp
//  
//
//  Created by Martino Milani on 26.05.19.
//

#ifndef Projection_h
#define Projection_h

#include "fastfilter.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

class Projection: public Filter{
    // Filter implementing the projection on one axis
public:
    Projection(int ax, int nthreads) : Filter(nthreads), axis(ax){};
    py::array_t<double> operator() (py::array_t<double>& data) override;
private:
    int axis;
};

#endif /* Projection_h */
