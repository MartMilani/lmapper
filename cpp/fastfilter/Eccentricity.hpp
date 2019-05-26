//
//  Eccentricity.hpp
//  
//
//  Created by Martino Milani on 26.05.19.
//

#ifndef Eccentricity_hpp
#define Eccentricity_hpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "Filter.hpp"

namespace py = pybind11;

class Eccentricity: public Filter{
    // Filter implementing the eccentricity
public:
    Eccentricity(int nthreads, int e, std::string m): Filter(nthreads), exponent(e), metric(m){};
    py::array_t<double> operator()(py::array_t<double>& data) override;
private:
    py::array_t<double> my_distance(py::array_t<double>& data) const;
    py::array_t<double> eccentricity(py::array_t<double>& dm) const;
    
    int exponent;
    std::string metric;
};

#endif /* Eccentricity_hpp */
