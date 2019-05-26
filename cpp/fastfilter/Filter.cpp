//
//  Filter.cpp
//  
//
//  Created by Martino Milani on 26.05.19.
//

#include <stdio.h>
#include "Filter.hpp"


std::unique_ptr<Filter> Filter::factory(std::string filter_name)
{
    if (filter_name == "Projection")
        return std::unique_ptr<Filter>(new Projection(0, 1));
    else if (filter_name == "Eccentricity")
        return std::unique_ptr<Filter>(new Eccentricity(1, 1, "euclidean"));
    
    std::cout<<"Wrong filter specifications: creating a Projection filter"<<std::endl;
    return std::unique_ptr<Filter>(new Projection(0, 1));
}
