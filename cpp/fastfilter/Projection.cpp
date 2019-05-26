//
//  Projection.cpp
//  
//
//  Created by Martino Milani
//

#include "Projection.hpp"


py::array_t<double> Projection::operator() (py::array_t<double>& data){
    // parallel implementation with OpenMP of the projection on one axis
    // Args:
    //      data (py::array_t<double>)
    // Returns:
    //      filter_values (py::array_t<double>)
    // Note:
    //      it is important to remember that NumPy arrays can be stored in memory
    //      not contiguously! this can break this code, since this implementations gives
    //       for granted that 'data' is actually stored contiguously in memory.
    
    // releasing the GIL before a CPU-bound snippet that
    // does not interact with any python object
    // (no data races on refcount: does not create neither destroy any PyObject)
    py::gil_scoped_release release;
    
    py::buffer_info data_info = data.request();
    /* Just to remember what a py::buffer_info is:
     struct buffer_info {
     void *ptr;
     size_t itemsize;
     std::string format;
     int ndim;
     std::vector<size_t> shape;
     std::vector<size_t> strides;
     }; */
    py::array_t<double> filter_values(data_info.shape[0]);
    py::buffer_info filter_info = filter_values.request();
    auto data_ptr = static_cast<double *>(data_info.ptr);
    auto filter_ptr = static_cast<double *>(filter_info.ptr);
    const int N = data_info.shape[0];
    const int PAD = data_info.shape[1];
    omp_set_num_threads(nthreads);
#pragma omp parallel
    {
        // splitting the work between threads
        const int whoami = omp_get_thread_num();
        const int stride = N/nthreads;
        const int start = stride*whoami;
        int end = 0;
        if (whoami!= omp_get_num_threads()-1)
            end = start + stride;
        else
            end = N;
        // projecting
        for (int i = start; i < end; i++)
            filter_ptr[i] = data_ptr[i*PAD + axis];
    }
    return filter_values;
    
};
