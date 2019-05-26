//
//  Eccentricity.cpp
//  
//
//  Created by Martino Milani
//

#include "Eccentricity.hpp"

py::array_t<double> Eccentricity::operator()(py::array_t<double>& data){
    // releasing the GIL before a CPU-bound code that
    // does not interact with any python object
    // (no data races on refcount: does not create neither destroy any PyObject)
    py::gil_scoped_release release;
    py::array_t<double> dm = my_distance(data);
    py::array_t<double> ecc = eccentricity(dm);
    return ecc;
}


py::array_t<double> Eccentricity::my_distance(py::array_t<double>& data) const{
    // Note:
    //      Beware: this method runs without holding the GIL
    std::cerr<<"yo"<<std::endl;
    py::buffer_info data_info = data.request();
    std::cerr<<"yo"<<std::endl;
    // allocating result
    int n = data_info.shape[0];
    py::array_t<double> dm({n,n});
    std::cerr<<"yo"<<std::endl;
    py::buffer_info dm_info = dm.request();
    auto data_ptr = static_cast<double *>(data_info.ptr);
    auto dm_ptr = static_cast<double *>(dm_info.ptr);
    
    const int N = data_info.shape[0];
    const int PAD = data_info.shape[1];
    // initiating intermediate data structures for correlation metric
    std::vector<double> data_normalized;
    std::vector<double> std_dev;
    
    omp_set_num_threads(nthreads);
    std::cerr<<"yo"<<std::endl;
    if (metric == "correlation"){
        // ---- calculating std_dev and data_normalized ----
        data_normalized.reserve(N*PAD);
        std_dev.reserve(N);
        
#pragma omp parallel
        {
            const int whoami = omp_get_thread_num();
            const int stride = N/nthreads;
            const int start = stride*whoami;
            
            int end = 0;
            if (whoami!= omp_get_num_threads()-1)
                end = start + stride;
            else
                end = N;
            
            double mean = 0.0;
            for (int i = start; i < end; i++){
                std_dev[i] = 0.0;
                mean = 0.0;
                for (int k = 0; k < PAD; k++)
                    mean += data_ptr[i*PAD+k];
                mean *= 1.0/PAD;
                for (int k = 0; k < PAD; k++){
                    data_normalized[i*PAD+k] = data_ptr[i*PAD+k]-mean;
                    std_dev[i] += std::pow(data_normalized[i*PAD+k], 2);
                }
                std_dev[i] = std::sqrt(std_dev[i]);
            }
        }
    }
    std::cerr<<"yo"<<std::endl;
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
        
        // here the computations come
        std::cerr<<"yo"<<std::endl;
        double sum = 0;
        if (metric == "euclidean"){
            for (int i = start; i < end; i++){
                for (int j = i+1; j < N; j++){
                    sum = 0.;
                    for (int k = 0; k < PAD; k++)
                        sum += std::pow(data_ptr[i*PAD+k]-data_ptr[j*PAD+k], 2);
                    dm_ptr[i*N + j] = std::sqrt(sum);
                    dm_ptr[j*N + i] = dm_ptr[i*N + j];
                }
            }
        }
        if (metric == "correlation")
            for (int i = start; i < end; i++){
                for (int j = i+1; j < N; j++){
                    sum = 0.;
                    for (int k = 0; k < PAD; k++)
                        sum += data_normalized[i*PAD+k] * data_normalized[j*PAD+k];
                    dm_ptr[i*N + j] = 1.0 - sum/(std_dev[i]*std_dev[j]);
                    dm_ptr[j*N + i] = dm_ptr[i*N + j];
                }
            }
    }
    return dm;
    
}

py::array_t<double> Eccentricity::eccentricity(py::array_t<double>& dm) const{
    // Note:
    //      this method runs without holding the GIL
    py::buffer_info dm_info = dm.request();
    // instantiating the result
    py::array_t<double> ecc(dm_info.shape[0]);
    py::buffer_info ecc_info = ecc.request();
    /*  Just to remember what a buffer_info is:
     struct buffer_info {
     void *ptr;
     size_t itemsize;
     std::string format;
     int ndim;
     std::vector<size_t> shape;
     std::vector<size_t> strides;
     }; */
    auto dm_ptr = static_cast<double *>(dm_info.ptr);
    auto ecc_ptr = static_cast<double *>(ecc_info.ptr);
    
    int N = dm_info.shape[0];
    omp_set_num_threads(nthreads);
#pragma omp parallel
    {
        // splitting the work between threads
        int whoami = omp_get_thread_num();
        int stride = N/nthreads;
        int start = stride*whoami;
        int end = 0;
        if (whoami!= omp_get_num_threads()-1)
            end = start + stride;
        else
            end = N;
        if (exponent == -1){  // -1 stands for inf
            for (int i = start; i <end; i++)
                for (int j = 0; j<N; j++){
                    double& el = dm_ptr[i*N+j];
                    if (el >= ecc_ptr[i])
                        ecc_ptr[i] = el;
                }
        }
        else if (exponent == 1){
            double sum = 0;
            for (int i = start; i <end; i++){
                sum = 0;
                for (int j = 0; j<N; j++)
                    sum += dm_ptr[i*N+j];
                ecc_ptr[i] = sum/N;
            }
        }
        else{
            double sum = 0;
            for (int i = start; i <end; i++){
                sum = 0;
                for (int j = 0; j<N; j++)
                    sum += std::pow(dm_ptr[i*N+j], exponent);
                ecc_ptr[i] = std::pow(sum/N, 1./exponent);
            }
        }
    }
    return ecc;
}
