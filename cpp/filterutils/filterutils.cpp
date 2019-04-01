#include <cmath>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <algorithm>
namespace py = pybind11;

// Passing in an array of doubles

void my_distance(py::array_t<double> data, py::array_t<double> dm, int nthreads, std::string metric){
    py::gil_scoped_acquire acquire;

    py::buffer_info data_info = data.request();
    py::buffer_info dm_info = dm.request();
    /* here we remind the structure of a buffer_info struct:

    struct buffer_info {
        void *ptr;
        size_t itemsize;
        std::string format;
        int ndim;
        std::vector<size_t> shape;
        std::vector<size_t> strides;
    }; */

    // data_ptr and dm_ptr are pointers to double to the array where
    // the data of the corresponding numpy array is stored
    auto data_ptr = static_cast<double *>(data_info.ptr);
    auto dm_ptr = static_cast<double *>(dm_info.ptr);

    // storing the dimensions of the arrays
    const int N = data_info.shape[0];
    const int PAD = data_info.shape[1];

    // initializing the data structures we need in case the
    // correlation distance is used
    std::vector<double> data_normalized;
    std::vector<double> std_dev;

    // setting the number of threads
    omp_set_num_threads(nthreads);

    if (metric == "correlation"){
        //
        // ---- calculating std_dev and data_normalized ----
        //

        // reserving the correct amount of memory
        data_normalized.reserve(N*PAD);
        std_dev.reserve(N);

        #pragma omp parallel
        {
        // these 8 lines split the job between threads
        const int whoami = omp_get_thread_num();
        const int stride = N/nthreads;
        const int start = stride*whoami;
        int end = 0;
        if (whoami!= omp_get_num_threads()-1)
            end = start + stride;
        else
            end = N;

        // here the data_normalized and the std_dev arrays are computed
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
    #pragma omp parallel
    {
    // these 8 lines split the job between threads
    const int whoami = omp_get_thread_num();
    const int stride = N/nthreads;
    const int start = stride*whoami;
    int end = 0;
    if (whoami!= omp_get_num_threads()-1)
        end = start + stride;
    else
        end = N;

    double sum = 0;

    // here we compute the distance matrix
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

}

void eccentricity(py::array_t<double> dm, py::array_t<double> ecc, int exponent, int nthreads) {
    py::gil_scoped_acquire acquire;

    py::buffer_info dm_info = dm.request();
    py::buffer_info ecc_info = ecc.request();

    auto dm_ptr = static_cast<double *>(dm_info.ptr);
    auto ecc_ptr = static_cast<double *>(ecc_info.ptr);

    int N = dm_info.shape[0];
    omp_set_num_threads(nthreads);

    #pragma omp parallel
    {
    // these 8 lines spit the job between threads
    int whoami = omp_get_thread_num();
    int stride = N/nthreads;
    int start = stride*whoami;
    int end = 0;
    if (whoami!= omp_get_num_threads()-1)
        end = start + stride;
    else
        end = N;

    // implementation of the eccentricity in case exponent==-1
    if (exponent == -1){
        for (int i = start; i <end; i++)
            for (int j = 0; j<N; j++){
                double el = dm_ptr[i*N+j];
                if (el >= ecc_ptr[i])
                    ecc_ptr[i] = el;
            }
    }
    // implementation of the eccentricity in case exponent==1
    else if (exponent == 1){
        double sum = 0;
        for (int i = start; i <end; i++){
            sum = 0;
            for (int j = 0; j<N; j++)
                sum += dm_ptr[i*N+j];
            ecc_ptr[i] = sum/N;
        }
    }
    // implementation of the eccentricity in case exponent!=1
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
}


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
