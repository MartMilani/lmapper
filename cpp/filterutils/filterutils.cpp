#include <cmath>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <algorithm>
namespace py = pybind11;
/*
N = np.alen(data)
ecc = np.empty(N)
if self.exponent in (np.inf, 'Inf', 'inf'):
    for i in range(N):
        ecc[i] = cdist(data[(i,), :], data, **self.metricpar).max()
elif self.exponent == 1.:
    for i in range(N):
        ecc[i] = cdist(data[(i,), :], data, **self.metricpar).sum()/float(N)
else:
    for i in range(N):
        dsum = np.power(cdist(data[(i,), :], data, **self.metricpar),
                        self.exponent).sum()
        ecc[i] = np.power(dsum/float(N), 1./self.exponent)
return ecc
*/
// Passing in an array of doubles

void my_distance(py::array_t<double> data, py::array_t<double> dm, int nthreads, std::string metric){
    py::gil_scoped_acquire acquire;

    py::buffer_info data_info = data.request();
    py::buffer_info dm_info = dm.request();
    /*
    struct buffer_info {
        void *ptr;
        size_t itemsize;
        std::string format;
        int ndim;
        std::vector<size_t> shape;
        std::vector<size_t> strides;
    }; */
    auto data_ptr = static_cast<double *>(data_info.ptr);
    auto dm_ptr = static_cast<double *>(dm_info.ptr);

    const int N = data_info.shape[0];
    const int PAD = data_info.shape[1];

    std::vector<double> data_normalized;
    std::vector<double> std_dev;

    omp_set_num_threads(nthreads);
    if (metric == "correlation"){
        //
        // ---- calculating std_dev and data_normalized ----
        //
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

}

void eccentricity(py::array_t<double> dm, py::array_t<double> ecc, int exponent, int nthreads) {
    py::gil_scoped_acquire acquire;

    py::buffer_info dm_info = dm.request();
    py::buffer_info ecc_info = ecc.request();
    /*
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
    if (exponent == -1){
        int whoami = omp_get_thread_num();
        int stride = N/nthreads;
        int start = stride*whoami;
        int end = 0;
        if (whoami!= omp_get_num_threads()-1)
            end = start + stride;
        else
            end = N;
        for (int i = start; i <end; i++)
            for (int j = 0; j<N; j++){
                double el = dm_ptr[i*N+j];
                if (el >= ecc_ptr[i])
                    ecc_ptr[i] = el;
            }
    }
    else if (exponent == 1){
        int whoami = omp_get_thread_num();
        int stride = N/nthreads;
        int start = stride*whoami;
        int end = 0;
        if (whoami!= omp_get_num_threads()-1)
            end = start + stride;
        else
            end = N;

        double sum = 0;
        for (int i = start; i <end; i++){
            sum = 0;
            for (int j = 0; j<N; j++)
                sum += dm_ptr[i*N+j];
            ecc_ptr[i] = sum/N;
        }
    }
    else{
        int whoami = omp_get_thread_num();
        int stride = N/nthreads;
        int start = stride*whoami;
        int end = 0;
        if (whoami!= omp_get_num_threads()-1)
            end = start + stride;
        else
            end = N;

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

PYBIND11_MODULE(filterutils, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("eccentricity", [](py::array_t<double> dm, py::array_t<double> ecc, int exponent, int nthreads) {
        /* Release GIL before calling into C++ code */
        py::gil_scoped_release release;
        return eccentricity(dm, ecc, exponent, nthreads);
    });
    m.def("my_distance", [](py::array_t<double> data, py::array_t<double> dm, int nthreads, std::string metric) {
        /* Release GIL before calling into C++ code */
        py::gil_scoped_release release;
        return my_distance(data, dm, nthreads, metric);
      });
}
