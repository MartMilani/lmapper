#include <cmath>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <algorithm>
namespace py = pybind11;

// Passing in an array of doubles

void my_distance(py::array_t<double> data, py::array_t<double> dm, int nthreads, std::string metric){
    /*Computes the distance matrix in squareform format of a data point cloud.

     Args:
         data: numpy array of doubles.
         dm: numpy array of doubles already initialized where the distance matrix will be saved.
            It has to be initialized in the Python code by >>>> dm = np.ndarray((N,N), dtype='float')
            This means that this function does not return the result, but modifies the content
            of the numpy array dm.
         nthreads: number of OpenMP threads to launch
         metric: string that can only assume the following two values: "euclidean" or "correlation"
     */

    // acquiring the Global Interpreter Lock (if not leads to SegmentationFault 11)
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
    std::cout << "Using "<<nthreads<<" threads"<<std::endl;

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

void eccentricity(py::array_t<double> dm, py::array_t<double> ecc, double exponent, int nthreads) {
    /*Computes the ecentricity of a data point cloud.

     Args:
         dm: squared form distance matrix NOTE: optimization possible: it could accept a compressed flat distance matrix
         ecc: numpy array already initialized where to store the result. This means that it has to be initialized in the Python code with >>>> ecc = numpy.zeros((N,1), dtype='float')
         exponent: if less than zero, it is interpreted as inf.
         nthreads: number of OpenMP threads to launch.
     */
    py::gil_scoped_acquire acquire; // acquiring the Global Interpreter Lock (if not leads to SegmentationFault 11)

    py::buffer_info dm_info = dm.request();
    py::buffer_info ecc_info = ecc.request();

    auto dm_ptr = static_cast<double *>(dm_info.ptr);
    auto ecc_ptr = static_cast<double *>(ecc_info.ptr);

    int N = dm_info.shape[0];
    omp_set_num_threads(nthreads);
    std::cout << "Using "<<nthreads<<" threads"<<std::endl;
    
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
    if (exponent < 0){
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
