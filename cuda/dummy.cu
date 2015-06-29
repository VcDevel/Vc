#include <cmath>
#include <cstdio>
#include <cstdlib>

//#include "math.h"
#include "vector.h"

struct fsqrt
{
    public:
        __host__ __device__ float operator()(float f)
        {
#ifdef __NVCC__
            return sqrtf(f);
#else
            return std::sqrt(f);
#endif
        }
};

template<class Op>
__global__ void apply_operator(const float *in, float *out, Op op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = op(in[idx]);
}

template<typename T, class Op>
Vc::CUDA::Vector<T> compute_on_device(const Vc::CUDA::Vector<T> &x, Op op)
{
    float *result;
    cudaMalloc(&result, sizeof(float) * CUDA_VECTOR_SIZE);
    apply_operator<<<1, CUDA_VECTOR_SIZE>>>(x.data(), result, op);
    Vc::CUDA::Vector<T> ret(result);
    cudaFree(result);
    return ret;
}

template<class Op>
void compute(float* data, float* result, Op p)
{
#ifdef __NVCC__
    // allocate memory on device
    float *devData;
    float *devResult;
    cudaMalloc(&devData, sizeof(float) * CUDA_VECTOR_SIZE);
    cudaMalloc(&devResult, sizeof(float) * CUDA_VECTOR_SIZE);

    // copy data to device
    cudaMemcpy(devData, data, sizeof(float) * CUDA_VECTOR_SIZE, cudaMemcpyHostToDevice);

    // create vector from data
    Vc::CUDA::Vector<float> inVec(devData);

    // computation
    Vc::CUDA::Vector<float> outVec = compute_on_device(inVec, p);

    // fetch result from vector
    outVec.store(devResult);

    // copy to host
    cudaMemcpy(result, devResult, sizeof(float) * CUDA_VECTOR_SIZE, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(devResult);
    cudaFree(devData);
#else
    // CPU code goes here
#endif
}

int main()
{
    float data[32];
    float result[32];
    // initialize data
    for(std::size_t i = 0; i < 32; ++i)
    {
        data[i] = i * i;
        result[i] = .0f;
    }

    fsqrt mySqrt;
    compute(data, result, mySqrt);

    // print results
    for(std::size_t i = 0; i < 32; ++i)
        printf("sqrt(%f) = %f\n", data[i], result[i]);
 
    return 0;
}

