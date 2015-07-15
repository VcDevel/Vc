#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "math.h"
#include "vector.h"
#include "macros.h"

__global__ void my_kernel(const float *in, float *out)
{
    Vc::CUDA::Vector<float> inVec(in);
    Vc::CUDA::Vector<float> outVec;
    outVec = sqrt(inVec);
    outVec.store(out);
}

template <typename F, typename... Arguments>
Vc_ALWAYS_INLINE void spawn(F&& kernel, Arguments&& ... args)
{
    kernel<<<1, CUDA_VECTOR_SIZE>>>(std::forward<Arguments>(args) ...);
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

    // allocate memory on device
    float *devData;
    float *devResult;
    cudaMalloc(&devData, sizeof(float) * CUDA_VECTOR_SIZE);
    cudaMalloc(&devResult, sizeof(float) * CUDA_VECTOR_SIZE);

    // copy data to device
    cudaMemcpy(devData, data, sizeof(float) * CUDA_VECTOR_SIZE, cudaMemcpyHostToDevice);

    // run kernel
    //my_kernel<<<1, CUDA_VECTOR_SIZE>>>(devData, devResult);
    spawn(my_kernel, devData, devResult);

    // copy result from device
    cudaMemcpy(result, devResult, sizeof(float) * CUDA_VECTOR_SIZE, cudaMemcpyDeviceToHost);
    
    // print results
    for(std::size_t i = 0; i < 32; ++i)
        printf("sqrt(%f) = %f\n", data[i], result[i]);
    
    // free memory
    cudaFree(devResult);
    cudaFree(devData);
 
    return 0;
}

#include "undomacros.h"

