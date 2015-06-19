#include <cstdio>

#include "math.h"
#include "vector.h"

int main()
{
    CUDA::Vector<float> foo(4.0f);
    CUDA::Vector<float> bar = sqrt(foo);

    float result = bar.data();
    printf("Result: %f\n", result);

    return 0;
}

