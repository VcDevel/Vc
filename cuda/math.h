/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_CUDA_MATH_H
#define VC_CUDA_MATH_H

#include "vector.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace CUDA
{
namespace Impl
{
    __global__ void sqrt(const float *in, float *out)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        out[idx] = ::sqrtf(in[idx]);
    }
} // namespace Impl

template <typename T> static Vc_ALWAYS_INLINE Vector<T> sqrt(const Vector<T> &x)
{
    float *result;
    cudaMalloc(&result, sizeof(float) * CUDA_VECTOR_SIZE);
    Impl::sqrt<<<1, CUDA_VECTOR_SIZE>>>(x.data(), result);
    Vector<T> ret(result);
    cudaFree(result);
    return ret;
}

} // namespace CUDA
} // namespace Vc

#include "undomacros.h"

#endif

