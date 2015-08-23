/*  This file is part of the Vc library. {{{
Copyright © 2015 Jan Stephan <jan.stephan.dd@gmail.com>
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

#ifndef VC_CUDA_DETAIL_H
#define VC_CUDA_DETAIL_H

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Detail
{

template<typename... Flags> struct InitFlags
{
};

using InternalInitTag = InitFlags<>;
constexpr InternalInitTag InternalInit;

__device__ Vc_ALWAYS_INLINE unsigned int getThreadId()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

template <typename Op, typename T>
__device__ Vc_ALWAYS_INLINE void reduce(volatile T *ptr, const T *data, unsigned int threadId)
{
    // ptr is a pointer to a single variable
    Op(ptr, data[threadId]);
}

template <typename Op, typename T>
__device__ Vc_ALWAYS_INLINE void reduce2(volatile T *ptr, const T *data1, const T *data2, unsigned int threadId)
{
    Op(ptr, data1[threadId], data2[threadId]);
}

} // namespace Detail
} // namespace Vc

#include "undomacros.h"

#endif // VC_CUDA_DETAIL_H

