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

#ifndef VC_CUDA_MATH_H
#define VC_CUDA_MATH_H

#include "detail.h"
#include "vector.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Detail
{
    template <typename T> __device__ Vc_ALWAYS_INLINE T reciprocal(T x)
    {
        return 1 / x;
    }
}

#ifndef MATH_FUNC_MACROS
#define MATH_FUNC_MACROS
#define FUNC1(name__, impl__) \
    template <typename T> __device__ static Vc_ALWAYS_INLINE Vector<T, VectorAbi::Cuda> name__(const Vector<T, VectorAbi::Cuda> &x) \
    { \
        return CALC1(impl__, x);\
    }

#define FUNC2(name__, impl__) \
    template <typename T> __device__ static Vc_ALWAYS_INLINE Vector<T, VectorAbi::Cuda> name__(const Vector<T, VectorAbi::Cuda> &x, const Vector<T, VectorAbi::Cuda> &y) \
    { \
        return CALC2(impl__, x, y); \
    }
#endif

#ifndef CALC_MACROS
#define CALC1(fun__, arg__) Vector<T, VectorAbi::Cuda>::internalInit(fun__(arg__[Detail::getThreadId()]))
#define CALC2(fun__, arg1__, arg2__) Vector<T, VectorAbi::Cuda>::internalInit(fun__(arg1__[Detail::getThreadId()], arg2__[Detail::getThreadId()]))
#define CALC_MACROS
#endif

FUNC1(sqrt, ::sqrtf)
FUNC1(rsqrt, ::rsqrtf)
FUNC1(reciprocal, Detail::reciprocal)
FUNC1(abs, ::fabsf)
FUNC1(round, ::roundf)
FUNC1(log, ::logf)
FUNC1(log2, ::log2f)
FUNC1(log10, ::log10f)
FUNC1(exp, ::expf)
FUNC1(sin, ::sinf)
FUNC1(cos, ::cosf)

template <typename T> __device__ static Vc_ALWAYS_INLINE void sincos(const Vector<T, VectorAbi::Cuda> &v, Vector<T, VectorAbi::Cuda>* sin, Vector<T, VectorAbi::Cuda>* cos)
{
    ::sincosf(v[Detail::getThreadId()], (*sin)[Detail::getThreadId()], (*cos)[Detail::getThreadId()]);
}

FUNC1(asin, ::asinf)
FUNC1(atan, ::atanf)
FUNC2(atan2, ::atan2f)
FUNC2(max, ::fmaxf)
FUNC2(min, ::fminf)

template <typename T> __device__ static Vc_ALWAYS_INLINE Vector<T, VectorAbi::Cuda> frexp(const Vector<T, VectorAbi::Cuda> &v, Vector<int, VectorAbi::Cuda>* e)
{
    return Vector<T, VectorAbi::Cuda>::internalInit(::frexpf(v[Detail::getThreadId()], (*e)[Detail::getThreadId()]));
}

template <typename T> __device__ static Vc_ALWAYS_INLINE Vector<T> ldexp(Vector<T, VectorAbi::Cuda> x, Vector<int, VectorAbi::Cuda> e)
{
    return Vector<T, VectorAbi::Cuda>::internalInit(::ldexpf(x[Detail::getThreadId()], e[Detail::getThreadId()]));
}

template <typename T> __device__ static Vc_ALWAYS_INLINE Mask<T> isfinite(const Vector<T, VectorAbi::Cuda> &v)
{
    return Mask<T, VectorAbi::Cuda>::internalInit(::isfinite((v[Detail::getThreadId()])));
}

template <typename T> __device__ static Vc_ALWAYS_INLINE Mask<T> isinf(const Vector<T, VectorAbi::Cuda> &v)
{
    return Mask<T, VectorAbi::Cuda>::internalInit(::isinf(v[Detail::getThreadId()]));
}

template <typename T> __device__ static Vc_ALWAYS_INLINE Mask<T> isnan(const Vector<T, VectorAbi::Cuda> &v)
{
    return Mask<T, VectorAbi::Cuda>::internalInit(::isnan(v[Detail::getThreadId()]));
}

#ifdef CALC_MACROS
#undef CALC1
#undef CALC2
#undef CALC_MACROS
#endif

#ifdef MATH_FUNC_MACROS
#undef FUNC1
#undef FUNC2
#undef MATH_FUNC_MACROS
#endif

__device__ Vc_ALWAYS_INLINE CUDA::double_v trunc(const CUDA::double_v& v)
{
    return CUDA::double_v::internalInit(::trunc(v[Detail::getThreadId()]));
}
__device__ Vc_ALWAYS_INLINE CUDA::float_v trunc(const CUDA::float_v& v)
{
    return CUDA::float_v::internalInit(::truncf(v[Detail::getThreadId()]));
}

__device__ Vc_ALWAYS_INLINE CUDA::double_v floor(const CUDA::double_v& v)
{
    return CUDA::double_v::internalInit(::floor(v[Detail::getThreadId()]));
}
__device__ Vc_ALWAYS_INLINE CUDA::float_v floor(const CUDA::float_v& v)
{
    return CUDA::float_v::internalInit(::floorf(v[Detail::getThreadId()]));
}

__device__ Vc_ALWAYS_INLINE CUDA::double_v ceil(const CUDA::double_v& v)
{
    return CUDA::double_v::internalInit(::ceil(v[Detail::getThreadId()]));
}
__device__ Vc_ALWAYS_INLINE CUDA::float_v ceil(const CUDA::float_v& v)
{
    return CUDA::float_v::internalInit(::ceilf(v[Detail::getThreadId()]));
}

} // namespace Vc

#include "undomacros.h"

#endif

