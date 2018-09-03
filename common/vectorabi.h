/*  This file is part of the Vc library. {{{
Copyright © 2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_VECTORABI_H_
#define VC_COMMON_VECTORABI_H_

namespace Vc_VERSIONED_NAMESPACE
{
namespace VectorAbi
{
template <typename T>
using Avx1Abi = typename std::conditional<std::is_integral<T>::value, VectorAbi::Sse,
                                          VectorAbi::Avx>::type;

template <typename T> struct DeduceCompatible {
#ifdef __x86_64__
    using type = Sse;
#else
    using type = Scalar;
#endif
};

template <typename T>
struct DeduceBest {
    using type = typename std::conditional<
        CurrentImplementation::is(ScalarImpl), Scalar,
        typename std::conditional<
            CurrentImplementation::is_between(SSE2Impl, SSE42Impl), Sse,
            typename std::conditional<
                CurrentImplementation::is(AVXImpl), Avx1Abi<T>,
                typename std::conditional<
                    CurrentImplementation::is(AVX2Impl), Avx,
                    typename std::conditional<CurrentImplementation::is(MICImpl), Mic,
                                              void>::type>::type>::type>::type>::type;
};
template <typename T> using Best = typename DeduceBest<T>::type;

#ifdef Vc_IMPL_AVX2
static_assert(std::is_same<Best<float>, Avx>::value, "");
static_assert(std::is_same<Best<int>, Avx>::value, "");
#elif defined Vc_IMPL_AVX
static_assert(std::is_same<Best<float>, Avx>::value, "");
static_assert(std::is_same<Best<int>, Sse>::value, "");
#elif defined Vc_IMPL_SSE
static_assert(CurrentImplementation::is_between(SSE2Impl, SSE42Impl), "");
static_assert(std::is_same<Best<float>, Sse>::value, "");
static_assert(std::is_same<Best<int>, Sse>::value, "");
#elif defined Vc_IMPL_MIC
static_assert(std::is_same<Best<float>, Mic>::value, "");
static_assert(std::is_same<Best<int>, Mic>::value, "");
#elif defined Vc_IMPL_Scalar
static_assert(std::is_same<Best<float>, Scalar>::value, "");
static_assert(std::is_same<Best<int>, Scalar>::value, "");
#endif
}  // namespace VectorAbi
}  // namespace Vc_VERSIONED_NAMESPACE

#include "simdarrayfwd.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace detail
{
template <class T> struct translate_to_simd<T, simd_abi::scalar> {
    using simd = Vector<T, VectorAbi::Scalar>;
    using mask = Mask<T, VectorAbi::Scalar>;
};
template <class T, int N> struct translate_to_simd<T, simd_abi::fixed_size<N>> {
    using simd = SimdArray<T, N>;
    using mask = SimdMaskArray<T, N>;
};
template <class T> struct translate_to_simd<T, simd_abi::compatible<T>> {
#ifdef Vc_IMPL_SSE2
    using simd = Vector<T, VectorAbi::Sse>;
    using mask = Mask<T, VectorAbi::Sse>;
#else
    using simd = Vector<T, VectorAbi::Scalar>;
    using mask = Mask<T, VectorAbi::Scalar>;
#endif
};
template <class T> struct translate_to_simd<T, simd_abi::native<T>> {
    using simd = Vector<T, VectorAbi::Best<T>>;
    using mask = Mask<T, VectorAbi::Best<T>>;
};
template <class T> struct translate_to_simd<T, simd_abi::__sse> {
    using simd = Vector<T, VectorAbi::Sse>;
    using mask = Mask<T, VectorAbi::Sse>;
};
template <class T> struct translate_to_simd<T, simd_abi::__avx> {
    using simd = Vector<T, VectorAbi::Avx>;
    using mask = Mask<T, VectorAbi::Avx>;
};
template <class T> struct translate_to_simd<T, simd_abi::__avx512> {
    // not implemented
};
template <class T> struct translate_to_simd<T, simd_abi::__neon> {
    // not implemented
};

// is_fixed_size_abi {{{
template <class T> struct is_fixed_size_abi : std::false_type {
};
template <int N> struct is_fixed_size_abi<simd_abi::fixed_size<N>> : std::true_type {
};
//}}}

template <class T>
using not_fixed_size_abi = typename std::enable_if<!is_fixed_size_abi<T>::value, T>::type;

}  // namespace detail
}  // namespace Vc

#endif  // VC_COMMON_VECTORABI_H_

// vim: foldmethod=marker
