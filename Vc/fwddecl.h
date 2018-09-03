/*  This file is part of the Vc library. {{{
Copyright © 2018 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_FWDDECL_H_
#define VC_FWDDECL_H_

#define Vc_VERSIONED_NAMESPACE Vc_1

namespace Vc_VERSIONED_NAMESPACE
{
namespace VectorAbi
{
struct Scalar {};
struct Sse {};
struct Avx {};
struct Mic {};
template <class T> struct DeduceCompatible;
template <class T> struct DeduceBest;
}  // namespace VectorAbi

template <class T, class Abi> class Mask;
template <class T, class Abi> class Vector;

namespace simd_abi
{
using scalar = VectorAbi::Scalar;
template <int N> struct fixed_size;
template <class T> using compatible = typename VectorAbi::DeduceCompatible<T>::type;
template <class T> using native = typename VectorAbi::DeduceBest<T>::type;
using __sse = VectorAbi::Sse;
using __avx = VectorAbi::Avx;
struct __avx512;
struct __neon;
}  // namespace simd_abi

namespace detail
{
template <class T, class Abi> struct translate_to_simd;
// specializations of translate_to_simd in common/vectorabi.h
}  // namespace detail

template <class T, class Abi = simd_abi::compatible<T>> using simd = Vector<T, Abi>;
template <class T, class Abi = simd_abi::compatible<T>> using simd_mask = Mask<T, Abi>;
template <class T> using native_simd = simd<T, simd_abi::native<T>>;
template <class T> using native_simd_mask = simd_mask<T, simd_abi::native<T>>;
template <class T, int N> using fixed_size_simd = simd<T, simd_abi::fixed_size<N>>;
template <class T, int N>
using fixed_size_simd_mask = simd_mask<T, simd_abi::fixed_size<N>>;

}  // namespace Vc_VERSIONED_NAMESPACE
namespace Vc = Vc_VERSIONED_NAMESPACE;

#endif  // VC_FWDDECL_H_

// vim: foldmethod=marker
