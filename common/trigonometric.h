/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_TRIGONOMETRIC_H_
#define VC_COMMON_TRIGONOMETRIC_H_

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Detail
{
template<Vc::Implementation Impl> struct MapImpl { enum Dummy { Value = Impl }; };
template<> struct MapImpl<Vc::SSE42Impl> { enum Dummy { Value = MapImpl<Vc::SSE41Impl>::Value }; };

template<Vc::Implementation Impl> using TrigonometricImplementation =
    ImplementationT<MapImpl<Impl>::Value
#if defined(Vc_IMPL_XOP) && defined(Vc_IMPL_FMA4)
    + Vc::XopInstructions
    + Vc::Fma4Instructions
#endif
    >;
}  // namespace Detail

namespace Common
{
template<typename Impl> struct Trigonometric
{
    template<typename T> static T sin(const T &_x);
    template<typename T> static T cos(const T &_x);
    template<typename T> static void sincos(const T &_x, T *_sin, T *_cos);
    template<typename T> static T asin (const T &_x);
    template<typename T> static T atan (const T &_x);
    template<typename T> static T atan2(const T &y, const T &x);
};
}  // namespace Common

#ifdef Vc_IMPL_SSE
// this is either SSE, AVX, or AVX2
namespace Detail
{
template <typename T, typename Abi>
using Trig = Common::Trigonometric<Detail::TrigonometricImplementation<
    (std::is_same<Abi, VectorAbi::Sse>::value
         ? SSE42Impl
         : std::is_same<Abi, VectorAbi::Avx>::value ? AVXImpl : ScalarImpl)>>;
}  // namespace Detail
template <typename T, typename Abi> Vc_INTRINSIC Vector<T, Abi> sin(const Vector<T, Abi> &x) { return Detail::Trig<T, Abi>::sin(x); }
template <typename T, typename Abi> Vc_INTRINSIC Vector<T, Abi> cos(const Vector<T, Abi> &x) { return Detail::Trig<T, Abi>::cos(x); }
template <typename T, typename Abi> Vc_INTRINSIC Vector<T, Abi> asin(const Vector<T, Abi> &x) { return Detail::Trig<T, Abi>::asin(x); }
template <typename T, typename Abi> Vc_INTRINSIC Vector<T, Abi> atan(const Vector<T, Abi> &x) { return Detail::Trig<T, Abi>::atan(x); }
template <typename T, typename Abi> Vc_INTRINSIC Vector<T, Abi> atan2(const Vector<T, Abi> &y, const Vector<T, Abi> &x) { return Detail::Trig<T, Abi>::atan2(y, x); }
template <typename T, typename Abi> Vc_INTRINSIC void sincos(const Vector<T, Abi> &x, Vector<T, Abi> *sin, Vector<T, Abi> *cos) { Detail::Trig<T, Abi>::sincos(x, sin, cos); }
#endif
}  // namespace Vc

#endif // VC_COMMON_TRIGONOMETRIC_H_
