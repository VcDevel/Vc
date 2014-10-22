/*  This file is part of the Vc library. {{{
Copyright © 2009-2014 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_COMMON_TRIGONOMETRIC_H
#define VC_COMMON_TRIGONOMETRIC_H

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Internal
{
template<Vc::Implementation Impl> struct MapImpl { enum Dummy { Value = Impl }; };
template<> struct MapImpl<Vc::SSE42Impl> { enum Dummy { Value = MapImpl<Vc::SSE41Impl>::Value }; };

template<Vc::Implementation Impl> using TrigonometricImplementation =
    ImplementationT<MapImpl<Impl>::Value
#if defined(VC_IMPL_XOP) && defined(VC_IMPL_FMA4)
    + Vc::XopInstructions
    + Vc::Fma4Instructions
#endif
    >;
}  // namespace Internal

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

#ifdef VC_IMPL_AVX
namespace AVX
{
using Trigonometric = Vc::Common::Trigonometric<Vc::Internal::TrigonometricImplementation<AVXImpl>>;
template<typename T> Vc_ALWAYS_INLINE T sin(const T &x) { return Trigonometric::sin(x); }
template<typename T> Vc_ALWAYS_INLINE T cos(const T &x) { return Trigonometric::cos(x); }
template<typename T> Vc_ALWAYS_INLINE void sincos(const T &x, T *sin, T *cos) { return Trigonometric::sincos(x, sin, cos); }
template<typename T> Vc_ALWAYS_INLINE T asin(const T &x) { return Trigonometric::asin(x); }
template<typename T> Vc_ALWAYS_INLINE T atan(const T &x) { return Trigonometric::atan(x); }
template<typename T> Vc_ALWAYS_INLINE T atan2(const T &y, const T &x) { return Trigonometric::atan2(y, x); }
}  // namespace AVX
#endif

#ifdef VC_IMPL_SSE
namespace SSE
{
// FIXME is SSE42Impl right? Probably yes, but explain why...
using Trigonometric = Vc::Common::Trigonometric<Vc::Internal::TrigonometricImplementation<SSE42Impl>>;
template<typename T> Vc_ALWAYS_INLINE T sin(const T &x) { return Trigonometric::sin(x); }
template<typename T> Vc_ALWAYS_INLINE T cos(const T &x) { return Trigonometric::cos(x); }
template<typename T> Vc_ALWAYS_INLINE void sincos(const T &x, T *sin, T *cos) { return Trigonometric::sincos(x, sin, cos); }
template<typename T> Vc_ALWAYS_INLINE T asin(const T &x) { return Trigonometric::asin(x); }
template<typename T> Vc_ALWAYS_INLINE T atan(const T &x) { return Trigonometric::atan(x); }
template<typename T> Vc_ALWAYS_INLINE T atan2(const T &y, const T &x) { return Trigonometric::atan2(y, x); }
}  // namespace SSE

// only needed for AVX2, AVX, or SSE:
namespace Vc_IMPL_NAMESPACE
{
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> sin(const Vector<T> &_x) {
    return Vc::Common::Trigonometric<Vc::Internal::TrigonometricImplementation<VC_IMPL>>::sin(_x);
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> cos(const Vector<T> &_x) {
    return Vc::Common::Trigonometric<Vc::Internal::TrigonometricImplementation<VC_IMPL>>::cos(_x);
}
template<typename T> Vc_ALWAYS_INLINE void sincos(const Vector<T> &_x, Vector<T> *_sin, Vector<T> *_cos) {
    Vc::Common::Trigonometric<Vc::Internal::TrigonometricImplementation<VC_IMPL>>::sincos(_x, _sin, _cos);
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> asin (const Vector<T> &_x) {
    return Vc::Common::Trigonometric<Vc::Internal::TrigonometricImplementation<VC_IMPL>>::asin(_x);
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> atan (const Vector<T> &_x) {
    return Vc::Common::Trigonometric<Vc::Internal::TrigonometricImplementation<VC_IMPL>>::atan(_x);
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> atan2(const Vector<T> &y, const Vector<T> &x) {
    return Vc::Common::Trigonometric<Vc::Internal::TrigonometricImplementation<VC_IMPL>>::atan2(y, x);
}
}  // namespace Vc_IMPL_NAMESPACE
#endif
}  // namespace Vc

#include "undomacros.h"
#endif // VC_COMMON_TRIGONOMETRIC_H
