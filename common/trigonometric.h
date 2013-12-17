/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

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
}
}

namespace Vc_VERSIONED_NAMESPACE
{
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
}
}

#ifdef VC_IMPL_AVX
namespace Vc_VERSIONED_NAMESPACE
{
namespace AVX
{
using Trigonometric = Vc::Common::Trigonometric<Vc::Internal::TrigonometricImplementation<AVXImpl>>;
template<typename T> Vc_ALWAYS_INLINE T sin(const T &x) { return Trigonometric::sin(x); }
template<typename T> Vc_ALWAYS_INLINE T cos(const T &x) { return Trigonometric::cos(x); }
template<typename T> Vc_ALWAYS_INLINE void sincos(const T &x, T *sin, T *cos) { return Trigonometric::sincos(x, sin, cos); }
template<typename T> Vc_ALWAYS_INLINE T asin(const T &x) { return Trigonometric::asin(x); }
template<typename T> Vc_ALWAYS_INLINE T atan(const T &x) { return Trigonometric::atan(x); }
template<typename T> Vc_ALWAYS_INLINE T atan2(const T &y, const T &x) { return Trigonometric::atan2(y, x); }
}
}
#endif

#ifdef VC_IMPL_SSE
namespace Vc_VERSIONED_NAMESPACE
{
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
}
}

// only needed for AVX2, AVX, or SSE:
namespace Vc_VERSIONED_NAMESPACE
{
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
}
}
#endif

#include "undomacros.h"
#endif // VC_COMMON_TRIGONOMETRIC_H
