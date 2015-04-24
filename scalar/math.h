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

#ifndef VC_SCALAR_MATH_H
#define VC_SCALAR_MATH_H

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Scalar
{
template <typename T> Vc_ALWAYS_INLINE Vector<T> copysign(Vector<T> a, Vector<T> b)
{
    return a.copySign(b);
}

#define VC_MINMAX(V) \
static Vc_ALWAYS_INLINE V min(const V &x, const V &y) { return V(std::min(x.data(), y.data())); } \
static Vc_ALWAYS_INLINE V max(const V &x, const V &y) { return V(std::max(x.data(), y.data())); }
VC_ALL_VECTOR_TYPES(VC_MINMAX)
#undef VC_MINMAX

template<typename T> static Vc_ALWAYS_INLINE Vector<T> sqrt (const Vector<T> &x)
{
    return Vector<T>(std::sqrt(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> rsqrt(const Vector<T> &x)
{
    const typename Vector<T>::EntryType one = 1; return Vector<T>(one / std::sqrt(x.data()));
}

template <typename T,
          typename = enable_if<std::is_same<T, double>::value || std::is_same<T, float>::value ||
                               std::is_same<T, short>::value ||
                               std::is_same<T, int>::value>>
Vc_ALWAYS_INLINE Vc_PURE Vector<T> abs(Vector<T> x)
{
    return std::abs(x.data());
}

template<typename T> static Vc_ALWAYS_INLINE void sincos(const Vector<T> &x, Vector<T> *sin, Vector<T> *cos)
{
#if (defined(VC_CLANG) && VC_HAS_BUILTIN(__builtin_sincosf)) || (!defined(VC_CLANG) && defined(__GNUC__) && !defined(_WIN32))
    __builtin_sincosf(x.data(), &sin->data(), &cos->data());
#elif defined(_WIN32)
    sin->data() = std::sin(x.data());
    cos->data() = std::cos(x.data());
#else
    sincosf(x.data(), &sin->data(), &cos->data());
#endif
}

template<> Vc_ALWAYS_INLINE void sincos(const Vector<double> &x, Vector<double> *sin, Vector<double> *cos)
{
#if (defined(VC_CLANG) && VC_HAS_BUILTIN(__builtin_sincos)) || (!defined(VC_CLANG) && defined(__GNUC__) && !defined(_WIN32))
    __builtin_sincos(x.data(), &sin->data(), &cos->data());
#elif defined(_WIN32)
    sin->data() = std::sin(x.data());
    cos->data() = std::cos(x.data());
#else
    ::sincos(x.data(), &sin->data(), &cos->data());
#endif
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> sin  (const Vector<T> &x)
{
    return Vector<T>(std::sin(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> asin (const Vector<T> &x)
{
    return Vector<T>(std::asin(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> cos  (const Vector<T> &x)
{
    return Vector<T>(std::cos(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> log  (const Vector<T> &x)
{
    return Vector<T>(std::log(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> log10(const Vector<T> &x)
{
    return Vector<T>(std::log10(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> log2(const Vector<T> &x)
{
    return Vector<T>(std::log2(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> exp (const Vector<T> &x)
{
    return Vector<T>(std::exp(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> atan (const Vector<T> &x)
{
    return Vector<T>(std::atan( x.data() ));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> atan2(const Vector<T> &x, const Vector<T> &y)
{
    return Vector<T>(std::atan2( x.data(), y.data() ));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> trunc(const Vector<T> &x)
{
    return std::trunc(x.data());
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> floor(const Vector<T> &x)
{
    return Vector<T>(std::floor(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> ceil(const Vector<T> &x)
{
    return Vector<T>(std::ceil(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> round(const Vector<T> &x)
{
    return x;
}

namespace
{
    template<typename T> bool _realIsEvenHalf(T x) {
        const T two = 2;
        const T half = 0.5;
        const T f = std::floor(x * half) * two;
        return (x - f) == half;
    }
} // namespace
template<> Vc_ALWAYS_INLINE Vector<float>  round(const Vector<float>  &x)
{
    return float_v(std::floor(x.data() + 0.5f) - (_realIsEvenHalf(x.data()) ? 1.f : 0.f));
}

template<> Vc_ALWAYS_INLINE Vector<double> round(const Vector<double> &x)
{
    return double_v(std::floor(x.data() + 0.5 ) - (_realIsEvenHalf(x.data()) ? 1.  : 0. ));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> reciprocal(const Vector<T> &x)
{
    const typename Vector<T>::EntryType one = 1; return Vector<T>(one / x.data());
}

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif
template<typename T> static Vc_ALWAYS_INLINE typename Vector<T>::Mask isfinite(const Vector<T> &x)
{
    return typename Vector<T>::Mask(
#ifdef _MSC_VER
            !!_finite(x.data())
#elif defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1500
            ::isfinite(x.data())
#else
            std::isfinite(x.data())
#endif
            );
}

template<typename T> Vc_ALWAYS_INLINE typename Vector<T>::Mask isinf(const Vector<T> &x)
{
    return typename Vector<T>::Mask(std::isinf(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE typename Vector<T>::Mask isnan(const Vector<T> &x)
{
    return typename Vector<T>::Mask(
#ifdef _MSC_VER
            !!_isnan(x.data())
#elif defined(__INTEL_COMPILER)
            ::isnan(x.data())
#else
            std::isnan(x.data())
#endif
            );
}

Vc_ALWAYS_INLINE Vector<float> frexp(Vector<float> x, simdarray<int, 1, Vector<int>, 1> *e) {
    return float_v(std::frexp(x.data(), &internal_data(*e).data()));
}
Vc_ALWAYS_INLINE Vector<double> frexp(Vector<double> x, simdarray<int, 1, Vector<int>, 1> *e) {
    return double_v(std::frexp(x.data(), &internal_data(*e).data()));
}

Vc_ALWAYS_INLINE Vector<float> ldexp(Vector<float> x, const simdarray<int, 1, Vector<int>, 1> &e) {
    return float_v(std::ldexp(x.data(), internal_data(e).data()));
}
Vc_ALWAYS_INLINE Vector<double> ldexp(Vector<double> x, const simdarray<int, 1, Vector<int>, 1> &e) {
    return double_v(std::ldexp(x.data(), internal_data(e).data()));
}

}  // namespace Scalar
}  // namespace Vc

#include "undomacros.h"

#endif // VC_SCALAR_MATH_H
