/*  This file is part of the Vc library. {{{
Copyright © 2014 Matthias Kretz <kretz@kde.org>
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
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_NEON_MATH_H_
#define VC_NEON_MATH_H_

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace NEON
{

Vc_ALWAYS_INLINE int_v min(const int_v &x, const int_v &y);
Vc_ALWAYS_INLINE uint_v min(const uint_v &x, const uint_v &y);
Vc_ALWAYS_INLINE short_v min(const short_v &x, const short_v &y);
Vc_ALWAYS_INLINE ushort_v min(const ushort_v &x, const ushort_v &y);
Vc_ALWAYS_INLINE float_v min(const float_v &x, const float_v &y);
Vc_ALWAYS_INLINE double_v min(const double_v &x, const double_v &y);
Vc_ALWAYS_INLINE int_v max(const int_v &x, const int_v &y);
Vc_ALWAYS_INLINE uint_v max(const uint_v &x, const uint_v &y);
Vc_ALWAYS_INLINE short_v max(const short_v &x, const short_v &y);
Vc_ALWAYS_INLINE ushort_v max(const ushort_v &x, const ushort_v &y);
Vc_ALWAYS_INLINE float_v max(const float_v &x, const float_v &y);
Vc_ALWAYS_INLINE double_v max(const double_v &x, const double_v &y);

template <typename T,
          typename = enable_if<std::is_same<T, double>::value || std::is_same<T, float>::value ||
                               std::is_same<T, short>::value ||
                               std::is_same<T, int>::value>>
Vc_ALWAYS_INLINE Vc_PURE Vector<T> abs(Vector<T> x);

template <typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> sqrt(const Vector<T> &x);
template <typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> rsqrt(const Vector<T> &x);
template <typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> reciprocal(const Vector<T> &x);
template <typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> round(const Vector<T> &x);

template <typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T>::Mask isfinite(const Vector<T> &x);
template <typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T>::Mask isinf(const Vector<T> &x);
template <typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T>::Mask isnan(const Vector<T> &x);

}
}

#include "undomacros.h"

#endif  // VC_NEON_MATH_H_

// vim: foldmethod=marker
