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
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
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

Vc_ALWAYS_INLINE NEON::int_v    min(const NEON::int_v    &x, const NEON::int_v &y);
Vc_ALWAYS_INLINE NEON::uint_v   min(const NEON::uint_v   &x, const NEON::uint_v &y);
Vc_ALWAYS_INLINE NEON::short_v  min(const NEON::short_v  &x, const NEON::short_v &y);
Vc_ALWAYS_INLINE NEON::ushort_v min(const NEON::ushort_v &x, const NEON::ushort_v &y);
Vc_ALWAYS_INLINE NEON::float_v  min(const NEON::float_v  &x, const NEON::float_v &y);
Vc_ALWAYS_INLINE NEON::double_v min(const NEON::double_v &x, const NEON::double_v &y);
Vc_ALWAYS_INLINE NEON::int_v    max(const NEON::int_v    &x, const NEON::int_v &y);
Vc_ALWAYS_INLINE NEON::uint_v   max(const NEON::uint_v   &x, const NEON::uint_v &y);
Vc_ALWAYS_INLINE NEON::short_v  max(const NEON::short_v  &x, const NEON::short_v &y);
Vc_ALWAYS_INLINE NEON::ushort_v max(const NEON::ushort_v &x, const NEON::ushort_v &y);
Vc_ALWAYS_INLINE NEON::float_v  max(const NEON::float_v  &x, const NEON::float_v &y);
Vc_ALWAYS_INLINE NEON::double_v max(const NEON::double_v &x, const NEON::double_v &y);

template <typename T,
          typename = enable_if<std::is_same<T, double>::value || std::is_same<T, float>::value ||
                               std::is_same<T, short>::value ||
                               std::is_same<T, int>::value>>
Vc_ALWAYS_INLINE Vc_PURE NEON::Vector<T> abs(NEON::Vector<T> x);

template <typename T> Vc_ALWAYS_INLINE Vc_PURE NEON::Vector<T> sqrt(const NEON::Vector<T> &x);
template <typename T> Vc_ALWAYS_INLINE Vc_PURE NEON::Vector<T> rsqrt(const NEON::Vector<T> &x);
template <typename T> Vc_ALWAYS_INLINE Vc_PURE NEON::Vector<T> reciprocal(const NEON::Vector<T> &x);
template <typename T> Vc_ALWAYS_INLINE Vc_PURE NEON::Vector<T> round(const NEON::Vector<T> &x);

template <typename T> Vc_ALWAYS_INLINE Vc_PURE NEON::Mask<T> isfinite(const NEON::Vector<T> &x);
template <typename T> Vc_ALWAYS_INLINE Vc_PURE NEON::Mask<T> isinf(const NEON::Vector<T> &x);
template <typename T> Vc_ALWAYS_INLINE Vc_PURE NEON::Mask<T> isnan(const NEON::Vector<T> &x);

}  // namespace Vc

#endif  // VC_NEON_MATH_H_

// vim: foldmethod=marker
