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

#ifndef VC_DETAIL_INTRINSICS_H_
#define VC_DETAIL_INTRINSICS_H_

#include "detail.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace detail
{

template <class A, class B, class T = std::common_type_t<A, B>,
          class Trait = builtin_traits<T>>
constexpr Vc_INTRINSIC T interleave_lo(A _a, B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const T a(_a);
    const T b(_b);
    if constexpr (sizeof(T) == 16 && needs_intrinsics) {
        if constexpr (Trait::width == 2) {
            if constexpr (std::is_integral_v<typename Trait::value_type>) {
                return reinterpret_cast<T>(
                    _mm_unpacklo_epi64(builtin_cast<llong>(a), builtin_cast<llong>(b)));
            } else {
                return reinterpret_cast<T>(
                    _mm_unpacklo_pd(builtin_cast<double>(a), builtin_cast<double>(b)));
            }
        } else if constexpr (Trait::width == 4) {
            if constexpr (std::is_integral_v<typename Trait::value_type>) {
                return reinterpret_cast<T>(
                    _mm_unpacklo_epi32(builtin_cast<llong>(a), builtin_cast<llong>(b)));
            } else {
                return reinterpret_cast<T>(
                    _mm_unpacklo_ps(builtin_cast<float>(a), builtin_cast<float>(b)));
            }
        } else if constexpr (Trait::width == 8) {
            return reinterpret_cast<T>(
                _mm_unpacklo_epi16(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        } else if constexpr (Trait::width == 16) {
            return reinterpret_cast<T>(
                _mm_unpacklo_epi8(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        }
    } else if constexpr (Trait::width == 2) {
        return T{a[0], b[0]};
    } else if constexpr (Trait::width == 4) {
        return T{a[0], b[0], a[1], b[1]};
    } else if constexpr (Trait::width == 8) {
        return T{a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]};
    } else if constexpr (Trait::width == 16) {
        return T{a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3],
                 a[4], b[4], a[5], b[5], a[6], b[6], a[7], b[7]};
    } else if constexpr (Trait::width == 32) {
        return T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],
                 a[4],  b[4],  a[5],  b[5],  a[6],  b[6],  a[7],  b[7],
                 a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11],
                 a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15]};
    } else if constexpr (Trait::width == 64) {
        return T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],  a[4],  b[4],
                 a[5],  b[5],  a[6],  b[6],  a[7],  b[7],  a[8],  b[8],  a[9],  b[9],
                 a[10], b[10], a[11], b[11], a[12], b[12], a[13], b[13], a[14], b[14],
                 a[15], b[15], a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19],
                 a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23], a[24], b[24],
                 a[25], b[25], a[26], b[26], a[27], b[27], a[28], b[28], a[29], b[29],
                 a[30], b[30], a[31], b[31]};
    } else {
        assert_unreachable<T>();
    }
}

template <class A, class B, class T = std::common_type_t<A, B>,
          class Trait = builtin_traits<T>>
constexpr Vc_INTRINSIC T interleave_hi(A _a, B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const T a(_a);
    const T b(_b);
    if constexpr (sizeof(T) == 16 && needs_intrinsics) {
        if constexpr (Trait::width == 2) {
            if constexpr (std::is_integral_v<typename Trait::value_type>) {
                return reinterpret_cast<T>(
                    _mm_unpackhi_epi64(builtin_cast<llong>(a), builtin_cast<llong>(b)));
            } else {
                return reinterpret_cast<T>(
                    _mm_unpackhi_pd(builtin_cast<double>(a), builtin_cast<double>(b)));
            }
        } else if constexpr (Trait::width == 4) {
            if constexpr (std::is_integral_v<typename Trait::value_type>) {
                return reinterpret_cast<T>(
                    _mm_unpackhi_epi32(builtin_cast<llong>(a), builtin_cast<llong>(b)));
            } else {
                return reinterpret_cast<T>(
                    _mm_unpackhi_ps(builtin_cast<float>(a), builtin_cast<float>(b)));
            }
        } else if constexpr (Trait::width == 8) {
            return reinterpret_cast<T>(
                _mm_unpackhi_epi16(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        } else if constexpr (Trait::width == 16) {
            return reinterpret_cast<T>(
                _mm_unpackhi_epi8(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        }
    } else if constexpr (Trait::width == 2) {
        return T{a[1], b[1]};
    } else if constexpr (Trait::width == 4) {
        return T{a[2], b[2], a[3], b[3]};
    } else if constexpr (Trait::width == 8) {
        return T{a[4], b[4], a[5], b[5], a[6], b[6], a[7], b[7]};
    } else if constexpr (Trait::width == 16) {
        return T{a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11],
                 a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15]};
    } else if constexpr (Trait::width == 32) {
        return T{a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19],
                 a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23],
                 a[24], b[24], a[25], b[25], a[26], b[26], a[27], b[27],
                 a[28], b[28], a[29], b[29], a[30], b[30], a[31], b[31]};
    } else if constexpr (Trait::width == 64) {
        return T{a[32], b[32], a[33], b[33], a[34], b[34], a[35], b[35],
                 a[36], b[36], a[37], b[37], a[38], b[38], a[39], b[39],
                 a[40], b[40], a[41], b[41], a[42], b[42], a[43], b[43],
                 a[44], b[44], a[45], b[45], a[46], b[46], a[47], b[47],
                 a[48], b[48], a[49], b[49], a[50], b[50], a[51], b[51],
                 a[52], b[52], a[53], b[53], a[54], b[54], a[55], b[55],
                 a[56], b[56], a[57], b[57], a[58], b[58], a[59], b[59],
                 a[60], b[60], a[61], b[61], a[62], b[62], a[63], b[63]};
    } else {
        assert_unreachable<T>();
    }
}

template <class A, class B, class T = std::common_type_t<A, B>,
          class Trait = builtin_traits<T>>
constexpr Vc_INTRINSIC T interleave128_lo(A _a, B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const T a(_a);
    const T b(_b);
    if constexpr (sizeof(T) == 16) {
        return interleave_lo(a, b);
    } else if constexpr (sizeof(T) == 32 && needs_intrinsics) {
        if constexpr (Trait::width == 4) {
            return reinterpret_cast<T>(
                _mm256_unpacklo_pd(builtin_cast<double>(a), builtin_cast<double>(b)));
        } else if constexpr (Trait::width == 8) {
            return reinterpret_cast<T>(
                _mm256_unpacklo_ps(builtin_cast<float>(a), builtin_cast<float>(b)));
        } else if constexpr (Trait::width == 16) {
            return reinterpret_cast<T>(
                _mm256_unpacklo_epi16(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        } else if constexpr (Trait::width == 32) {
            return reinterpret_cast<T>(
                _mm256_unpacklo_epi8(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        }
    } else if constexpr (sizeof(T) == 32) {
        if constexpr (Trait::width == 4) {
            return T{a[0], b[0], a[2], b[2]};
        } else if constexpr (Trait::width == 8) {
            return T{a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5]};
        } else if constexpr (Trait::width == 16) {
            return T{a[0], b[0], a[1], b[1], a[2],  b[2],  a[3],  b[3],
                     a[8], b[8], a[9], b[9], a[10], b[10], a[11], b[11]};
        } else if constexpr (Trait::width == 32) {
            return T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],
                     a[4],  b[4],  a[5],  b[5],  a[6],  b[6],  a[7],  b[7],
                     a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19],
                     a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23]};
        } else if constexpr (Trait::width == 64) {
            return T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],  a[4],  b[4],
                     a[5],  b[5],  a[6],  b[6],  a[7],  b[7],  a[8],  b[8],  a[9],  b[9],
                     a[10], b[10], a[11], b[11], a[12], b[12], a[13], b[13], a[14], b[14],
                     a[15], b[15], a[32], b[32], a[33], b[33], a[34], b[34], a[35], b[35],
                     a[36], b[36], a[37], b[37], a[38], b[38], a[39], b[39], a[40], b[40],
                     a[41], b[41], a[42], b[42], a[43], b[43], a[44], b[44], a[45], b[45],
                     a[46], b[46], a[47], b[47]};
        } else {
            assert_unreachable<T>();
        }
    } else if constexpr (sizeof(T) == 64 && needs_intrinsics) {
        if constexpr (Trait::width == 8) {
            return reinterpret_cast<T>(
                _mm512_unpacklo_pd(builtin_cast<double>(a), builtin_cast<double>(b)));
        } else if constexpr (Trait::width == 16) {
            return reinterpret_cast<T>(
                _mm512_unpacklo_ps(builtin_cast<float>(a), builtin_cast<float>(b)));
        } else if constexpr (Trait::width == 32) {
            return reinterpret_cast<T>(
                _mm512_unpacklo_epi16(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        } else if constexpr (Trait::width == 64) {
            return reinterpret_cast<T>(
                _mm512_unpacklo_epi8(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        }
    } else if constexpr (sizeof(T) == 64) {
        if constexpr (Trait::width == 8) {
            return T{a[0], b[0], a[2], b[2], a[4], b[4], a[6], b[6]};
        } else if constexpr (Trait::width == 16) {
            return T{a[0], b[0], a[1], b[1], a[4],  b[4],  a[5],  b[5],
                     a[8], b[8], a[9], b[9], a[12], b[12], a[13], b[13]};
        } else if constexpr (Trait::width == 32) {
            return T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],
                     a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11],
                     a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19],
                     a[24], b[24], a[25], b[25], a[26], b[26], a[27], b[27]};
        } else if constexpr (Trait::width == 64) {
            return T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],  a[4],  b[4],
                     a[5],  b[5],  a[6],  b[6],  a[7],  b[7],  a[16], b[16], a[17], b[17],
                     a[18], b[18], a[19], b[19], a[20], b[20], a[21], b[21], a[22], b[22],
                     a[23], b[23], a[32], b[32], a[33], b[33], a[34], b[34], a[35], b[35],
                     a[36], b[36], a[37], b[37], a[38], b[38], a[39], b[39], a[48], b[48],
                     a[49], b[49], a[50], b[50], a[51], b[51], a[52], b[52], a[53], b[53],
                     a[54], b[54], a[55], b[55]};
        } else {
            assert_unreachable<T>();
        }
    }
}

template <class A, class B, class T = std::common_type_t<A, B>,
          class Trait = builtin_traits<T>>
constexpr Vc_INTRINSIC T interleave128_hi(A _a, B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const T a(_a);
    const T b(_b);
    if constexpr (sizeof(T) == 16) {
        return interleave_hi(a, b);
    } else if constexpr (sizeof(T) == 32 && needs_intrinsics) {
        if constexpr (Trait::width == 4) {
            return reinterpret_cast<T>(
                _mm256_unpackhi_pd(builtin_cast<double>(a), builtin_cast<double>(b)));
        } else if constexpr (Trait::width == 8) {
            return reinterpret_cast<T>(
                _mm256_unpackhi_ps(builtin_cast<float>(a), builtin_cast<float>(b)));
        } else if constexpr (Trait::width == 16) {
            return reinterpret_cast<T>(
                _mm256_unpackhi_epi16(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        } else if constexpr (Trait::width == 32) {
            return reinterpret_cast<T>(
                _mm256_unpackhi_epi8(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        }
    } else if constexpr (sizeof(T) == 32) {
        if constexpr (Trait::width == 4) {
            return T{a[1], b[1], a[3], b[3]};
        } else if constexpr (Trait::width == 8) {
            return T{a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7]};
        } else if constexpr (Trait::width == 16) {
            return T{a[4],  b[4],  a[5],  b[5],  a[6],  b[6],  a[7],  b[7],
                     a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15]};
        } else if constexpr (Trait::width == 32) {
            return T{a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11],
                     a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15],
                     a[24], b[24], a[25], b[25], a[26], b[26], a[27], b[27],
                     a[28], b[28], a[29], b[29], a[30], b[30], a[31], b[31]};
        } else if constexpr (Trait::width == 64) {
            return T{a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19], a[20], b[20],
                     a[21], b[21], a[22], b[22], a[23], b[23], a[24], b[24], a[25], b[25],
                     a[26], b[26], a[27], b[27], a[28], b[28], a[29], b[29], a[30], b[30],
                     a[31], b[31], a[48], b[48], a[49], b[49], a[50], b[50], a[51], b[51],
                     a[52], b[52], a[53], b[53], a[54], b[54], a[55], b[55], a[56], b[56],
                     a[57], b[57], a[58], b[58], a[59], b[59], a[60], b[60], a[61], b[61],
                     a[62], b[62], a[63], b[63]};
        } else {
            assert_unreachable<T>();
        }
    } else if constexpr (sizeof(T) == 64 && needs_intrinsics) {
        if constexpr (Trait::width == 8) {
            return reinterpret_cast<T>(
                _mm512_unpackhi_pd(builtin_cast<double>(a), builtin_cast<double>(b)));
        } else if constexpr (Trait::width == 16) {
            return reinterpret_cast<T>(
                _mm512_unpackhi_ps(builtin_cast<float>(a), builtin_cast<float>(b)));
        } else if constexpr (Trait::width == 32) {
            return reinterpret_cast<T>(
                _mm512_unpackhi_epi16(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        } else if constexpr (Trait::width == 64) {
            return reinterpret_cast<T>(
                _mm512_unpackhi_epi8(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        }
    } else if constexpr (sizeof(T) == 64) {
        if constexpr (Trait::width == 8) {
            return T{a[1], b[1], a[3], b[3], a[5], b[5], a[7], b[7]};
        } else if constexpr (Trait::width == 16) {
            return T{a[2],  b[2],  a[3],  b[3],  a[6],  b[6],  a[7],  b[7],
                     a[10], b[10], a[11], b[11], a[14], b[14], a[15], b[15]};
        } else if constexpr (Trait::width == 32) {
            return T{a[4],  b[4],  a[5],  b[5],  a[6],  b[6],  a[7],  b[7],
                     a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15],
                     a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23],
                     a[28], b[28], a[29], b[29], a[30], b[30], a[31], b[31]};
        } else if constexpr (Trait::width == 64) {
            return T{a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11], a[12], b[12],
                     a[13], b[13], a[14], b[14], a[15], b[15], a[24], b[24], a[25], b[25],
                     a[26], b[26], a[27], b[27], a[28], b[28], a[29], b[29], a[30], b[30],
                     a[31], b[31], a[40], b[40], a[41], b[41], a[42], b[42], a[43], b[43],
                     a[44], b[44], a[45], b[45], a[46], b[46], a[47], b[47], a[56], b[56],
                     a[57], b[57], a[58], b[58], a[59], b[59], a[60], b[60], a[61], b[61],
                     a[62], b[62], a[63], b[63]};
        } else {
            assert_unreachable<T>();
        }
    }
}

template <class T> struct interleaved_pair {
    T lo, hi;
};

template <class A, class B, class T = std::common_type_t<A, B>,
          class Trait = builtin_traits<T>>
constexpr Vc_INTRINSIC interleaved_pair<T> interleave(A a, B b)
{
    return {interleave_lo(a, b), interleave_hi(a, b)};
}

template <class A, class B, class T = std::common_type_t<A, B>,
          class Trait = builtin_traits<T>>
constexpr Vc_INTRINSIC interleaved_pair<T> interleave128(A a, B b)
{
    return {interleave128_lo(a, b), interleave128_hi(a, b)};
}
}  // namespace detail
}  // namespace Vc_VERSIONED_NAMESPACE

#endif  // VC_DETAIL_INTRINSICS_H_

// vim: foldmethod=marker
