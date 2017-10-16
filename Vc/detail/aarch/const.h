/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DETAIL_AARCH_CONST_H_
#define VC_DETAIL_AARCH_CONST_H_

#include "../const.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
#ifdef Vc_HAVE_NEON
#ifdef Vc_WORK_AROUND_ICE
namespace aarch
{
namespace neon_const
{
#define constexpr const
#else
template <class X> struct constants<simd_abi::neon, X> {
#endif
    alignas(64) static constexpr int    absMaskFloat[4] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
    alignas(16) static constexpr uint   signMaskFloat[4] = {0x80000000, 0x80000000, 0x80000000, 0x80000000};
    alignas(16) static constexpr uint   highMaskFloat[4] = {0xfffff000u, 0xfffff000u, 0xfffff000u, 0xfffff000u};
    alignas(16) static constexpr float  oneFloat[4] = {1.f, 1.f, 1.f, 1.f};

    alignas(16) static constexpr short  minShort[8] = {-0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000};
    alignas(16) static constexpr uchar  one8[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    alignas(16) static constexpr ushort one16[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    alignas(16) static constexpr uint   one32[4] = {1, 1, 1, 1};
    alignas(16) static constexpr ullong one64[2] = {1, 1};

    alignas(16) static constexpr double oneDouble[2] = {1., 1.};
    alignas(16) static constexpr ullong highMaskDouble[2] = {0xfffffffff8000000ull, 0xfffffffff8000000ull};
    alignas(16) static constexpr llong  absMaskDouble[2] = {0x7fffffffffffffffll, 0x7fffffffffffffffll};
    alignas(16) static constexpr ullong signMaskDouble[2] = {0x8000000000000000ull, 0x8000000000000000ull};
    alignas(16) static constexpr ullong frexpMask[2] = {0xbfefffffffffffffull, 0xbfefffffffffffffull};

    alignas(16) static constexpr uint   IndexesFromZero4[4] = { 0, 1, 2, 3 };
    alignas(16) static constexpr ushort IndexesFromZero8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    alignas(16) static constexpr uchar  IndexesFromZero16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    alignas(16) static constexpr uint   AllBitsSet[4] = { 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU };
#ifdef Vc_WORK_AROUND_ICE
#undef constexpr
}  // namespace neon_const
}  // namespace aarch
#else   // Vc_WORK_AROUND_ICE
};
template <class X> alignas(64) constexpr int    constants<simd_abi::neon, X>::absMaskFloat[4];
template <class X> alignas(16) constexpr uint   constants<simd_abi::neon, X>::signMaskFloat[4];
template <class X> alignas(16) constexpr uint   constants<simd_abi::neon, X>::highMaskFloat[4];
template <class X> alignas(16) constexpr float  constants<simd_abi::neon, X>::oneFloat[4];
template <class X> alignas(16) constexpr short  constants<simd_abi::neon, X>::minShort[8];
template <class X> alignas(16) constexpr uchar  constants<simd_abi::neon, X>::one8[16];
template <class X> alignas(16) constexpr ushort constants<simd_abi::neon, X>::one16[8];
template <class X> alignas(16) constexpr uint   constants<simd_abi::neon, X>::one32[4];
template <class X> alignas(16) constexpr ullong constants<simd_abi::neon, X>::one64[2];
template <class X> alignas(16) constexpr double constants<simd_abi::neon, X>::oneDouble[2];
template <class X> alignas(16) constexpr ullong constants<simd_abi::neon, X>::highMaskDouble[2];
template <class X> alignas(16) constexpr llong  constants<simd_abi::neon, X>::absMaskDouble[2];
template <class X> alignas(16) constexpr ullong constants<simd_abi::neon, X>::signMaskDouble[2];
template <class X> alignas(16) constexpr ullong constants<simd_abi::neon, X>::frexpMask[2];
template <class X> alignas(16) constexpr uint   constants<simd_abi::neon, X>::IndexesFromZero4[4];
template <class X> alignas(16) constexpr ushort constants<simd_abi::neon, X>::IndexesFromZero8[8];
template <class X> alignas(16) constexpr uchar  constants<simd_abi::neon, X>::IndexesFromZero16[16];
template <class X> alignas(16) constexpr uint   constants<simd_abi::neon, X>::AllBitsSet[4];
namespace aarch
{
using neon_const = constants<simd_abi::neon>;
}  // namespace aarch
#endif  // Vc_WORK_AROUND_ICE
#endif  // Vc_HAVE_NEON

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_AARCH_CONST_H_

// vim: foldmethod=marker
