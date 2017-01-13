/*  This file is part of the Vc library. {{{
Copyright © 2016 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DATAPAR_CONST_H_
#define VC_DATAPAR_CONST_H_

#include "synopsis.h"

#if defined Vc_MSVC && Vc_MSVC < 191024903
#define Vc_WORK_AROUND_ICE
#endif

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {

template <class T, class = void> struct constants;

#ifdef Vc_HAVE_SSE_ABI
#ifdef Vc_WORK_AROUND_ICE
namespace x86
{
namespace sse_const
{
#define constexpr const
#else
template <class X> struct constants<datapar_abi::sse, X> {
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
    alignas(16) static constexpr uchar cvti16_i08_shuffle[16] = {
        0, 2, 4, 6, 8, 10, 12, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
#ifdef Vc_WORK_AROUND_ICE
#undef constexpr
}  // namespace sse_const
}  // namespace x86
#else   // Vc_WORK_AROUND_ICE
};
template <class X> alignas(64) constexpr int    constants<datapar_abi::sse, X>::absMaskFloat[4];
template <class X> alignas(16) constexpr uint   constants<datapar_abi::sse, X>::signMaskFloat[4];
template <class X> alignas(16) constexpr uint   constants<datapar_abi::sse, X>::highMaskFloat[4];
template <class X> alignas(16) constexpr float  constants<datapar_abi::sse, X>::oneFloat[4];
template <class X> alignas(16) constexpr short  constants<datapar_abi::sse, X>::minShort[8];
template <class X> alignas(16) constexpr uchar  constants<datapar_abi::sse, X>::one8[16];
template <class X> alignas(16) constexpr ushort constants<datapar_abi::sse, X>::one16[8];
template <class X> alignas(16) constexpr uint   constants<datapar_abi::sse, X>::one32[4];
template <class X> alignas(16) constexpr ullong constants<datapar_abi::sse, X>::one64[2];
template <class X> alignas(16) constexpr double constants<datapar_abi::sse, X>::oneDouble[2];
template <class X> alignas(16) constexpr ullong constants<datapar_abi::sse, X>::highMaskDouble[2];
template <class X> alignas(16) constexpr llong  constants<datapar_abi::sse, X>::absMaskDouble[2];
template <class X> alignas(16) constexpr ullong constants<datapar_abi::sse, X>::signMaskDouble[2];
template <class X> alignas(16) constexpr ullong constants<datapar_abi::sse, X>::frexpMask[2];
template <class X> alignas(16) constexpr uint   constants<datapar_abi::sse, X>::IndexesFromZero4[4];
template <class X> alignas(16) constexpr ushort constants<datapar_abi::sse, X>::IndexesFromZero8[8];
template <class X> alignas(16) constexpr uchar  constants<datapar_abi::sse, X>::IndexesFromZero16[16];
template <class X> alignas(16) constexpr uint   constants<datapar_abi::sse, X>::AllBitsSet[4];
template <class X> alignas(16) constexpr uchar  constants<datapar_abi::sse, X>::cvti16_i08_shuffle[16];
namespace x86
{
using sse_const = constants<datapar_abi::sse>;
}  // namespace x86
#endif  // Vc_WORK_AROUND_ICE
#endif  // Vc_HAVE_SSE_ABI

#ifdef Vc_HAVE_AVX
#ifdef Vc_WORK_AROUND_ICE
namespace x86
{
namespace avx_const
{
#define constexpr const
#else   // Vc_WORK_AROUND_ICE
template <class X> struct constants<datapar_abi::avx, X> {
#endif  // Vc_WORK_AROUND_ICE
    alignas(64) static constexpr ullong IndexesFromZero64[ 4] = { 0, 1, 2, 3 };
    alignas(32) static constexpr uint   IndexesFromZero32[ 8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    alignas(32) static constexpr ushort IndexesFromZero16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    alignas(32) static constexpr uchar  IndexesFromZero8 [32] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };

    alignas(32) static constexpr uint AllBitsSet[8] = {
        0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU
    };

    static constexpr   uint absMaskFloat[2] = { 0xffffffffu, 0x7fffffffu };
    static constexpr   uint signMaskFloat[2] = { 0x0u, 0x80000000u };
    static constexpr   uint highMaskFloat = 0xfffff000u;
    static constexpr  float oneFloat = 1.f;
    alignas(float) static constexpr ushort one16[2] = { 1, 1 };
    alignas(float) static constexpr  uchar one8[4] = { 1, 1, 1, 1 };
    static constexpr  float _2_pow_31 = 1u << 31;
    static constexpr ullong highMaskDouble = 0xfffffffff8000000ull;
    static constexpr double oneDouble = 1.;
    static constexpr ullong frexpMask = 0xbfefffffffffffffull;
#ifdef Vc_WORK_AROUND_ICE
#undef constexpr
#undef Vc_WORK_AROUND_ICE
}  // namespace avx_const
}  // namespace x86
#else   // Vc_WORK_AROUND_ICE
};
template <class X> alignas(64) constexpr ullong constants<datapar_abi::avx, X>::IndexesFromZero64[ 4];
template <class X> alignas(32) constexpr uint   constants<datapar_abi::avx, X>::IndexesFromZero32[ 8];
template <class X> alignas(32) constexpr ushort constants<datapar_abi::avx, X>::IndexesFromZero16[16];
template <class X> alignas(32) constexpr uchar  constants<datapar_abi::avx, X>::IndexesFromZero8 [32];
template <class X> alignas(32) constexpr uint   constants<datapar_abi::avx, X>::AllBitsSet[8];
template <class X> constexpr   uint constants<datapar_abi::avx, X>::absMaskFloat[2];
template <class X> constexpr   uint constants<datapar_abi::avx, X>::signMaskFloat[2];
template <class X> constexpr   uint constants<datapar_abi::avx, X>::highMaskFloat;
template <class X> constexpr  float constants<datapar_abi::avx, X>::oneFloat;
template <class X> alignas(float) constexpr ushort constants<datapar_abi::avx, X>::one16[2];
template <class X> alignas(float) constexpr  uchar constants<datapar_abi::avx, X>::one8[4];
template <class X> constexpr  float constants<datapar_abi::avx, X>::_2_pow_31;
template <class X> constexpr ullong constants<datapar_abi::avx, X>::highMaskDouble;
template <class X> constexpr double constants<datapar_abi::avx, X>::oneDouble;
template <class X> constexpr ullong constants<datapar_abi::avx, X>::frexpMask;
namespace x86
{
using avx_const = constants<datapar_abi::avx>;
}  // namespace x86
#endif  // Vc_WORK_AROUND_ICE
#endif  // Vc_HAVE_AVX

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END
#endif  // VC_DATAPAR_CONST_H_
