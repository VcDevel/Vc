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

#ifndef VC_DETAIL_CONVERT_MASK_H_
#define VC_DETAIL_CONVERT_MASK_H_

#include "macros.h"
#include "intrinsics.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace detail
{
// is_bitset {{{
template <class T> struct is_bitset : std::false_type {};
template <size_t N> struct is_bitset<std::bitset<N>> : std::true_type {};
template <class T> inline constexpr bool is_bitset_v = is_bitset<T>::value;

// }}}
// is_storage {{{
template <class T> struct is_storage : std::false_type {};
template <class T, size_t N> struct is_storage<Storage<T, N>> : std::true_type {};
template <class T> inline constexpr bool is_storage_v = is_storage<T>::value;

// }}}
// convert_mask{{{
template <class To, class From> inline To convert_mask(From k) {
    if constexpr (std::is_same_v<To, From>) {  // also covers bool -> bool
        return k;
    } else if constexpr (std::is_unsigned_v<From> && std::is_unsigned_v<To>) {
        // bits -> bits
        return k;  // zero-extends or truncates
    } else if constexpr (is_bitset_v<From>) {
        // from std::bitset {{{
        static_assert(k.size() <= sizeof(ullong) * CHAR_BIT);
        using T = std::conditional_t<
            (k.size() <= sizeof(ushort) * CHAR_BIT),
            std::conditional_t<(k.size() <= CHAR_BIT), uchar, ushort>,
            std::conditional_t<(k.size() <= sizeof(uint) * CHAR_BIT), uint, ullong>>;
        return convert_mask<To>(static_cast<T>(k.to_ullong()));
        // }}}
    } else if constexpr (is_bitset_v<To>) {
        // to std::bitset {{{
        static_assert(To().size() <= sizeof(ullong) * CHAR_BIT);
        using T = std::conditional_t<
            (To().size() <= sizeof(ushort) * CHAR_BIT),
            std::conditional_t<(To().size() <= CHAR_BIT), uchar, ushort>,
            std::conditional_t<(To().size() <= sizeof(uint) * CHAR_BIT), uint, ullong>>;
        return convert_mask<T>(k);
        // }}}
    } else if constexpr (is_storage_v<From>) {
        return convert_mask<To>(k.d);
    } else if constexpr (is_storage_v<To>) {
        return convert_mask<typename To::register_type>(k);
    } else if constexpr (std::is_unsigned_v<From> && is_builtin_vector_v<To>) {
        // bits -> vector {{{
        using Trait = builtin_traits<To>;
        constexpr size_t N_in = sizeof(From) * CHAR_BIT;
        using ToT = typename Trait::value_type;
        constexpr size_t N_out = Trait::width;
        constexpr size_t N = std::min(N_in, N_out);
        constexpr size_t bytes_per_output_element = sizeof(ToT);
        if constexpr (have_avx512f) {
            if constexpr (bytes_per_output_element == 1 && sizeof(To) == 16) {
                if constexpr (have_avx512bw_vl) {
                    return builtin_cast<ToT>(_mm_movm_epi8(k));
                } else if constexpr (have_avx512bw) {
                    return builtin_cast<ToT>(lo128(_mm512_movm_epi8(k)));
                } else {
                    auto as32bits = _mm512_maskz_mov_epi32(k, ~__m512i());
                    auto as16bits = fixup_avx_xzyw(
                        _mm256_packs_epi32(lo256(as32bits), hi256(as32bits)));
                    return builtin_cast<ToT>(
                        _mm_packs_epi16(lo128(as16bits), hi128(as16bits)));
                }
            } else if constexpr (bytes_per_output_element == 1 && sizeof(To) == 32) {
                if constexpr (have_avx512bw_vl) {
                    return builtin_cast<ToT>(_mm256_movm_epi8(k));
                } else if constexpr (have_avx512bw) {
                    return builtin_cast<ToT>(lo256(_mm512_movm_epi8(k)));
                } else {
                    auto as16bits =  // 0 16 1 17 ... 15 31
                        _mm512_srli_epi32(_mm512_maskz_mov_epi32(k, ~__m512i()), 16) |
                        _mm512_slli_epi32(_mm512_maskz_mov_epi32(k >> 16, ~__m512i()),
                                          16);
                    auto _0_16_1_17 = fixup_avx_xzyw(_mm256_packs_epi16(
                        lo256(as16bits),
                        hi256(as16bits))  // 0 16 1 17 2 18 3 19 8 24 9 25 ...
                    );
                    // deinterleave:
                    return builtin_cast<ToT>(fixup_avx_xzyw(_mm256_shuffle_epi8(
                        _0_16_1_17,  // 0 16 1 17 2 ...
                        _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13,
                                         15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11,
                                         13, 15))));  // 0-7 16-23 8-15 24-31 -> xzyw
                                                      // 0-3  8-11 16-19 24-27
                                                      // 4-7 12-15 20-23 28-31
                }
            } else if constexpr (bytes_per_output_element == 1 && sizeof(To) == 64) {
                return reinterpret_cast<builtin_type_t<schar, 64>>(_mm512_movm_epi8(k));
            } else if constexpr (bytes_per_output_element == 2 && sizeof(To) == 16) {
                if constexpr (have_avx512bw_vl) {
                    return builtin_cast<ToT>(_mm_movm_epi16(k));
                } else if constexpr (have_avx512bw) {
                    return builtin_cast<ToT>(lo128(_mm512_movm_epi16(k)));
                } else {
                    __m256i as32bits;
                    if constexpr (have_avx512vl) {
                        as32bits = _mm256_maskz_mov_epi32(k, ~__m256i());
                    } else {
                        as32bits = lo256(_mm512_maskz_mov_epi32(k, ~__m512i()));
                    }
                    return builtin_cast<ToT>(
                        _mm_packs_epi32(lo128(as32bits), hi128(as32bits)));
                }
            } else if constexpr (bytes_per_output_element == 2 && sizeof(To) == 32) {
                if constexpr (have_avx512bw_vl) {
                    return builtin_cast<ToT>(_mm256_movm_epi16(k));
                } else if constexpr (have_avx512bw) {
                    return builtin_cast<ToT>(lo256(_mm512_movm_epi16(k)));
                } else {
                    auto as32bits = _mm512_maskz_mov_epi32(k, ~__m512i());
                    return builtin_cast<ToT>(fixup_avx_xzyw(
                        _mm256_packs_epi32(lo256(as32bits), hi256(as32bits))));
                }
            } else if constexpr (bytes_per_output_element == 2 && sizeof(To) == 64) {
                return builtin_cast<ToT>(_mm512_movm_epi16(k));
            } else if constexpr (bytes_per_output_element == 4 && sizeof(To) == 16) {
                return builtin_cast<ToT>(
                    have_avx512dq_vl
                        ? _mm_movm_epi32(k)
                        : have_avx512dq
                              ? lo128(_mm512_movm_epi32(k))
                              : have_avx512vl
                                    ? _mm_maskz_mov_epi32(k, ~__m128i())
                                    : lo128(_mm512_maskz_mov_epi32(k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 4 && sizeof(To) == 32) {
                return builtin_cast<ToT>(
                    have_avx512dq_vl
                        ? _mm256_movm_epi32(k)
                        : have_avx512dq
                              ? lo256(_mm512_movm_epi32(k))
                              : have_avx512vl
                                    ? _mm256_maskz_mov_epi32(k, ~__m256i())
                                    : lo256(_mm512_maskz_mov_epi32(k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 4 && sizeof(To) == 64) {
                return builtin_cast<ToT>(have_avx512dq
                                             ? _mm512_movm_epi32(k)
                                             : _mm512_maskz_mov_epi32(k, ~__m512i()));
            } else if constexpr (bytes_per_output_element == 8 && sizeof(To) == 16) {
                return builtin_cast<ToT>(
                    have_avx512dq_vl
                        ? _mm_movm_epi64(k)
                        : have_avx512dq
                              ? lo128(_mm512_movm_epi64(k))
                              : have_avx512vl
                                    ? _mm_maskz_mov_epi64(k, ~__m128i())
                                    : lo128(_mm512_maskz_mov_epi64(k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 8 && sizeof(To) == 32) {
                return builtin_cast<ToT>(
                    have_avx512dq_vl
                        ? _mm256_movm_epi64(k)
                        : have_avx512dq
                              ? lo256(_mm512_movm_epi64(k))
                              : have_avx512vl
                                    ? _mm256_maskz_mov_epi64(k, ~__m256i())
                                    : lo256(_mm512_maskz_mov_epi64(k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 8 && sizeof(To) == 64) {
                return builtin_cast<ToT>(have_avx512dq
                                             ? _mm512_movm_epi64(k)
                                             : _mm512_maskz_mov_epi64(k, ~__m512i()));
            } else {
                assert_unreachable<To>();
            }
        } else if constexpr (have_sse) {
            using U = std::make_unsigned_t<detail::int_for_sizeof_t<ToT>>;
            using V = builtin_type_t<U, N>;  // simd<U, Abi>;
            static_assert(sizeof(V) <= 32);  // can't be AVX512
            constexpr size_t bits_per_element = sizeof(U) * CHAR_BIT;
            if constexpr (!have_avx2 && have_avx && sizeof(V) == 32) {
                if constexpr (N == 8) {
                    return _mm256_cmp_ps(
                        _mm256_and_ps(
                            _mm256_castsi256_ps(_mm256_set1_epi32(k)),
                            _mm256_castsi256_ps(_mm256_setr_epi32(
                                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80))),
                        _mm256_setzero_ps(), _CMP_NEQ_UQ);
                } else if constexpr (N == 4) {
                    return _mm256_cmp_pd(
                        _mm256_and_pd(
                            _mm256_castsi256_pd(_mm256_set1_epi64x(k)),
                            _mm256_castsi256_pd(
                                _mm256_setr_epi64x(0x01, 0x02, 0x04, 0x08))),
                        _mm256_setzero_pd(), _CMP_NEQ_UQ);
                } else {
                    assert_unreachable<To>();
                }
            } else if constexpr (bits_per_element >= N) {
                constexpr auto bitmask = generate_builtin<builtin_type_t<U, N>>(
                    [](auto i) -> U { return 1ull << i; });
                return builtin_cast<ToT>(
                    (builtin_broadcast<N, U>(k) & bitmask) != 0);
            } else if constexpr (sizeof(V) == 16 && sizeof(ToT) == 1 && have_ssse3) {
                const auto bitmask = to_intrin(make_builtin<uchar>(
                    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128));
                return builtin_cast<ToT>(
                    builtin_cast<ToT>(
                        _mm_shuffle_epi8(
                            to_intrin(builtin_type_t<ullong, 2>{k}),
                            _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                                          1)) &
                        bitmask) != 0);
            } else if constexpr (sizeof(V) == 32 && sizeof(ToT) == 1 && have_avx2) {
                const auto bitmask =
                    _mm256_broadcastsi128_si256(to_intrin(make_builtin<uchar>(
                        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128)));
                return builtin_cast<ToT>(
                    builtin_cast<ToT>(_mm256_shuffle_epi8(
                                        _mm256_broadcastsi128_si256(to_intrin(
                                            builtin_type_t<ullong, 2>{k})),
                                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                                                         1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                                         2, 2, 3, 3, 3, 3, 3, 3, 3, 3)) &
                                    bitmask) != 0);
                /* TODO:
                } else if constexpr (sizeof(V) == 32 && sizeof(ToT) == 2 && have_avx2) {
                    constexpr auto bitmask = _mm256_broadcastsi128_si256(
                        _mm_setr_epi8(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000)); return
                builtin_cast<ToT>( _mm256_shuffle_epi8(
                                   _mm256_broadcastsi128_si256(__m128i{k}),
                                   _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)) & bitmask) != 0;
                */
            } else {
                const V tmp = generate_builtin<V>([&](auto i) {
                                  return static_cast<U>(
                                      k >> (bits_per_element * (i / bits_per_element)));
                              }) &
                              generate_builtin<V>([](auto i) {
                                  return static_cast<U>(1ull << (i % bits_per_element));
                              });  // mask bit index
                return builtin_cast<ToT>(tmp != V());
            }
        } else {
            assert_unreachable<To>();
        } // }}}
    } else if constexpr (is_builtin_vector_v<From> && std::is_unsigned_v<To>) {
        // vector -> bits {{{
        using Trait = builtin_traits<From>;
        using T = typename Trait::value_type;
        constexpr size_t FromN = Trait::width;
        constexpr size_t cvt_id = FromN * 10 + sizeof(T);
        constexpr bool have_avx512_int = have_avx512f && std::is_integral_v<T>;
        [[maybe_unused]]  // PR85827
        const auto intrin = to_intrin(k);

             if constexpr (cvt_id == 16'1 && have_avx512bw_vl) { return    _mm_movepi8_mask(intrin); }
        else if constexpr (cvt_id == 16'1 && have_avx512bw   ) { return _mm512_movepi8_mask(zeroExtend(intrin)); }
        else if constexpr (cvt_id == 16'1                    ) { return    _mm_movemask_epi8(intrin); }
        else if constexpr (cvt_id == 32'1 && have_avx512bw_vl) { return _mm256_movepi8_mask(intrin); }
        else if constexpr (cvt_id == 32'1 && have_avx512bw   ) { return _mm512_movepi8_mask(zeroExtend(intrin)); }
        else if constexpr (cvt_id == 32'1                    ) { return _mm256_movemask_epi8(intrin); }
        else if constexpr (cvt_id == 64'1 && have_avx512bw   ) { return _mm512_movepi8_mask(intrin); }
        else if constexpr (cvt_id ==  8'2 && have_avx512bw_vl) { return    _mm_movepi16_mask(intrin); }
        else if constexpr (cvt_id ==  8'2 && have_avx512bw   ) { return _mm512_movepi16_mask(zeroExtend(intrin)); }
        else if constexpr (cvt_id ==  8'2                    ) { return x86::movemask_epi16(intrin); }
        else if constexpr (cvt_id == 16'2 && have_avx512bw_vl) { return _mm256_movepi16_mask(intrin); }
        else if constexpr (cvt_id == 16'2 && have_avx512bw   ) { return _mm512_movepi16_mask(zeroExtend(intrin)); }
        else if constexpr (cvt_id == 16'2                    ) { return x86::movemask_epi16(intrin); }
        else if constexpr (cvt_id == 32'2 && have_avx512bw   ) { return _mm512_movepi16_mask(intrin); }
        else if constexpr (cvt_id ==  4'4 && have_avx512dq_vl) { return    _mm_movepi32_mask(builtin_cast<llong>(k)); }
        else if constexpr (cvt_id ==  4'4 && have_avx512dq   ) { return _mm512_movepi32_mask(zeroExtend(builtin_cast<llong>(k))); }
        else if constexpr (cvt_id ==  4'4 && have_avx512vl   ) { return    _mm_cmp_epi32_mask(builtin_cast<llong>(k), __m128i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'4 && have_avx512_int ) { return _mm512_cmp_epi32_mask(zeroExtend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'4                    ) { return    _mm_movemask_ps(k); }
        else if constexpr (cvt_id ==  8'4 && have_avx512dq_vl) { return _mm256_movepi32_mask(builtin_cast<llong>(k)); }
        else if constexpr (cvt_id ==  8'4 && have_avx512dq   ) { return _mm512_movepi32_mask(zeroExtend(builtin_cast<llong>(k))); }
        else if constexpr (cvt_id ==  8'4 && have_avx512vl   ) { return _mm256_cmp_epi32_mask(builtin_cast<llong>(k), __m256i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  8'4 && have_avx512_int ) { return _mm512_cmp_epi32_mask(zeroExtend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  8'4                    ) { return _mm256_movemask_ps(k); }
        else if constexpr (cvt_id == 16'4 && have_avx512dq   ) { return _mm512_movepi32_mask(builtin_cast<llong>(k)); }
        else if constexpr (cvt_id == 16'4                    ) { return _mm512_cmp_epi32_mask(builtin_cast<llong>(k), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8 && have_avx512dq_vl) { return    _mm_movepi64_mask(builtin_cast<llong>(k)); }
        else if constexpr (cvt_id ==  2'8 && have_avx512dq   ) { return _mm512_movepi64_mask(zeroExtend(builtin_cast<llong>(k))); }
        else if constexpr (cvt_id ==  2'8 && have_avx512vl   ) { return    _mm_cmp_epi64_mask(builtin_cast<llong>(k), __m128i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8 && have_avx512_int ) { return _mm512_cmp_epi64_mask(zeroExtend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8                    ) { return    _mm_movemask_pd(k); }
        else if constexpr (cvt_id ==  4'8 && have_avx512dq_vl) { return _mm256_movepi64_mask(builtin_cast<llong>(k)); }
        else if constexpr (cvt_id ==  4'8 && have_avx512dq   ) { return _mm512_movepi64_mask(zeroExtend(builtin_cast<llong>(k))); }
        else if constexpr (cvt_id ==  4'8 && have_avx512vl   ) { return _mm256_cmp_epi64_mask(builtin_cast<llong>(k), __m256i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'8 && have_avx512_int ) { return _mm512_cmp_epi64_mask(zeroExtend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'8                    ) { return _mm256_movemask_pd(k); }
        else if constexpr (cvt_id ==  8'8 && have_avx512dq   ) { return _mm512_movepi64_mask(builtin_cast<llong>(k)); }
        else if constexpr (cvt_id ==  8'8                    ) { return _mm512_cmp_epi64_mask(builtin_cast<llong>(k), __m512i(), _MM_CMPINT_LT); }
        else { assert_unreachable<To>(); }
        // }}}
    } else if constexpr (is_builtin_vector_v<From> && is_builtin_vector_v<To>) {
        // vector -> vector {{{
        using ToTrait = builtin_traits<To>;
        using FromTrait = builtin_traits<From>;
        using ToT = typename ToTrait::value_type;
        using T = typename FromTrait::value_type;
        constexpr size_t FromN = FromTrait::width;
        constexpr size_t ToN = ToTrait::width;
        constexpr int FromBytes = sizeof(T);
        constexpr int ToBytes = sizeof(ToT);

        if constexpr (FromN == ToN && sizeof(From) == sizeof(To)) {
            // reinterpret the bits
            return reinterpret_cast<To>(k);
        } else if constexpr (sizeof(To) == 16 && sizeof(k) == 16) {
            // SSE -> SSE {{{
            if constexpr (FromBytes == 4 && ToBytes == 8) {
                if constexpr(std::is_integral_v<T>) {
                    return builtin_cast<ToT>(interleave128_lo(k, k));
                } else {
                    return builtin_cast<ToT>(interleave128_lo(k, k));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 8) {
                const auto y = builtin_cast<int>(interleave128_lo(k, k));
                return builtin_cast<ToT>(interleave128_lo(y, y));
            } else if constexpr (FromBytes == 1 && ToBytes == 8) {
                auto y = builtin_cast<short>(interleave128_lo(k, k));
                auto z = builtin_cast<int>(interleave128_lo(y, y));
                return builtin_cast<ToT>(interleave128_lo(z, z));
            } else if constexpr (FromBytes == 8 && ToBytes == 4) {
                if constexpr (std::is_floating_point_v<T>) {
                    return builtin_cast<ToT>(_mm_shuffle_ps(builtin_cast<float>(k), __m128(),
                                                     make_immediate<4>(1, 3, 1, 3)));
                } else {
                    auto y = builtin_cast<llong>(k);
                    return builtin_cast<ToT>(_mm_packs_epi32(y, __m128i()));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 4) {
                return builtin_cast<ToT>(interleave128_lo(k, k));
            } else if constexpr (FromBytes == 1 && ToBytes == 4) {
                const auto y = builtin_cast<short>(interleave128_lo(k, k));
                return builtin_cast<ToT>(interleave128_lo(y, y));
            } else if constexpr (FromBytes == 8 && ToBytes == 2) {
                if constexpr(have_ssse3) {
                    return builtin_cast<ToT>(
                        _mm_shuffle_epi8(builtin_cast<llong>(k),
                                         _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    const auto y = _mm_packs_epi32(builtin_cast<llong>(k), __m128i());
                    return builtin_cast<ToT>(_mm_packs_epi32(y, __m128i()));
                }
            } else if constexpr (FromBytes == 4 && ToBytes == 2) {
                return builtin_cast<ToT>(
                    _mm_packs_epi32(builtin_cast<llong>(k), __m128i()));
            } else if constexpr (FromBytes == 1 && ToBytes == 2) {
                return builtin_cast<ToT>(interleave128_lo(k, k));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {
                if constexpr(have_ssse3) {
                    return builtin_cast<ToT>(
                        _mm_shuffle_epi8(builtin_cast<llong>(k),
                                         _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    auto y = _mm_packs_epi32(builtin_cast<llong>(k), __m128i());
                    y = _mm_packs_epi32(y, __m128i());
                    return builtin_cast<ToT>(_mm_packs_epi16(y, __m128i()));
                }
            } else if constexpr (FromBytes == 4 && ToBytes == 1) {
                if constexpr(have_ssse3) {
                    return builtin_cast<ToT>(
                        _mm_shuffle_epi8(builtin_cast<llong>(k),
                                         _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    const auto y = _mm_packs_epi32(builtin_cast<llong>(k), __m128i());
                    return builtin_cast<ToT>(_mm_packs_epi16(y, __m128i()));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 1) {
                return builtin_cast<ToT>(_mm_packs_epi16(builtin_cast<llong>(k), __m128i()));
            } else {
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(To) == 32 && sizeof(k) == 32) {
            // AVX -> AVX {{{
            if constexpr (FromBytes == ToBytes) {  // keep low 1/2
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            } else if constexpr (FromBytes == ToBytes * 2) {
                const auto y = builtin_cast<llong>(k);
                return builtin_cast<ToT>(
                    _mm256_castsi128_si256(_mm_packs_epi16(lo128(y), hi128(y))));
            } else if constexpr (FromBytes == ToBytes * 4) {
                const auto y = builtin_cast<llong>(k);
                return builtin_cast<ToT>(_mm256_castsi128_si256(
                    _mm_packs_epi16(_mm_packs_epi16(lo128(y), hi128(y)), __m128i())));
            } else if constexpr (FromBytes == ToBytes * 8) {
                const auto y = builtin_cast<llong>(k);
                return builtin_cast<ToT>(_mm256_castsi128_si256(
                    _mm_shuffle_epi8(_mm_packs_epi16(lo128(y), hi128(y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1))));
            } else if constexpr (FromBytes * 2 == ToBytes) {
                auto y = fixup_avx_xzyw(to_intrin(k));
                if constexpr(std::is_floating_point_v<T>) {
                    return builtin_cast<ToT>(_mm256_unpacklo_ps(y, y));
                } else {
                    return builtin_cast<ToT>(_mm256_unpacklo_epi8(y, y));
                }
            } else if constexpr (FromBytes * 4 == ToBytes) {
                auto y = _mm_unpacklo_epi8(lo128(builtin_cast<llong>(k)),
                                           lo128(builtin_cast<llong>(k)));  // drops 3/4 of input
                return builtin_cast<ToT>(
                    concat(_mm_unpacklo_epi16(y, y), _mm_unpackhi_epi16(y, y)));
            } else if constexpr (FromBytes == 1 && ToBytes == 8) {
                auto y = _mm_unpacklo_epi8(lo128(builtin_cast<llong>(k)),
                                           lo128(builtin_cast<llong>(k)));  // drops 3/4 of input
                y = _mm_unpacklo_epi16(y, y);  // drops another 1/2 => 7/8 total
                return builtin_cast<ToT>(
                    concat(_mm_unpacklo_epi32(y, y), _mm_unpackhi_epi32(y, y)));
            } else {
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(To) == 32 && sizeof(k) == 16) {
            // SSE -> AVX {{{
            if constexpr (FromBytes == ToBytes) {
                return builtin_cast<ToT>(
                    intrinsic_type_t<T, 32 / sizeof(T)>(zeroExtend(to_intrin(k))));
            } else if constexpr (FromBytes * 2 == ToBytes) {  // keep all
                return builtin_cast<ToT>(concat(_mm_unpacklo_epi8(builtin_cast<llong>(k), builtin_cast<llong>(k)),
                                         _mm_unpackhi_epi8(builtin_cast<llong>(k), builtin_cast<llong>(k))));
            } else if constexpr (FromBytes * 4 == ToBytes) {
                if constexpr (have_avx2) {
                    return builtin_cast<ToT>(_mm256_shuffle_epi8(
                        concat(builtin_cast<llong>(k), builtin_cast<llong>(k)),
                        _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7,
                                         7)));
                } else {
                    return builtin_cast<ToT>(
                        concat(_mm_shuffle_epi8(builtin_cast<llong>(k),
                                                _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2,
                                                              2, 2, 2, 3, 3, 3, 3)),
                               _mm_shuffle_epi8(builtin_cast<llong>(k),
                                                _mm_setr_epi8(4, 4, 4, 4, 5, 5, 5, 5, 6,
                                                              6, 6, 6, 7, 7, 7, 7))));
                }
            } else if constexpr (FromBytes * 8 == ToBytes) {
                if constexpr (have_avx2) {
                    return builtin_cast<ToT>(_mm256_shuffle_epi8(
                        concat(builtin_cast<llong>(k), builtin_cast<llong>(k)),
                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                                         3)));
                } else {
                    return builtin_cast<ToT>(
                        concat(_mm_shuffle_epi8(builtin_cast<llong>(k),
                                                _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                              1, 1, 1, 1, 1, 1, 1)),
                               _mm_shuffle_epi8(builtin_cast<llong>(k),
                                                _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3,
                                                              3, 3, 3, 3, 3, 3, 3))));
                }
            } else if constexpr (FromBytes == ToBytes * 2) {
                return builtin_cast<ToT>(
                    __m256i(zeroExtend(_mm_packs_epi16(builtin_cast<llong>(k), __m128i()))));
            } else if constexpr (FromBytes == 8 && ToBytes == 2) {
                return builtin_cast<ToT>(__m256i(zeroExtend(
                    _mm_shuffle_epi8(builtin_cast<llong>(k),
                                     _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else if constexpr (FromBytes == 4 && ToBytes == 1) {
                return builtin_cast<ToT>(__m256i(zeroExtend(
                    _mm_shuffle_epi8(builtin_cast<llong>(k),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {
                return builtin_cast<ToT>(__m256i(zeroExtend(
                    _mm_shuffle_epi8(builtin_cast<llong>(k),
                                     _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else {
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(To) == 16 && sizeof(k) == 32) {
            // AVX -> SSE {{{
            if constexpr (FromBytes == ToBytes) {  // keep low 1/2
                return builtin_cast<ToT>(lo128(k));
            } else if constexpr (FromBytes == ToBytes * 2) {  // keep all
                auto y = builtin_cast<llong>(k);
                return builtin_cast<ToT>(_mm_packs_epi16(lo128(y), hi128(y)));
            } else if constexpr (FromBytes == ToBytes * 4) {  // add 1/2 undef
                auto y = builtin_cast<llong>(k);
                return builtin_cast<ToT>(
                    _mm_packs_epi16(_mm_packs_epi16(lo128(y), hi128(y)), __m128i()));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {  // add 3/4 undef
                auto y = builtin_cast<llong>(k);
                return builtin_cast<ToT>(
                    _mm_shuffle_epi8(_mm_packs_epi16(lo128(y), hi128(y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)));
            } else if constexpr (FromBytes * 2 == ToBytes) {  // keep low 1/4
                auto y = lo128(builtin_cast<llong>(k));
                return builtin_cast<ToT>(_mm_unpacklo_epi8(y, y));
            } else if constexpr (FromBytes * 4 == ToBytes) {  // keep low 1/8
                auto y = lo128(builtin_cast<llong>(k));
                y = _mm_unpacklo_epi8(y, y);
                return builtin_cast<ToT>(_mm_unpacklo_epi8(y, y));
            } else if constexpr (FromBytes * 8 == ToBytes) {  // keep low 1/16
                auto y = lo128(builtin_cast<llong>(k));
                y = _mm_unpacklo_epi8(y, y);
                y = _mm_unpacklo_epi8(y, y);
                return builtin_cast<ToT>(_mm_unpacklo_epi8(y, y));
            } else {
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            }
            // }}}
        }
        // }}}
    } else {
        assert_unreachable<To>();
    }
}

// }}}
}  // namespace detail
}  // namespace Vc_VERSIONED_NAMESPACE

#endif  // VC_DETAIL_CONVERT_MASK_H_

// vim: foldmethod=marker
