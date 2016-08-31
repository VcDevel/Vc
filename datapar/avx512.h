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

#ifndef VC_DATAPAR_AVX512_H_
#define VC_DATAPAR_AVX512_H_

#include "macros.h"
#include "storage.h"
#include "x86/intrinsics.h"
#include "x86/convert.h"
#include "x86/compares.h"

namespace Vc_VERSIONED_NAMESPACE::detail
{
struct avx512_mask_impl;
struct avx512_datapar_impl;

// bool_storage_member_type{{{1
#ifdef Vc_HAVE_AVX512F
template <> struct bool_storage_member_type< 2> { using type = __mmask8 ; };
template <> struct bool_storage_member_type< 4> { using type = __mmask8 ; };
template <> struct bool_storage_member_type< 8> { using type = __mmask8 ; };
template <> struct bool_storage_member_type<16> { using type = __mmask16; };
template <> struct bool_storage_member_type<32> { using type = __mmask32; };
template <> struct bool_storage_member_type<64> { using type = __mmask64; };
#endif  // Vc_HAVE_AVX512F

// traits<T, datapar_abi::avx512>{{{1
template <class T> using avx512_datapar_member_type = Storage<T, 64 / sizeof(T)>;
template <class T> using avx512_mask_member_type = Storage<bool, 64 / sizeof(T)>;

template <class T> struct traits<T, datapar_abi::avx512> {
    static_assert(sizeof(T) <= 8,
                  "AVX can only implement operations on element types with sizeof <= 8");
    static constexpr size_t size() noexcept { return 64 / sizeof(T); }

    using datapar_member_type = avx512_datapar_member_type<T>;
    using datapar_impl_type = avx512_datapar_impl;
    static constexpr size_t datapar_member_alignment = alignof(datapar_member_type);
    using datapar_cast_type = typename datapar_member_type::VectorType;

    using mask_member_type = avx512_mask_member_type<T>;
    using mask_impl_type = avx512_mask_impl;
    static constexpr size_t mask_member_alignment = alignof(mask_member_type);
    using mask_cast_type = typename mask_member_type::VectorType;
};

template <>
struct traits<long double, datapar_abi::avx512>
    : public traits<long double, datapar_abi::scalar> {
};
}  // namespace Vc_VERSIONED_NAMESPACE::detail

#ifdef Vc_HAVE_AVX512_ABI
namespace Vc_VERSIONED_NAMESPACE::detail
{
// datapar impl {{{1
struct avx512_datapar_impl {
    // member types {{{2
    using abi = datapar_abi::avx512;
    template <class T> static constexpr size_t size = datapar_size_v<T, abi>;
    template <class T> using datapar_member_type = avx512_datapar_member_type<T>;
    template <class T> using intrinsic_type = typename datapar_member_type<T>::VectorType;
    template <class T> using mask_member_type = avx512_mask_member_type<T>;
    template <class T> using datapar = Vc::datapar<T, abi>;
    template <class T> using mask = Vc::mask<T, abi>;
    template <size_t N> using size_tag = std::integral_constant<size_t, N>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    static Vc_INTRINSIC intrinsic_type<double> broadcast(double x, size_tag<8>) noexcept
    {
        return _mm512_set1_pd(x);
    }
    static Vc_INTRINSIC intrinsic_type<float> broadcast(float x, size_tag<16>) noexcept
    {
        return _mm512_set1_ps(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<8>) noexcept
    {
        return _mm512_set1_epi64x(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<16>) noexcept
    {
        return _mm512_set1_epi32(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<32>) noexcept
    {
        return _mm512_set1_epi16(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<64>) noexcept
    {
        return _mm512_set1_epi8(x);
    }

    // negation {{{2
    template <class T> static Vc_INTRINSIC mask<T> negate(datapar<T> x) noexcept
    {
#if defined Vc_GCC && defined Vc_USE_BUILTIN_VECTOR_TYPES
        return {private_init, !x.d.builtin()};
#else
        return equal_to(x, datapar<T>(0));
#endif
    }

    // compares {{{2
#if defined Vc_USE_BUILTIN_VECTOR_TYPES
    template <class T>
    static Vc_INTRINSIC mask<T> equal_to(datapar<T> x, datapar<T> y)
    {
        return {private_init, x.d.builtin() == y.d.builtin()};
    }
    template <class T>
    static Vc_INTRINSIC mask<T> not_equal_to(datapar<T> x, datapar<T> y)
    {
        return {private_init, x.d.builtin() != y.d.builtin()};
    }
    template <class T>
    static Vc_INTRINSIC mask<T> less(datapar<T> x, datapar<T> y)
    {
        return {private_init, x.d.builtin() < y.d.builtin()};
    }
    template <class T>
    static Vc_INTRINSIC mask<T> less_equal(datapar<T> x, datapar<T> y)
    {
        return {private_init, x.d.builtin() <= y.d.builtin()};
    }
#else  // Vc_USE_BUILTIN_VECTOR_TYPES
    static Vc_INTRINSIC mask<double> equal_to    (datapar<double> x, datapar<double> y) { return {private_init, _mm512_cmpeq_pd_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<double> not_equal_to(datapar<double> x, datapar<double> y) { return {private_init, _mm512_cmpneq_pd_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<double> less        (datapar<double> x, datapar<double> y) { return {private_init, _mm512_cmplt_pd_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<double> less_equal  (datapar<double> x, datapar<double> y) { return {private_init, _mm512_cmple_pd_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< float> equal_to    (datapar< float> x, datapar< float> y) { return {private_init, _mm512_cmpeq_ps_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< float> not_equal_to(datapar< float> x, datapar< float> y) { return {private_init, _mm512_cmpnew_ps_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< float> less        (datapar< float> x, datapar< float> y) { return {private_init, _mm512_cmplt_ps_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< float> less_equal  (datapar< float> x, datapar< float> y) { return {private_init, _mm512_cmple_ps_mask(x.d, y.d)}; }

#ifdef Vc_HAVE_FULL_AVX512_ABI
    static Vc_INTRINSIC mask< llong> equal_to(datapar< llong> x, datapar< llong> y) { return {private_init, _mm512_cmpeq_epi64_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ullong> equal_to(datapar<ullong> x, datapar<ullong> y) { return {private_init, _mm512_cmpeq_epi64_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  long> equal_to(datapar<  long> x, datapar<  long> y) { return {private_init, sizeof(long) == 8 ? _mm512_cmpeq_epi64_mask(x.d, y.d) : _mm512_cmpeq_epi32_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< ulong> equal_to(datapar< ulong> x, datapar< ulong> y) { return {private_init, sizeof(long) == 8 ? _mm512_cmpeq_epi64_mask(x.d, y.d) : _mm512_cmpeq_epi32_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<   int> equal_to(datapar<   int> x, datapar<   int> y) { return {private_init, _mm512_cmpeq_epi32_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  uint> equal_to(datapar<  uint> x, datapar<  uint> y) { return {private_init, _mm512_cmpeq_epi32_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< short> equal_to(datapar< short> x, datapar< short> y) { return {private_init, _mm512_cmpeq_epi16_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ushort> equal_to(datapar<ushort> x, datapar<ushort> y) { return {private_init, _mm512_cmpeq_epi16_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< schar> equal_to(datapar< schar> x, datapar< schar> y) { return {private_init, _mm512_cmpeq_epi8_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< uchar> equal_to(datapar< uchar> x, datapar< uchar> y) { return {private_init, _mm512_cmpeq_epi8_mask(x.d, y.d)}; }

    static Vc_INTRINSIC mask< llong> not_equal_to(datapar< llong> x, datapar< llong> y) { return !equal_to(x, y); }
    static Vc_INTRINSIC mask<ullong> not_equal_to(datapar<ullong> x, datapar<ullong> y) { return !equal_to(x, y); }
    static Vc_INTRINSIC mask<  long> not_equal_to(datapar<  long> x, datapar<  long> y) { return !equal_to(x, y); }
    static Vc_INTRINSIC mask< ulong> not_equal_to(datapar< ulong> x, datapar< ulong> y) { return !equal_to(x, y); }
    static Vc_INTRINSIC mask<   int> not_equal_to(datapar<   int> x, datapar<   int> y) { return !equal_to(x, y); }
    static Vc_INTRINSIC mask<  uint> not_equal_to(datapar<  uint> x, datapar<  uint> y) { return !equal_to(x, y); }
    static Vc_INTRINSIC mask< short> not_equal_to(datapar< short> x, datapar< short> y) { return !equal_to(x, y); }
    static Vc_INTRINSIC mask<ushort> not_equal_to(datapar<ushort> x, datapar<ushort> y) { return !equal_to(x, y); }
    static Vc_INTRINSIC mask< schar> not_equal_to(datapar< schar> x, datapar< schar> y) { return !equal_to(x, y); }
    static Vc_INTRINSIC mask< uchar> not_equal_to(datapar< uchar> x, datapar< uchar> y) { return !equal_to(x, y); }

    static Vc_INTRINSIC mask< llong> less(datapar< llong> x, datapar< llong> y) { return {private_init, _mm512_cmpgt_epi64_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask<ullong> less(datapar<ullong> x, datapar<ullong> y) { return {private_init, _mm512_cmpgt_epu64_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask<  long> less(datapar<  long> x, datapar<  long> y) { return {private_init, sizeof(long) == 8 ? _mm512_cmpgt_epi64_mask(y.d, x.d) : _mm512_cmpgt_epi32_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask< ulong> less(datapar< ulong> x, datapar< ulong> y) { return {private_init, sizeof(long) == 8 ? _mm512_cmpgt_epu64_mask(y.d, x.d) : _mm512_cmpgt_epu32_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask<   int> less(datapar<   int> x, datapar<   int> y) { return {private_init, _mm512_cmpgt_epi32_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask<  uint> less(datapar<  uint> x, datapar<  uint> y) { return {private_init, _mm512_cmpgt_epu32_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask< short> less(datapar< short> x, datapar< short> y) { return {private_init, _mm512_cmpgt_epi16_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask<ushort> less(datapar<ushort> x, datapar<ushort> y) { return {private_init, _mm512_cmpgt_epu16_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask< schar> less(datapar< schar> x, datapar< schar> y) { return {private_init, _mm512_cmpgt_epi8_mask (y.d, x.d)}; }
    static Vc_INTRINSIC mask< uchar> less(datapar< uchar> x, datapar< uchar> y) { return {private_init, _mm512_cmpgt_epu8_mask (y.d, x.d)}; }

    static Vc_INTRINSIC mask< llong> less_equal(datapar< llong> x, datapar< llong> y) { return {private_init, _mm512_cmple_epi64_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask<ullong> less_equal(datapar<ullong> x, datapar<ullong> y) { return {private_init, _mm512_cmple_epu64_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask<  long> less_equal(datapar<  long> x, datapar<  long> y) { return {private_init, sizeof(long) == 8 ? _mm512_cmple_epi64_mask(y.d, x.d) : _mm512_cmple_epi32_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask< ulong> less_equal(datapar< ulong> x, datapar< ulong> y) { return {private_init, sizeof(long) == 8 ? _mm512_cmple_epu64_mask(y.d, x.d) : _mm512_cmple_epu32_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask<   int> less_equal(datapar<   int> x, datapar<   int> y) { return {private_init, _mm512_cmple_epi32_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask<  uint> less_equal(datapar<  uint> x, datapar<  uint> y) { return {private_init, _mm512_cmple_epu32_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask< short> less_equal(datapar< short> x, datapar< short> y) { return {private_init, _mm512_cmple_epi16_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask<ushort> less_equal(datapar<ushort> x, datapar<ushort> y) { return {private_init, _mm512_cmple_epu16_mask(y.d, x.d)}; }
    static Vc_INTRINSIC mask< schar> less_equal(datapar< schar> x, datapar< schar> y) { return {private_init, _mm512_cmple_epi8_mask (y.d, x.d)}; }
    static Vc_INTRINSIC mask< uchar> less_equal(datapar< uchar> x, datapar< uchar> y) { return {private_init, _mm512_cmple_epu8_mask (y.d, x.d)}; }
#endif  // Vc_HAVE_FULL_AVX512_ABI
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

    // smart_reference access {{{2
    template <class T, class A>
    static Vc_INTRINSIC T get(Vc::datapar<T, A> v, int i) noexcept
    {
        return v.d.m(i);
    }
    template <class T, class A, class U>
    static Vc_INTRINSIC void set(Vc::datapar<T, A> &v, int i, U &&x) noexcept
    {
        v.d.set(i, std::forward<U>(x));
    }
    // }}}2
};

// mask impl {{{1
struct avx512_mask_impl {
    // member types {{{2
    using abi = datapar_abi::avx512;
    template <class T> static constexpr size_t size = datapar_size_v<T, abi>;
    template <class T> using mask_member_type = avx512_mask_member_type<T>;
    template <class T> using mask = Vc::mask<T, abi>;
    template <class T> using mask_bool = MaskBool<sizeof(T)>;
    template <size_t N> using size_tag = std::integral_constant<size_t, N>;

    // broadcast {{{2
    static Vc_INTRINSIC __mmask8 broadcast(bool x, size_tag<8>) noexcept
    {
        return static_cast<__mmask8>(x) * ~__mmask8();
    }
    static Vc_INTRINSIC __mmask16 broadcast(bool x, size_tag<16>) noexcept
    {
        return static_cast<__mmask16>(x) * ~__mmask16();
    }
    static Vc_INTRINSIC __mmask32 broadcast(bool x, size_tag<32>) noexcept
    {
        return static_cast<__mmask32>(x) * ~__mmask32();
    }
    static Vc_INTRINSIC __mmask64 broadcast(bool x, size_tag<64>) noexcept
    {
        return static_cast<__mmask64>(x) * ~__mmask64();
    }

    // load {{{2
    template <class F>
    static Vc_INTRINSIC __mmask8 load(const bool *mem, F, size_tag<8>) noexcept
    {
        const auto a = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem));
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        return _mm_test_epi8_mask(a, a);
#else
        const auto b = _mm512_cvtepi8_epi64(a);
        return _mm_test_epi64_mask(b, b);
#endif  // Vc_HAVE_AVX512BW
    }
    template <class F>
    static Vc_INTRINSIC __mmask16 load(const bool *mem, F, size_tag<16>) noexcept
    {
        const auto a = load16(mem, f);
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        return _mm_test_epi8_mask(a, a);
#else
        const auto b = _mm512_cvtepi8_epi32(a);
        return _mm_test_epi32_mask(b, b);
#endif  // Vc_HAVE_AVX512BW
    }
    template <class F>
    static Vc_INTRINSIC __mmask32 load(const bool *mem, F f, size_tag<32>) noexcept
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        const auto a = load32(mem, f);
        return _mm256_test_epi8_mask(a, a);
#else
        const auto a = _mm512_cvtepi8_epi32(load16(mem, f));
        const auto b = _mm512_cvtepi8_epi32(load16(mem + 16, f));
        return _mm_test_epi32_mask(a, a) | (_mm_test_epi32_mask(b, b) << 16);
#endif  // Vc_HAVE_AVX512BW
    }
    template <class F>
    static Vc_INTRINSIC __mmask64 load(const bool *mem, F f, size_tag<64>) noexcept
    {
#ifdef Vc_HAVE_AVX512BW
        const auto a = load64(mem, f);
        return _mm512_test_epi8_mask(a, a);
#else
        const auto a = _mm512_cvtepi8_epi32(load16(mem, f));
        const auto b = _mm512_cvtepi8_epi32(load16(mem + 16, f));
        const auto c = _mm512_cvtepi8_epi32(load16(mem + 32, f));
        const auto d = _mm512_cvtepi8_epi32(load16(mem + 48, f));
        return _mm_test_epi32_mask(a, a) | (_mm_test_epi32_mask(b, b) << 16) |
               (_mm_test_epi32_mask(b, b) << 32) | (_mm_test_epi32_mask(b, b) << 48);
#endif  // Vc_HAVE_AVX512BW
    }

    // masked load {{{2
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
    template <class T, class F>
    static Vc_INTRINSIC void masked_load(mask_member_type<T> &merge,
                                         mask_member_type<T> mask, const bool *mem, F,
                                         size_tag<8>) noexcept
    {
        const auto a = _mm_mask_loadu_epi8(zero<__m128i>(), mask.v(), mem);
        return _mm_test_epi8_mask(a, a);
    }

    template <class T, class F>
    static Vc_INTRINSIC void masked_load(mask_member_type<T> &merge,
                                         mask_member_type<T> mask, const bool *mem, F,
                                         size_tag<16>) noexcept
    {
        const auto a = _mm_mask_loadu_epi8(zero<__m128i>(), mask.v(), mem);
        return _mm_test_epi8_mask(a, a);
    }

    template <class T, class F>
    static Vc_INTRINSIC void masked_load(mask_member_type<T> &merge,
                                         mask_member_type<T> mask, const bool *mem, F,
                                         size_tag<32>) noexcept
    {
        const auto a = _mm256_mask_loadu_epi8(zero<__m256i>(), mask.v(), mem);
        return _mm256_test_epi8_mask(a, a);
    }

    template <class T, class F>
    static Vc_INTRINSIC void masked_load(mask_member_type<T> &merge,
                                         mask_member_type<T> mask, const bool *mem, F,
                                         size_tag<64>) noexcept
    {
        const auto a = _mm512_mask_loadu_epi8(zero<__m512i>(), mask.v(), mem);
        return _mm512_test_epi8_mask(a, a);
    }

#else
    template <class T, class F, class SizeTag>
    static Vc_INTRINSIC void masked_load(mask_member_type<T> &merge,
                                         mask_member_type<T> mask, const bool *mem, F,
                                         SizeTag) noexcept
    {
        for (std::size_t i = 0; i < size<T>; ++i) {
            if (mask[i]) {
                merge.set(i, mem[i]);
            }
        }
    }
#endif

    // store {{{2
    template <class T, class F>
    static constexpr void store(mask_member_type<T> v, bool *mem, F, size_tag<8>) noexcept
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        _mm_storel_epi64(reinterpret_cast<__m128i *>(mem),
                         and_(one16(uchar), _mm_movm_epi8(v.v())));
#elif defined Vc_HAVE_AVX512DQ
        _mm_storel_epi64(
            reinterpret_cast<__m128i *>(mem),
            _mm512_cvtepi64_epi8(_mm512_srli_epi64(_mm512_movm_epi64(v.v()), 63)));
#else
        *reinterpret_cast<may_alias<ullong> *>(mem) =
            _pdep_u64(v.v(), 0x0101010101010101ULL);
#endif
    }
    template <class T, class F>
    static constexpr void store(mask_member_type<T> v, bool *mem, F f, size_tag<16>) noexcept
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        _mm_store_si128(reinterpret_cast<__m128i *>(mem),
                        and_(one16(uchar), _mm_movm_epi8(v.v())));
        unused(f);
#elif defined Vc_HAVE_AVX512DQ
        store16(_mm512_cvtepi32_epi8(_mm512_srli_epi32(_mm512_movm_epi32(v.v()), 31)),
                mem, f);
#else
        *reinterpret_cast<may_alias<ullong> *>(mem) =
            _pdep_u64(v.v(), 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 8) =
            _pdep_u64(v.v() >> 8, 0x0101010101010101ULL);
#endif
    }
    template <class T, class F>
    static constexpr void store(mask_member_type<T> v, bool *mem, F, size_tag<32>) noexcept
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        _mm256_store_si256(reinterpret_cast<__m256i *>(mem),
                           and_(one32(uchar), _mm256_movm_epi8(v.v())));
#elif defined Vc_HAVE_AVX512DQ
        store32(
            concat(_mm512_cvtepi32_epi8(_mm512_srli_epi32(_mm512_movm_epi32(v.v()), 31)),
                   _mm512_cvtepi32_epi8(
                       _mm512_srli_epi32(_mm512_movm_epi32(v.v() >> 16), 31))),
            mem, f);
#else
        *reinterpret_cast<may_alias<ullong> *>(mem) =
            _pdep_u64(v.v(), 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 8) =
            _pdep_u64(v.v() >> 8, 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 16) =
            _pdep_u64(v.v() >> 16, 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 24) =
            _pdep_u64(v.v() >> 24, 0x0101010101010101ULL);
#endif
    }
    template <class T, class F>
    static constexpr void store(mask_member_type<T> v, bool *mem, F, size_tag<64>) noexcept
    {
#if defined Vc_HAVE_AVX512BW
        _mm512_store_si512(mem, and_(one64(uchar), _mm512_movm_epi8(v.v())));
#elif defined Vc_HAVE_AVX512DQ
        store64(concat(concat(_mm512_cvtepi32_epi8(
                                  _mm512_srli_epi32(_mm512_movm_epi32(v.v()), 31)),
                              _mm512_cvtepi32_epi8(
                                  _mm512_srli_epi32(_mm512_movm_epi32(v.v() >> 16), 31))),
                       concat(_mm512_cvtepi32_epi8(
                                  _mm512_srli_epi32(_mm512_movm_epi32(v.v() >> 32), 31)),
                              _mm512_cvtepi32_epi8(
                                  _mm512_srli_epi32(_mm512_movm_epi32(v.v() >> 48), 31)))),
                mem, f);
#else
        *reinterpret_cast<may_alias<ullong> *>(mem) =
            _pdep_u64(v.v(), 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 8) =
            _pdep_u64(v.v() >> 8, 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 16) =
            _pdep_u64(v.v() >> 16, 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 24) =
            _pdep_u64(v.v() >> 24, 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 32) =
            _pdep_u64(v.v() >> 32, 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 40) =
            _pdep_u64(v.v() >> 40, 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 48) =
            _pdep_u64(v.v() >> 48, 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 56) =
            _pdep_u64(v.v() >> 56, 0x0101010101010101ULL);
#endif
    }

    // masked store {{{2
#if defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL
    template <class T, class F>
    static Vc_INTRINSIC void masked_store(mask_member_type<T> v, bool *mem, F,
                                          mask_member_type<T> k, size_tag<8>) noexcept
    {
        _mm_mask_storeu_epi8(mem, k.v(), and_(one16(uchar), _mm_movm_epi8(v.v())));
    }

    template <class T, class F>
    static Vc_INTRINSIC void masked_store(mask_member_type<T> v, bool *mem, F,
                                          mask_member_type<T> k, size_tag<16>) noexcept
    {
        _mm_mask_storeu_epi8(mem, k.v(), and_(one16(uchar), _mm_movm_epi8(v.v())));
    }

    template <class T, class F>
    static Vc_INTRINSIC void masked_store(mask_member_type<T> v, bool *mem, F,
                                          mask_member_type<T> k, size_tag<32>) noexcept
    {
        _mm256_mask_storeu_epi8(mem, k.v(), and_(one32(uchar), _mm256_movm_epi8(v.v())));
    }

    template <class T, class F>
    static Vc_INTRINSIC void masked_store(mask_member_type<T> v, bool *mem, F,
                                          mask_member_type<T> k, size_tag<64>) noexcept
    {
        _mm512_mask_storeu_epi8(mem, k.v(), and_(one64(uchar), _mm512_movm_epi8(v.v())));
    }

#else   // defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL
    template <class T, class F, class SizeTag>
    static Vc_INTRINSIC void masked_store(mask_member_type<T> v, bool *mem, F,
                                          mask_member_type<T> k, SizeTag) noexcept
    {
        for (std::size_t i = 0; i < size<T>; ++i) {
            if (k[i]) {
                mem[i] = v.m(i);
            }
        }
    }
#endif  // defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL

    // negation {{{2
    template <class T, class SizeTag>
    static Vc_INTRINSIC mask_member_type<T> negate(const mask_member_type<T> &x,
                                                   SizeTag) noexcept
    {
        return ~x.v();
    }

    // smart_reference access {{{2
    template <class T> static bool get(const mask<T> &k, int i) noexcept
    {
        return k.d.m(i);
    }
    template <class T> static void set(mask<T> &k, int i, bool x) noexcept
    {
        k.d.set(i, x);
    }
    // }}}2
};

// mask compare base {{{1
struct avx512_compare_base {
protected:
    using abi = Vc::datapar_abi::avx512;
    template <class T> using V = Vc::datapar<T, abi>;
    template <class T> using M = Vc::mask<T, abi>;
    template <class T> using S = typename Vc::detail::traits<T, abi>::mask_cast_type;
    template <class T> static constexpr size_t size = M<T>::size();
};
// }}}1
}  // namespace Vc_VERSIONED_NAMESPACE::detail

// [mask.reductions] {{{
namespace Vc_VERSIONED_NAMESPACE
{
template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE bool all_of(mask<T, datapar_abi::avx512> k)
{
    switch (k.size()) {
    case 8:  return k.d.v() == 0xffU;
    case 16: return k.d.v() == 0xffffU;
    case 32: return k.d.v() == 0xffffffffU;
    case 64: return k.d.v() == 0xffffffffffffffffULL;
    default: Vc_UNREACHABLE();
    }
    // TODO: use _mm512_kortestc somehow?
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE bool any_of(mask<T, datapar_abi::avx512> k)
{
    return k.d.v() != 0U;
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE bool none_of(mask<T, datapar_abi::avx512> k)
{
    return k.d.v() == 0U;
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE bool some_of(mask<T, datapar_abi::avx512> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx512>::mask_cast_type>(k);
    return 0 != detail::testnzc(d, detail::allone_poly);
}

Vc_ALWAYS_INLINE int popcount(mask<short, datapar_abi::avx512> k)
{
    return detail::popcount32(detail::mask_to_int<32>(
               static_cast<
                   typename detail::traits<T, datapar_abi::avx512>::mask_cast_type>(
                   k))) >>
           1;
}

Vc_ALWAYS_INLINE int popcount(mask<ushort, datapar_abi::avx512> k)
{
    return detail::popcount32(detail::mask_to_int<32>(
               static_cast<
                   typename detail::traits<T, datapar_abi::avx512>::mask_cast_type>(
                   k))) >>
           1;
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE int popcount(mask<T, datapar_abi::avx512> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx512>::mask_cast_type>(k);
    const auto n = mask_to_int<k.size()>(d);
    switch (k.size()) {
    case  4: return detail::popcnt4(n);
    case  8: return detail::popcnt8(n);
    case 32: return detail::popcnt32(n);
    default: Vc_UNREACHABLE();
    }
}

Vc_ALWAYS_INLINE int find_first_set(mask<short, datapar_abi::avx512> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx512>::mask_cast_type>(k);
    return detail::bit_scan_forward(detail::mask_to_int<32>(d)) >> 1;
}

Vc_ALWAYS_INLINE int find_first_set(mask<ushort, datapar_abi::avx512> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx512>::mask_cast_type>(k);
    return detail::bit_scan_forward(detail::mask_to_int<32>(d)) >> 1;
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE int find_first_set(mask<T, datapar_abi::avx512> k)
{
    const auto d = detail::intrin_cast<__m512i>(
        static_cast<typename detail::traits<T, datapar_abi::avx512>::mask_cast_type>(k));
    return detail::bit_scan_forward(detail::mask_to_int<k.size()>(d));
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE int find_last_set(mask<T, datapar_abi::avx512> k)
{
    const auto d = detail::intrin_cast<__m512i>(
        static_cast<typename detail::traits<T, datapar_abi::avx512>::mask_cast_type>(k));
    return detail::bit_scan_reverse(detail::mask_to_int<k.size()>(d));
}
}  // namespace Vc_VERSIONED_NAMESPACE
// }}}

namespace std
{
// mask operators {{{1
template <class T>
struct equal_to<Vc::mask<T, Vc::datapar_abi::avx512>>
    : private Vc::detail::avx512_compare_base {
public:
    Vc_ALWAYS_INLINE bool operator()(const M<T> &x, const M<T> &y) const
    {
        return static_cast<S<T>>(x).v() == static_cast<S<T>>(y).v();
    }
};
template <>
struct equal_to<Vc::mask<long double, Vc::datapar_abi::avx512>>
    : public equal_to<Vc::mask<long double, Vc::datapar_abi::scalar>> {
};
// }}}1
}  // namespace std
#endif  // Vc_HAVE_AVX512_ABI

#endif  // VC_DATAPAR_AVX512_H_

// vim: foldmethod=marker
