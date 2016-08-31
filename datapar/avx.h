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

#ifndef VC_DATAPAR_AVX_H_
#define VC_DATAPAR_AVX_H_

#include "macros.h"
#include "storage.h"
#include "x86/intrinsics.h"
#include "x86/convert.h"
#include "x86/compares.h"

namespace Vc_VERSIONED_NAMESPACE::detail
{
struct avx_mask_impl;
struct avx_datapar_impl;

template <class T> using avx_datapar_member_type = Storage<T, 32 / sizeof(T)>;
template <class T> using avx_mask_member_type = Storage<T, 32 / sizeof(T)>;

template <class T> struct traits<T, datapar_abi::avx> {
    static_assert(sizeof(T) <= 8,
                  "AVX can only implement operations on element types with sizeof <= 8");
    static constexpr size_t size() noexcept { return 32 / sizeof(T); }

    using datapar_member_type = avx_datapar_member_type<T>;
    using datapar_impl_type = avx_datapar_impl;
    static constexpr size_t datapar_member_alignment = alignof(datapar_member_type);
    using datapar_cast_type = typename datapar_member_type::VectorType;

    using mask_member_type = avx_mask_member_type<T>;
    using mask_impl_type = avx_mask_impl;
    static constexpr size_t mask_member_alignment = alignof(mask_member_type);
    using mask_cast_type = typename mask_member_type::VectorType;
};

template <>
struct traits<long double, datapar_abi::avx>
    : public traits<long double, datapar_abi::scalar> {
};
}  // namespace Vc_VERSIONED_NAMESPACE::detail

#ifdef Vc_HAVE_AVX_ABI
namespace Vc_VERSIONED_NAMESPACE::detail
{
// datapar impl {{{1
struct avx_datapar_impl {
    // member types {{{2
    using abi = datapar_abi::avx;
    template <class T> static constexpr size_t size = datapar_size_v<T, abi>;
    template <class T> using datapar_member_type = avx_datapar_member_type<T>;
    template <class T> using intrinsic_type = typename datapar_member_type<T>::VectorType;
    template <class T> using mask_member_type = avx_mask_member_type<T>;
    template <class T> using datapar = Vc::datapar<T, abi>;
    template <class T> using mask = Vc::mask<T, abi>;
    template <size_t N> using size_tag = std::integral_constant<size_t, N>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    static Vc_INTRINSIC intrinsic_type<double> broadcast(double x, size_tag<4>) noexcept
    {
        return _mm256_set1_pd(x);
    }
    static Vc_INTRINSIC intrinsic_type<float> broadcast(float x, size_tag<8>) noexcept
    {
        return _mm256_set1_ps(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<4>) noexcept
    {
        return _mm256_set1_epi64x(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<8>) noexcept
    {
        return _mm256_set1_epi32(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<16>) noexcept
    {
        return _mm256_set1_epi16(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<32>) noexcept
    {
        return _mm256_set1_epi8(x);
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
#else
    static Vc_INTRINSIC mask<double> equal_to    (datapar<double> x, datapar<double> y) { return {private_init, _mm256_cmp_pd(x.d, y.d, _CMP_EQ_OQ)}; }
    static Vc_INTRINSIC mask<double> not_equal_to(datapar<double> x, datapar<double> y) { return {private_init, _mm256_cmp_pd(x.d, y.d, _CMP_NEQ_UQ)}; }
    static Vc_INTRINSIC mask<double> less        (datapar<double> x, datapar<double> y) { return {private_init, _mm256_cmp_pd(x.d, y.d, _CMP_LT_OS)}; }
    static Vc_INTRINSIC mask<double> less_equal  (datapar<double> x, datapar<double> y) { return {private_init, _mm256_cmp_pd(x.d, y.d, _CMP_LE_OS)}; }
    static Vc_INTRINSIC mask< float> equal_to    (datapar< float> x, datapar< float> y) { return {private_init, _mm256_cmp_ps(x.d, y.d, _CMP_EQ_OQ)}; }
    static Vc_INTRINSIC mask< float> not_equal_to(datapar< float> x, datapar< float> y) { return {private_init, _mm256_cmp_ps(x.d, y.d, _CMP_NEQ_UQ)}; }
    static Vc_INTRINSIC mask< float> less        (datapar< float> x, datapar< float> y) { return {private_init, _mm256_cmp_ps(x.d, y.d, _CMP_LT_OS)}; }
    static Vc_INTRINSIC mask< float> less_equal  (datapar< float> x, datapar< float> y) { return {private_init, _mm256_cmp_ps(x.d, y.d, _CMP_LE_OS)}; }

#ifdef Vc_HAVE_FULL_AVX_ABI
    static Vc_INTRINSIC mask< llong> equal_to(datapar< llong> x, datapar< llong> y) { return {private_init, _mm256_cmpeq_epi64(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ullong> equal_to(datapar<ullong> x, datapar<ullong> y) { return {private_init, _mm256_cmpeq_epi64(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  long> equal_to(datapar<  long> x, datapar<  long> y) { return {private_init, sizeof(long) == 8 ? _mm256_cmpeq_epi64(x.d, y.d) : _mm256_cmpeq_epi32(x.d, y.d)}; }
    static Vc_INTRINSIC mask< ulong> equal_to(datapar< ulong> x, datapar< ulong> y) { return {private_init, sizeof(long) == 8 ? _mm256_cmpeq_epi64(x.d, y.d) : _mm256_cmpeq_epi32(x.d, y.d)}; }
    static Vc_INTRINSIC mask<   int> equal_to(datapar<   int> x, datapar<   int> y) { return {private_init, _mm256_cmpeq_epi32(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  uint> equal_to(datapar<  uint> x, datapar<  uint> y) { return {private_init, _mm256_cmpeq_epi32(x.d, y.d)}; }
    static Vc_INTRINSIC mask< short> equal_to(datapar< short> x, datapar< short> y) { return {private_init, _mm256_cmpeq_epi16(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ushort> equal_to(datapar<ushort> x, datapar<ushort> y) { return {private_init, _mm256_cmpeq_epi16(x.d, y.d)}; }
    static Vc_INTRINSIC mask< schar> equal_to(datapar< schar> x, datapar< schar> y) { return {private_init, _mm256_cmpeq_epi8(x.d, y.d)}; }
    static Vc_INTRINSIC mask< uchar> equal_to(datapar< uchar> x, datapar< uchar> y) { return {private_init, _mm256_cmpeq_epi8(x.d, y.d)}; }

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

    static Vc_INTRINSIC mask< llong> less(datapar< llong> x, datapar< llong> y) { return {private_init, _mm256_cmpgt_epi64(y.d, x.d)}; }
    static Vc_INTRINSIC mask<ullong> less(datapar<ullong> x, datapar<ullong> y) { return {private_init, cmpgt(y.d, x.d)}; }
    static Vc_INTRINSIC mask<  long> less(datapar<  long> x, datapar<  long> y) { return {private_init, sizeof(long) == 8 ? _mm256_cmpgt_epi64(y.d, x.d) : _mm256_cmpgt_epi32(y.d, x.d)}; }
    static Vc_INTRINSIC mask< ulong> less(datapar< ulong> x, datapar< ulong> y) { return {private_init, cmpgt(y.d, x.d)}; }
    static Vc_INTRINSIC mask<   int> less(datapar<   int> x, datapar<   int> y) { return {private_init, _mm256_cmpgt_epi32(y.d, x.d)}; }
    static Vc_INTRINSIC mask<  uint> less(datapar<  uint> x, datapar<  uint> y) { return {private_init, cmpgt(y.d, x.d)}; }
    static Vc_INTRINSIC mask< short> less(datapar< short> x, datapar< short> y) { return {private_init, _mm256_cmpgt_epi16(y.d, x.d)}; }
    static Vc_INTRINSIC mask<ushort> less(datapar<ushort> x, datapar<ushort> y) { return {private_init, cmpgt(y.d, x.d)}; }
    static Vc_INTRINSIC mask< schar> less(datapar< schar> x, datapar< schar> y) { return {private_init, _mm256_cmpgt_epi8 (y.d, x.d)}; }
    static Vc_INTRINSIC mask< uchar> less(datapar< uchar> x, datapar< uchar> y) { return {private_init, cmpgt(y.d, x.d)}; }

    static Vc_INTRINSIC mask< llong> less_equal(datapar< llong> x, datapar< llong> y) { return {private_init, detail::not_(_mm256_cmpgt_epi64(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ullong> less_equal(datapar<ullong> x, datapar<ullong> y) { return {private_init, detail::not_(cmpgt(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  long> less_equal(datapar<  long> x, datapar<  long> y) { return {private_init, detail::not_(sizeof(long) == 8 ? _mm256_cmpgt_epi64(x.d, y.d) : _mm256_cmpgt_epi32(x.d, y.d))}; }
    static Vc_INTRINSIC mask< ulong> less_equal(datapar< ulong> x, datapar< ulong> y) { return {private_init, detail::not_(cmpgt(x.d, y.d))}; }
    static Vc_INTRINSIC mask<   int> less_equal(datapar<   int> x, datapar<   int> y) { return {private_init, detail::not_(_mm256_cmpgt_epi32(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  uint> less_equal(datapar<  uint> x, datapar<  uint> y) { return {private_init, detail::not_(cmpgt(x.d, y.d))}; }
    static Vc_INTRINSIC mask< short> less_equal(datapar< short> x, datapar< short> y) { return {private_init, detail::not_(_mm256_cmpgt_epi16(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ushort> less_equal(datapar<ushort> x, datapar<ushort> y) { return {private_init, detail::not_(cmpgt(x.d, y.d))}; }
    static Vc_INTRINSIC mask< schar> less_equal(datapar< schar> x, datapar< schar> y) { return {private_init, detail::not_(_mm256_cmpgt_epi8 (x.d, y.d))}; }
    static Vc_INTRINSIC mask< uchar> less_equal(datapar< uchar> x, datapar< uchar> y) { return {private_init, detail::not_(cmpgt (x.d, y.d))}; }
#endif
#endif

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
struct avx_mask_impl {
    // member types {{{2
    using abi = datapar_abi::avx;
    template <class T> static constexpr size_t size = datapar_size_v<T, abi>;
    template <class T> using mask_member_type = avx_mask_member_type<T>;
    template <class T> using mask = Vc::mask<T, datapar_abi::avx>;
    template <class T> using mask_bool = MaskBool<sizeof(T)>;
    template <size_t N> using size_tag = std::integral_constant<size_t, N>;

    // broadcast {{{2
    static Vc_INTRINSIC __m256d broadcast(bool x, size_tag<4>) noexcept
    {
        return _mm256_set1_pd(mask_bool<double>{x});
    }
    static Vc_INTRINSIC __m256 broadcast(bool x, size_tag<8>) noexcept
    {
        return _mm256_set1_ps(mask_bool<float>{x});
    }
    static Vc_INTRINSIC __m256i broadcast(bool x, size_tag<16>) noexcept
    {
#ifdef Vc_HAVE_AVX2
        return _mm256_set1_epi16(mask_bool<std::int16_t>{x});
#else
        const std::uint32_t tmp = x ? 0x00010001u : 0u;
        return intrin_cast<__m256i>(
            _mm256_set1_ps(reinterpret_cast<const may_alias<float> &>(tmp)));
#endif
    }
    static __m256i broadcast(bool x, size_tag<32>) noexcept
    {
#ifdef Vc_HAVE_AVX2
        return _mm256_set1_epi8(mask_bool<std::int8_t>{x});
#else
        const std::uint32_t tmp = x ? 0x01010101u : 0u;
        return intrin_cast<__m256i>(
            _mm256_set1_ps(reinterpret_cast<const may_alias<float> &>(tmp)));
#endif
    }

    // load {{{2
    template <class F>
    static Vc_INTRINSIC __m256 load(const bool *mem, F, size_tag<4>) noexcept
    {
        __m128i k = intrin_cast<__m128i>(_mm_and_ps(
            _mm_set1_ps(*reinterpret_cast<const may_alias<float> *>(mem)),
            intrin_cast<__m128>(_mm_setr_epi32(0x1, 0x100, 0x10000, 0x1000000))));
        k = _mm_cmpgt_epi32(k, _mm_setzero_si128());
        return intrin_cast<__m256>(
            concat(_mm_unpacklo_epi32(k, k), _mm_unpackhi_epi32(k, k)));
    }
    template <class F>
    static Vc_INTRINSIC __m256 load(const bool *mem, F, size_tag<8>) noexcept
    {
#ifdef Vc_IS_AMD64
        __m128i k = _mm_cvtsi64_si128(*reinterpret_cast<const may_alias<int64_t> *>(mem));
#else
        __m128i k = _mm_castpd_si128(
            _mm_load_sd(reinterpret_cast<const may_alias<double> *>(mem)));
#endif
        k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128());
        return intrin_cast<__m256>(
            concat(_mm_unpacklo_epi16(k, k), _mm_unpackhi_epi16(k, k)));
    }
    template <class F>
    static Vc_INTRINSIC __m256i load(const bool *mem, F f, size_tag<16>) noexcept
    {
        const auto k128 = _mm_cmpgt_epi8(load16(mem, f), zero<__m128i>());
        return concat(_mm_unpacklo_epi8(k128, k128), _mm_unpackhi_epi8(k128, k128));
    }
    template <class F>
    static Vc_INTRINSIC __m256i load(const bool *mem, F f, size_tag<32>) noexcept
    {
        return _mm256_cmpgt_epi8(load32(mem, f), zero<__m256i>());
    }

    // masked load {{{2
    template <class T, class F, class SizeTag>
    static Vc_INTRINSIC void masked_load(mask_member_type<T> &merge,
                                         mask_member_type<T> mask, const bool *mem, F,
                                         SizeTag) noexcept
    {
        for (std::size_t i = 0; i < size<T>; ++i) {
            if (mask.m(i)) {
                merge.set(i, mask_bool<T>{mem[i]});
            }
        }
    }

    // store {{{2
    template <class T, class F>
    static constexpr void store(mask_member_type<T> v, bool *mem, F, size_tag<4>) noexcept
    {
        auto k = intrin_cast<__m256i>(v.v());
#ifdef Vc_HAVE_AVX2
        *reinterpret_cast<may_alias<int32_t> *>(mem) = _mm256_movemask_epi8(k) & 0x01010101;
#else
        *reinterpret_cast<may_alias<int32_t> *>(mem) =
            (_mm_movemask_epi8(lo128(k)) |
             (_mm_movemask_epi8(hi128(k)) << 16)) &
            0x01010101;
#endif
    }
    template <class T, class F>
    static constexpr void store(mask_member_type<T> v, bool *mem, F, size_tag<8>) noexcept
    {
        auto k = intrin_cast<__m256i>(v.v());
        const auto k2 =
            _mm_srli_epi16(_mm_packs_epi16(lo128(k), hi128(k)), 15);
        const auto k3 = _mm_packs_epi16(k2, _mm_setzero_si128());
#ifdef Vc_IS_AMD64
        *reinterpret_cast<may_alias<int64_t> *>(mem) = _mm_cvtsi128_si64(k3);
#else
        *reinterpret_cast<may_alias<int32_t> *>(mem) = _mm_cvtsi128_si32(k3);
        *reinterpret_cast<may_alias<int32_t> *>(mem + 4) = _mm_extract_epi32(k3, 1);
#endif
    }
    template <class T, class F>
    static constexpr void store(mask_member_type<T> v, bool *mem, F f,
                                size_tag<16>) noexcept
    {
#ifdef Vc_HAVE_AVX2
        auto x =_mm256_srli_epi16(v, 15);
        const auto bools = _mm_packs_epi16(lo128(x), hi128(x));
#else
        const auto bools =
            detail::and_(one16(uchar()), _mm_packs_epi16(lo128(v.v()), hi128(v.v())));
#endif
        store16(bools, mem, f);
    }
    template <class T, class F>
    static constexpr void store(mask_member_type<T> v, bool *mem, F f,
                                size_tag<32>) noexcept
    {
        const auto bools = detail::and_(one32(uchar()), v.v());
        store32(bools, mem, f);
    }

    // masked store {{{2
    template <class T, class F, class SizeTag>
    static Vc_INTRINSIC void masked_store(mask_member_type<T> v, bool *mem, F,
                                          mask_member_type<T> k, SizeTag) noexcept
    {
        for (std::size_t i = 0; i < size<T>; ++i) {
            if (k.m(i)) {
                mem[i] = v.m(i);
            }
        }
    }

    // negation {{{2
    template <class T, class SizeTag>
    static Vc_INTRINSIC mask_member_type<T> negate(const mask_member_type<T> &x,
                                                   SizeTag) noexcept
    {
#if defined Vc_GCC && defined Vc_USE_BUILTIN_VECTOR_TYPES
        return !x.builtin();
#else
        return detail::not_(x.v());
#endif
    }

    // smart_reference access {{{2
    template <class T> static bool get(const mask<T> &k, int i) noexcept
    {
        return k.d.m(i);
    }
    template <class T> static void set(mask<T> &k, int i, bool x) noexcept
    {
        k.d.set(i, mask_bool<T>(x));
    }
    // }}}2
};

// mask compare base {{{1
struct avx_compare_base {
protected:
    template <class T> using V = Vc::datapar<T, Vc::datapar_abi::avx>;
    template <class T> using M = Vc::mask<T, Vc::datapar_abi::avx>;
    template <class T>
    using S = typename Vc::detail::traits<T, Vc::datapar_abi::avx>::mask_cast_type;
    template <class T> static constexpr size_t size = M<T>::size();
};
// }}}1
constexpr struct {
    template <class T> operator T() const { return detail::allone<T>(); }
} allone_poly = {};
}  // namespace Vc_VERSIONED_NAMESPACE::detail

// [mask.reductions] {{{
namespace Vc_VERSIONED_NAMESPACE
{
template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE bool all_of(mask<T, datapar_abi::avx> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k);
    return 0 != detail::testc(d, detail::allone_poly);
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE bool any_of(mask<T, datapar_abi::avx> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k);
    return 0 == detail::testz(d, d);
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE bool none_of(mask<T, datapar_abi::avx> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k);
    return 0 != detail::testz(d, d);
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE bool some_of(mask<T, datapar_abi::avx> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k);
    return 0 != detail::testnzc(d, detail::allone_poly);
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE int popcount(mask<T, datapar_abi::avx> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k);
    switch (k.size()) {
    case 4:
        return detail::popcnt4(detail::mask_to_int<k.size()>(d));
    case 8:
        return detail::popcnt8(detail::mask_to_int<k.size()>(d));
    case 16:
        return detail::popcnt32(detail::mask_to_int<32>(d)) >> 1;
    case 32:
        return detail::popcnt32(detail::mask_to_int<k.size()>(d));
    default:
        Vc_UNREACHABLE();
    }
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE int find_first_set(mask<T, datapar_abi::avx> k)
{
    const auto d = detail::intrin_cast<__m256i>(
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k));
    return detail::bit_scan_forward(detail::mask_to_int<k.size()>(d));
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE int find_last_set(mask<T, datapar_abi::avx> k)
{
    const auto d = detail::intrin_cast<__m256i>(
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k));
    return detail::bit_scan_reverse(detail::mask_to_int<k.size()>(d));
}
}  // namespace Vc_VERSIONED_NAMESPACE
// }}}

namespace std
{
// mask operators {{{1
template <class T>
struct equal_to<Vc::mask<T, Vc::datapar_abi::avx>>
    : private Vc::detail::avx_compare_base {
public:
    Vc_ALWAYS_INLINE bool operator()(const M<T> &x, const M<T> &y) const
    {
        switch (sizeof(T)) {
        case 1:
        case 2:
            return Vc::detail::movemask(
                       Vc::detail::intrin_cast<__m256i>(static_cast<S<T>>(x))) ==
                   Vc::detail::movemask(
                       Vc::detail::intrin_cast<__m256i>(static_cast<S<T>>(y)));
        case 4:
            return Vc::detail::movemask(
                       Vc::detail::intrin_cast<__m256>(static_cast<S<T>>(x))) ==
                   Vc::detail::movemask(
                       Vc::detail::intrin_cast<__m256>(static_cast<S<T>>(y)));
        case 8:
            return Vc::detail::movemask(
                       Vc::detail::intrin_cast<__m256d>(static_cast<S<T>>(x))) ==
                   Vc::detail::movemask(
                       Vc::detail::intrin_cast<__m256d>(static_cast<S<T>>(y)));
        default:
            Vc_UNREACHABLE();
        }
    }
};
template <>
struct equal_to<Vc::mask<long double, Vc::datapar_abi::avx>>
    : public equal_to<Vc::mask<long double, Vc::datapar_abi::scalar>> {
};
// }}}1
}  // namespace std
#endif  // Vc_HAVE_AVX_ABI

#endif  // VC_DATAPAR_AVX_H_

// vim: foldmethod=marker
