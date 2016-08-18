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

#ifndef VC_DATAPAR_SSE_H_
#define VC_DATAPAR_SSE_H_

#include "macros.h"
#include "../common/storage.h"

#ifdef Vc_HAVE_SSE_ABI
#include "../sse/casts.h"
#endif

namespace Vc_VERSIONED_NAMESPACE::detail
{
struct sse_mask_impl;
struct sse_datapar_impl;
using Vc::Common::Storage;

template <class T> using sse_datapar_member_type = Storage<T, 16 / sizeof(T)>;
template <class T> using sse_mask_member_type = Storage<T, 16 / sizeof(T)>;

template <class T> struct traits<T, datapar_abi::sse> {
    static_assert(sizeof(T) <= 8,
                  "SSE can only implement operations on element types with sizeof <= 8");
    static constexpr size_t size() noexcept { return 16 / sizeof(T); }

    using datapar_member_type = sse_datapar_member_type<T>;
    using datapar_impl_type = sse_datapar_impl;
    static constexpr size_t datapar_member_alignment = alignof(datapar_member_type);
    using datapar_cast_type = typename datapar_member_type::VectorType;

    using mask_member_type = sse_mask_member_type<T>;
    using mask_impl_type = sse_mask_impl;
    static constexpr size_t mask_member_alignment = alignof(mask_member_type);
    using mask_cast_type = typename mask_member_type::VectorType;
};

template <>
struct traits<long double, datapar_abi::sse>
    : public traits<long double, datapar_abi::scalar> {
};
}  // namespace Vc_VERSIONED_NAMESPACE::detail

#ifdef Vc_HAVE_SSE_ABI
namespace Vc_VERSIONED_NAMESPACE::detail
{
using SSE::sse_cast;
// datapar impl {{{1
struct sse_datapar_impl {
    // member types {{{2
    using abi = datapar_abi::sse;
    template <class T> static constexpr size_t size = datapar_size_v<T, abi>;
    template <class T> using datapar_member_type = sse_datapar_member_type<T>;
    template <class T> using mask_member_type = sse_mask_member_type<T>;
    template <class T> using datapar = Vc::datapar<T, abi>;
    template <class T> using mask = Vc::mask<T, abi>;
    template <size_t N> using size_tag = std::integral_constant<size_t, N>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    static Vc_INTRINSIC datapar_member_type<double> broadcast(double x,
                                                              size_tag<2>) noexcept
    {
        return _mm_set1_pd(x);
    }
    static Vc_INTRINSIC datapar_member_type<float> broadcast(float x,
                                                             size_tag<4>) noexcept
    {
        return _mm_set1_ps(x);
    }
    template <class T>
    static Vc_INTRINSIC datapar_member_type<T> broadcast(T x, size_tag<2>) noexcept
    {
        return _mm_set1_epi64x(x);
    }
    template <class T>
    static Vc_INTRINSIC datapar_member_type<T> broadcast(T x, size_tag<4>) noexcept
    {
        return _mm_set1_epi32(x);
    }
    template <class T>
    static Vc_INTRINSIC datapar_member_type<T> broadcast(T x, size_tag<8>) noexcept
    {
        return _mm_set1_epi16(x);
    }
    template <class T>
    static Vc_INTRINSIC datapar_member_type<T> broadcast(T x, size_tag<16>) noexcept
    {
        return _mm_set1_epi8(x);
    }

    // negation {{{2
    template <class T> static Vc_INTRINSIC mask<T> negate(datapar<T> x) noexcept
    {
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
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
    static Vc_INTRINSIC mask<double> equal_to(datapar<double> x, datapar<double> y) { return {private_init, _mm_cmpeq_pd(x.d, y.d)}; }
    static Vc_INTRINSIC mask< float> equal_to(datapar< float> x, datapar< float> y) { return {private_init, _mm_cmpeq_ps(x.d, y.d)}; }
    static Vc_INTRINSIC mask< llong> equal_to(datapar< llong> x, datapar< llong> y) { return {private_init, SSE::cmpeq_epi64(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ullong> equal_to(datapar<ullong> x, datapar<ullong> y) { return {private_init, SSE::cmpeq_epi64(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  long> equal_to(datapar<  long> x, datapar<  long> y) { return {private_init, sizeof(long) == 8 ? SSE::cmpeq_epi64(x.d, y.d) : _mm_cmpeq_epi32(x.d, y.d)}; }
    static Vc_INTRINSIC mask< ulong> equal_to(datapar< ulong> x, datapar< ulong> y) { return {private_init, sizeof(long) == 8 ? SSE::cmpeq_epi64(x.d, y.d) : _mm_cmpeq_epi32(x.d, y.d)}; }
    static Vc_INTRINSIC mask<   int> equal_to(datapar<   int> x, datapar<   int> y) { return {private_init, _mm_cmpeq_epi32(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  uint> equal_to(datapar<  uint> x, datapar<  uint> y) { return {private_init, _mm_cmpeq_epi32(x.d, y.d)}; }
    static Vc_INTRINSIC mask< short> equal_to(datapar< short> x, datapar< short> y) { return {private_init, _mm_cmpeq_epi16(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ushort> equal_to(datapar<ushort> x, datapar<ushort> y) { return {private_init, _mm_cmpeq_epi16(x.d, y.d)}; }
    static Vc_INTRINSIC mask< schar> equal_to(datapar< schar> x, datapar< schar> y) { return {private_init, _mm_cmpeq_epi8(x.d, y.d)}; }
    static Vc_INTRINSIC mask< uchar> equal_to(datapar< uchar> x, datapar< uchar> y) { return {private_init, _mm_cmpeq_epi8(x.d, y.d)}; }

    static Vc_INTRINSIC mask<double> not_equal_to(datapar<double> x, datapar<double> y) { return {private_init, _mm_cmpneq_pd(x.d, y.d)}; }
    static Vc_INTRINSIC mask< float> not_equal_to(datapar< float> x, datapar< float> y) { return {private_init, _mm_cmpneq_ps(x.d, y.d)}; }
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

    static Vc_INTRINSIC mask<double> less(datapar<double> x, datapar<double> y) { return {private_init, _mm_cmplt_pd(x.d, y.d)}; }
    static Vc_INTRINSIC mask< float> less(datapar< float> x, datapar< float> y) { return {private_init, _mm_cmplt_ps(x.d, y.d)}; }
    static Vc_INTRINSIC mask< llong> less(datapar< llong> x, datapar< llong> y) { return {private_init, SSE::cmpgt_epi64(y.d, x.d)}; }
    static Vc_INTRINSIC mask<ullong> less(datapar<ullong> x, datapar<ullong> y) { return {private_init, SSE::cmpgt_epu64(y.d, x.d)}; }
    static Vc_INTRINSIC mask<  long> less(datapar<  long> x, datapar<  long> y) { return {private_init, sizeof(long) == 8 ? SSE::cmpgt_epi64(y.d, x.d) :  _mm_cmpgt_epi32(y.d, x.d)}; }
    static Vc_INTRINSIC mask< ulong> less(datapar< ulong> x, datapar< ulong> y) { return {private_init, sizeof(long) == 8 ? SSE::cmpgt_epu64(y.d, x.d) : SSE::cmpgt_epu32(y.d, x.d)}; }
    static Vc_INTRINSIC mask<   int> less(datapar<   int> x, datapar<   int> y) { return {private_init,  _mm_cmpgt_epi32(y.d, x.d)}; }
    static Vc_INTRINSIC mask<  uint> less(datapar<  uint> x, datapar<  uint> y) { return {private_init, SSE::cmpgt_epu32(y.d, x.d)}; }
    static Vc_INTRINSIC mask< short> less(datapar< short> x, datapar< short> y) { return {private_init,  _mm_cmpgt_epi16(y.d, x.d)}; }
    static Vc_INTRINSIC mask<ushort> less(datapar<ushort> x, datapar<ushort> y) { return {private_init, SSE::cmpgt_epu16(y.d, x.d)}; }
    static Vc_INTRINSIC mask< schar> less(datapar< schar> x, datapar< schar> y) { return {private_init,  _mm_cmpgt_epi8 (y.d, x.d)}; }
    static Vc_INTRINSIC mask< uchar> less(datapar< uchar> x, datapar< uchar> y) { return {private_init, SSE::cmpgt_epu8 (y.d, x.d)}; }

    static Vc_INTRINSIC mask<double> less_equal(datapar<double> x, datapar<double> y) { return {private_init, _mm_cmple_pd(x.d, y.d)}; }
    static Vc_INTRINSIC mask< float> less_equal(datapar< float> x, datapar< float> y) { return {private_init, _mm_cmple_ps(x.d, y.d)}; }
    static Vc_INTRINSIC mask< llong> less_equal(datapar< llong> x, datapar< llong> y) { return {private_init, Detail::not_(SSE::cmpgt_epi64(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ullong> less_equal(datapar<ullong> x, datapar<ullong> y) { return {private_init, Detail::not_(SSE::cmpgt_epu64(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  long> less_equal(datapar<  long> x, datapar<  long> y) { return {private_init, Detail::not_(sizeof(long) == 8 ? SSE::cmpgt_epi64(x.d, y.d) :  _mm_cmpgt_epi32(x.d, y.d))}; }
    static Vc_INTRINSIC mask< ulong> less_equal(datapar< ulong> x, datapar< ulong> y) { return {private_init, Detail::not_(sizeof(long) == 8 ? SSE::cmpgt_epu64(x.d, y.d) : SSE::cmpgt_epu32(x.d, y.d))}; }
    static Vc_INTRINSIC mask<   int> less_equal(datapar<   int> x, datapar<   int> y) { return {private_init, Detail::not_( _mm_cmpgt_epi32(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  uint> less_equal(datapar<  uint> x, datapar<  uint> y) { return {private_init, Detail::not_(SSE::cmpgt_epu32(x.d, y.d))}; }
    static Vc_INTRINSIC mask< short> less_equal(datapar< short> x, datapar< short> y) { return {private_init, Detail::not_( _mm_cmpgt_epi16(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ushort> less_equal(datapar<ushort> x, datapar<ushort> y) { return {private_init, Detail::not_(SSE::cmpgt_epu16(x.d, y.d))}; }
    static Vc_INTRINSIC mask< schar> less_equal(datapar< schar> x, datapar< schar> y) { return {private_init, Detail::not_( _mm_cmpgt_epi8 (x.d, y.d))}; }
    static Vc_INTRINSIC mask< uchar> less_equal(datapar< uchar> x, datapar< uchar> y) { return {private_init, Detail::not_(SSE::cmpgt_epu8 (x.d, y.d))}; }
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
struct sse_mask_impl {
    // member types {{{2
    using abi = datapar_abi::sse;
    template <class T> static constexpr size_t size = datapar_size_v<T, abi>;
    template <class T> using mask_member_type = sse_mask_member_type<T>;
    template <class T> using mask = Vc::mask<T, datapar_abi::sse>;
    template <class T> using mask_bool = Common::MaskBool<sizeof(T)>;
    template <size_t N> using size_tag = std::integral_constant<size_t, N>;

    // broadcast {{{2
    static Vc_INTRINSIC auto broadcast(bool x, size_tag<2>) noexcept
    {
        return _mm_set1_pd(mask_bool<double>{x});
    }
    static Vc_INTRINSIC auto broadcast(bool x, size_tag<4>) noexcept
    {
        return _mm_set1_ps(mask_bool<float>{x});
    }
    static Vc_INTRINSIC auto broadcast(bool x, size_tag<8>) noexcept
    {
        return _mm_set1_epi16(mask_bool<std::int16_t>{x});
    }
    static Vc_INTRINSIC auto broadcast(bool x, size_tag<16>) noexcept
    {
        return _mm_set1_epi8(mask_bool<std::int8_t>{x});
    }

    // load {{{2
    template <class F>
    static Vc_INTRINSIC auto load(const bool *mem, F, size_tag<2>) noexcept
    {
        return _mm_set_epi32(-mem[1], -mem[1], -mem[0], -mem[0]);
    }
    template <class F>
    static Vc_INTRINSIC auto load(const bool *mem, F, size_tag<4>) noexcept
    {
        __m128i k = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem));
        k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128());
        return sse_cast<__m128>(_mm_unpacklo_epi16(k, k));
    }
    template <class F>
    static Vc_INTRINSIC auto load(const bool *mem, F, size_tag<8>) noexcept
    {
#ifdef Vc_IS_AMD64
        __m128i k = _mm_cvtsi64_si128(*reinterpret_cast<const int64_t *>(mem));
#else
        __m128i k = _mm_castpd_si128(_mm_load_sd(reinterpret_cast<const double *>(mem)));
#endif
        return sse_cast<__m128>(
            _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128()));
    }
    template <class F>
    static Vc_INTRINSIC auto load(const bool *mem, F, size_tag<16>) noexcept
    {
        return sse_cast<__m128>(
            _mm_cmpgt_epi8(std::is_same<F, flags::vector_aligned_tag>::value
                               ? _mm_load_si128(reinterpret_cast<const __m128i *>(mem))
                               : _mm_loadu_si128(reinterpret_cast<const __m128i *>(mem)),
                           _mm_setzero_si128()));
    }

    // masked load {{{2
    template <class T, class F, class SizeTag>
    static Vc_INTRINSIC void masked_load(mask_member_type<T> &merge,
                                         mask_member_type<T> mask, const bool *mem, F,
                                         SizeTag s) noexcept
    {
        for (std::size_t i = 0; i < s; ++i) {
            if (mask.m(i)) {
                merge.set(i, mask_bool<T>{mem[i]});
            }
        }
    }

    // store {{{2
    template <class T, class F>
    static Vc_INTRINSIC void store(mask_member_type<T> v, bool *mem, F,
                                   size_tag<2>) noexcept
    {
        const auto k = sse_cast<__m128i>(v.v());
        mem[0] = -SseIntrinsics::extract_epi32<1>(k);
        mem[1] = -SseIntrinsics::extract_epi32<3>(k);
    }
    template <class T, class F>
    static Vc_INTRINSIC void store(mask_member_type<T> v, bool *mem, F,
                                   size_tag<4>) noexcept
    {
        const auto k = sse_cast<__m128i>(v.v());
        *reinterpret_cast<MayAlias<int32_t> *>(mem) = _mm_cvtsi128_si32(
            _mm_packs_epi16(_mm_srli_epi16(_mm_packs_epi32(k, _mm_setzero_si128()), 15),
                            _mm_setzero_si128()));
    }
    template <class T, class F>
    static Vc_INTRINSIC void store(mask_member_type<T> v, bool *mem, F,
                                   size_tag<8>) noexcept
    {
        auto k = sse_cast<__m128i>(v.v());
        k = _mm_srli_epi16(k, 15);
        const auto k2 = _mm_packs_epi16(k, _mm_setzero_si128());
#ifdef Vc_IS_AMD64
        *reinterpret_cast<MayAlias<int64_t> *>(mem) = _mm_cvtsi128_si64(k2);
#else
        _mm_store_sd(reinterpret_cast<MayAlias<double> *>(mem), _mm_castsi128_pd(k2));
#endif
    }
    template <class T, class F>
    static Vc_INTRINSIC void store(mask_member_type<T> v, bool *mem, F,
                                   size_tag<16>) noexcept
    {
        auto k = sse_cast<__m128i>(v.v());
        k = _mm_and_si128(k, _mm_set1_epi32(0x01010101));
        if (std::is_same<F, flags::vector_aligned_tag>::value) {
            _mm_store_si128(reinterpret_cast<__m128i *>(mem), k);
        } else {
            _mm_storeu_si128(reinterpret_cast<__m128i *>(mem), k);
        }
    }

    // masked store {{{2
    template <class T, class F, class SizeTag>
    static Vc_INTRINSIC void masked_store(mask_member_type<T> v, bool *mem, F, SizeTag,
                                          mask_member_type<T> k) noexcept
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
        return Detail::not_(x.v());
        //return !x.builtin();
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
struct sse_compare_base {
protected:
    template <class T> using V = Vc::datapar<T, Vc::datapar_abi::sse>;
    template <class T> using M = Vc::mask<T, Vc::datapar_abi::sse>;
    template <class T>
    using S = typename Vc::detail::traits<T, Vc::datapar_abi::sse>::mask_cast_type;
};
// }}}1
}  // namespace Vc_VERSIONED_NAMESPACE::detail

// [mask.reductions] {{{
namespace Vc_VERSIONED_NAMESPACE
{
template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE bool all_of(mask<T, datapar_abi::sse> k)
{
    const auto d = detail::sse_cast<__m128i>(
        static_cast<typename detail::traits<T, datapar_abi::sse>::mask_cast_type>(k));
#ifdef Vc_USE_PTEST
    return _mm_testc_si128(d, SSE::_mm_setallone_si128());  // return 1 if (0xffffffff,
                                                            // 0xffffffff, 0xffffffff,
                                                            // 0xffffffff) == (~0 & d.v())
#else
    return _mm_movemask_epi8(d) == 0xffff;
#endif
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE bool any_of(mask<T, datapar_abi::sse> k)
{
    const auto d = detail::sse_cast<__m128i>(
        static_cast<typename detail::traits<T, datapar_abi::sse>::mask_cast_type>(k));
#ifdef Vc_USE_PTEST
    return 0 == _mm_testz_si128(d, d);  // return 1 if (0, 0, 0, 0) == (d.v() & d.v())
#else
    return _mm_movemask_epi8(d) != 0x0000;
#endif
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE bool none_of(mask<T, datapar_abi::sse> k)
{
    const auto d = detail::sse_cast<__m128i>(
        static_cast<typename detail::traits<T, datapar_abi::sse>::mask_cast_type>(k));
#ifdef Vc_USE_PTEST
    return 0 != _mm_testz_si128(d, d);  // return 1 if (0, 0, 0, 0) == (d.v() & d.v())
#else
    return _mm_movemask_epi8(d) == 0x0000;
#endif
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE bool some_of(mask<T, datapar_abi::sse> k)
{
    const auto d = detail::sse_cast<__m128i>(
        static_cast<typename detail::traits<T, datapar_abi::sse>::mask_cast_type>(k));
#ifdef Vc_USE_PTEST
    return _mm_test_mix_ones_zeros(d, SSE::_mm_setallone_si128());
#else
    const int tmp = _mm_movemask_epi8(d);
    return tmp != 0 && (tmp ^ 0xffff) != 0;
#endif
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE int popcount(mask<T, datapar_abi::sse> k)
{
    const auto d = detail::sse_cast<__m128i>(
        static_cast<typename detail::traits<T, datapar_abi::sse>::mask_cast_type>(k));
    return Detail::mask_count<k.size()>(d);
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE int find_first_set(mask<T, datapar_abi::sse> k)
{
    const auto d = detail::sse_cast<__m128i>(
        static_cast<typename detail::traits<T, datapar_abi::sse>::mask_cast_type>(k));
    return _bit_scan_forward(Detail::mask_to_int<k.size()>(d));
}

template <class T, class = enable_if<sizeof(T) <= 8>>
Vc_ALWAYS_INLINE int find_last_set(mask<T, datapar_abi::sse> k)
{
    const auto d = detail::sse_cast<__m128i>(
        static_cast<typename detail::traits<T, datapar_abi::sse>::mask_cast_type>(k));
    return _bit_scan_reverse(Detail::mask_to_int<k.size()>(d));
}
}  // namespace Vc_VERSIONED_NAMESPACE
// }}}

namespace std
{
// mask operators {{{1
template <class T>
struct equal_to<Vc::mask<T, Vc::datapar_abi::sse>>
    : private Vc::detail::sse_compare_base {
public:
    bool operator()(const M<T> &x, const M<T> &y) const noexcept
    {
        return Vc::Detail::is_equal<M<T>::size()>(
            Vc::sse_cast<__m128>(static_cast<S<T>>(x)),
            Vc::sse_cast<__m128>(static_cast<S<T>>(y)));
    }
};
template <>
struct equal_to<Vc::mask<long double, Vc::datapar_abi::sse>>
    : public equal_to<Vc::mask<long double, Vc::datapar_abi::scalar>> {
};
// }}}1
}  // namespace std
#endif  // Vc_HAVE_SSE_ABI

#endif  // VC_DATAPAR_SSE_H_

// vim: foldmethod=marker
