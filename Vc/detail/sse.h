/*  This file is part of the Vc library. {{{
Copyright © 2016-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SIMD_SSE_H_
#define VC_SIMD_SSE_H_

#include "macros.h"
#ifdef Vc_HAVE_SSE_ABI
#include "storage.h"
#include "x86/intrinsics.h"
#include "x86/convert.h"
#include "x86/arithmetics.h"
#include "maskbool.h"
#include "genericimpl.h"
#include "simd_tuple.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// simd_mask impl {{{1
struct sse_mask_impl : public generic_mask_impl<simd_abi::Sse, sse_mask_member_type> {
    // member types {{{2
    using abi = simd_abi::Sse;
    template <class T> static constexpr size_t size() { return simd_size_v<T, abi>; }
    template <class T> using mask_member_type = sse_mask_member_type<T>;
    template <class T>
    using int_builtin_type = builtin_type16_t<detail::int_for_sizeof_t<T>>;
    template <class T> using simd_mask = Vc::simd_mask<T, simd_abi::Sse>;
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> broadcast(bool x, type_tag<T>) noexcept
    {
        return to_storage(x ? ~int_builtin_type<T>() : int_builtin_type<T>());
    }

    // load {{{2
    template <class F>
    static Vc_INTRINSIC auto load(const bool *mem, F, size_tag<4>) noexcept
    {
#ifdef Vc_HAVE_SSE2
        __m128i k = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem));
        k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128());
        return intrin_cast<__m128>(_mm_unpacklo_epi16(k, k));
#elif defined Vc_HAVE_MMX
        __m128 k = _mm_cvtpi8_ps(_mm_cvtsi32_si64(*reinterpret_cast<const int *>(mem)));
        _mm_empty();
        return _mm_cmpgt_ps(k, detail::zero<__m128>());
#endif  // Vc_HAVE_SSE2
    }
#ifdef Vc_HAVE_SSE2
    template <class F>
    static Vc_INTRINSIC auto load(const bool *mem, F, size_tag<2>) noexcept
    {
        return _mm_set_epi32(-int(mem[1]), -int(mem[1]), -int(mem[0]), -int(mem[0]));
    }
    template <class F>
    static Vc_INTRINSIC auto load(const bool *mem, F, size_tag<8>) noexcept
    {
#ifdef __x86_64__
        __m128i k = _mm_cvtsi64_si128(*reinterpret_cast<const int64_t *>(mem));
#else
        __m128i k = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem));
#endif
        return _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128());
    }
    template <class F>
    static Vc_INTRINSIC auto load(const bool *mem, F f, size_tag<16>) noexcept
    {
        return _mm_cmpgt_epi8(load16(mem, f), _mm_setzero_si128());
    }
#endif  // Vc_HAVE_SSE2

    // store {{{2
#if !defined Vc_HAVE_SSE2 && defined Vc_HAVE_MMX
    template <class F>
    static Vc_INTRINSIC void store(mask_member_type<float> v, bool *mem, F,
                                   size_tag<4>) noexcept
    {
        const __m128 k(v);
        const __m64 kk = _mm_cvtps_pi8(and_(k, detail::one16(float())));
        *reinterpret_cast<may_alias<int32_t> *>(mem) = _mm_cvtsi64_si32(kk);
        _mm_empty();
    }
#endif  // Vc_HAVE_MMX
#ifdef Vc_HAVE_SSE2
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F,
                                            size_tag<2>) noexcept
    {
        const auto k = to_m128i(v);
        mem[0] = -extract_epi32<1>(k);
        mem[1] = -extract_epi32<3>(k);
    }
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F,
                                            size_tag<4>) noexcept
    {
        const auto k = to_m128i(v);
        __m128i k2 = _mm_packs_epi32(k, _mm_setzero_si128());
        *reinterpret_cast<may_alias<int32_t> *>(mem) = _mm_cvtsi128_si32(
            _mm_packs_epi16(x86::srli_epi16<15>(k2), _mm_setzero_si128()));
    }
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F,
                                            size_tag<8>) noexcept
    {
        auto k = to_m128i(v);
        k = x86::srli_epi16<15>(k);
        const auto k2 = _mm_packs_epi16(k, _mm_setzero_si128());
#ifdef __x86_64__
        *reinterpret_cast<may_alias<int64_t> *>(mem) = _mm_cvtsi128_si64(k2);
#else
        _mm_store_sd(reinterpret_cast<may_alias<double> *>(mem), _mm_castsi128_pd(k2));
#endif
    }
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F f,
                                            size_tag<16>) noexcept
    {
        auto k = to_m128i(v);
        k = _mm_and_si128(k, _mm_set1_epi32(0x01010101));
        x86::store16(k, mem, f);
    }
#endif  // Vc_HAVE_SSE2

    // negation {{{2
    template <class T, class SizeTag>
    static Vc_INTRINSIC mask_member_type<T> negate(const mask_member_type<T> &x,
                                                   SizeTag) noexcept
    {
        return to_storage(~storage_bitcast<uint>(x).d);
    }

    // logical and bitwise operators {{{2
    template <class T>
    static Vc_INTRINSIC simd_mask<T> logical_and(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, detail::and_(x.d, y.d)};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> logical_or(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, detail::or_(x.d, y.d)};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> bit_and(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, detail::and_(x.d, y.d)};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> bit_or(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, detail::or_(x.d, y.d)};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> bit_xor(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, detail::xor_(x.d, y.d)};
    }

    // smart_reference access {{{2
    template <class T> static bool get(const mask_member_type<T> k, int i) noexcept
    {
        return k.d[i];
    }
    template <class T> static void set(mask_member_type<T> &k, int i, bool x) noexcept
    {
        auto tmp = reinterpret_cast<int_builtin_type<T>>(k.d);
        tmp[i] = -x;
        k = reinterpret_cast<builtin_type16_t<T>>(tmp);
    }
    // }}}2
};

// }}}1
}  // namespace detail

// [simd_mask.reductions] {{{
Vc_ALWAYS_INLINE bool Vc_VDECL all_of(simd_mask<float, simd_abi::Sse> k)
{
    const __m128 d(k);
#if defined Vc_HAVE_AVX
    return _mm_testc_ps(d, detail::allone<__m128>());
#elif defined Vc_HAVE_SSE4_1
    const auto dd = detail::intrin_cast<__m128i>(d);
    return _mm_testc_si128(dd, detail::allone<__m128i>());
#else
    return _mm_movemask_ps(d) == 0xf;
#endif
}

Vc_ALWAYS_INLINE bool Vc_VDECL any_of(simd_mask<float, simd_abi::Sse> k)
{
    const __m128 d(k);
#if defined Vc_HAVE_AVX
    return 0 == _mm_testz_ps(d, d);
#elif defined Vc_HAVE_SSE4_1
    const auto dd = detail::intrin_cast<__m128i>(d);
    return 0 == _mm_testz_si128(dd, dd);
#else
    return _mm_movemask_ps(d) != 0;
#endif
}

Vc_ALWAYS_INLINE bool Vc_VDECL none_of(simd_mask<float, simd_abi::Sse> k)
{
    const __m128 d(k);
#if defined Vc_HAVE_AVX
    return 0 != _mm_testz_ps(d, d);
#elif defined Vc_HAVE_SSE4_1
    const auto dd = detail::intrin_cast<__m128i>(d);
    return 0 != _mm_testz_si128(dd, dd);
#else
    return _mm_movemask_ps(d) == 0;
#endif
}

Vc_ALWAYS_INLINE bool Vc_VDECL some_of(simd_mask<float, simd_abi::Sse> k)
{
    const __m128 d(k);
#if defined Vc_HAVE_AVX
    return _mm_testnzc_ps(d, detail::allone<__m128>());
#elif defined Vc_HAVE_SSE4_1
    const auto dd = detail::intrin_cast<__m128i>(d);
    return _mm_testnzc_si128(dd, detail::allone<__m128i>());
#else
    const int tmp = _mm_movemask_ps(d);
    return tmp != 0 && (tmp ^ 0xf) != 0;
#endif
}

#ifdef Vc_HAVE_SSE2
Vc_ALWAYS_INLINE bool Vc_VDECL all_of(simd_mask<double, simd_abi::Sse> k)
{
    __m128d d(k);
#ifdef Vc_HAVE_SSE4_1
#ifdef Vc_HAVE_AVX
    return _mm_testc_pd(d, detail::allone<__m128d>());
#else
    const auto dd = detail::intrin_cast<__m128i>(d);
    return _mm_testc_si128(dd, detail::allone<__m128i>());
#endif
#else
    return _mm_movemask_pd(d) == 0x3;
#endif
}

Vc_ALWAYS_INLINE bool Vc_VDECL any_of(simd_mask<double, simd_abi::Sse> k)
{
    const __m128d d(k);
#if defined Vc_HAVE_AVX
    return 0 == _mm_testz_pd(d, d);
#elif defined Vc_HAVE_SSE4_1
    const auto dd = detail::intrin_cast<__m128i>(d);
    return 0 == _mm_testz_si128(dd, dd);
#else
    return _mm_movemask_pd(d) != 0;
#endif
}

Vc_ALWAYS_INLINE bool Vc_VDECL none_of(simd_mask<double, simd_abi::Sse> k)
{
    const __m128d d(k);
#if defined Vc_HAVE_AVX
    return 0 != _mm_testz_pd(d, d);
#elif defined Vc_HAVE_SSE4_1
    const auto dd = detail::intrin_cast<__m128i>(d);
    return 0 != _mm_testz_si128(dd, dd);
#else
    return _mm_movemask_pd(d) == 0;
#endif
}

Vc_ALWAYS_INLINE bool Vc_VDECL some_of(simd_mask<double, simd_abi::Sse> k)
{
    const __m128d d(k);
#if defined Vc_HAVE_AVX
    return _mm_testnzc_pd(d, detail::allone<__m128d>());
#elif defined Vc_HAVE_SSE4_1
    const auto dd = detail::intrin_cast<__m128i>(d);
    return _mm_testnzc_si128(dd, detail::allone<__m128i>());
#else
    const int tmp = _mm_movemask_pd(d);
    return tmp == 1 || tmp == 2;
#endif
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL all_of(simd_mask<T, simd_abi::Sse> k)
{
    const __m128i d(k);
#ifdef Vc_HAVE_SSE4_1
    return _mm_testc_si128(d, ~__m128i());  // return 1 if (0xffffffff,
                                            // 0xffffffff, 0xffffffff,
                                            // 0xffffffff) == (~0 & d.intrin())
#else
    return _mm_movemask_epi8(d) == 0xffff;
#endif
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL any_of(simd_mask<T, simd_abi::Sse> k)
{
    const __m128i d(k);
#ifdef Vc_HAVE_SSE4_1
    return 0 == _mm_testz_si128(d, d);  // return 1 if (0, 0, 0, 0) == (d.intrin() & d.intrin())
#else
    return _mm_movemask_epi8(d) != 0x0000;
#endif
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL none_of(simd_mask<T, simd_abi::Sse> k)
{
    const __m128i d(k);
#ifdef Vc_HAVE_SSE4_1
    return 0 != _mm_testz_si128(d, d);  // return 1 if (0, 0, 0, 0) == (d.intrin() & d.intrin())
#else
    return _mm_movemask_epi8(d) == 0x0000;
#endif
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL some_of(simd_mask<T, simd_abi::Sse> k)
{
    const __m128i d(k);
#ifdef Vc_HAVE_SSE4_1
    return _mm_test_mix_ones_zeros(d, detail::allone<__m128i>());
#else
    const int tmp = _mm_movemask_epi8(d);
    return tmp != 0 && (tmp ^ 0xffff) != 0;
#endif
}
#endif

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL popcount(simd_mask<T, simd_abi::Sse> k)
{
    const auto d = detail::data(k);
    return detail::mask_count<k.size()>(d);
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL find_first_set(simd_mask<T, simd_abi::Sse> k)
{
    const auto d = detail::data(k);
    return detail::firstbit(detail::mask_to_int<k.size()>(d));
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL find_last_set(simd_mask<T, simd_abi::Sse> k)
{
    const auto d = detail::data(k);
    return detail::lastbit(detail::mask_to_int<k.size()>(d));
}

// }}}

namespace detail
{
// simd impl {{{1
struct sse_simd_impl : public generic_simd_impl<sse_simd_impl> {
    // member types {{{2
    using abi = simd_abi::Sse;
    template <class T> static constexpr size_t size() { return simd_size_v<T, abi>; }
    template <class T> using simd_member_type = sse_simd_member_type<T>;
    template <class T> using intrinsic_type = typename simd_member_type<T>::register_type;
    template <class T> using mask_member_type = sse_mask_member_type<T>;
    template <class T> using simd = Vc::simd<T, abi>;
    template <class T> using simd_mask = Vc::simd_mask<T, abi>;
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;

    // make_simd {{{2
    template <class T>
    static Vc_INTRINSIC simd<T> make_simd(simd_member_type<T> x)
    {
        return {detail::private_init, x};
    }

    // load {{{2
    // from long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC simd_member_type<T> load(const long double *mem, F,
                                                    type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        return generate_storage<T, size<T>()>(
            [&](auto i) { return static_cast<T>(mem[i]); });
    }

    // load without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC simd_member_type<T> load(const T *mem, F f,
                                                    type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        return detail::load16(mem, f);
    }

    // convert from an SSE load{{{3
    template <class T, class U, class F>
    static inline simd_member_type<T> load(
        const convertible_memory<U, sizeof(T), T> *mem, F f, type_tag<T>,
        tag<1> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_SSE_ABI
        return x86::convert<simd_member_type<T>, simd_member_type<U>>(
            detail::load16(mem, f));
#else
        unused(f);
        return generate_storage<T, size<T>()>(
            [&](auto i) { return static_cast<T>(mem[i]); });
#endif
    }

    // convert from a half SSE load{{{3
    template <class T, class U, class F>
    static inline simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 2, T> *mem, F f, type_tag<T>,
        tag<2> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_SSE_ABI
        return x86::convert<simd_member_type<T>, simd_member_type<U>>(
            intrin_cast<detail::intrinsic_type_t<U, size<U>()>>(load8(mem, f)));
#else
        return generate_storage<T, size<T>()>(
            [&](auto i) { return static_cast<T>(mem[i]); });
        unused(f);
#endif
    }

    // convert from a quarter SSE load{{{3
    template <class T, class U, class F>
    static inline simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 4, T> *mem, F f, type_tag<T>,
        tag<3> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_SSE_ABI
        return x86::convert<simd_member_type<T>, simd_member_type<U>>(
            intrin_cast<detail::intrinsic_type_t<U, size<U>()>>(load4(mem, f)));
#else
        return generate_storage<T, size<T>()>(
            [&](auto i) { return static_cast<T>(mem[i]); });
        unused(f);
#endif
    }

    // convert from a 1/8th SSE load{{{3
#ifdef Vc_HAVE_FULL_SSE_ABI
    template <class T, class U>
    static Vc_INTRINSIC simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 8, T> *mem,
        when_aligned<alignof(uint16_t)>, type_tag<T>, tag<4> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return x86::convert<simd_member_type<T>, simd_member_type<U>>(
            intrin_cast<detail::intrinsic_type_t<U, size<U>()>>(
                load2(mem, vector_aligned)));
    }

    template <class T, class U>
    static Vc_INTRINSIC simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 8, T> *mem,
        when_unaligned<alignof(uint16_t)>, type_tag<T>,
        tag<4> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return make_storage<T>(T(mem[0]), T(mem[1]));
    }
#else   // Vc_HAVE_FULL_SSE_ABI
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 8, T> *mem, F, type_tag<T>,
        tag<4> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return make_storage<T>(T(mem[0]), T(mem[1]));
    }
#endif  // Vc_HAVE_FULL_SSE_ABI

    // AVX and AVX-512 simd_member_type aliases{{{3
    template <class T> using avx_member_type = avx_simd_member_type<T>;
    template <class T> using avx512_member_type = avx512_simd_member_type<T>;

    // convert from an AVX/2-SSE load{{{3
    template <class T, class U, class F>
    static inline simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) * 2, T> *mem, F f, type_tag<T>,
        tag<5> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX
        return x86::convert<simd_member_type<T>, avx_member_type<U>>(
            detail::load32(mem, f));
#elif defined Vc_HAVE_FULL_SSE_ABI
        return x86::convert<simd_member_type<T>, simd_member_type<U>>(
            load(mem, f, type_tag<U>()), load(mem + size<U>(), f, type_tag<U>()));
#else
        unused(f);
        return generate_storage<T, size<T>()>(
            [&](auto i) { return static_cast<T>(mem[i]); });
#endif
    }

    // convert from an AVX512/2-AVX/4-SSE load{{{3
    template <class T, class U, class F>
    static inline simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) * 4, T> *mem, F f, type_tag<T>,
        tag<6> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX512F
        return x86::convert<simd_member_type<T>, avx512_member_type<U>>(load64(mem, f));
#elif defined Vc_HAVE_AVX
        return x86::convert<simd_member_type<T>, avx_member_type<U>>(
            detail::load32(mem, f), detail::load32(mem + 2 * size<U>(), f));
#else
        return x86::convert<simd_member_type<T>, simd_member_type<U>>(
            load(mem, f, type_tag<U>()), load(mem + size<U>(), f, type_tag<U>()),
            load(mem + 2 * size<U>(), f, type_tag<U>()),
            load(mem + 3 * size<U>(), f, type_tag<U>()));
#endif
    }

    // convert from a 2-AVX512/4-AVX/8-SSE load{{{3
    template <class T, class U, class F>
    static inline simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) * 8, T> *mem, F f, type_tag<T>,
        tag<7> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX512F
        return x86::convert<simd_member_type<T>, avx512_member_type<U>>(
            load64(mem, f), load64(mem + 4 * size<U>(), f));
#elif defined Vc_HAVE_AVX
        return x86::convert<simd_member_type<T>, avx_member_type<U>>(
            load32(mem, f), load32(mem + 2 * size<U>(), f),
            load32(mem + 4 * size<U>(), f), load32(mem + 6 * size<U>(), f));
#else
        return x86::convert<simd_member_type<T>, simd_member_type<U>>(
            load16(mem, f), load16(mem + size<U>(), f), load16(mem + 2 * size<U>(), f),
            load16(mem + 3 * size<U>(), f), load16(mem + 4 * size<U>(), f),
            load16(mem + 5 * size<U>(), f), load16(mem + 6 * size<U>(), f),
            load16(mem + 7 * size<U>(), f));
#endif
    }

    // masked load {{{2
    // fallback {{{3
    template <class T, class U, class F>
    static inline void Vc_VDECL masked_load(simd_member_type<T> &merge,
                                            mask_member_type<T> k, const U *mem,
                                            F) Vc_NOEXCEPT_OR_IN_TEST
    {
        execute_n_times<size<T>()>([&](auto i) {
            if (k[i]) {
                merge.set(i, static_cast<T>(mem[i]));
            }
        });
    }

    // 8-bit and 16-bit integers with AVX512VL/BW {{{3
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<schar> &merge,
                                                  mask_member_type<schar> k, const schar *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm_mask_loadu_epi8(merge, _mm_movemask_epi8(k), mem);
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<uchar> &merge,
                                                  mask_member_type<uchar> k, const uchar *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm_mask_loadu_epi8(merge, _mm_movemask_epi8(k), mem);
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<short> &merge,
                                                  mask_member_type<short> k, const short *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm_mask_loadu_epi16(merge, x86::movemask_epi16(k), mem);
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<ushort> &merge,
                                                  mask_member_type<ushort> k, const ushort *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm_mask_loadu_epi16(merge, x86::movemask_epi16(k), mem);
    }

#endif  // AVX512VL && AVX512BW

    // 32-bit and 64-bit integers with AVX2 {{{3
#ifdef Vc_HAVE_AVX2
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<int> &merge,
                                                  mask_member_type<int> k, const int *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge), _mm_maskload_epi32(mem, k));
    }
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<uint> &merge,
                                                  mask_member_type<uint> k, const uint *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge),
                    _mm_maskload_epi32(
                        reinterpret_cast<const detail::may_alias<int> *>(mem), k));
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<llong> &merge,
                                                  mask_member_type<llong> k, const llong *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge), _mm_maskload_epi64(mem, k));
    }
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<ullong> &merge,
                                                  mask_member_type<ullong> k, const ullong *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge),
                    _mm_maskload_epi64(
                        reinterpret_cast<const may_alias<long long> *>(mem), k));
    }
#endif  // Vc_HAVE_AVX2

    // 32-bit and 64-bit floats with AVX {{{3
#ifdef Vc_HAVE_AVX
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<double> &merge,
                                                  mask_member_type<double> k, const double *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge), _mm_maskload_pd(mem, _mm_castpd_si128(k)));
    }
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<float> &merge,
                                                  mask_member_type<float> k,
                                                  const float *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge), _mm_maskload_ps(mem, _mm_castps_si128(k)));
    }
#endif  // Vc_HAVE_AVX

    // store {{{2
    // store to long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(simd_member_type<T> v, long double *mem, F,
                                            type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        // alignment F doesn't matter
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v[i]; });
    }

    // store without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(simd_member_type<T> v, T *mem, F f,
                                            type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        store16(v, mem, f);
    }

    // convert and 16-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 8> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        store2(x86::convert<simd_member_type<U>>(v), mem, f);
    }

    // convert and 32-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 4> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_SSE_ABI
        store4(x86::convert<simd_member_type<U>>(v), mem, f);
#else
        unused(f);
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
#endif
    }

    // convert and 64-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 2> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_SSE_ABI
        store8(x86::convert<simd_member_type<U>>(v), mem, f);
#else
        unused(f);
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
#endif
    }

    // convert and 128-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_SSE_ABI
        store16(x86::convert<simd_member_type<U>>(v), mem, f);
#else
        unused(f);
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
#endif
    }

    // convert and 256-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) * 2 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX
        store32(x86::convert<avx_member_type<U>>(v), mem, f);
#elif defined Vc_HAVE_FULL_SSE_ABI
        // without the full SSE ABI there cannot be any vectorized converting loads
        // because only float vectors exist
        const auto tmp = convert_all<simd_member_type<U>>(v);
        store16(tmp[0], mem, f);
        store16(tmp[1], mem + size<T>() / 2, f);
#else
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
        detail::unused(f);
#endif
    }

    // convert and 512-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) * 4 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX512F
        store64(convert_all<avx512_member_type<U>>(v), mem, f);
#elif defined Vc_HAVE_AVX
        const auto tmp = convert_all<avx_member_type<U>>(v);
        store32(tmp[0], mem, f);
        store32(tmp[1], mem + size<T>() / 2, f);
#else
        const auto tmp = convert_all<simd_member_type<U>>(v);
        store16(tmp[0], mem, f);
        store16(tmp[1], mem + size<T>() * 1 / 4, f);
        store16(tmp[2], mem + size<T>() * 2 / 4, f);
        store16(tmp[3], mem + size<T>() * 3 / 4, f);
#endif
    }

    // convert and 1024-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) * 8 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX512F
        const auto tmp = convert_all<avx512_member_type<U>>(v);
        store64(tmp[0], mem, f);
        store64(tmp[1], mem + size<T>() / 2, f);
#elif defined Vc_HAVE_AVX
        const auto tmp = convert_all<avx_member_type<U>>(v);
        store32(tmp[0], mem, f);
        store32(tmp[1], mem + size<T>() * 1 / 4, f);
        store32(tmp[2], mem + size<T>() * 2 / 4, f);
        store32(tmp[3], mem + size<T>() * 3 / 4, f);
#else
        const auto tmp = convert_all<simd_member_type<U>>(v);
        store16(tmp[0], mem, f);
        store16(tmp[1], mem + size<T>() * 1 / 8, f);
        store16(tmp[2], mem + size<T>() * 2 / 8, f);
        store16(tmp[3], mem + size<T>() * 3 / 8, f);
        store16(tmp[4], mem + size<T>() * 4 / 8, f);
        store16(tmp[5], mem + size<T>() * 5 / 8, f);
        store16(tmp[6], mem + size<T>() * 6 / 8, f);
        store16(tmp[7], mem + size<T>() * 7 / 8, f);
#endif
    }

    // masked store {{{2
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL
    masked_store(const simd_member_type<T> v, long double *mem, F,
                 const mask_member_type<T> k) Vc_NOEXCEPT_OR_IN_TEST
    {
        // no SSE support for long double
        execute_n_times<size<T>()>([&](auto i) {
            if (k[i]) {
                mem[i] = v[i];
            }
        });
    }

    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL masked_store(const simd_member_type<T> v, U *mem, F,
                                                   const mask_member_type<T> k)
        Vc_NOEXCEPT_OR_IN_TEST
    {
        constexpr int N = simd_member_type<T>::width;
        constexpr bool truncate = have_avx512vl && std::is_integral_v<T> && std::is_integral_v<U> && sizeof(T) > sizeof(U);
        if constexpr (std::is_same_v<T, U> ||
                      (std::is_integral_v<T> && std::is_integral_v<U> &&
                       sizeof(T) == sizeof(U))) {
            x86::maskstore(storage_bitcast<U>(v), mem, F(), storage_bitcast<U>(k));
        } else if constexpr (truncate && sizeof(T) == 8) {
            auto kk = convert_any_mask<Storage<bool, N>>(k);
            if constexpr (sizeof(U) == 4) {
                _mm_mask_cvtepi64_storeu_epi32(mem, kk, v);
            } else if constexpr (sizeof(U) == 2) {
                _mm_mask_cvtepi64_storeu_epi16(mem, kk, v);
            } else if constexpr (sizeof(U) == 1) {
                _mm_mask_cvtepi64_storeu_epi8(mem, kk, v);
            }
        } else if constexpr (truncate && sizeof(T) == 4) {
            auto kk = convert_any_mask<Storage<bool, N>>(k);
            if constexpr (sizeof(U) == 2) {
                _mm_mask_cvtepi32_storeu_epi16(mem, kk, v);
            } else if constexpr (sizeof(U) == 1) {
                _mm_mask_cvtepi32_storeu_epi8(mem, kk, v);
            }
        } else if constexpr (truncate && have_avx512bw && sizeof(T) == 2) {
            auto kk = convert_any_mask<Storage<bool, N>>(k);
            _mm_mask_cvtepi16_storeu_epi8(mem, kk, v);
        /* TODO:
        } else if constexpr (sizeof(T) * 2 == sizeof(U)) {
            if constexpr(have_avx512vl) {
                x86::maskstore(convert<storage32_t<U>>(v), mem, F(), convert_any_mask<Storage<bool, N>>(k));
            } else if constexpr(have_avx2 || (have_avx && std::is_floating_point_v<U>)) {
                x86::maskstore(convert<storage32_t<U>>(v), mem, F(), convert_any_mask<storage32_t<U>>(k));
            } else {
                using V = storage16_t<U>;
                const std::array<V, 2> converted = convert_all<V>(v);
                _mm_maskmoveu_si128(converted[0], 
                x86::maskstore(converted[0], mem, F(), M(k >> 0));
                x86::maskstore(converted[1], mem + V::width, F(), M(k >> V::width));
            }
        } else if constexpr (sizeof(T) * 4 == sizeof(U)) {
            const std::array<V, 4> converted = convert_all<V>(v);
            x86::maskstore(converted[0], mem, F(), M(k >> 0));
            x86::maskstore(converted[1], mem + 1 * V::width, F(), M(k >> 1 * V::width));
            x86::maskstore(converted[2], mem + 2 * V::width, F(), M(k >> 2 * V::width));
            x86::maskstore(converted[3], mem + 3 * V::width, F(), M(k >> 3 * V::width));
        } else if constexpr (sizeof(T) * 8 == sizeof(U)) {
            const std::array<V, 8> converted = convert_all<V>(v);
            x86::maskstore(converted[0], mem, F(), M(k >> 0));
            x86::maskstore(converted[1], mem + 1 * V::width, F(), M(k >> 1 * V::width));
            x86::maskstore(converted[2], mem + 2 * V::width, F(), M(k >> 2 * V::width));
            x86::maskstore(converted[3], mem + 3 * V::width, F(), M(k >> 3 * V::width));
            x86::maskstore(converted[4], mem + 4 * V::width, F(), M(k >> 4 * V::width));
            x86::maskstore(converted[5], mem + 5 * V::width, F(), M(k >> 5 * V::width));
            x86::maskstore(converted[6], mem + 6 * V::width, F(), M(k >> 6 * V::width));
            x86::maskstore(converted[7], mem + 7 * V::width, F(), M(k >> 7 * V::width));
        } else if constexpr (sizeof(T) > sizeof(U)) {
            x86::maskstore(convert<V>(v), mem, F(), k);
        */
        } else {
            execute_n_times<size<T>()>([&](auto i) {
                if (k[i]) {
                    mem[i] = static_cast<T>(v[i]);
                }
            });
        }
    }

    // negation {{{2
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> Vc_VDECL
    negate(simd_member_type<T> x) noexcept
    {
        return detail::to_storage(!x.d);
    }

    // reductions {{{2
    template <class BinaryOperation>
    static Vc_INTRINSIC double Vc_VDECL reduce(size_tag<2>, simd<double> x,
                                               BinaryOperation &binary_op)
    {
        using V = simd<double>;
        auto intrin_ = data(x);
        intrin_ = data(
            binary_op(x, V(detail::private_init, _mm_unpackhi_pd(intrin_, intrin_))));
        return _mm_cvtsd_f64(intrin_);
    }

    template <class BinaryOperation>
    static Vc_INTRINSIC float Vc_VDECL reduce(size_tag<4>, simd<float> x,
                                              BinaryOperation &binary_op)
    {
        using V = simd<float>;
        auto intrin_ = data(x);
        intrin_ = data(
            binary_op(x, V(detail::private_init,
                           _mm_shuffle_ps(intrin_, intrin_, _MM_SHUFFLE(0, 1, 2, 3)))));
        intrin_ =
            data(binary_op(V(detail::private_init, intrin_),
                           V(detail::private_init, _mm_unpackhi_ps(intrin_, intrin_))));
        return _mm_cvtss_f32(intrin_);
    }

    template <class T, class BinaryOperation>
    static Vc_INTRINSIC T Vc_VDECL reduce(size_tag<2>, const simd<T> x,
                                          BinaryOperation &binary_op)
    {
        return binary_op(x[0], x[1]);
    }

    template <class T, class BinaryOperation>
    static Vc_INTRINSIC T Vc_VDECL reduce(size_tag<4>, simd<T> x,
                                          BinaryOperation &binary_op)
    {
        using V = simd<T>;
        auto intrin_ = data(x);
        intrin_ =
            data(binary_op(x, V(detail::private_init,
                                _mm_shuffle_epi32(intrin_, _MM_SHUFFLE(0, 1, 2, 3)))));
        intrin_ = data(
            binary_op(V(detail::private_init, intrin_),
                      V(detail::private_init, _mm_unpackhi_epi64(intrin_, intrin_))));
        return _mm_cvtsi128_si32(intrin_);
    }

    template <class T, class BinaryOperation>
    static Vc_INTRINSIC T Vc_VDECL reduce(size_tag<8>, simd<T> x,
                                          BinaryOperation &binary_op)
    {
        using V = simd<T>;
        auto intrin_ = data(x);
        intrin_ = data(
            binary_op(V(detail::private_init, _mm_unpacklo_epi16(intrin_, intrin_)),
                      V(detail::private_init, _mm_unpackhi_epi16(intrin_, intrin_))));
        intrin_ = data(
            binary_op(V(detail::private_init, _mm_unpacklo_epi32(intrin_, intrin_)),
                      V(detail::private_init, _mm_unpackhi_epi32(intrin_, intrin_))));
        return binary_op(
            V(detail::private_init, intrin_),
            V(detail::private_init, _mm_unpackhi_epi64(intrin_, intrin_)))[0];
    }

    template <class T, class BinaryOperation>
    static Vc_INTRINSIC T Vc_VDECL reduce(size_tag<16>, simd<T> x,
                                          BinaryOperation &binary_op)
    {
        using V = simd<T>;
        auto intrin_ = data(x);
        intrin_ =
            data(binary_op(V(detail::private_init, _mm_unpacklo_epi8(intrin_, intrin_)),
                           V(detail::private_init, _mm_unpackhi_epi8(intrin_, intrin_))));
        intrin_ = data(
            binary_op(V(detail::private_init, _mm_unpacklo_epi16(intrin_, intrin_)),
                      V(detail::private_init, _mm_unpackhi_epi16(intrin_, intrin_))));
        intrin_ = data(
            binary_op(V(detail::private_init, _mm_unpacklo_epi32(intrin_, intrin_)),
                      V(detail::private_init, _mm_unpackhi_epi32(intrin_, intrin_))));
        return binary_op(
            V(detail::private_init, intrin_),
            V(detail::private_init, _mm_unpackhi_epi64(intrin_, intrin_)))[0];
    }

    // compares {{{2
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> equal_to(simd_member_type<T> x, simd_member_type<T> y)
    {
        return to_storage(x.d == y.d);
    }
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> not_equal_to(simd_member_type<T> x, simd_member_type<T> y)
    {
        return to_storage(x.d != y.d);
    }
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> less(simd_member_type<T> x, simd_member_type<T> y)
    {
        return to_storage(x.d < y.d);
    }
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> less_equal(simd_member_type<T> x, simd_member_type<T> y)
    {
        return to_storage(x.d <= y.d);
    }

    // math {{{2
    // sqrt {{{3
    static Vc_INTRINSIC simd_member_type<float> sqrt(simd_member_type<float> x)
    {
        return _mm_sqrt_ps(x);
    }
    static Vc_INTRINSIC simd_member_type<double> sqrt(simd_member_type<double> x)
    {
        return _mm_sqrt_pd(x);
    }

    // logb {{{3
    static Vc_INTRINSIC Vc_CONST simd_member_type<float> logb_positive(simd_member_type<float> v)
    {
#ifdef Vc_HAVE_AVX512VL
        return _mm_getexp_ps(v);
#else   // Vc_HAVE_AVX512VL
        __m128i tmp = _mm_srli_epi32(_mm_castps_si128(v), 23);
        tmp = _mm_sub_epi32(tmp, _mm_set1_epi32(0x7f));
        return _mm_cvtepi32_ps(tmp);
#endif  // Vc_HAVE_AVX512VL
    }

    static Vc_INTRINSIC Vc_CONST simd_member_type<double> logb_positive(simd_member_type<double> v)
    {
#ifdef Vc_HAVE_AVX512VL
        return _mm_getexp_pd(v);
#else   // Vc_HAVE_AVX512VL
        __m128i tmp = _mm_srli_epi64(_mm_castpd_si128(v), 52);
        tmp = _mm_sub_epi32(tmp, _mm_set1_epi32(0x3ff));
        return _mm_cvtepi32_pd(_mm_shuffle_epi32(tmp, 0x08));
#endif  // Vc_HAVE_AVX512VL
    }

#ifdef Vc_HAVE_AVX512VL
    static Vc_INTRINSIC Vc_CONST simd_member_type<float> logb(simd_member_type<float> v)
    {
        return _mm_fixupimm_ps(_mm_getexp_ps(abs(v)), v, broadcast16(0x00550433), 0x00);
    }
    static Vc_INTRINSIC Vc_CONST simd_member_type<double> logb(simd_member_type<double> v)
    {
        return _mm_fixupimm_pd(_mm_getexp_pd(abs(v)), v, broadcast16(0x00550433), 0x00);
    }
#endif  // Vc_HAVE_AVX512VL

    // trunc {{{3
#ifdef Vc_HAVE_SSE4_1
    static Vc_INTRINSIC simd_member_type<float> trunc(simd_member_type<float> x)
    {
        return _mm_round_ps(x, 0x3);
    }
    static Vc_INTRINSIC simd_member_type<double> trunc(simd_member_type<double> x)
    {
        return _mm_round_pd(x, 0x3);
    }
#else
    static Vc_INTRINSIC simd_member_type<float> trunc(simd_member_type<float> x)
    {
        auto truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
        auto mask = intrin_cast<__m128>(_mm_cmplt_epi32(
            and_(to_m128i(x), broadcast16(0x7f800000u)),
            broadcast16(0x4b000000)));  // exponent is so large that no mantissa bits
                                        // signify fractional values (0x3f8 + 23*8 =
                                        // 0x4b0)
        return blendv_ps(x, truncated, mask);
    }

    static Vc_INTRINSIC simd_member_type<double> trunc(simd_member_type<double> x)
    {
        const auto abs_x = abs(x);
        const auto min_no_fractional_bits = intrin_cast<__m128d>(
            broadcast16(0x4330'0000'0000'0000ull));  // 0x3ff + 52 = 0x433
        simd_member_type<double> truncated =
            _mm_sub_pd(_mm_add_pd(abs_x, min_no_fractional_bits), min_no_fractional_bits);
        // due to rounding, the result can be too large. In this case `truncated > abs(x)`
        // holds, so subtract 1 to truncated if `abs(x) < truncated`
        truncated = _mm_sub_pd(truncated, and_(less(abs_x, truncated), broadcast16(1.)));
        // finally, fix the sign bit:
        return or_(and_(intrin_cast<__m128d>(broadcast16(0x8000'0000'0000'0000ull)), x),
                   truncated);
    }
#endif

    // floor {{{3
#ifdef Vc_HAVE_SSE4_1
    static Vc_INTRINSIC simd_member_type<float> floor(simd_member_type<float> x)
    {
        return _mm_floor_ps(x);
    }
    static Vc_INTRINSIC simd_member_type<double> floor(simd_member_type<double> x)
    {
        return _mm_floor_pd(x);
    }
#else
    template<class T>
    static Vc_INTRINSIC simd_member_type<T> floor(simd_member_type<T> x)
    {
        auto y = trunc(x);
        const auto negative_input = less(x, simd_member_type<T>::broadcast(T(0)));
        const auto mask = andnot_(equal_to(y, x), negative_input);
        return or_(andnot_(mask, y),
                   and_(mask, minus(y, simd_member_type<T>::broadcast(T(1)))));
    }
#endif

    // ceil {{{3
#ifdef Vc_HAVE_SSE4_1
    static Vc_INTRINSIC simd_member_type<float> ceil(simd_member_type<float> x)
    {
        return _mm_ceil_ps(x);
    }
    static Vc_INTRINSIC simd_member_type<double> ceil(simd_member_type<double> x)
    {
        return _mm_ceil_pd(x);
    }
#else
    template<class T>
    static Vc_INTRINSIC simd_member_type<T> ceil(simd_member_type<T> x)
    {
        auto y = trunc(x);
        const auto negative_input = less(x, simd_member_type<T>::broadcast(T(0)));
        const auto inv_mask = or_(equal_to(y, x), negative_input);
        return or_(and_(inv_mask, y),
                   andnot_(inv_mask, plus(y, simd_member_type<T>::broadcast(T(1)))));
    }
#endif

    // frexp {{{3
    /**
     * splits \p v into exponent and mantissa, the sign is kept with the mantissa
     *
     * The return value will be in the range [0.5, 1.0[
     * The \p e value will be an integer defining the power-of-two exponent
     */
#ifdef Vc_HAVE_AVX512VL
    static inline simd_member_type<double> frexp(
        simd_member_type<double> v,
        simd_tuple<int, simd_abi::scalar, simd_abi::scalar> &exp)
    {
        const __mmask8 isnonzerovalue = _mm_cmp_pd_mask(
            _mm_mul_pd(broadcast16(std::numeric_limits<double>::infinity()),
                       v),                                 // NaN if v == 0
            _mm_mul_pd(_mm_setzero_pd(), v), _CMP_ORD_Q);  // NaN if v == inf
        if (Vc_IS_LIKELY(isnonzerovalue == 0x03)) {
            const x_i32 e =
                _mm_add_epi32(broadcast16(1), _mm_cvttpd_epi32(_mm_getexp_pd(v)));
            exp.first = e[0];
            exp.second.first = e[1];
            return _mm_getmant_pd(v, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
        }
        const x_i32 e =
            _mm_mask_add_epi32(_mm_setzero_si128(), isnonzerovalue, broadcast16(1),
                               _mm_cvttpd_epi32(_mm_getexp_pd(v)));
        exp.first = e[0];
        exp.second.first = e[1];
        return _mm_mask_getmant_pd(v, isnonzerovalue, v, _MM_MANT_NORM_p5_1,
                                   _MM_MANT_SIGN_src);

    }

    static inline simd_member_type<float> frexp(simd_member_type<float> v,
                                                simd_member_type<int> &exp)
    {
        const __mmask8 isnonzerovalue = _mm_cmp_ps_mask(
            _mm_mul_ps(broadcast16(std::numeric_limits<float>::infinity()),
                       v),                                 // NaN if v == 0 / NaN
            _mm_mul_ps(_mm_setzero_ps(), v), _CMP_ORD_Q);  // NaN if v == inf / NaN
        if (Vc_IS_LIKELY(isnonzerovalue == 0x0f)) {
            exp = _mm_add_epi32(broadcast16(1), _mm_cvttps_epi32(_mm_getexp_ps(v)));
            return _mm_getmant_ps(v, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
        }
        exp = _mm_mask_add_epi32(_mm_setzero_si128(), isnonzerovalue, broadcast16(1),
                                 _mm_cvttps_epi32(_mm_getexp_ps(v)));
        return _mm_mask_getmant_ps(v, isnonzerovalue, v, _MM_MANT_NORM_p5_1,
                                   _MM_MANT_SIGN_src);
    }
    static Vc_INTRINSIC simd_member_type<float> frexp(simd_member_type<float> v,
                                                simd_tuple<int, simd_abi::Sse> &exp)
    {
        return frexp(v, exp.first);
    }
#endif  // Vc_HAVE_AVX512VL

    // isnan {{{3
    static Vc_INTRINSIC mask_member_type<float> isnan(simd_member_type<float> x)
    {
        return _mm_cmpunord_ps(x, x);
    }
    static Vc_INTRINSIC mask_member_type<double> isnan(simd_member_type<double> x)
    {
        return _mm_cmpunord_pd(x, x);
    }

    // isfinite {{{3
    static Vc_INTRINSIC mask_member_type<float> isfinite(simd_member_type<float> x)
    {
        return _mm_cmpord_ps(x, _mm_mul_ps(_mm_setzero_ps(), x));
    }
    static Vc_INTRINSIC mask_member_type<double> isfinite(simd_member_type<double> x)
    {
        return _mm_cmpord_pd(x, _mm_mul_pd(_mm_setzero_pd(), x));
    }

    // isinf {{{3
    static Vc_INTRINSIC mask_member_type<float> isinf(simd_member_type<float> x)
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
        return to_storage(
            __mmask8(_mm_fpclass_ps_mask(x, 0x08) | _mm_fpclass_ps_mask(x, 0x10)));
#else
        return to_storage(
            _mm_cmpeq_epi32(_mm_castps_si128(abs(x)), broadcast16(0x7f800000u)));
#endif
    }
    static Vc_INTRINSIC mask_member_type<double> isinf(simd_member_type<double> x)
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
        return to_storage(
            __mmask8(_mm_fpclass_pd_mask(x, 0x08) | _mm_fpclass_pd_mask(x, 0x10)));
#else
        return storage_bitcast<double>(
            equal_to(storage_bitcast<llong>(abs(x)),
                     Storage<llong, 2>::broadcast(0x7ff0'0000'0000'0000ull)));
#endif
    }

    // isnormal {{{3
    static Vc_INTRINSIC mask_member_type<float> isnormal(simd_member_type<float> x)
    {
        // subnormals -> 0
        // 0 -> 0
        // inf -> inf
        // -inf -> inf
        // nan -> inf
        // normal value -> positive value / not 0
        const auto tmp =
            and_(x, intrin_cast<__m128>(broadcast16(0x7f800000u)));
        return _mm_cmpord_ps(
            _mm_mul_ps(broadcast16(std::numeric_limits<float>::infinity()),
                       tmp),                     // NaN if tmp == 0
            _mm_mul_ps(_mm_setzero_ps(), tmp));  // NaN if tmp == inf
    }
    static Vc_INTRINSIC mask_member_type<double> isnormal(simd_member_type<double> x)
    {
        const auto tmp =
            and_(x, intrin_cast<__m128d>(broadcast16(0x7ff0'0000'0000'0000ull)));
        return _mm_cmpord_pd(
            _mm_mul_pd(broadcast16(std::numeric_limits<double>::infinity()),
                       tmp),                     // NaN if tmp == 0
            _mm_mul_pd(_mm_setzero_pd(), tmp));  // NaN if tmp == inf
    }

    // signbit {{{3
    static Vc_INTRINSIC mask_member_type<float> signbit(simd_member_type<float> x)
    {
        return _mm_castsi128_ps(
            _mm_srai_epi32(and_(to_m128i(x), broadcast16(0x80000000u)), 31));
    }
    static Vc_INTRINSIC mask_member_type<double> signbit(simd_member_type<double> x)
    {
        const auto signbit = broadcast16(0x8000000000000000ull);
#ifdef Vc_HAVE_AVX512VL
        return _mm_castsi128_pd(
            _mm_srai_epi64(and_(to_m128i(x), signbit), 63));
#elif defined Vc_HAVE_SSSE3
        return _mm_castsi128_pd(
            _mm_cmpeq_epi64(and_(to_m128i(x), signbit), signbit));
#else
        return _mm_cmpneq_pd(or_(and_(intrin_cast<__m128d>(signbit), x), broadcast16(1.)),
                             broadcast16(1.));
#endif
    }

    // isunordered {{{3
    static Vc_INTRINSIC mask_member_type<float> isunordered(simd_member_type<float> x,
                                                            simd_member_type<float> y)
    {
        return _mm_cmpunord_ps(x, y);
    }
    static Vc_INTRINSIC mask_member_type<double> isunordered(simd_member_type<double> x,
                                                             simd_member_type<double> y)
    {
        return _mm_cmpunord_pd(x, y);
    }

    // fpclassify {{{3
    static Vc_INTRINSIC simd_tuple<int, simd_abi::Sse> fpclassify(
        simd_member_type<float> x)
    {
        auto &&b = [](int y) { return intrin_cast<__m128>(broadcast16(y)); };
        return {_mm_castps_si128(blendv_ps(
            blendv_ps(blendv_ps(b(FP_NORMAL), b(FP_NAN), isnan(x)), b(FP_INFINITE),
                      isinf(x)),
            blendv_ps(b(FP_SUBNORMAL), b(FP_ZERO), _mm_cmpeq_ps(x, _mm_setzero_ps())),
            _mm_cmplt_ps(abs(x), broadcast16(std::numeric_limits<float>::min()))))};
    }
    static Vc_INTRINSIC simd_tuple<int, simd_abi::scalar, simd_abi::scalar> fpclassify(
        simd_member_type<double> x)
    {
        auto &&b = [](llong y) { return intrin_cast<__m128d>(broadcast16(y)); };
        const __m128i tmp = intrin_cast<__m128i>(blendv_pd(
            blendv_pd(blendv_pd(b(FP_NORMAL), b(FP_NAN), isnan(x)), b(FP_INFINITE),
                      isinf(x)),
            blendv_pd(b(FP_SUBNORMAL), b(FP_ZERO), _mm_cmpeq_pd(x, _mm_setzero_pd())),
            _mm_cmplt_pd(abs(x), broadcast16(std::numeric_limits<double>::min()))));
        return {_mm_cvtsi128_si32(tmp), {_mm_cvtsi128_si32(_mm_unpackhi_epi64(tmp, tmp))}};
    }

    // smart_reference access {{{2
    template <class T>
    static Vc_INTRINSIC T Vc_VDECL get(simd_member_type<T> v, int i) noexcept
    {
        return v.d[i];
    }
    template <class T, class U>
    static Vc_INTRINSIC void set(simd_member_type<T> &v, int i, U &&x) noexcept
    {
        v.set(i, std::forward<U>(x));
    }
    // }}}2
};

// simd_converter Sse -> scalar {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::Sse, To, simd_abi::scalar> {
    using Arg = sse_simd_member_type<From>;

    Vc_INTRINSIC std::array<To, Arg::width> operator()(Arg a)
    {
        return impl(std::make_index_sequence<Arg::width>(), a);
    }

    template <size_t... Indexes>
    Vc_INTRINSIC std::array<To, Arg::width> impl(std::index_sequence<Indexes...>, Arg a)
    {
        return {static_cast<To>(a[Indexes])...};
    }
};

// }}}1
// simd_converter scalar -> Sse {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::scalar, To, simd_abi::Sse> {
    using R = sse_simd_member_type<To>;

    Vc_INTRINSIC R operator()(From a)
    {
        R r{};
        r.set(0, static_cast<To>(a));
        return r;
    }
    Vc_INTRINSIC R operator()(From a, From b)
    {
        R r{};
        r.set(0, static_cast<To>(a));
        r.set(1, static_cast<To>(b));
        return r;
    }
    Vc_INTRINSIC R operator()(From a, From b, From c, From d)
    {
        R r{};
        r.set(0, static_cast<To>(a));
        r.set(1, static_cast<To>(b));
        r.set(2, static_cast<To>(c));
        r.set(3, static_cast<To>(d));
        return r;
    }
    Vc_INTRINSIC R operator()(From a, From b, From c, From d, From e, From f, From g,
                              From h)
    {
        R r{};
        r.set(0, static_cast<To>(a));
        r.set(1, static_cast<To>(b));
        r.set(2, static_cast<To>(c));
        r.set(3, static_cast<To>(d));
        r.set(4, static_cast<To>(e));
        r.set(5, static_cast<To>(f));
        r.set(6, static_cast<To>(g));
        r.set(7, static_cast<To>(h));
        return r;
    }
    Vc_INTRINSIC R operator()(From x0, From x1, From x2, From x3, From x4, From x5,
                              From x6, From x7, From x8, From x9, From x10, From x11,
                              From x12, From x13, From x14, From x15)
    {
        return R(static_cast<To>(x0), static_cast<To>(x1), static_cast<To>(x2),
                 static_cast<To>(x3), static_cast<To>(x4), static_cast<To>(x5),
                 static_cast<To>(x6), static_cast<To>(x7), static_cast<To>(x8),
                 static_cast<To>(x9), static_cast<To>(x10), static_cast<To>(x11),
                 static_cast<To>(x12), static_cast<To>(x13), static_cast<To>(x14),
                 static_cast<To>(x15));
    }
};

// }}}1
// simd_converter Sse -> Sse {{{1
template <class T> struct simd_converter<T, simd_abi::Sse, T, simd_abi::Sse> {
    using Arg = sse_simd_member_type<T>;
    Vc_INTRINSIC const Arg &operator()(const Arg &x) { return x; }
};

template <class From, class To>
struct simd_converter<From, simd_abi::Sse, To, simd_abi::Sse> {
    using Arg = sse_simd_member_type<From>;

    Vc_INTRINSIC auto operator()(Arg a)
    {
        return x86::convert_all<sse_simd_member_type<To>>(a);
    }
    Vc_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return x86::convert<sse_simd_member_type<To>>(a, b);
    }
    Vc_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<sse_simd_member_type<To>>(a, b, c, d);
    }
    Vc_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d, Arg e,
                                                     Arg f, Arg g, Arg h)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<sse_simd_member_type<To>>(a, b, c, d, e, f, g, h);
    }
};

// }}}1
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // Vc_HAVE_SSE
#endif  // VC_SIMD_SSE_H_

// vim: foldmethod=marker
