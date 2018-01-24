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

#ifndef VC_SIMD_AVX_H_
#define VC_SIMD_AVX_H_

#include "sse.h"
#include "macros.h"
#ifdef Vc_HAVE_SSE
#include "storage.h"
#include "concepts.h"
#include "x86/intrinsics.h"
#include "x86/convert.h"
#include "x86/compares.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
struct avx_mask_impl;
struct avx_simd_impl;

// avx_traits {{{1
template <class T> struct avx_traits {
    static_assert(sizeof(T) <= 8,
                  "AVX can only implement operations on element types with sizeof <= 8");

    using simd_member_type = avx_simd_member_type<T>;
    using simd_impl_type = avx_simd_impl;
    static constexpr size_t simd_member_alignment = alignof(simd_member_type);
    using simd_cast_type = typename simd_member_type::VectorType;
    struct simd_base {
        explicit operator simd_cast_type() const
        {
            return data(*static_cast<const simd<T, simd_abi::avx> *>(this));
        }
    };

    using mask_member_type = avx_mask_member_type<T>;
    using mask_impl_type = avx_mask_impl;
    static constexpr size_t mask_member_alignment = alignof(mask_member_type);
    class mask_cast_type
    {
        using U = typename mask_member_type::VectorType;
        U d;

    public:
        mask_cast_type(U x) : d(x) {}
        operator mask_member_type() const { return d; }
    };
    struct mask_base {
        explicit operator typename mask_member_type::VectorType() const
        {
            return data(*static_cast<const simd_mask<T, simd_abi::avx> *>(this));
        }
    };
};

#ifdef Vc_HAVE_AVX_ABI
template <> struct traits<double, simd_abi::avx> : public avx_traits<double> {};
template <> struct traits< float, simd_abi::avx> : public avx_traits< float> {};
#ifdef Vc_HAVE_FULL_AVX_ABI
template <> struct traits<ullong, simd_abi::avx> : public avx_traits<ullong> {};
template <> struct traits< llong, simd_abi::avx> : public avx_traits< llong> {};
template <> struct traits< ulong, simd_abi::avx> : public avx_traits< ulong> {};
template <> struct traits<  long, simd_abi::avx> : public avx_traits<  long> {};
template <> struct traits<  uint, simd_abi::avx> : public avx_traits<  uint> {};
template <> struct traits<   int, simd_abi::avx> : public avx_traits<   int> {};
template <> struct traits<ushort, simd_abi::avx> : public avx_traits<ushort> {};
template <> struct traits< short, simd_abi::avx> : public avx_traits< short> {};
template <> struct traits< uchar, simd_abi::avx> : public avx_traits< uchar> {};
template <> struct traits< schar, simd_abi::avx> : public avx_traits< schar> {};
template <> struct traits<  char, simd_abi::avx> : public avx_traits<  char> {};
#endif  // Vc_HAVE_FULL_AVX_ABI
#endif  // Vc_HAVE_AVX_ABI
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#ifdef Vc_HAVE_AVX_ABI
Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// simd_mask impl {{{1
struct avx_mask_impl : public generic_mask_impl<simd_abi::avx, avx_mask_member_type> {
    // member types {{{2
    using abi = simd_abi::avx;
    template <class T> static constexpr size_t size() { return simd_size_v<T, abi>; }
    template <class T> using mask_member_type = avx_mask_member_type<T>;
    template <class T> using simd_mask = Vc::simd_mask<T, simd_abi::avx>;
    template <class T> using mask_bool = MaskBool<sizeof(T)>;
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    template <typename T> static Vc_INTRINSIC auto broadcast(bool x, type_tag<T>) noexcept
    {
        return detail::broadcast32(T(mask_bool<T>{x}));
    }

    // from_bitset overloads {{{2
    using generic_mask_impl<abi, avx_mask_member_type>::from_bitset;

    static Vc_INTRINSIC mask_member_type<float> from_bitset(std::bitset<8> bits,
                                                            type_tag<float>)
    {
        return _mm256_cmp_ps(
            _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(bits.to_ulong())),
                          _mm256_castsi256_ps(_mm256_setr_epi32(0x01, 0x02, 0x04, 0x08,
                                                                0x10, 0x20, 0x40, 0x80))),
            _mm256_setzero_ps(), _CMP_NEQ_UQ);
    }

    static Vc_INTRINSIC mask_member_type<double> from_bitset(std::bitset<4> bits,
                                                            type_tag<double>)
    {
        return _mm256_cmp_pd(
            _mm256_and_pd(
                _mm256_castsi256_pd(_mm256_set1_epi64x(bits.to_ulong())),
                _mm256_castsi256_pd(_mm256_setr_epi64x(0x01, 0x02, 0x04, 0x08))),
            _mm256_setzero_pd(), _CMP_NEQ_UQ);
    }

    // load {{{2
    template <class F>
    static Vc_INTRINSIC __m256 load(const bool *mem, F, size_tag<4>) noexcept
    {
#ifdef Vc_MSVC
        return intrin_cast<__m256>(x86::set(mem[0] ? 0xffffffffffffffffULL : 0ULL,
                                            mem[1] ? 0xffffffffffffffffULL : 0ULL,
                                            mem[2] ? 0xffffffffffffffffULL : 0ULL,
                                            mem[3] ? 0xffffffffffffffffULL : 0ULL));
#else
        __m128i k = intrin_cast<__m128i>(_mm_and_ps(
            _mm_set1_ps(*reinterpret_cast<const may_alias<float> *>(mem)),
            intrin_cast<__m128>(_mm_setr_epi32(0x1, 0x100, 0x10000, 0x1000000))));
        k = _mm_cmpgt_epi32(k, _mm_setzero_si128());
        return intrin_cast<__m256>(
            concat(_mm_unpacklo_epi32(k, k), _mm_unpackhi_epi32(k, k)));
#endif
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

    // store {{{2
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F,
                                            size_tag<4>) noexcept
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
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F,
                                            size_tag<8>) noexcept
    {
        const auto k = intrin_cast<__m256i>(v.v());
        const auto k2 = x86::srli_epi16<15>(_mm_packs_epi16(lo128(k), hi128(k)));
        const auto k3 = _mm_packs_epi16(k2, _mm_setzero_si128());
#ifdef Vc_IS_AMD64
        *reinterpret_cast<may_alias<int64_t> *>(mem) = _mm_cvtsi128_si64(k3);
#else
        *reinterpret_cast<may_alias<int32_t> *>(mem) = _mm_cvtsi128_si32(k3);
        *reinterpret_cast<may_alias<int32_t> *>(mem + 4) = _mm_extract_epi32(k3, 1);
#endif
    }
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F f,
                                            size_tag<16>) noexcept
    {
#ifdef Vc_HAVE_AVX2
        const auto x = x86::srli_epi16<15>(v);
        const auto bools = _mm_packs_epi16(lo128(x), hi128(x));
#else
        const auto bools =
            detail::and_(one16(uchar()), _mm_packs_epi16(lo128(v.v()), hi128(v.v())));
#endif
        store16(bools, mem, f);
    }
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F f,
                                            size_tag<32>) noexcept
    {
        const auto bools = detail::and_(one32(uchar()), v.v());
        store32(bools, mem, f);
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
        return k.m(i);
    }
    template <class T> static void set(mask_member_type<T> &k, int i, bool x) noexcept
    {
        k.set(i, mask_bool<T>(x));
    }
    // }}}2
};

// }}}1

constexpr struct {
    template <class T> operator T() const { return detail::allone<T>(); }
} allone_poly = {};
}  // namespace detail

// [simd_mask.reductions] {{{
template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL all_of(simd_mask<T, simd_abi::avx> k)
{
    const auto d = detail::data(k);
    return 0 != detail::testc(d, detail::allone_poly);
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL any_of(simd_mask<T, simd_abi::avx> k)
{
    const auto d = detail::data(k);
    return 0 == detail::testz(d, d);
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL none_of(simd_mask<T, simd_abi::avx> k)
{
    const auto d = detail::data(k);
    return 0 != detail::testz(d, d);
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL some_of(simd_mask<T, simd_abi::avx> k)
{
    const auto d = detail::data(k);
    return 0 != detail::testnzc(d, detail::allone_poly);
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL popcount(simd_mask<T, simd_abi::avx> k)
{
    const auto d = detail::data(k);
    switch (k.size()) {
    case 4:
        return detail::popcnt4(detail::mask_to_int<k.size()>(d));
    case 8:
        return detail::popcnt8(detail::mask_to_int<k.size()>(d));
    case 16:
        return detail::popcnt32(detail::mask_to_int<32>(d)) / 2;
    case 32:
        return detail::popcnt32(detail::mask_to_int<k.size()>(d));
    default:
        Vc_UNREACHABLE();
        return 0;
    }
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL find_first_set(simd_mask<T, simd_abi::avx> k)
{
    const auto d = detail::data(k);
    return detail::firstbit(detail::mask_to_int<k.size()>(d));
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL find_last_set(simd_mask<T, simd_abi::avx> k)
{
    const auto d = detail::data(k);
    if (k.size() == 16) {
        return detail::lastbit(detail::mask_to_int<32>(d)) / 2;
    }
    return detail::lastbit(detail::mask_to_int<k.size()>(d));
}
// }}}

namespace detail
{
// simd impl {{{1
struct avx_simd_impl : public generic_simd_impl<avx_simd_impl> {
    // member types {{{2
    using abi = simd_abi::avx;
    template <class T> static constexpr size_t size() { return simd_size_v<T, abi>; }
    template <class T> using simd_member_type = avx_simd_member_type<T>;
    template <class T> using intrinsic_type = typename simd_member_type<T>::VectorType;
    template <class T> using mask_member_type = avx_mask_member_type<T>;
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

    // load {{{2
    // from long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC simd_member_type<T> load(const long double *mem, F,
                                                    type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        return generate_from_n_evaluations<size<T>(), simd_member_type<T>>(
            [&](auto i) { return static_cast<T>(mem[i]); });
    }

    // load without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(const T *mem, F f, type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        return detail::load32(mem, f);
    }

    // convert from an AVX load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const convertible_memory<U, sizeof(T), T> *mem, F f, type_tag<T>,
        tag<1> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<simd_member_type<U>, simd_member_type<T>>(load32(mem, f));
    }

    // convert from an SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const convertible_memory<U, sizeof(T) / 2, T> *mem, F f, type_tag<T>,
        tag<2> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<sse_simd_member_type<U>, simd_member_type<T>>(
            load16(mem, f));
    }

    // convert from a half SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const convertible_memory<U, sizeof(T) / 4, T> *mem, F f, type_tag<T>,
        tag<3> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<sse_simd_member_type<U>, simd_member_type<T>>(load8(mem, f));
    }

    // convert from a 1/4th SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const convertible_memory<U, sizeof(T) / 8, T> *mem, F f, type_tag<T>,
        tag<4> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<sse_simd_member_type<U>, simd_member_type<T>>(load4(mem, f));
    }

    // convert from an AVX512/2-AVX load{{{3
    template <class T> using avx512_member_type = avx512_simd_member_type<T>;

    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const convertible_memory<U, sizeof(T) * 2, T> *mem, F f, type_tag<T>,
        tag<5> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX512F
        return convert<avx512_member_type<U>, simd_member_type<T>>(
            load64(mem, f));
#else
        return convert<simd_member_type<U>, simd_member_type<T>>(
            load32(mem, f), load32(mem + size<U>(), f));
#endif
    }

    // convert from an 2-AVX512/4-AVX load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const convertible_memory<U, sizeof(T) * 4, T> *mem, F f, type_tag<T>,
        tag<6> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX512F
        using LoadT = avx512_member_type<U>;
        constexpr auto N = LoadT::size();
        return convert<LoadT, simd_member_type<T>>(load64(mem, f), load64(mem + N, f));
#else
        return convert<simd_member_type<U>, simd_member_type<T>>(
            load32(mem, f), load32(mem + size<U>(), f), load32(mem + 2 * size<U>(), f),
            load32(mem + 3 * size<U>(), f));
#endif
    }

    // convert from a 4-AVX512/8-AVX load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const convertible_memory<U, sizeof(T) * 8, T> *mem, F f, type_tag<T>,
        tag<7> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX512F
        using LoadT = avx512_member_type<U>;
        constexpr auto N = LoadT::size();
        return convert<LoadT, simd_member_type<T>>(load64(mem, f), load64(mem + N, f),
                                                      load64(mem + 2 * N, f),
                                                      load64(mem + 3 * N, f));
#else
        using LoadT = simd_member_type<U>;
        constexpr auto N = LoadT::size();
        return convert<simd_member_type<U>, simd_member_type<T>>(
            load32(mem, f), load32(mem + N, f), load32(mem + 2 * N, f),
            load32(mem + 3 * N, f), load32(mem + 4 * N, f), load32(mem + 5 * N, f),
            load32(mem + 6 * N, f), load32(mem + 7 * N, f));
#endif
    }

    // masked load {{{2
    // fallback {{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<T> &merge, mask_member_type<T> k,
                                                  const U *mem, F) Vc_NOEXCEPT_OR_IN_TEST
    {
        execute_n_times<size<T>()>([&](auto i) {
            if (k.m(i)) {
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
        merge = _mm256_mask_loadu_epi8(merge, _mm256_movemask_epi8(k), mem);
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<uchar> &merge,
                                                  mask_member_type<uchar> k, const uchar *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm256_mask_loadu_epi8(merge, _mm256_movemask_epi8(k), mem);
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<short> &merge,
                                                  mask_member_type<short> k, const short *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm256_mask_loadu_epi16(merge, x86::movemask_epi16(k), mem);
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<ushort> &merge,
                                                  mask_member_type<ushort> k, const ushort *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm256_mask_loadu_epi16(merge, x86::movemask_epi16(k), mem);
    }

#endif  // AVX512VL && AVX512BW

    // 32-bit and 64-bit integers with AVX2 {{{3
#ifdef Vc_HAVE_AVX2
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<int> &merge,
                                                  mask_member_type<int> k, const int *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge), _mm256_maskload_epi32(mem, k));
    }
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<uint> &merge,
                                                  mask_member_type<uint> k, const uint *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge),
                    _mm256_maskload_epi32(
                        reinterpret_cast<const detail::may_alias<int> *>(mem), k));
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<llong> &merge,
                                                  mask_member_type<llong> k, const llong *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge), _mm256_maskload_epi64(mem, k));
    }
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<ullong> &merge,
                                                  mask_member_type<ullong> k, const ullong *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge),
                    _mm256_maskload_epi64(
                        reinterpret_cast<const may_alias<long long> *>(mem), k));
    }
#endif  // Vc_HAVE_AVX2

    // 32-bit and 64-bit floats {{{3
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<double> &merge,
                                                  mask_member_type<double> k, const double *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge),
                    _mm256_maskload_pd(mem, _mm256_castpd_si256(k)));
    }
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(simd_member_type<float> &merge,
                                                  mask_member_type<float> k, const float *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge),
                    _mm256_maskload_ps(mem, _mm256_castps_si256(k)));
    }

    // store {{{2
    // store to long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(simd_member_type<T> v, long double *mem, F,
                                            type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        // alignment F doesn't matter
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // store without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(simd_member_type<T> v, T *mem, F f,
                                            type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        store32(v, mem, f);
    }

    // convert and 32-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 8> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 64-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 4> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 128-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 2> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 256-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 512-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) * 2 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 1024-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) * 4 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 2048-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) * 8 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // masked store {{{2
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL
    masked_store(simd_member_type<T> v, long double *mem, F,
                 mask_member_type<T> k) Vc_NOEXCEPT_OR_IN_TEST
    {
        // no SSE support for long double
        execute_n_times<size<T>()>([&](auto i) {
            if (k.m(i)) {
                mem[i] = v.m(i);
            }
        });
    }
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL masked_store(
        simd_member_type<T> v, U *mem, F, mask_member_type<T> k) Vc_NOEXCEPT_OR_IN_TEST
    {
        //TODO: detail::masked_store(mem, v.v(), k.d.v(), f);
        execute_n_times<size<T>()>([&](auto i) {
            if (k.m(i)) {
                mem[i] = static_cast<T>(v.m(i));
            }
        });
    }

    // negation {{{2
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> Vc_VDECL
    negate(simd_member_type<T> x) noexcept
    {
#if defined Vc_GCC && defined Vc_USE_BUILTIN_VECTOR_TYPES
        return !x.builtin();
#else
        return equal_to(x, simd_member_type<T>(x86::zero<intrinsic_type<T>>()));
#endif
    }

    // reductions {{{2
    template <class T, class BinaryOperation, size_t N>
    static Vc_INTRINSIC T Vc_VDECL reduce(size_tag<N>, simd<T> x,
                                          BinaryOperation &binary_op)
    {
        using V = Vc::simd<T, simd_abi::sse>;
        return sse_simd_impl::reduce(size_tag<N / 2>(),
                                        binary_op(V(lo128(data(x))), V(hi128(data(x)))),
                                        binary_op);
    }

    // min, max {{{2
#define Vc_MINMAX_(T_, suffix_)                                                          \
    static Vc_INTRINSIC simd_member_type<T_> min(simd_member_type<T_> a,           \
                                                    simd_member_type<T_> b)           \
    {                                                                                    \
        return _mm256_min_##suffix_(a, b);                                               \
    }                                                                                    \
    static Vc_INTRINSIC simd_member_type<T_> max(simd_member_type<T_> a,           \
                                                    simd_member_type<T_> b)           \
    {                                                                                    \
        return _mm256_max_##suffix_(a, b);                                               \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
    Vc_MINMAX_(double, pd);
    Vc_MINMAX_( float, ps);
#ifdef Vc_HAVE_AVX2
    Vc_MINMAX_(   int, epi32);
    Vc_MINMAX_(  uint, epu32);
    Vc_MINMAX_( short, epi16);
    Vc_MINMAX_(ushort, epu16);
    Vc_MINMAX_( schar, epi8);
    Vc_MINMAX_( uchar, epu8);
#endif  // Vc_HAVE_AVX2
#ifdef Vc_HAVE_AVX512VL
    Vc_MINMAX_( llong, epi64);
    Vc_MINMAX_(ullong, epu64);
#elif defined Vc_HAVE_AVX2
    static Vc_INTRINSIC simd_member_type<llong> min(simd_member_type<llong> a,
                                                       simd_member_type<llong> b)
    {
        return _mm256_blendv_epi8(a, b, _mm256_cmpgt_epi64(a, b));
    }
    static Vc_INTRINSIC simd_member_type<llong> max(simd_member_type<llong> a,
                                                       simd_member_type<llong> b)
    {
        return _mm256_blendv_epi8(b, a, _mm256_cmpgt_epi64(a, b));
    } static Vc_INTRINSIC simd_member_type<ullong> min(simd_member_type<ullong> a,
                                                          simd_member_type<ullong> b)
    {
        return _mm256_blendv_epi8(a, b, cmpgt(a, b));
    }
    static Vc_INTRINSIC simd_member_type<ullong> max(simd_member_type<ullong> a,
                                                        simd_member_type<ullong> b)
    {
        return _mm256_blendv_epi8(b, a, cmpgt(a, b));
    }
#endif
#undef Vc_MINMAX_

#if defined Vc_HAVE_AVX2
    static Vc_INTRINSIC simd_member_type<long> min(simd_member_type<long> a,
                                                      simd_member_type<long> b)
    {
        return min(simd_member_type<equal_int_type_t<long>>(a.v()),
                   simd_member_type<equal_int_type_t<long>>(b.v()))
            .v();
    }
    static Vc_INTRINSIC simd_member_type<long> max(simd_member_type<long> a,
                                                      simd_member_type<long> b)
    {
        return max(simd_member_type<equal_int_type_t<long>>(a.v()),
                   simd_member_type<equal_int_type_t<long>>(b.v()))
            .v();
    }

    static Vc_INTRINSIC simd_member_type<ulong> min(simd_member_type<ulong> a,
                                                       simd_member_type<ulong> b)
    {
        return min(simd_member_type<equal_int_type_t<ulong>>(a.v()),
                   simd_member_type<equal_int_type_t<ulong>>(b.v()))
            .v();
    }
    static Vc_INTRINSIC simd_member_type<ulong> max(simd_member_type<ulong> a,
                                                       simd_member_type<ulong> b)
    {
        return max(simd_member_type<equal_int_type_t<ulong>>(a.v()),
                   simd_member_type<equal_int_type_t<ulong>>(b.v()))
            .v();
    }
#endif  // Vc_HAVE_AVX2

    template <class T>
    static Vc_INTRINSIC std::pair<simd_member_type<T>, simd_member_type<T>> minmax(
        simd_member_type<T> a, simd_member_type<T> b)
    {
        return {min(a, b), max(a, b)};
    }

    // compares {{{2
#if defined Vc_USE_BUILTIN_VECTOR_TYPES
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> equal_to(simd_member_type<T> x, simd_member_type<T> y)
    {
        return x.builtin() == y.builtin();
    }
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> not_equal_to(simd_member_type<T> x, simd_member_type<T> y)
    {
        return x.builtin() != y.builtin();
    }
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> less(simd_member_type<T> x, simd_member_type<T> y)
    {
        return x.builtin() < y.builtin();
    }
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> less_equal(simd_member_type<T> x, simd_member_type<T> y)
    {
        return x.builtin() <= y.builtin();
    }
#else
    static Vc_INTRINSIC mask_member_type<double> Vc_VDECL equal_to    (simd_member_type<double> x, simd_member_type<double> y) { return _mm256_cmp_pd(x, y, _CMP_EQ_OQ); }
    static Vc_INTRINSIC mask_member_type<double> Vc_VDECL not_equal_to(simd_member_type<double> x, simd_member_type<double> y) { return _mm256_cmp_pd(x, y, _CMP_NEQ_UQ); }
    static Vc_INTRINSIC mask_member_type<double> Vc_VDECL less        (simd_member_type<double> x, simd_member_type<double> y) { return _mm256_cmp_pd(x, y, _CMP_LT_OS); }
    static Vc_INTRINSIC mask_member_type<double> Vc_VDECL less_equal  (simd_member_type<double> x, simd_member_type<double> y) { return _mm256_cmp_pd(x, y, _CMP_LE_OS); }
    static Vc_INTRINSIC mask_member_type< float> Vc_VDECL equal_to    (simd_member_type< float> x, simd_member_type< float> y) { return _mm256_cmp_ps(x, y, _CMP_EQ_OQ); }
    static Vc_INTRINSIC mask_member_type< float> Vc_VDECL not_equal_to(simd_member_type< float> x, simd_member_type< float> y) { return _mm256_cmp_ps(x, y, _CMP_NEQ_UQ); }
    static Vc_INTRINSIC mask_member_type< float> Vc_VDECL less        (simd_member_type< float> x, simd_member_type< float> y) { return _mm256_cmp_ps(x, y, _CMP_LT_OS); }
    static Vc_INTRINSIC mask_member_type< float> Vc_VDECL less_equal  (simd_member_type< float> x, simd_member_type< float> y) { return _mm256_cmp_ps(x, y, _CMP_LE_OS); }

#ifdef Vc_HAVE_FULL_AVX_ABI
    static Vc_INTRINSIC mask_member_type< llong> Vc_VDECL equal_to(simd_member_type< llong> x, simd_member_type< llong> y) { return _mm256_cmpeq_epi64(x, y); }
    static Vc_INTRINSIC mask_member_type<ullong> Vc_VDECL equal_to(simd_member_type<ullong> x, simd_member_type<ullong> y) { return _mm256_cmpeq_epi64(x, y); }
    static Vc_INTRINSIC mask_member_type<  long> Vc_VDECL equal_to(simd_member_type<  long> x, simd_member_type<  long> y) { return sizeof(long) == 8 ? _mm256_cmpeq_epi64(x, y) : _mm256_cmpeq_epi32(x, y); }
    static Vc_INTRINSIC mask_member_type< ulong> Vc_VDECL equal_to(simd_member_type< ulong> x, simd_member_type< ulong> y) { return sizeof(long) == 8 ? _mm256_cmpeq_epi64(x, y) : _mm256_cmpeq_epi32(x, y); }
    static Vc_INTRINSIC mask_member_type<   int> Vc_VDECL equal_to(simd_member_type<   int> x, simd_member_type<   int> y) { return _mm256_cmpeq_epi32(x, y); }
    static Vc_INTRINSIC mask_member_type<  uint> Vc_VDECL equal_to(simd_member_type<  uint> x, simd_member_type<  uint> y) { return _mm256_cmpeq_epi32(x, y); }
    static Vc_INTRINSIC mask_member_type< short> Vc_VDECL equal_to(simd_member_type< short> x, simd_member_type< short> y) { return _mm256_cmpeq_epi16(x, y); }
    static Vc_INTRINSIC mask_member_type<ushort> Vc_VDECL equal_to(simd_member_type<ushort> x, simd_member_type<ushort> y) { return _mm256_cmpeq_epi16(x, y); }
    static Vc_INTRINSIC mask_member_type< schar> Vc_VDECL equal_to(simd_member_type< schar> x, simd_member_type< schar> y) { return _mm256_cmpeq_epi8(x, y); }
    static Vc_INTRINSIC mask_member_type< uchar> Vc_VDECL equal_to(simd_member_type< uchar> x, simd_member_type< uchar> y) { return _mm256_cmpeq_epi8(x, y); }

    static Vc_INTRINSIC mask_member_type< llong> Vc_VDECL not_equal_to(simd_member_type< llong> x, simd_member_type< llong> y) { return detail::not_(_mm256_cmpeq_epi64(x, y)); }
    static Vc_INTRINSIC mask_member_type<ullong> Vc_VDECL not_equal_to(simd_member_type<ullong> x, simd_member_type<ullong> y) { return detail::not_(_mm256_cmpeq_epi64(x, y)); }
    static Vc_INTRINSIC mask_member_type<  long> Vc_VDECL not_equal_to(simd_member_type<  long> x, simd_member_type<  long> y) { return detail::not_(sizeof(long) == 8 ? _mm256_cmpeq_epi64(x, y) : _mm256_cmpeq_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type< ulong> Vc_VDECL not_equal_to(simd_member_type< ulong> x, simd_member_type< ulong> y) { return detail::not_(sizeof(long) == 8 ? _mm256_cmpeq_epi64(x, y) : _mm256_cmpeq_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type<   int> Vc_VDECL not_equal_to(simd_member_type<   int> x, simd_member_type<   int> y) { return detail::not_(_mm256_cmpeq_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type<  uint> Vc_VDECL not_equal_to(simd_member_type<  uint> x, simd_member_type<  uint> y) { return detail::not_(_mm256_cmpeq_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type< short> Vc_VDECL not_equal_to(simd_member_type< short> x, simd_member_type< short> y) { return detail::not_(_mm256_cmpeq_epi16(x, y)); }
    static Vc_INTRINSIC mask_member_type<ushort> Vc_VDECL not_equal_to(simd_member_type<ushort> x, simd_member_type<ushort> y) { return detail::not_(_mm256_cmpeq_epi16(x, y)); }
    static Vc_INTRINSIC mask_member_type< schar> Vc_VDECL not_equal_to(simd_member_type< schar> x, simd_member_type< schar> y) { return detail::not_(_mm256_cmpeq_epi8(x, y)); }
    static Vc_INTRINSIC mask_member_type< uchar> Vc_VDECL not_equal_to(simd_member_type< uchar> x, simd_member_type< uchar> y) { return detail::not_(_mm256_cmpeq_epi8(x, y)); }

    static Vc_INTRINSIC mask_member_type< llong> Vc_VDECL less(simd_member_type< llong> x, simd_member_type< llong> y) { return _mm256_cmpgt_epi64(y, x); }
    static Vc_INTRINSIC mask_member_type<ullong> Vc_VDECL less(simd_member_type<ullong> x, simd_member_type<ullong> y) { return cmpgt(y, x); }
    static Vc_INTRINSIC mask_member_type<  long> Vc_VDECL less(simd_member_type<  long> x, simd_member_type<  long> y) { return sizeof(long) == 8 ? _mm256_cmpgt_epi64(y, x) : _mm256_cmpgt_epi32(y, x); }
    static Vc_INTRINSIC mask_member_type< ulong> Vc_VDECL less(simd_member_type< ulong> x, simd_member_type< ulong> y) { return cmpgt(y, x); }
    static Vc_INTRINSIC mask_member_type<   int> Vc_VDECL less(simd_member_type<   int> x, simd_member_type<   int> y) { return _mm256_cmpgt_epi32(y, x); }
    static Vc_INTRINSIC mask_member_type<  uint> Vc_VDECL less(simd_member_type<  uint> x, simd_member_type<  uint> y) { return cmpgt(y, x); }
    static Vc_INTRINSIC mask_member_type< short> Vc_VDECL less(simd_member_type< short> x, simd_member_type< short> y) { return _mm256_cmpgt_epi16(y, x); }
    static Vc_INTRINSIC mask_member_type<ushort> Vc_VDECL less(simd_member_type<ushort> x, simd_member_type<ushort> y) { return cmpgt(y, x); }
    static Vc_INTRINSIC mask_member_type< schar> Vc_VDECL less(simd_member_type< schar> x, simd_member_type< schar> y) { return _mm256_cmpgt_epi8 (y, x); }
    static Vc_INTRINSIC mask_member_type< uchar> Vc_VDECL less(simd_member_type< uchar> x, simd_member_type< uchar> y) { return cmpgt(y, x); }

    static Vc_INTRINSIC mask_member_type< llong> Vc_VDECL less_equal(simd_member_type< llong> x, simd_member_type< llong> y) { return detail::not_(_mm256_cmpgt_epi64(x, y)); }
    static Vc_INTRINSIC mask_member_type<ullong> Vc_VDECL less_equal(simd_member_type<ullong> x, simd_member_type<ullong> y) { return detail::not_(cmpgt(x, y)); }
    static Vc_INTRINSIC mask_member_type<  long> Vc_VDECL less_equal(simd_member_type<  long> x, simd_member_type<  long> y) { return detail::not_(sizeof(long) == 8 ? _mm256_cmpgt_epi64(x, y) : _mm256_cmpgt_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type< ulong> Vc_VDECL less_equal(simd_member_type< ulong> x, simd_member_type< ulong> y) { return detail::not_(cmpgt(x, y)); }
    static Vc_INTRINSIC mask_member_type<   int> Vc_VDECL less_equal(simd_member_type<   int> x, simd_member_type<   int> y) { return detail::not_(_mm256_cmpgt_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type<  uint> Vc_VDECL less_equal(simd_member_type<  uint> x, simd_member_type<  uint> y) { return detail::not_(cmpgt(x, y)); }
    static Vc_INTRINSIC mask_member_type< short> Vc_VDECL less_equal(simd_member_type< short> x, simd_member_type< short> y) { return detail::not_(_mm256_cmpgt_epi16(x, y)); }
    static Vc_INTRINSIC mask_member_type<ushort> Vc_VDECL less_equal(simd_member_type<ushort> x, simd_member_type<ushort> y) { return detail::not_(cmpgt(x, y)); }
    static Vc_INTRINSIC mask_member_type< schar> Vc_VDECL less_equal(simd_member_type< schar> x, simd_member_type< schar> y) { return detail::not_(_mm256_cmpgt_epi8 (x, y)); }
    static Vc_INTRINSIC mask_member_type< uchar> Vc_VDECL less_equal(simd_member_type< uchar> x, simd_member_type< uchar> y) { return detail::not_(cmpgt (x, y)); }
#endif
#endif

    // math {{{2
    // sqrt {{{3
    static Vc_INTRINSIC simd_member_type<float> sqrt(simd_member_type<float> x)
    {
        return _mm256_sqrt_ps(x);
    }
    static Vc_INTRINSIC simd_member_type<double> sqrt(simd_member_type<double> x)
    {
        return _mm256_sqrt_pd(x);
    }

    // logb {{{3
#ifdef Vc_HAVE_AVX512F
    static Vc_INTRINSIC Vc_CONST simd_member_type<float> logb(simd_member_type<float> v)
    {
#ifdef Vc_HAVE_AVX512VL
        return _mm256_fixupimm_ps(_mm256_getexp_ps(abs(v)), v, broadcast32(0x00550433),
                                  0x00);
#else
        const __m512 vv = intrin_cast<__m512>(v);
        return lo256(_mm512_fixupimm_ps(_mm512_getexp_ps(_mm512_abs_ps(vv)), vv,
                                        broadcast64(0x00550433), 0x00));
#endif
    }
    static Vc_INTRINSIC Vc_CONST simd_member_type<double> logb(simd_member_type<double> v)
    {
#ifdef Vc_HAVE_AVX512VL
        return _mm256_fixupimm_pd(_mm256_getexp_pd(abs(v)), v, broadcast32(0x00550433),
                                  0x00);
#else
        const auto vv = intrin_cast<__m512d>(v);
        return lo256(_mm512_fixupimm_pd(_mm512_getexp_pd(_mm512_abs_pd(auto_cvt(v))), vv,
                                        broadcast64(0x00550433), 0x00));
#endif
    }
#endif  // Vc_HAVE_AVX512F

    // trunc {{{3
    static Vc_INTRINSIC simd_member_type<float> trunc(simd_member_type<float> x)
    {
        return _mm256_round_ps(x, 0x3);
    }
    static Vc_INTRINSIC simd_member_type<double> trunc(simd_member_type<double> x)
    {
        return _mm256_round_pd(x, 0x3);
    }

    // floor {{{3
    static Vc_INTRINSIC simd_member_type<float> floor(simd_member_type<float> x)
    {
        return _mm256_round_ps(x, 0x1);
    }
    static Vc_INTRINSIC simd_member_type<double> floor(simd_member_type<double> x)
    {
        return _mm256_round_pd(x, 0x1);
    }

    // ceil {{{3
    static Vc_INTRINSIC simd_member_type<float> ceil(simd_member_type<float> x)
    {
        return _mm256_round_ps(x, 0x2);
    }
    static Vc_INTRINSIC simd_member_type<double> ceil(simd_member_type<double> x)
    {
        return _mm256_round_pd(x, 0x2);
    }

    // frexp {{{3
    /**
     * splits \p v into exponent and mantissa, the sign is kept with the mantissa
     *
     * The return value will be in the range [0.5, 1.0[
     * The \p e value will be an integer defining the power-of-two exponent
     */
#ifdef Vc_HAVE_AVX512VL
    static inline simd_member_type<double> frexp(simd_member_type<double> v,
                                                 sse_simd_member_type<int> &exp)
    {
        const __mmask8 isnonzerovalue = _mm256_cmp_pd_mask(
            _mm256_mul_pd(broadcast32(std::numeric_limits<double>::infinity()),
                          v),                                    // NaN if v == 0
            _mm256_mul_pd(_mm256_setzero_pd(), v), _CMP_ORD_Q);  // NaN if v == inf
        if (Vc_IS_LIKELY(isnonzerovalue == 0xf)) {
            exp = _mm_add_epi32(broadcast16(1), _mm256_cvttpd_epi32(_mm256_getexp_pd(v)));
            return _mm256_getmant_pd(v, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
        }
        exp = _mm_mask_add_epi32(_mm_setzero_si128(), isnonzerovalue, broadcast16(1),
                                 _mm256_cvttpd_epi32(_mm256_getexp_pd(v)));
        return _mm256_mask_getmant_pd(v, isnonzerovalue, v, _MM_MANT_NORM_p5_1,
                                      _MM_MANT_SIGN_src);
    }
    static Vc_INTRINSIC simd_member_type<double> frexp(
        simd_member_type<double> v, simd_tuple<int, simd_abi::sse> &exp)
    {
        return frexp(v, exp.first);
    }

    static inline simd_member_type<float> frexp(simd_member_type<float> v,
                                                avx_simd_member_type<int> &exp)
    {
        const __mmask8 isnonzerovalue = _mm256_cmp_ps_mask(
            _mm256_mul_ps(broadcast32(std::numeric_limits<float>::infinity()),
                          v),                                    // NaN if v == 0 / NaN
            _mm256_mul_ps(_mm256_setzero_ps(), v), _CMP_ORD_Q);  // NaN if v == inf / NaN
        if (Vc_IS_LIKELY(isnonzerovalue == 0xff)) {
            exp = _mm256_add_epi32(broadcast32(1),
                                   _mm256_cvttps_epi32(_mm256_getexp_ps(v)));
            return _mm256_getmant_ps(v, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
        }
        exp =
            _mm256_mask_add_epi32(_mm256_setzero_si256(), isnonzerovalue, broadcast32(1),
                                  _mm256_cvttps_epi32(_mm256_getexp_ps(v)));
        return _mm256_mask_getmant_ps(v, isnonzerovalue, v, _MM_MANT_NORM_p5_1,
                                      _MM_MANT_SIGN_src);
    }
    static Vc_INTRINSIC simd_member_type<float> frexp(simd_member_type<float> v,
                                                      simd_tuple<int, simd_abi::avx> &exp)
    {
        return frexp(v, exp.first);
    }
#endif  // Vc_HAVE_AVX512VL


    // isfinite {{{3
    static Vc_INTRINSIC mask_member_type<float> isfinite(simd_member_type<float> x)
    {
        return _mm256_cmp_ps(x, _mm256_mul_ps(_mm256_setzero_ps(), x), _CMP_ORD_Q);
    }
    static Vc_INTRINSIC mask_member_type<double> isfinite(simd_member_type<double> x)
    {
        return _mm256_cmp_pd(x, _mm256_mul_pd(_mm256_setzero_pd(), x), _CMP_ORD_Q);
    }

    // isinf {{{3
    static Vc_INTRINSIC mask_member_type<float> isinf(simd_member_type<float> x)
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
        return __mmask8(_mm256_fpclass_ps_mask(x, 0x08) |
                        _mm256_fpclass_ps_mask(x, 0x10));
#else
        return less(y_f32(broadcast32(std::numeric_limits<float>::max())), abs(x));
#endif
    }
    static Vc_INTRINSIC mask_member_type<double> isinf(simd_member_type<double> x)
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
        return __mmask8(_mm256_fpclass_pd_mask(x, 0x08) |
                        _mm256_fpclass_pd_mask(x, 0x10));
#else
        return less(y_f64(broadcast32(std::numeric_limits<double>::max())), abs(x));
#endif
    }

    // isnan {{{3
    static Vc_INTRINSIC mask_member_type<float> isnan(simd_member_type<float> x)
    {
        return _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    }
    static Vc_INTRINSIC mask_member_type<double> isnan(simd_member_type<double> x)
    {
        return _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
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
            and_(x, intrin_cast<__m256>(broadcast32(0x7f800000u)));
        return _mm256_cmp_ps(
            _mm256_mul_ps(broadcast32(std::numeric_limits<float>::infinity()),
                          tmp),                                    // NaN if tmp == 0
            _mm256_mul_ps(_mm256_setzero_ps(), tmp), _CMP_ORD_Q);  // NaN if tmp == inf
    }
    static Vc_INTRINSIC mask_member_type<double> isnormal(simd_member_type<double> x)
    {
        const auto tmp =
            and_(x, intrin_cast<__m256d>(broadcast32(0x7ff0'0000'0000'0000ull)));
        return _mm256_cmp_pd(
            _mm256_mul_pd(broadcast32(std::numeric_limits<double>::infinity()),
                          tmp),                                    // NaN if tmp == 0
            _mm256_mul_pd(_mm256_setzero_pd(), tmp), _CMP_ORD_Q);  // NaN if tmp == inf
    }

    // signbit {{{3
    static Vc_INTRINSIC mask_member_type<float> signbit(simd_member_type<float> x)
    {
        const auto signbit = broadcast32(0x80000000u);
#ifdef Vc_HAVE_AVX2
        return _mm256_castsi256_ps(
            _mm256_srai_epi32(and_(intrin_cast<__m256i>(x), signbit), 31));
#else   // Vc_HAVE_AVX2
        return not_equal_to(
            y_f32(or_(and_(x, _mm256_castsi256_ps(signbit)), broadcast32(1.f))),
            y_f32(broadcast32(1.f)));
#endif  // Vc_HAVE_AVX2
    }
    static Vc_INTRINSIC mask_member_type<double> signbit(simd_member_type<double> x)
    {
        const auto signbit = broadcast32(0x8000000000000000ull);
#ifdef Vc_HAVE_AVX512VL
        return _mm256_castsi256_pd(
            _mm256_srai_epi64(and_(intrin_cast<__m256i>(x), signbit), 63));
#elif defined Vc_HAVE_AVX2
        return _mm256_castsi256_pd(
            _mm256_cmpeq_epi64(and_(intrin_cast<__m256i>(x), signbit), signbit));
#else
        return not_equal_to(
            y_f64(or_(and_(x, _mm256_castsi256_pd(signbit)), broadcast32(1.))),
            y_f64(broadcast32(1.)));
#endif
    }

    // isunordered {{{3
    static Vc_INTRINSIC mask_member_type<float> isunordered(simd_member_type<float> x,
                                                            simd_member_type<float> y)
    {
        return _mm256_cmp_ps(x, y, _CMP_UNORD_Q);
    }
    static Vc_INTRINSIC mask_member_type<double> isunordered(simd_member_type<double> x,
                                                             simd_member_type<double> y)
    {
        return _mm256_cmp_pd(x, y, _CMP_UNORD_Q);
    }

    // fpclassify {{{3
#ifdef Vc_HAVE_AVX2
    static Vc_INTRINSIC simd_tuple<int, simd_abi::avx> fpclassify(
        simd_member_type<float> x)
    {
        auto &&b = [](int y) { return intrin_cast<__m256>(broadcast32(y)); };
        return {_mm256_castps_si256(_mm256_blendv_ps(
            _mm256_blendv_ps(_mm256_blendv_ps(b(FP_NORMAL), b(FP_NAN), isnan(x)),
                             b(FP_INFINITE), isinf(x)),
            _mm256_blendv_ps(b(FP_SUBNORMAL), b(FP_ZERO),
                             _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ)),
            _mm256_cmp_ps(abs(x), broadcast32(std::numeric_limits<float>::min()),
                          _CMP_LT_OS)))};
    }
#else  // Vc_HAVE_AVX2
    static Vc_INTRINSIC simd_tuple<int, simd_abi::sse, simd_abi::sse> fpclassify(
        simd_member_type<float> x)
    {
        auto &&b = [](int y) { return intrin_cast<__m256>(broadcast32(y)); };
        const auto tmp = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_blendv_ps(_mm256_blendv_ps(b(FP_NORMAL), b(FP_NAN), isnan(x)),
                             b(FP_INFINITE), isinf(x)),
            _mm256_blendv_ps(b(FP_SUBNORMAL), b(FP_ZERO),
                             _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ)),
            _mm256_cmp_ps(abs(x), broadcast32(std::numeric_limits<float>::min()),
                          _CMP_LT_OS)));
        return {lo128(tmp), hi128(tmp)};
    }
#endif  // Vc_HAVE_AVX2

    static Vc_INTRINSIC simd_tuple<int, simd_abi::sse> fpclassify(
        simd_member_type<double> x)
    {
        auto &&b = [](llong y) { return intrin_cast<__m256d>(broadcast32(y)); };
        const __m256i tmp = intrin_cast<__m256i>(_mm256_blendv_pd(
            _mm256_blendv_pd(_mm256_blendv_pd(b(FP_NORMAL), b(FP_NAN), isnan(x)),
                             b(FP_INFINITE), isinf(x)),
            _mm256_blendv_pd(b(FP_SUBNORMAL), b(FP_ZERO),
                             _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_EQ_OQ)),
            _mm256_cmp_pd(abs(x), broadcast32(std::numeric_limits<double>::min()),
                          _CMP_LT_OS)));
        return {_mm_packs_epi32(lo128(tmp), hi128(tmp))};
    }

    // smart_reference access {{{2
    template <class T>
    static Vc_INTRINSIC T Vc_VDECL get(simd_member_type<T> v, int i) noexcept
    {
        return v.m(i);
    }
    template <class T, class U>
    static Vc_INTRINSIC void set(simd_member_type<T> &v, int i, U &&x) noexcept
    {
        v.set(i, std::forward<U>(x));
    }
    // }}}2
    };

    // simd_converter avx -> scalar {{{1
    template <class From, class To>
    struct simd_converter<From, simd_abi::avx, To, simd_abi::scalar> {
        using Arg = avx_simd_member_type<From>;

        Vc_INTRINSIC std::array<To, Arg::size()> operator()(Arg a)
        {
            return impl(std::make_index_sequence<Arg::size()>(), a);
        }

        template <size_t... Indexes>
        Vc_INTRINSIC std::array<To, Arg::size()> impl(std::index_sequence<Indexes...>,
                                                      Arg a)
        {
            return {static_cast<To>(a[Indexes])...};
        }
    };

    // }}}1
    // simd_converter scalar -> avx {{{1
    template <class From, class To>
    struct simd_converter<From, simd_abi::scalar, To, simd_abi::avx> {
        using R = avx_simd_member_type<To>;

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
            R r{};
            r.set(0, static_cast<To>(x0));
            r.set(1, static_cast<To>(x1));
            r.set(2, static_cast<To>(x2));
            r.set(3, static_cast<To>(x3));
            r.set(4, static_cast<To>(x4));
            r.set(5, static_cast<To>(x5));
            r.set(6, static_cast<To>(x6));
            r.set(7, static_cast<To>(x7));
            r.set(8, static_cast<To>(x8));
            r.set(9, static_cast<To>(x9));
            r.set(10, static_cast<To>(x10));
            r.set(11, static_cast<To>(x11));
            r.set(12, static_cast<To>(x12));
            r.set(13, static_cast<To>(x13));
            r.set(14, static_cast<To>(x14));
            r.set(15, static_cast<To>(x15));
            return r;
        }
        Vc_INTRINSIC R operator()(From x0, From x1, From x2, From x3, From x4, From x5,
                                  From x6, From x7, From x8, From x9, From x10, From x11,
                                  From x12, From x13, From x14, From x15, From x16,
                                  From x17, From x18, From x19, From x20, From x21,
                                  From x22, From x23, From x24, From x25, From x26,
                                  From x27, From x28, From x29, From x30, From x31)
        {
            return R(static_cast<To>(x0), static_cast<To>(x1), static_cast<To>(x2),
                     static_cast<To>(x3), static_cast<To>(x4), static_cast<To>(x5),
                     static_cast<To>(x6), static_cast<To>(x7), static_cast<To>(x8),
                     static_cast<To>(x9), static_cast<To>(x10), static_cast<To>(x11),
                     static_cast<To>(x12), static_cast<To>(x13), static_cast<To>(x14),
                     static_cast<To>(x15), static_cast<To>(x16), static_cast<To>(x17),
                     static_cast<To>(x18), static_cast<To>(x19), static_cast<To>(x20),
                     static_cast<To>(x21), static_cast<To>(x22), static_cast<To>(x23),
                     static_cast<To>(x24), static_cast<To>(x25), static_cast<To>(x26),
                     static_cast<To>(x27), static_cast<To>(x28), static_cast<To>(x29),
                     static_cast<To>(x30), static_cast<To>(x31));
        }
    };

    // }}}1
    // simd_converter sse -> avx {{{1
    template <class From, class To>
    struct simd_converter<From, simd_abi::sse, To, simd_abi::avx> {
        using Arg = sse_simd_member_type<From>;

        Vc_INTRINSIC auto operator()(Arg a)
        {
            return x86::convert_all<avx_simd_member_type<To>>(a);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b)
        {
            static_assert(sizeof(From) >= 1 * sizeof(To), "");
            return x86::convert<Arg, avx_simd_member_type<To>>(a, b);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
        {
            static_assert(sizeof(From) >= 2 * sizeof(To), "");
            return x86::convert<Arg, avx_simd_member_type<To>>(a, b, c, d);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                         Arg x4, Arg x5, Arg x6, Arg x7)
        {
            static_assert(sizeof(From) >= 4 * sizeof(To), "");
            return x86::convert<Arg, avx_simd_member_type<To>>(x0, x1, x2, x3, x4, x5, x6,
                                                               x7);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                         Arg x4, Arg x5, Arg x6, Arg x7,
                                                         Arg x8, Arg x9, Arg x10, Arg x11,
                                                         Arg x12, Arg x13, Arg x14,
                                                         Arg x15)
        {
            static_assert(sizeof(From) >= 8 * sizeof(To), "");
            return x86::convert<Arg, avx_simd_member_type<To>>(
                x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
        }
    };

    // }}}1
    // simd_converter avx -> sse {{{1
    template <class From, class To>
    struct simd_converter<From, simd_abi::avx, To, simd_abi::sse> {
        using Arg = avx_simd_member_type<From>;

        Vc_INTRINSIC auto operator()(Arg a)
        {
            return x86::convert_all<sse_simd_member_type<To>>(a);
        }
        Vc_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b)
        {
            static_assert(sizeof(From) >= 4 * sizeof(To), "");
            return x86::convert<Arg, sse_simd_member_type<To>>(a, b);
        }
        Vc_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
        {
            static_assert(sizeof(From) >= 8 * sizeof(To), "");
            return x86::convert<Arg, sse_simd_member_type<To>>(a, b, c, d);
        }
    };

    // }}}1
    // simd_converter avx -> avx {{{1
    template <class T> struct simd_converter<T, simd_abi::avx, T, simd_abi::avx> {
        using Arg = avx_simd_member_type<T>;
        Vc_INTRINSIC const Arg &operator()(const Arg &x) { return x; }
    };

    template <class From, class To>
    struct simd_converter<From, simd_abi::avx, To, simd_abi::avx> {
        using Arg = avx_simd_member_type<From>;

        Vc_INTRINSIC auto operator()(Arg a)
        {
            return x86::convert_all<avx_simd_member_type<To>>(a);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b)
        {
            static_assert(sizeof(From) >= 2 * sizeof(To), "");
            return x86::convert<Arg, avx_simd_member_type<To>>(a, b);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
        {
            static_assert(sizeof(From) >= 4 * sizeof(To), "");
            return x86::convert<Arg, avx_simd_member_type<To>>(a, b, c, d);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d,
                                                         Arg e, Arg f, Arg g, Arg h)
        {
            static_assert(sizeof(From) >= 8 * sizeof(To), "");
            return x86::convert<Arg, avx_simd_member_type<To>>(a, b, c, d, e, f, g, h);
        }
    };

    // split_to_array {{{1
    template <class T> struct split_to_array<simd<T, simd_abi::sse>, 2> {
        using V = simd<T, simd_abi::sse>;
        std::array<V, 2> operator()(simd<T, simd_abi::avx> x, std::index_sequence<0, 1>)
        {
            const auto xx = detail::data(x);
            return {V(detail::private_init, lo128(xx)),
                    V(detail::private_init, hi128(xx))};
        }
    };

    // split_to_tuple {{{1
    template <class T>
    struct split_to_tuple<std::tuple<simd<T, simd_abi::sse>, simd<T, simd_abi::sse>>,
                          simd_abi::avx> {
        using V = simd<T, simd_abi::sse>;
        std::tuple<V, V> operator()(simd<T, simd_abi::avx> x)
        {
            const auto xx = detail::data(x);
            return {V(detail::private_init, lo128(xx)),
                    V(detail::private_init, hi128(xx))};
        }
    };

    // }}}1
    }  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // Vc_HAVE_AVX_ABI

#endif  // Vc_HAVE_SSE
#endif  // VC_SIMD_AVX_H_

// vim: foldmethod=marker
