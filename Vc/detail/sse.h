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

#ifndef VC_DATAPAR_SSE_H_
#define VC_DATAPAR_SSE_H_

#include "macros.h"
#ifdef Vc_HAVE_SSE
#include "storage.h"
#include "x86/intrinsics.h"
#include "x86/convert.h"
#include "x86/arithmetics.h"
#include "maskbool.h"
#include "genericimpl.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
struct sse_mask_impl;
struct sse_datapar_impl;

// sse_traits {{{1
template <class T> struct sse_traits {
    static_assert(sizeof(T) <= 8,
                  "SSE can only implement operations on element types with sizeof <= 8");
    static_assert(std::is_arithmetic<T>::value,
                  "SSE can only vectorize arithmetic types");
    static_assert(!std::is_same<T, bool>::value, "SSE cannot vectorize bool");

    using datapar_member_type = sse_datapar_member_type<T>;
    using datapar_impl_type = sse_datapar_impl;
    static constexpr size_t datapar_member_alignment = alignof(datapar_member_type);
    using datapar_cast_type = typename datapar_member_type::VectorType;
    struct datapar_base {
        explicit operator datapar_cast_type() const
        {
            return data(*static_cast<const datapar<T, datapar_abi::sse> *>(this));
        }
    };

    using mask_member_type = sse_mask_member_type<T>;
    using mask_impl_type = sse_mask_impl;
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
            return data(*static_cast<const mask<T, datapar_abi::sse> *>(this));
        }
    };
};

#ifdef Vc_HAVE_SSE_ABI
template <> struct traits< float, datapar_abi::sse> : public sse_traits< float> {};
#ifdef Vc_HAVE_FULL_SSE_ABI
template <> struct traits<double, datapar_abi::sse> : public sse_traits<double> {};
template <> struct traits<ullong, datapar_abi::sse> : public sse_traits<ullong> {};
template <> struct traits< llong, datapar_abi::sse> : public sse_traits< llong> {};
template <> struct traits< ulong, datapar_abi::sse> : public sse_traits< ulong> {};
template <> struct traits<  long, datapar_abi::sse> : public sse_traits<  long> {};
template <> struct traits<  uint, datapar_abi::sse> : public sse_traits<  uint> {};
template <> struct traits<   int, datapar_abi::sse> : public sse_traits<   int> {};
template <> struct traits<ushort, datapar_abi::sse> : public sse_traits<ushort> {};
template <> struct traits< short, datapar_abi::sse> : public sse_traits< short> {};
template <> struct traits< uchar, datapar_abi::sse> : public sse_traits< uchar> {};
template <> struct traits< schar, datapar_abi::sse> : public sse_traits< schar> {};
template <> struct traits<  char, datapar_abi::sse> : public sse_traits<  char> {};
#endif  // Vc_HAVE_FULL_SSE_ABI
#endif  // Vc_HAVE_SSE_ABI
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#ifdef Vc_HAVE_SSE_ABI
Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// datapar impl {{{1
struct sse_datapar_impl : public generic_datapar_impl<sse_datapar_impl> {
    // member types {{{2
    using abi = datapar_abi::sse;
    template <class T> static constexpr size_t size() { return datapar_size_v<T, abi>; }
    template <class T> using datapar_member_type = sse_datapar_member_type<T>;
    template <class T> using intrinsic_type = typename datapar_member_type<T>::VectorType;
    template <class T> using mask_member_type = sse_mask_member_type<T>;
    template <class T> using datapar = Vc::datapar<T, abi>;
    template <class T> using mask = Vc::mask<T, abi>;
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;

    // make_datapar {{{2
    template <class T>
    static Vc_INTRINSIC datapar<T> make_datapar(datapar_member_type<T> x)
    {
        return {detail::private_init, x};
    }

    // broadcast {{{2
    static Vc_INTRINSIC intrinsic_type<float> broadcast(float x, size_tag<4>) noexcept
    {
        return _mm_set1_ps(x);
    }
#ifdef Vc_HAVE_SSE2
    static Vc_INTRINSIC intrinsic_type<double> broadcast(double x, size_tag<2>) noexcept
    {
        return _mm_set1_pd(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<2>) noexcept
    {
        return _mm_set1_epi64x(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<4>) noexcept
    {
        return _mm_set1_epi32(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<8>) noexcept
    {
        return _mm_set1_epi16(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<16>) noexcept
    {
        return _mm_set1_epi8(x);
    }
#endif

    // load {{{2
    // from long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC datapar_member_type<T> load(const long double *mem, F,
                                                    type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        return generate_from_n_evaluations<size<T>(), datapar_member_type<T>>(
            [&](auto i) { return static_cast<T>(mem[i]); });
    }

    // load without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC datapar_member_type<T> load(const T *mem, F f,
                                                    type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        return detail::load16(mem, f);
    }

    // convert from an SSE load{{{3
    template <class T, class U, class F>
    static inline datapar_member_type<T> load(
        const convertible_memory<U, sizeof(T), T> *mem, F f, type_tag<T>,
        tag<1> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_SSE_ABI
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
            detail::load16(mem, f));
#else
        unused(f);
        return generate_from_n_evaluations<size<T>(), intrinsic_type<T>>(
            [&](auto i) { return static_cast<T>(mem[i]); });
#endif
    }

    // convert from a half SSE load{{{3
    template <class T, class U, class F>
    static inline datapar_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 2, T> *mem, F f, type_tag<T>,
        tag<2> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_SSE_ABI
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
            intrin_cast<detail::intrinsic_type<U, size<U>()>>(load8(mem, f)));
#else
        return generate_from_n_evaluations<size<T>(), intrinsic_type<T>>(
            [&](auto i) { return static_cast<T>(mem[i]); });
        unused(f);
#endif
    }

    // convert from a quarter SSE load{{{3
    template <class T, class U, class F>
    static inline datapar_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 4, T> *mem, F f, type_tag<T>,
        tag<3> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_SSE_ABI
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
            intrin_cast<detail::intrinsic_type<U, size<U>()>>(load4(mem, f)));
#else
        return generate_from_n_evaluations<size<T>(), intrinsic_type<T>>(
            [&](auto i) { return static_cast<T>(mem[i]); });
        unused(f);
#endif
    }

    // convert from a 1/8th SSE load{{{3
#ifdef Vc_HAVE_FULL_SSE_ABI
    template <class T, class U>
    static Vc_INTRINSIC datapar_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 8, T> *mem,
        when_aligned<alignof(uint16_t)>, type_tag<T>, tag<4> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
            intrin_cast<detail::intrinsic_type<U, size<U>()>>(load2(mem, flags::vector_aligned)));
    }

    template <class T, class U>
    static Vc_INTRINSIC datapar_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 8, T> *mem,
        when_unaligned<alignof(uint16_t)>, type_tag<T>,
        tag<4> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return datapar_member_type<T>(T(mem[0]), T(mem[1]));
    }
#else   // Vc_HAVE_FULL_SSE_ABI
    template <class T, class U, class F>
    static Vc_INTRINSIC datapar_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 8, T> *mem, F, type_tag<T>,
        tag<4> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return datapar_member_type<T>(T(mem[0]), T(mem[1]));
    }
#endif  // Vc_HAVE_FULL_SSE_ABI

    // AVX and AVX-512 datapar_member_type aliases{{{3
    template <class T> using avx_member_type = avx_datapar_member_type<T>;
    template <class T> using avx512_member_type = avx512_datapar_member_type<T>;

    // convert from an AVX/2-SSE load{{{3
    template <class T, class U, class F>
    static inline datapar_member_type<T> load(
        const convertible_memory<U, sizeof(T) * 2, T> *mem, F f, type_tag<T>,
        tag<5> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX
        return convert<avx_member_type<U>, datapar_member_type<T>>(
            detail::load32(mem, f));
#elif defined Vc_HAVE_FULL_SSE_ABI
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
            load(mem, f, type_tag<U>()), load(mem + size<U>(), f, type_tag<U>()));
#else
        unused(f);
        return generate_from_n_evaluations<size<T>(), intrinsic_type<T>>(
            [&](auto i) { return static_cast<T>(mem[i]); });
#endif
    }

    // convert from an AVX512/2-AVX/4-SSE load{{{3
    template <class T, class U, class F>
    static inline datapar_member_type<T> load(
        const convertible_memory<U, sizeof(T) * 4, T> *mem, F f, type_tag<T>,
        tag<6> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX512F
        return convert<avx512_member_type<U>, datapar_member_type<T>>(load64(mem, f));
#elif defined Vc_HAVE_AVX
        return convert<avx_member_type<U>, datapar_member_type<T>>(
            detail::load32(mem, f), detail::load32(mem + 2 * size<U>(), f));
#else
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
            load(mem, f, type_tag<U>()), load(mem + size<U>(), f, type_tag<U>()),
            load(mem + 2 * size<U>(), f, type_tag<U>()),
            load(mem + 3 * size<U>(), f, type_tag<U>()));
#endif
    }

    // convert from a 2-AVX512/4-AVX/8-SSE load{{{3
    template <class T, class U, class F>
    static inline datapar_member_type<T> load(
        const convertible_memory<U, sizeof(T) * 8, T> *mem, F f, type_tag<T>,
        tag<7> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX512F
        return convert<avx512_member_type<U>, datapar_member_type<T>>(
            load64(mem, f), load64(mem + 4 * size<U>(), f));
#elif defined Vc_HAVE_AVX
        return convert<avx_member_type<U>, datapar_member_type<T>>(
            load32(mem, f), load32(mem + 2 * size<U>(), f), load32(mem + 4 * size<U>(), f),
            load32(mem + 6 * size<U>(), f));
#else
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
            load16(mem, f), load16(mem + size<U>(), f), load16(mem + 2 * size<U>(), f),
            load16(mem + 3 * size<U>(), f), load16(mem + 4 * size<U>(), f),
            load16(mem + 5 * size<U>(), f), load16(mem + 6 * size<U>(), f),
            load16(mem + 7 * size<U>(), f));
#endif
    }

    // masked load {{{2
    // fallback {{{3
    template <class T, class U, class F>
    static inline void Vc_VDECL masked_load(datapar_member_type<T> &merge,
                                            mask_member_type<T> k, const U *mem,
                                            F) Vc_NOEXCEPT_OR_IN_TEST
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
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<schar> &merge,
                                                  mask_member_type<schar> k, const schar *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm_mask_loadu_epi8(merge, _mm_movemask_epi8(k), mem);
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<uchar> &merge,
                                                  mask_member_type<uchar> k, const uchar *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm_mask_loadu_epi8(merge, _mm_movemask_epi8(k), mem);
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<short> &merge,
                                                  mask_member_type<short> k, const short *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm_mask_loadu_epi16(merge, x86::movemask_epi16(k), mem);
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<ushort> &merge,
                                                  mask_member_type<ushort> k, const ushort *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm_mask_loadu_epi16(merge, x86::movemask_epi16(k), mem);
    }

#endif  // AVX512VL && AVX512BW

    // 32-bit and 64-bit integers with AVX2 {{{3
#ifdef Vc_HAVE_AVX2
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<int> &merge,
                                                  mask_member_type<int> k, const int *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge), _mm_maskload_epi32(mem, k));
    }
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<uint> &merge,
                                                  mask_member_type<uint> k, const uint *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge),
                    _mm_maskload_epi32(
                        reinterpret_cast<const detail::may_alias<int> *>(mem), k));
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<llong> &merge,
                                                  mask_member_type<llong> k, const llong *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge), _mm_maskload_epi64(mem, k));
    }
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<ullong> &merge,
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
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<double> &merge,
                                                  mask_member_type<double> k, const double *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge), _mm_maskload_pd(mem, _mm_castpd_si128(k)));
    }
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<float> &merge,
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
    static Vc_INTRINSIC void Vc_VDECL store(datapar_member_type<T> v, long double *mem, F,
                                            type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        // alignment F doesn't matter
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // store without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(datapar_member_type<T> v, T *mem, F f,
                                            type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        store16(v, mem, f);
    }

    // convert and 16-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 8> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        store2(convert<datapar_member_type<T>, datapar_member_type<U>>(v), mem, f);
    }

    // convert and 32-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 4> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_SSE_ABI
        store4(convert<datapar_member_type<T>, datapar_member_type<U>>(v), mem, f);
#else
        unused(f);
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
#endif
    }

    // convert and 64-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 2> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_SSE_ABI
        store8(convert<datapar_member_type<T>, datapar_member_type<U>>(v), mem, f);
#else
        unused(f);
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
#endif
    }

    // convert and 128-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_SSE_ABI
        store16(convert<datapar_member_type<T>, datapar_member_type<U>>(v), mem, f);
#else
        unused(f);
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
#endif
    }

    // convert and 256-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) * 2 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX
        store32(convert<datapar_member_type<T>, avx_member_type<U>>(v), mem, f);
#elif defined Vc_HAVE_FULL_SSE_ABI
        // without the full SSE ABI there cannot be any vectorized converting loads
        // because only float vectors exist
        const auto tmp = convert_all<datapar_member_type<U>>(v);
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
    store(datapar_member_type<T> v, U *mem, F f, type_tag<T>,
          enable_if<sizeof(T) * 4 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX512F
        store64(convert_all<avx512_member_type<U>>(v), mem, f);
#elif defined Vc_HAVE_AVX
        const auto tmp = convert_all<avx_member_type<U>>(v);
        store32(tmp[0], mem, f);
        store32(tmp[1], mem + size<T>() / 2, f);
#else
        const auto tmp = convert_all<datapar_member_type<U>>(v);
        store16(tmp[0], mem, f);
        store16(tmp[1], mem + size<T>() * 1 / 4, f);
        store16(tmp[2], mem + size<T>() * 2 / 4, f);
        store16(tmp[3], mem + size<T>() * 3 / 4, f);
#endif
    }

    // convert and 1024-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F f, type_tag<T>,
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
        const auto tmp = convert_all<datapar_member_type<U>>(v);
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
    masked_store(const datapar_member_type<T> v, long double *mem, F,
                 const mask_member_type<T> k) Vc_NOEXCEPT_OR_IN_TEST
    {
        // no SSE support for long double
        execute_n_times<size<T>()>([&](auto i) {
            if (k.m(i)) {
                mem[i] = v.m(i);
            }
        });
    }
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL masked_store(const datapar_member_type<T> v, U *mem,
                                                   F, const mask_member_type<T> k)
        Vc_NOEXCEPT_OR_IN_TEST
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
    negate(datapar_member_type<T> x) noexcept
    {
#if defined Vc_GCC && defined Vc_USE_BUILTIN_VECTOR_TYPES
        return !x.builtin();
#else
        return equal_to(x, datapar_member_type<T>(x86::zero<intrinsic_type<T>>()));
#endif
    }

    // reductions {{{2
    template <class BinaryOperation>
    static Vc_INTRINSIC double Vc_VDECL reduce(size_tag<2>, datapar<double> x,
                                               BinaryOperation &binary_op)
    {
        using V = datapar<double>;
        auto intrin_ = data(x);
        intrin_ = data(binary_op(x, V(_mm_unpackhi_pd(intrin_, intrin_))));
        return _mm_cvtsd_f64(intrin_);
    }

    template <class BinaryOperation>
    static Vc_INTRINSIC float Vc_VDECL reduce(size_tag<4>, datapar<float> x,
                                              BinaryOperation &binary_op)
    {
        using V = datapar<float>;
        auto intrin_ = data(x);
        intrin_ = data(
            binary_op(x, V(_mm_shuffle_ps(intrin_, intrin_, _MM_SHUFFLE(0, 1, 2, 3)))));
        intrin_ = data(binary_op(V(intrin_), V(_mm_unpackhi_ps(intrin_, intrin_))));
        return _mm_cvtss_f32(intrin_);
    }

    template <class T, class BinaryOperation>
    static Vc_INTRINSIC T Vc_VDECL reduce(size_tag<2>, const datapar<T> x,
                                          BinaryOperation &binary_op)
    {
        return binary_op(x[0], x[1]);
    }

    template <class T, class BinaryOperation>
    static Vc_INTRINSIC T Vc_VDECL reduce(size_tag<4>, datapar<T> x,
                                          BinaryOperation &binary_op)
    {
        using V = datapar<T>;
        auto intrin_ = data(x);
        intrin_ =
            data(binary_op(x, V(_mm_shuffle_epi32(intrin_, _MM_SHUFFLE(0, 1, 2, 3)))));
        intrin_ = data(binary_op(V(intrin_), V(_mm_unpackhi_epi64(intrin_, intrin_))));
        return _mm_cvtsi128_si32(intrin_);
    }

    template <class T, class BinaryOperation>
    static Vc_INTRINSIC T Vc_VDECL reduce(size_tag<8>, datapar<T> x,
                                          BinaryOperation &binary_op)
    {
        using V = datapar<T>;
        auto intrin_ = data(x);
        intrin_ = data(binary_op(V(_mm_unpacklo_epi16(intrin_, intrin_)),
                                 V(_mm_unpackhi_epi16(intrin_, intrin_))));
        intrin_ = data(binary_op(V(_mm_unpacklo_epi32(intrin_, intrin_)),
                                 V(_mm_unpackhi_epi32(intrin_, intrin_))));
        return binary_op(V(intrin_), V(_mm_unpackhi_epi64(intrin_, intrin_)))[0];
    }

    template <class T, class BinaryOperation>
    static Vc_INTRINSIC T Vc_VDECL reduce(size_tag<16>, datapar<T> x,
                                          BinaryOperation &binary_op)
    {
        using V = datapar<T>;
        auto intrin_ = data(x);
        intrin_ = data(binary_op(V(_mm_unpacklo_epi8(intrin_, intrin_)),
                                 V(_mm_unpackhi_epi8(intrin_, intrin_))));
        intrin_ = data(binary_op(V(_mm_unpacklo_epi16(intrin_, intrin_)),
                                 V(_mm_unpackhi_epi16(intrin_, intrin_))));
        intrin_ = data(binary_op(V(_mm_unpacklo_epi32(intrin_, intrin_)),
                                 V(_mm_unpackhi_epi32(intrin_, intrin_))));
        return binary_op(V(intrin_), V(_mm_unpackhi_epi64(intrin_, intrin_)))[0];
    }

    // min, max, clamp {{{2
    static Vc_INTRINSIC datapar_member_type<double> min(datapar_member_type<double> a,
                                                        datapar_member_type<double> b)
    {
        return _mm_min_pd(a, b);
    }

    static Vc_INTRINSIC datapar_member_type<float> min(datapar_member_type<float> a,
                                                       datapar_member_type<float> b)
    {
        return _mm_min_ps(a, b);
    }

    static Vc_INTRINSIC datapar_member_type<llong> min(datapar_member_type<llong> a,
                                                       datapar_member_type<llong> b)
    {
#if defined Vc_HAVE_AVX512F && defined Vc_HAVE_AVX512VL
        return _mm_min_epi64(a, b);
#else
        return blendv_epi8(a, b, cmpgt_epi64(a, b));
#endif
    }

    static Vc_INTRINSIC datapar_member_type<ullong> min(datapar_member_type<ullong> a,
                                                        datapar_member_type<ullong> b)
    {
#if defined Vc_HAVE_AVX512F && defined Vc_HAVE_AVX512VL
        return _mm_min_epu64(a, b);
#else
        return blendv_epi8(a, b, cmpgt_epu64(a, b));
#endif
    }

    static Vc_INTRINSIC datapar_member_type<int> min(datapar_member_type<int> a,
                                                     datapar_member_type<int> b)
    {
#ifdef Vc_HAVE_SSE4_1
        return _mm_min_epi32(a, b);
#else
        return blendv_epi8(a, b, _mm_cmpgt_epi32(a, b));
#endif
    }

    static Vc_INTRINSIC datapar_member_type<uint> min(datapar_member_type<uint> a,
                                                      datapar_member_type<uint> b)
    {
#ifdef Vc_HAVE_SSE4_1
        return _mm_min_epu32(a, b);
#else
        return blendv_epi8(a, b, cmpgt_epu32(a, b));
#endif
    }

    static Vc_INTRINSIC datapar_member_type<short> min(datapar_member_type<short> a,
                                                       datapar_member_type<short> b)
    {
        return _mm_min_epi16(a, b);
    }

    static Vc_INTRINSIC datapar_member_type<ushort> min(datapar_member_type<ushort> a,
                                                        datapar_member_type<ushort> b)
    {
#ifdef Vc_HAVE_SSE4_1
        return _mm_min_epu16(a, b);
#else
        return blendv_epi8(a, b, cmpgt_epu16(a, b));
#endif
    }

    static Vc_INTRINSIC datapar_member_type<schar> min(datapar_member_type<schar> a,
                                                       datapar_member_type<schar> b)
    {
#ifdef Vc_HAVE_SSE4_1
        return _mm_min_epi8(a, b);
#else
        return blendv_epi8(a, b, _mm_cmpgt_epi8(a, b));
#endif
    }

    static Vc_INTRINSIC datapar_member_type<uchar> min(datapar_member_type<uchar> a,
                                                       datapar_member_type<uchar> b)
    {
        return _mm_min_epu8(a, b);
    }

    static Vc_INTRINSIC datapar_member_type<double> max(datapar_member_type<double> a,
                                                        datapar_member_type<double> b)
    {
        return _mm_max_pd(a, b);
    }

    static Vc_INTRINSIC datapar_member_type<float> max(datapar_member_type<float> a,
                                                       datapar_member_type<float> b)
    {
        return _mm_max_ps(a, b);
    }

    static Vc_INTRINSIC datapar_member_type<llong> max(datapar_member_type<llong> a,
                                                       datapar_member_type<llong> b)
    {
#if defined Vc_HAVE_AVX512F && defined Vc_HAVE_AVX512VL
        return _mm_max_epi64(a, b);
#else
        return blendv_epi8(b, a, cmpgt_epi64(a, b));
#endif
    }

    static Vc_INTRINSIC datapar_member_type<ullong> max(datapar_member_type<ullong> a,
                                                        datapar_member_type<ullong> b)
    {
#if defined Vc_HAVE_AVX512F && defined Vc_HAVE_AVX512VL
        return _mm_max_epu64(a, b);
#else
        return blendv_epi8(b, a, cmpgt_epu64(a, b));
#endif
    }

    static Vc_INTRINSIC datapar_member_type<int> max(datapar_member_type<int> a,
                                                     datapar_member_type<int> b){
#ifdef Vc_HAVE_SSE4_1
        return _mm_max_epi32(a, b);
#else
        return blendv_epi8(b, a, _mm_cmpgt_epi32(a, b));
#endif
    }

    static Vc_INTRINSIC datapar_member_type<uint> max(datapar_member_type<uint> a,
                                                      datapar_member_type<uint> b){
#ifdef Vc_HAVE_SSE4_1
        return _mm_max_epu32(a, b);
#else
        return blendv_epi8(b, a, cmpgt_epu32(a, b));
#endif
    }

    static Vc_INTRINSIC datapar_member_type<short> max(datapar_member_type<short> a,
                                                       datapar_member_type<short> b)
    {
        return _mm_max_epi16(a, b);
    }

    static Vc_INTRINSIC datapar_member_type<ushort> max(datapar_member_type<ushort> a,
                                                        datapar_member_type<ushort> b)
    {
#ifdef Vc_HAVE_SSE4_1
        return _mm_max_epu16(a, b);
#else
        return blendv_epi8(b, a, cmpgt_epu16(a, b));
#endif
    }

    static Vc_INTRINSIC datapar_member_type<schar> max(datapar_member_type<schar> a,
                                                       datapar_member_type<schar> b)
    {
#ifdef Vc_HAVE_SSE4_1
        return _mm_max_epi8(a, b);
#else
        return blendv_epi8(b, a, _mm_cmpgt_epi8(a, b));
#endif
    }

    static Vc_INTRINSIC datapar_member_type<uchar> max(datapar_member_type<uchar> a,
                                                       datapar_member_type<uchar> b)
    {
        return _mm_max_epu8(a, b);
    }

    static Vc_INTRINSIC datapar_member_type<long> min(datapar_member_type<long> a,
                                                      datapar_member_type<long> b)
    {
        return min(datapar_member_type<equal_int_type_t<long>>(a.v()),
                   datapar_member_type<equal_int_type_t<long>>(b.v()))
            .v();
    }
    static Vc_INTRINSIC datapar_member_type<long> max(datapar_member_type<long> a,
                                                      datapar_member_type<long> b)
    {
        return max(datapar_member_type<equal_int_type_t<long>>(a.v()),
                   datapar_member_type<equal_int_type_t<long>>(b.v()))
            .v();
    }

    static Vc_INTRINSIC datapar_member_type<ulong> min(datapar_member_type<ulong> a,
                                                       datapar_member_type<ulong> b)
    {
        return min(datapar_member_type<equal_int_type_t<ulong>>(a.v()),
                   datapar_member_type<equal_int_type_t<ulong>>(b.v()))
            .v();
    }
    static Vc_INTRINSIC datapar_member_type<ulong> max(datapar_member_type<ulong> a,
                                                       datapar_member_type<ulong> b)
    {
        return max(datapar_member_type<equal_int_type_t<ulong>>(a.v()),
                   datapar_member_type<equal_int_type_t<ulong>>(b.v()))
            .v();
    }

    template <class T>
    static Vc_INTRINSIC std::pair<datapar_member_type<T>, datapar_member_type<T>> minmax(
        datapar_member_type<T> a, datapar_member_type<T> b)
    {
        return {min(a, b), max(a, b)};
    }

    // compares {{{2
#if defined Vc_USE_BUILTIN_VECTOR_TYPES
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> equal_to(datapar_member_type<T> x, datapar_member_type<T> y)
    {
        return x.builtin() == y.builtin();
    }
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> not_equal_to(datapar_member_type<T> x, datapar_member_type<T> y)
    {
        return x.builtin() != y.builtin();
    }
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> less(datapar_member_type<T> x, datapar_member_type<T> y)
    {
        return x.builtin() < y.builtin();
    }
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> less_equal(datapar_member_type<T> x, datapar_member_type<T> y)
    {
        return x.builtin() <= y.builtin();
    }
#else
    static Vc_INTRINSIC mask_member_type<double> Vc_VDECL equal_to(datapar_member_type<double> x, datapar_member_type<double> y) { return _mm_cmpeq_pd(x, y); }
    static Vc_INTRINSIC mask_member_type< float> Vc_VDECL equal_to(datapar_member_type< float> x, datapar_member_type< float> y) { return _mm_cmpeq_ps(x, y); }
    static Vc_INTRINSIC mask_member_type< llong> Vc_VDECL equal_to(datapar_member_type< llong> x, datapar_member_type< llong> y) { return cmpeq_epi64(x, y); }
    static Vc_INTRINSIC mask_member_type<ullong> Vc_VDECL equal_to(datapar_member_type<ullong> x, datapar_member_type<ullong> y) { return cmpeq_epi64(x, y); }
    static Vc_INTRINSIC mask_member_type<  long> Vc_VDECL equal_to(datapar_member_type<  long> x, datapar_member_type<  long> y) { return sizeof(long) == 8 ? cmpeq_epi64(x, y) : _mm_cmpeq_epi32(x, y); }
    static Vc_INTRINSIC mask_member_type< ulong> Vc_VDECL equal_to(datapar_member_type< ulong> x, datapar_member_type< ulong> y) { return sizeof(long) == 8 ? cmpeq_epi64(x, y) : _mm_cmpeq_epi32(x, y); }
    static Vc_INTRINSIC mask_member_type<   int> Vc_VDECL equal_to(datapar_member_type<   int> x, datapar_member_type<   int> y) { return _mm_cmpeq_epi32(x, y); }
    static Vc_INTRINSIC mask_member_type<  uint> Vc_VDECL equal_to(datapar_member_type<  uint> x, datapar_member_type<  uint> y) { return _mm_cmpeq_epi32(x, y); }
    static Vc_INTRINSIC mask_member_type< short> Vc_VDECL equal_to(datapar_member_type< short> x, datapar_member_type< short> y) { return _mm_cmpeq_epi16(x, y); }
    static Vc_INTRINSIC mask_member_type<ushort> Vc_VDECL equal_to(datapar_member_type<ushort> x, datapar_member_type<ushort> y) { return _mm_cmpeq_epi16(x, y); }
    static Vc_INTRINSIC mask_member_type< schar> Vc_VDECL equal_to(datapar_member_type< schar> x, datapar_member_type< schar> y) { return _mm_cmpeq_epi8(x, y); }
    static Vc_INTRINSIC mask_member_type< uchar> Vc_VDECL equal_to(datapar_member_type< uchar> x, datapar_member_type< uchar> y) { return _mm_cmpeq_epi8(x, y); }

    static Vc_INTRINSIC mask_member_type<double> Vc_VDECL not_equal_to(datapar_member_type<double> x, datapar_member_type<double> y) { return _mm_cmpneq_pd(x, y); }
    static Vc_INTRINSIC mask_member_type< float> Vc_VDECL not_equal_to(datapar_member_type< float> x, datapar_member_type< float> y) { return _mm_cmpneq_ps(x, y); }
    static Vc_INTRINSIC mask_member_type< llong> Vc_VDECL not_equal_to(datapar_member_type< llong> x, datapar_member_type< llong> y) { return detail::not_(cmpeq_epi64(x, y)); }
    static Vc_INTRINSIC mask_member_type<ullong> Vc_VDECL not_equal_to(datapar_member_type<ullong> x, datapar_member_type<ullong> y) { return detail::not_(cmpeq_epi64(x, y)); }
    static Vc_INTRINSIC mask_member_type<  long> Vc_VDECL not_equal_to(datapar_member_type<  long> x, datapar_member_type<  long> y) { return detail::not_(sizeof(long) == 8 ? cmpeq_epi64(x, y) : _mm_cmpeq_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type< ulong> Vc_VDECL not_equal_to(datapar_member_type< ulong> x, datapar_member_type< ulong> y) { return detail::not_(sizeof(long) == 8 ? cmpeq_epi64(x, y) : _mm_cmpeq_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type<   int> Vc_VDECL not_equal_to(datapar_member_type<   int> x, datapar_member_type<   int> y) { return detail::not_(_mm_cmpeq_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type<  uint> Vc_VDECL not_equal_to(datapar_member_type<  uint> x, datapar_member_type<  uint> y) { return detail::not_(_mm_cmpeq_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type< short> Vc_VDECL not_equal_to(datapar_member_type< short> x, datapar_member_type< short> y) { return detail::not_(_mm_cmpeq_epi16(x, y)); }
    static Vc_INTRINSIC mask_member_type<ushort> Vc_VDECL not_equal_to(datapar_member_type<ushort> x, datapar_member_type<ushort> y) { return detail::not_(_mm_cmpeq_epi16(x, y)); }
    static Vc_INTRINSIC mask_member_type< schar> Vc_VDECL not_equal_to(datapar_member_type< schar> x, datapar_member_type< schar> y) { return detail::not_(_mm_cmpeq_epi8(x, y)); }
    static Vc_INTRINSIC mask_member_type< uchar> Vc_VDECL not_equal_to(datapar_member_type< uchar> x, datapar_member_type< uchar> y) { return detail::not_(_mm_cmpeq_epi8(x, y)); }

    static Vc_INTRINSIC mask_member_type<double> Vc_VDECL less(datapar_member_type<double> x, datapar_member_type<double> y) { return _mm_cmplt_pd(x, y); }
    static Vc_INTRINSIC mask_member_type< float> Vc_VDECL less(datapar_member_type< float> x, datapar_member_type< float> y) { return _mm_cmplt_ps(x, y); }
    static Vc_INTRINSIC mask_member_type< llong> Vc_VDECL less(datapar_member_type< llong> x, datapar_member_type< llong> y) { return cmpgt_epi64(y, x); }
    static Vc_INTRINSIC mask_member_type<ullong> Vc_VDECL less(datapar_member_type<ullong> x, datapar_member_type<ullong> y) { return cmpgt_epu64(y, x); }
    static Vc_INTRINSIC mask_member_type<  long> Vc_VDECL less(datapar_member_type<  long> x, datapar_member_type<  long> y) { return sizeof(long) == 8 ? cmpgt_epi64(y, x) :  _mm_cmpgt_epi32(y, x); }
    static Vc_INTRINSIC mask_member_type< ulong> Vc_VDECL less(datapar_member_type< ulong> x, datapar_member_type< ulong> y) { return sizeof(long) == 8 ? cmpgt_epu64(y, x) : cmpgt_epu32(y, x); }
    static Vc_INTRINSIC mask_member_type<   int> Vc_VDECL less(datapar_member_type<   int> x, datapar_member_type<   int> y) { return  _mm_cmpgt_epi32(y, x); }
    static Vc_INTRINSIC mask_member_type<  uint> Vc_VDECL less(datapar_member_type<  uint> x, datapar_member_type<  uint> y) { return cmpgt_epu32(y, x); }
    static Vc_INTRINSIC mask_member_type< short> Vc_VDECL less(datapar_member_type< short> x, datapar_member_type< short> y) { return  _mm_cmpgt_epi16(y, x); }
    static Vc_INTRINSIC mask_member_type<ushort> Vc_VDECL less(datapar_member_type<ushort> x, datapar_member_type<ushort> y) { return cmpgt_epu16(y, x); }
    static Vc_INTRINSIC mask_member_type< schar> Vc_VDECL less(datapar_member_type< schar> x, datapar_member_type< schar> y) { return  _mm_cmpgt_epi8 (y, x); }
    static Vc_INTRINSIC mask_member_type< uchar> Vc_VDECL less(datapar_member_type< uchar> x, datapar_member_type< uchar> y) { return cmpgt_epu8 (y, x); }

    static Vc_INTRINSIC mask_member_type<double> Vc_VDECL less_equal(datapar_member_type<double> x, datapar_member_type<double> y) { return _mm_cmple_pd(x, y); }
    static Vc_INTRINSIC mask_member_type< float> Vc_VDECL less_equal(datapar_member_type< float> x, datapar_member_type< float> y) { return _mm_cmple_ps(x, y); }
    static Vc_INTRINSIC mask_member_type< llong> Vc_VDECL less_equal(datapar_member_type< llong> x, datapar_member_type< llong> y) { return detail::not_(cmpgt_epi64(x, y)); }
    static Vc_INTRINSIC mask_member_type<ullong> Vc_VDECL less_equal(datapar_member_type<ullong> x, datapar_member_type<ullong> y) { return detail::not_(cmpgt_epu64(x, y)); }
    static Vc_INTRINSIC mask_member_type<  long> Vc_VDECL less_equal(datapar_member_type<  long> x, datapar_member_type<  long> y) { return detail::not_(sizeof(long) == 8 ? cmpgt_epi64(x, y) :  _mm_cmpgt_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type< ulong> Vc_VDECL less_equal(datapar_member_type< ulong> x, datapar_member_type< ulong> y) { return detail::not_(sizeof(long) == 8 ? cmpgt_epu64(x, y) : cmpgt_epu32(x, y)); }
    static Vc_INTRINSIC mask_member_type<   int> Vc_VDECL less_equal(datapar_member_type<   int> x, datapar_member_type<   int> y) { return detail::not_( _mm_cmpgt_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type<  uint> Vc_VDECL less_equal(datapar_member_type<  uint> x, datapar_member_type<  uint> y) { return detail::not_(cmpgt_epu32(x, y)); }
    static Vc_INTRINSIC mask_member_type< short> Vc_VDECL less_equal(datapar_member_type< short> x, datapar_member_type< short> y) { return detail::not_( _mm_cmpgt_epi16(x, y)); }
    static Vc_INTRINSIC mask_member_type<ushort> Vc_VDECL less_equal(datapar_member_type<ushort> x, datapar_member_type<ushort> y) { return detail::not_(cmpgt_epu16(x, y)); }
    static Vc_INTRINSIC mask_member_type< schar> Vc_VDECL less_equal(datapar_member_type< schar> x, datapar_member_type< schar> y) { return detail::not_( _mm_cmpgt_epi8 (x, y)); }
    static Vc_INTRINSIC mask_member_type< uchar> Vc_VDECL less_equal(datapar_member_type< uchar> x, datapar_member_type< uchar> y) { return detail::not_(cmpgt_epu8 (x, y)); }
#endif

    // smart_reference access {{{2
    template <class T>
    static Vc_INTRINSIC T Vc_VDECL get(datapar_member_type<T> v, int i) noexcept
    {
        return v.m(i);
    }
    template <class T, class U>
    static Vc_INTRINSIC void set(datapar_member_type<T> &v, int i, U &&x) noexcept
    {
        v.set(i, std::forward<U>(x));
    }
    // }}}2
};

// mask impl {{{1
struct sse_mask_impl : public generic_mask_impl<datapar_abi::sse, sse_mask_member_type> {
    // member types {{{2
    using abi = datapar_abi::sse;
    template <class T> static constexpr size_t size() { return datapar_size_v<T, abi>; }
    template <class T> using mask_member_type = sse_mask_member_type<T>;
    template <class T> using mask = Vc::mask<T, datapar_abi::sse>;
    template <class T> using mask_bool = MaskBool<sizeof(T)>;
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    template <class T> static Vc_INTRINSIC auto broadcast(bool x, type_tag<T>) noexcept
    {
        return detail::broadcast16(T(mask_bool<T>{x}));
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
#ifdef Vc_IS_AMD64
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
        const auto k = intrin_cast<__m128i>(v.v());
        mem[0] = -extract_epi32<1>(k);
        mem[1] = -extract_epi32<3>(k);
    }
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F,
                                            size_tag<4>) noexcept
    {
        const auto k = intrin_cast<__m128i>(v.v());
        __m128i k2 = _mm_packs_epi32(k, _mm_setzero_si128());
        *reinterpret_cast<may_alias<int32_t> *>(mem) = _mm_cvtsi128_si32(
            _mm_packs_epi16(x86::srli_epi16<15>(k2), _mm_setzero_si128()));
    }
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F,
                                            size_tag<8>) noexcept
    {
        auto k = intrin_cast<__m128i>(v.v());
        k = x86::srli_epi16<15>(k);
        const auto k2 = _mm_packs_epi16(k, _mm_setzero_si128());
#ifdef Vc_IS_AMD64
        *reinterpret_cast<may_alias<int64_t> *>(mem) = _mm_cvtsi128_si64(k2);
#else
        _mm_store_sd(reinterpret_cast<may_alias<double> *>(mem), _mm_castsi128_pd(k2));
#endif
    }
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F f,
                                            size_tag<16>) noexcept
    {
        auto k = intrin_cast<__m128i>(v.v());
        k = _mm_and_si128(k, _mm_set1_epi32(0x01010101));
        x86::store16(k, mem, f);
    }
#endif  // Vc_HAVE_SSE2

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
    static Vc_INTRINSIC mask<T> logical_and(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, detail::and_(x.d, y.d)};
    }

    template <class T>
    static Vc_INTRINSIC mask<T> logical_or(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, detail::or_(x.d, y.d)};
    }

    template <class T>
    static Vc_INTRINSIC mask<T> bit_and(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, detail::and_(x.d, y.d)};
    }

    template <class T>
    static Vc_INTRINSIC mask<T> bit_or(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, detail::or_(x.d, y.d)};
    }

    template <class T>
    static Vc_INTRINSIC mask<T> bit_xor(const mask<T> &x, const mask<T> &y)
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
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

// [mask.reductions] {{{
Vc_VERSIONED_NAMESPACE_BEGIN
Vc_ALWAYS_INLINE bool Vc_VDECL all_of(mask<float, datapar_abi::sse> k)
{
    const __m128 d(k);
#if defined Vc_USE_PTEST && defined Vc_HAVE_AVX
    return _mm_testc_ps(d, detail::allone<__m128>());
#elif defined Vc_USE_PTEST
    const auto dd = detail::intrin_cast<__m128i>(d);
    return _mm_testc_si128(dd, detail::allone<__m128i>());
#else
    return _mm_movemask_ps(d) == 0xf;
#endif
}

Vc_ALWAYS_INLINE bool Vc_VDECL any_of(mask<float, datapar_abi::sse> k)
{
    const __m128 d(k);
#if defined Vc_USE_PTEST && defined Vc_HAVE_AVX
    return 0 == _mm_testz_ps(d, d);
#elif defined Vc_USE_PTEST
    const auto dd = detail::intrin_cast<__m128i>(d);
    return 0 == _mm_testz_si128(dd, dd);
#else
    return _mm_movemask_ps(d) != 0;
#endif
}

Vc_ALWAYS_INLINE bool Vc_VDECL none_of(mask<float, datapar_abi::sse> k)
{
    const __m128 d(k);
#if defined Vc_USE_PTEST && defined Vc_HAVE_AVX
    return 0 != _mm_testz_ps(d, d);
#elif defined Vc_USE_PTEST
    const auto dd = detail::intrin_cast<__m128i>(d);
    return 0 != _mm_testz_si128(dd, dd);
#else
    return _mm_movemask_ps(d) == 0;
#endif
}

Vc_ALWAYS_INLINE bool Vc_VDECL some_of(mask<float, datapar_abi::sse> k)
{
    const __m128 d(k);
#if defined Vc_USE_PTEST && defined Vc_HAVE_AVX
    return _mm_testnzc_ps(d, detail::allone<__m128>());
#elif defined Vc_USE_PTEST
    const auto dd = detail::intrin_cast<__m128i>(d);
    return _mm_testnzc_si128(dd, detail::allone<__m128i>());
#else
    const int tmp = _mm_movemask_ps(d);
    return tmp != 0 && (tmp ^ 0xf) != 0;
#endif
}

#ifdef Vc_HAVE_SSE2
Vc_ALWAYS_INLINE bool Vc_VDECL all_of(mask<double, datapar_abi::sse> k)
{
    __m128d d(k);
#ifdef Vc_USE_PTEST
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

Vc_ALWAYS_INLINE bool Vc_VDECL any_of(mask<double, datapar_abi::sse> k)
{
    const __m128d d(k);
#if defined Vc_USE_PTEST && defined Vc_HAVE_AVX
    return 0 == _mm_testz_pd(d, d);
#elif defined Vc_USE_PTEST
    const auto dd = detail::intrin_cast<__m128i>(d);
    return 0 == _mm_testz_si128(dd, dd);
#else
    return _mm_movemask_pd(d) != 0;
#endif
}

Vc_ALWAYS_INLINE bool Vc_VDECL none_of(mask<double, datapar_abi::sse> k)
{
    const __m128d d(k);
#if defined Vc_USE_PTEST && defined Vc_HAVE_AVX
    return 0 != _mm_testz_pd(d, d);
#elif defined Vc_USE_PTEST
    const auto dd = detail::intrin_cast<__m128i>(d);
    return 0 != _mm_testz_si128(dd, dd);
#else
    return _mm_movemask_pd(d) == 0;
#endif
}

Vc_ALWAYS_INLINE bool Vc_VDECL some_of(mask<double, datapar_abi::sse> k)
{
    const __m128d d(k);
#if defined Vc_USE_PTEST && defined Vc_HAVE_AVX
    return _mm_testnzc_pd(d, detail::allone<__m128d>());
#elif defined Vc_USE_PTEST
    const auto dd = detail::intrin_cast<__m128i>(d);
    return _mm_testnzc_si128(dd, detail::allone<__m128i>());
#else
    const int tmp = _mm_movemask_pd(d);
    return tmp == 1 || tmp == 2;
#endif
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL all_of(mask<T, datapar_abi::sse> k)
{
    const __m128i d(k);
#ifdef Vc_USE_PTEST
    return _mm_testc_si128(d, detail::allone<__m128i>());  // return 1 if (0xffffffff,
                                                           // 0xffffffff, 0xffffffff,
                                                           // 0xffffffff) == (~0 & d.v())
#else
    return _mm_movemask_epi8(d) == 0xffff;
#endif
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL any_of(mask<T, datapar_abi::sse> k)
{
    const __m128i d(k);
#ifdef Vc_USE_PTEST
    return 0 == _mm_testz_si128(d, d);  // return 1 if (0, 0, 0, 0) == (d.v() & d.v())
#else
    return _mm_movemask_epi8(d) != 0x0000;
#endif
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL none_of(mask<T, datapar_abi::sse> k)
{
    const __m128i d(k);
#ifdef Vc_USE_PTEST
    return 0 != _mm_testz_si128(d, d);  // return 1 if (0, 0, 0, 0) == (d.v() & d.v())
#else
    return _mm_movemask_epi8(d) == 0x0000;
#endif
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL some_of(mask<T, datapar_abi::sse> k)
{
    const __m128i d(k);
#ifdef Vc_USE_PTEST
    return _mm_test_mix_ones_zeros(d, detail::allone<__m128i>());
#else
    const int tmp = _mm_movemask_epi8(d);
    return tmp != 0 && (tmp ^ 0xffff) != 0;
#endif
}
#endif

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL popcount(mask<T, datapar_abi::sse> k)
{
    const auto d = detail::data(k);
    return detail::mask_count<k.size()>(d);
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL find_first_set(mask<T, datapar_abi::sse> k)
{
    const auto d = detail::data(k);
    return detail::firstbit(detail::mask_to_int<k.size()>(d));
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL find_last_set(mask<T, datapar_abi::sse> k)
{
    const auto d = detail::data(k);
    return detail::lastbit(detail::mask_to_int<k.size()>(d));
}

Vc_VERSIONED_NAMESPACE_END
// }}}

#endif  // Vc_HAVE_SSE_ABI
#endif  // Vc_HAVE_SSE

#endif  // VC_DATAPAR_SSE_H_

// vim: foldmethod=marker
