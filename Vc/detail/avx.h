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

#ifndef VC_DATAPAR_AVX_H_
#define VC_DATAPAR_AVX_H_

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
struct avx_datapar_impl;

// avx_traits {{{1
template <class T> struct avx_traits {
    static_assert(sizeof(T) <= 8,
                  "AVX can only implement operations on element types with sizeof <= 8");

    using datapar_member_type = avx_datapar_member_type<T>;
    using datapar_impl_type = avx_datapar_impl;
    static constexpr size_t datapar_member_alignment = alignof(datapar_member_type);
    using datapar_cast_type = typename datapar_member_type::VectorType;
    struct datapar_base {
        explicit operator datapar_cast_type() const
        {
            return data(*static_cast<const datapar<T, datapar_abi::avx> *>(this));
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
            return data(*static_cast<const mask<T, datapar_abi::avx> *>(this));
        }
    };
};

#ifdef Vc_HAVE_AVX_ABI
template <> struct traits<double, datapar_abi::avx> : public avx_traits<double> {};
template <> struct traits< float, datapar_abi::avx> : public avx_traits< float> {};
#ifdef Vc_HAVE_FULL_AVX_ABI
template <> struct traits<ullong, datapar_abi::avx> : public avx_traits<ullong> {};
template <> struct traits< llong, datapar_abi::avx> : public avx_traits< llong> {};
template <> struct traits< ulong, datapar_abi::avx> : public avx_traits< ulong> {};
template <> struct traits<  long, datapar_abi::avx> : public avx_traits<  long> {};
template <> struct traits<  uint, datapar_abi::avx> : public avx_traits<  uint> {};
template <> struct traits<   int, datapar_abi::avx> : public avx_traits<   int> {};
template <> struct traits<ushort, datapar_abi::avx> : public avx_traits<ushort> {};
template <> struct traits< short, datapar_abi::avx> : public avx_traits< short> {};
template <> struct traits< uchar, datapar_abi::avx> : public avx_traits< uchar> {};
template <> struct traits< schar, datapar_abi::avx> : public avx_traits< schar> {};
template <> struct traits<  char, datapar_abi::avx> : public avx_traits<  char> {};
#endif  // Vc_HAVE_FULL_AVX_ABI
#endif  // Vc_HAVE_AVX_ABI
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#ifdef Vc_HAVE_AVX_ABI
Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// datapar impl {{{1
struct avx_datapar_impl : public generic_datapar_impl<avx_datapar_impl> {
    // member types {{{2
    using abi = datapar_abi::avx;
    template <class T> static constexpr size_t size() { return datapar_size_v<T, abi>; }
    template <class T> using datapar_member_type = avx_datapar_member_type<T>;
    template <class T> using intrinsic_type = typename datapar_member_type<T>::VectorType;
    template <class T> using mask_member_type = avx_mask_member_type<T>;
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
    static Vc_INTRINSIC datapar_member_type<T> load(const long double *mem, F,
                                                    type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        return generate_from_n_evaluations<size<T>(), datapar_member_type<T>>(
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
        return convert<datapar_member_type<U>, datapar_member_type<T>>(load32(mem, f));
    }

    // convert from an SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const convertible_memory<U, sizeof(T) / 2, T> *mem, F f, type_tag<T>,
        tag<2> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<sse_datapar_member_type<U>, datapar_member_type<T>>(
            load16(mem, f));
    }

    // convert from a half SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const convertible_memory<U, sizeof(T) / 4, T> *mem, F f, type_tag<T>,
        tag<3> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<sse_datapar_member_type<U>, datapar_member_type<T>>(load8(mem, f));
    }

    // convert from a 1/4th SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const convertible_memory<U, sizeof(T) / 8, T> *mem, F f, type_tag<T>,
        tag<4> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<sse_datapar_member_type<U>, datapar_member_type<T>>(load4(mem, f));
    }

    // convert from an AVX512/2-AVX load{{{3
    template <class T> using avx512_member_type = avx512_datapar_member_type<T>;

    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const convertible_memory<U, sizeof(T) * 2, T> *mem, F f, type_tag<T>,
        tag<5> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_AVX512F
        return convert<avx512_member_type<U>, datapar_member_type<T>>(
            load64(mem, f));
#else
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
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
        return convert<LoadT, datapar_member_type<T>>(load64(mem, f), load64(mem + N, f));
#else
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
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
        return convert<LoadT, datapar_member_type<T>>(load64(mem, f), load64(mem + N, f),
                                                      load64(mem + 2 * N, f),
                                                      load64(mem + 3 * N, f));
#else
        using LoadT = datapar_member_type<U>;
        constexpr auto N = LoadT::size();
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
            load32(mem, f), load32(mem + N, f), load32(mem + 2 * N, f),
            load32(mem + 3 * N, f), load32(mem + 4 * N, f), load32(mem + 5 * N, f),
            load32(mem + 6 * N, f), load32(mem + 7 * N, f));
#endif
    }

    // masked load {{{2
    // fallback {{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<T> &merge, mask_member_type<T> k,
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
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<schar> &merge,
                                                  mask_member_type<schar> k, const schar *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm256_mask_loadu_epi8(merge, _mm256_movemask_epi8(k), mem);
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<uchar> &merge,
                                                  mask_member_type<uchar> k, const uchar *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm256_mask_loadu_epi8(merge, _mm256_movemask_epi8(k), mem);
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<short> &merge,
                                                  mask_member_type<short> k, const short *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm256_mask_loadu_epi16(merge, x86::movemask_epi16(k), mem);
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<ushort> &merge,
                                                  mask_member_type<ushort> k, const ushort *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = _mm256_mask_loadu_epi16(merge, x86::movemask_epi16(k), mem);
    }

#endif  // AVX512VL && AVX512BW

    // 32-bit and 64-bit integers with AVX2 {{{3
#ifdef Vc_HAVE_AVX2
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<int> &merge,
                                                  mask_member_type<int> k, const int *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge), _mm256_maskload_epi32(mem, k));
    }
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<uint> &merge,
                                                  mask_member_type<uint> k, const uint *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge),
                    _mm256_maskload_epi32(
                        reinterpret_cast<const detail::may_alias<int> *>(mem), k));
    }

    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<llong> &merge,
                                                  mask_member_type<llong> k, const llong *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge), _mm256_maskload_epi64(mem, k));
    }
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<ullong> &merge,
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
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<double> &merge,
                                                  mask_member_type<double> k, const double *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge),
                    _mm256_maskload_pd(mem, _mm256_castpd_si256(k)));
    }
    template <class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<float> &merge,
                                                  mask_member_type<float> k, const float *mem,
                                                  F) Vc_NOEXCEPT_OR_IN_TEST
    {
        merge = or_(andnot_(k, merge),
                    _mm256_maskload_ps(mem, _mm256_castps_si256(k)));
    }

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
        store32(v, mem, f);
    }

    // convert and 32-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 8> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 64-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 4> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 128-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 2> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 256-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 512-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) * 2 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 1024-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) * 4 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 2048-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) * 8 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // masked store {{{2
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL
    masked_store(datapar_member_type<T> v, long double *mem, F,
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
        datapar_member_type<T> v, U *mem, F, mask_member_type<T> k) Vc_NOEXCEPT_OR_IN_TEST
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
    template <class T, class BinaryOperation, size_t N>
    static Vc_INTRINSIC T Vc_VDECL reduce(size_tag<N>, datapar<T> x,
                                          BinaryOperation &binary_op)
    {
        using V = Vc::datapar<T, datapar_abi::sse>;
        return sse_datapar_impl::reduce(size_tag<N / 2>(),
                                        binary_op(V(lo128(data(x))), V(hi128(data(x)))),
                                        binary_op);
    }

    // min, max {{{2
#define Vc_MINMAX_(T_, suffix_)                                                          \
    static Vc_INTRINSIC datapar_member_type<T_> min(datapar_member_type<T_> a,           \
                                                    datapar_member_type<T_> b)           \
    {                                                                                    \
        return _mm256_min_##suffix_(a, b);                                               \
    }                                                                                    \
    static Vc_INTRINSIC datapar_member_type<T_> max(datapar_member_type<T_> a,           \
                                                    datapar_member_type<T_> b)           \
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
    static Vc_INTRINSIC datapar_member_type<llong> min(datapar_member_type<llong> a,
                                                       datapar_member_type<llong> b)
    {
        return _mm256_blendv_epi8(a, b, _mm256_cmpgt_epi64(a, b));
    }
    static Vc_INTRINSIC datapar_member_type<llong> max(datapar_member_type<llong> a,
                                                       datapar_member_type<llong> b)
    {
        return _mm256_blendv_epi8(b, a, _mm256_cmpgt_epi64(a, b));
    } static Vc_INTRINSIC datapar_member_type<ullong> min(datapar_member_type<ullong> a,
                                                          datapar_member_type<ullong> b)
    {
        return _mm256_blendv_epi8(a, b, cmpgt(a, b));
    }
    static Vc_INTRINSIC datapar_member_type<ullong> max(datapar_member_type<ullong> a,
                                                        datapar_member_type<ullong> b)
    {
        return _mm256_blendv_epi8(b, a, cmpgt(a, b));
    }
#endif
#undef Vc_MINMAX_

#if defined Vc_HAVE_AVX2
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
#endif  // Vc_HAVE_AVX2

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
    static Vc_INTRINSIC mask_member_type<double> Vc_VDECL equal_to    (datapar_member_type<double> x, datapar_member_type<double> y) { return _mm256_cmp_pd(x, y, _CMP_EQ_OQ); }
    static Vc_INTRINSIC mask_member_type<double> Vc_VDECL not_equal_to(datapar_member_type<double> x, datapar_member_type<double> y) { return _mm256_cmp_pd(x, y, _CMP_NEQ_UQ); }
    static Vc_INTRINSIC mask_member_type<double> Vc_VDECL less        (datapar_member_type<double> x, datapar_member_type<double> y) { return _mm256_cmp_pd(x, y, _CMP_LT_OS); }
    static Vc_INTRINSIC mask_member_type<double> Vc_VDECL less_equal  (datapar_member_type<double> x, datapar_member_type<double> y) { return _mm256_cmp_pd(x, y, _CMP_LE_OS); }
    static Vc_INTRINSIC mask_member_type< float> Vc_VDECL equal_to    (datapar_member_type< float> x, datapar_member_type< float> y) { return _mm256_cmp_ps(x, y, _CMP_EQ_OQ); }
    static Vc_INTRINSIC mask_member_type< float> Vc_VDECL not_equal_to(datapar_member_type< float> x, datapar_member_type< float> y) { return _mm256_cmp_ps(x, y, _CMP_NEQ_UQ); }
    static Vc_INTRINSIC mask_member_type< float> Vc_VDECL less        (datapar_member_type< float> x, datapar_member_type< float> y) { return _mm256_cmp_ps(x, y, _CMP_LT_OS); }
    static Vc_INTRINSIC mask_member_type< float> Vc_VDECL less_equal  (datapar_member_type< float> x, datapar_member_type< float> y) { return _mm256_cmp_ps(x, y, _CMP_LE_OS); }

#ifdef Vc_HAVE_FULL_AVX_ABI
    static Vc_INTRINSIC mask_member_type< llong> Vc_VDECL equal_to(datapar_member_type< llong> x, datapar_member_type< llong> y) { return _mm256_cmpeq_epi64(x, y); }
    static Vc_INTRINSIC mask_member_type<ullong> Vc_VDECL equal_to(datapar_member_type<ullong> x, datapar_member_type<ullong> y) { return _mm256_cmpeq_epi64(x, y); }
    static Vc_INTRINSIC mask_member_type<  long> Vc_VDECL equal_to(datapar_member_type<  long> x, datapar_member_type<  long> y) { return sizeof(long) == 8 ? _mm256_cmpeq_epi64(x, y) : _mm256_cmpeq_epi32(x, y); }
    static Vc_INTRINSIC mask_member_type< ulong> Vc_VDECL equal_to(datapar_member_type< ulong> x, datapar_member_type< ulong> y) { return sizeof(long) == 8 ? _mm256_cmpeq_epi64(x, y) : _mm256_cmpeq_epi32(x, y); }
    static Vc_INTRINSIC mask_member_type<   int> Vc_VDECL equal_to(datapar_member_type<   int> x, datapar_member_type<   int> y) { return _mm256_cmpeq_epi32(x, y); }
    static Vc_INTRINSIC mask_member_type<  uint> Vc_VDECL equal_to(datapar_member_type<  uint> x, datapar_member_type<  uint> y) { return _mm256_cmpeq_epi32(x, y); }
    static Vc_INTRINSIC mask_member_type< short> Vc_VDECL equal_to(datapar_member_type< short> x, datapar_member_type< short> y) { return _mm256_cmpeq_epi16(x, y); }
    static Vc_INTRINSIC mask_member_type<ushort> Vc_VDECL equal_to(datapar_member_type<ushort> x, datapar_member_type<ushort> y) { return _mm256_cmpeq_epi16(x, y); }
    static Vc_INTRINSIC mask_member_type< schar> Vc_VDECL equal_to(datapar_member_type< schar> x, datapar_member_type< schar> y) { return _mm256_cmpeq_epi8(x, y); }
    static Vc_INTRINSIC mask_member_type< uchar> Vc_VDECL equal_to(datapar_member_type< uchar> x, datapar_member_type< uchar> y) { return _mm256_cmpeq_epi8(x, y); }

    static Vc_INTRINSIC mask_member_type< llong> Vc_VDECL not_equal_to(datapar_member_type< llong> x, datapar_member_type< llong> y) { return detail::not_(_mm256_cmpeq_epi64(x, y)); }
    static Vc_INTRINSIC mask_member_type<ullong> Vc_VDECL not_equal_to(datapar_member_type<ullong> x, datapar_member_type<ullong> y) { return detail::not_(_mm256_cmpeq_epi64(x, y)); }
    static Vc_INTRINSIC mask_member_type<  long> Vc_VDECL not_equal_to(datapar_member_type<  long> x, datapar_member_type<  long> y) { return detail::not_(sizeof(long) == 8 ? _mm256_cmpeq_epi64(x, y) : _mm256_cmpeq_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type< ulong> Vc_VDECL not_equal_to(datapar_member_type< ulong> x, datapar_member_type< ulong> y) { return detail::not_(sizeof(long) == 8 ? _mm256_cmpeq_epi64(x, y) : _mm256_cmpeq_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type<   int> Vc_VDECL not_equal_to(datapar_member_type<   int> x, datapar_member_type<   int> y) { return detail::not_(_mm256_cmpeq_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type<  uint> Vc_VDECL not_equal_to(datapar_member_type<  uint> x, datapar_member_type<  uint> y) { return detail::not_(_mm256_cmpeq_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type< short> Vc_VDECL not_equal_to(datapar_member_type< short> x, datapar_member_type< short> y) { return detail::not_(_mm256_cmpeq_epi16(x, y)); }
    static Vc_INTRINSIC mask_member_type<ushort> Vc_VDECL not_equal_to(datapar_member_type<ushort> x, datapar_member_type<ushort> y) { return detail::not_(_mm256_cmpeq_epi16(x, y)); }
    static Vc_INTRINSIC mask_member_type< schar> Vc_VDECL not_equal_to(datapar_member_type< schar> x, datapar_member_type< schar> y) { return detail::not_(_mm256_cmpeq_epi8(x, y)); }
    static Vc_INTRINSIC mask_member_type< uchar> Vc_VDECL not_equal_to(datapar_member_type< uchar> x, datapar_member_type< uchar> y) { return detail::not_(_mm256_cmpeq_epi8(x, y)); }

    static Vc_INTRINSIC mask_member_type< llong> Vc_VDECL less(datapar_member_type< llong> x, datapar_member_type< llong> y) { return _mm256_cmpgt_epi64(y, x); }
    static Vc_INTRINSIC mask_member_type<ullong> Vc_VDECL less(datapar_member_type<ullong> x, datapar_member_type<ullong> y) { return cmpgt(y, x); }
    static Vc_INTRINSIC mask_member_type<  long> Vc_VDECL less(datapar_member_type<  long> x, datapar_member_type<  long> y) { return sizeof(long) == 8 ? _mm256_cmpgt_epi64(y, x) : _mm256_cmpgt_epi32(y, x); }
    static Vc_INTRINSIC mask_member_type< ulong> Vc_VDECL less(datapar_member_type< ulong> x, datapar_member_type< ulong> y) { return cmpgt(y, x); }
    static Vc_INTRINSIC mask_member_type<   int> Vc_VDECL less(datapar_member_type<   int> x, datapar_member_type<   int> y) { return _mm256_cmpgt_epi32(y, x); }
    static Vc_INTRINSIC mask_member_type<  uint> Vc_VDECL less(datapar_member_type<  uint> x, datapar_member_type<  uint> y) { return cmpgt(y, x); }
    static Vc_INTRINSIC mask_member_type< short> Vc_VDECL less(datapar_member_type< short> x, datapar_member_type< short> y) { return _mm256_cmpgt_epi16(y, x); }
    static Vc_INTRINSIC mask_member_type<ushort> Vc_VDECL less(datapar_member_type<ushort> x, datapar_member_type<ushort> y) { return cmpgt(y, x); }
    static Vc_INTRINSIC mask_member_type< schar> Vc_VDECL less(datapar_member_type< schar> x, datapar_member_type< schar> y) { return _mm256_cmpgt_epi8 (y, x); }
    static Vc_INTRINSIC mask_member_type< uchar> Vc_VDECL less(datapar_member_type< uchar> x, datapar_member_type< uchar> y) { return cmpgt(y, x); }

    static Vc_INTRINSIC mask_member_type< llong> Vc_VDECL less_equal(datapar_member_type< llong> x, datapar_member_type< llong> y) { return detail::not_(_mm256_cmpgt_epi64(x, y)); }
    static Vc_INTRINSIC mask_member_type<ullong> Vc_VDECL less_equal(datapar_member_type<ullong> x, datapar_member_type<ullong> y) { return detail::not_(cmpgt(x, y)); }
    static Vc_INTRINSIC mask_member_type<  long> Vc_VDECL less_equal(datapar_member_type<  long> x, datapar_member_type<  long> y) { return detail::not_(sizeof(long) == 8 ? _mm256_cmpgt_epi64(x, y) : _mm256_cmpgt_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type< ulong> Vc_VDECL less_equal(datapar_member_type< ulong> x, datapar_member_type< ulong> y) { return detail::not_(cmpgt(x, y)); }
    static Vc_INTRINSIC mask_member_type<   int> Vc_VDECL less_equal(datapar_member_type<   int> x, datapar_member_type<   int> y) { return detail::not_(_mm256_cmpgt_epi32(x, y)); }
    static Vc_INTRINSIC mask_member_type<  uint> Vc_VDECL less_equal(datapar_member_type<  uint> x, datapar_member_type<  uint> y) { return detail::not_(cmpgt(x, y)); }
    static Vc_INTRINSIC mask_member_type< short> Vc_VDECL less_equal(datapar_member_type< short> x, datapar_member_type< short> y) { return detail::not_(_mm256_cmpgt_epi16(x, y)); }
    static Vc_INTRINSIC mask_member_type<ushort> Vc_VDECL less_equal(datapar_member_type<ushort> x, datapar_member_type<ushort> y) { return detail::not_(cmpgt(x, y)); }
    static Vc_INTRINSIC mask_member_type< schar> Vc_VDECL less_equal(datapar_member_type< schar> x, datapar_member_type< schar> y) { return detail::not_(_mm256_cmpgt_epi8 (x, y)); }
    static Vc_INTRINSIC mask_member_type< uchar> Vc_VDECL less_equal(datapar_member_type< uchar> x, datapar_member_type< uchar> y) { return detail::not_(cmpgt (x, y)); }
#endif
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
struct avx_mask_impl : public generic_mask_impl<datapar_abi::avx, avx_mask_member_type> {
    // member types {{{2
    using abi = datapar_abi::avx;
    template <class T> static constexpr size_t size() { return datapar_size_v<T, abi>; }
    template <class T> using mask_member_type = avx_mask_member_type<T>;
    template <class T> using mask = Vc::mask<T, datapar_abi::avx>;
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
constexpr struct {
    template <class T> operator T() const { return detail::allone<T>(); }
} allone_poly = {};
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

// [mask.reductions] {{{
Vc_VERSIONED_NAMESPACE_BEGIN
template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL all_of(mask<T, datapar_abi::avx> k)
{
    const auto d = detail::data(k);
    return 0 != detail::testc(d, detail::allone_poly);
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL any_of(mask<T, datapar_abi::avx> k)
{
    const auto d = detail::data(k);
    return 0 == detail::testz(d, d);
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL none_of(mask<T, datapar_abi::avx> k)
{
    const auto d = detail::data(k);
    return 0 != detail::testz(d, d);
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL some_of(mask<T, datapar_abi::avx> k)
{
    const auto d = detail::data(k);
    return 0 != detail::testnzc(d, detail::allone_poly);
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL popcount(mask<T, datapar_abi::avx> k)
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

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL find_first_set(mask<T, datapar_abi::avx> k)
{
    const auto d = detail::data(k);
    return detail::firstbit(detail::mask_to_int<k.size()>(d));
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL find_last_set(mask<T, datapar_abi::avx> k)
{
    const auto d = detail::data(k);
    if (k.size() == 16) {
        return detail::lastbit(detail::mask_to_int<32>(d)) / 2;
    }
    return detail::lastbit(detail::mask_to_int<k.size()>(d));
}

Vc_VERSIONED_NAMESPACE_END
// }}}

#endif  // Vc_HAVE_AVX_ABI

#endif  // Vc_HAVE_SSE
#endif  // VC_DATAPAR_AVX_H_

// vim: foldmethod=marker
