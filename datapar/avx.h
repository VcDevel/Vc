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
#ifdef Vc_HAVE_SSE
#include "storage.h"
#include "x86/intrinsics.h"
#include "x86/convert.h"
#include "x86/compares.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
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

    // load {{{2
    // from long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC datapar_member_type<T> load(const long double *mem, F,
                                                    type_tag<T>) noexcept
    {
        return generate_from_n_evaluations<size<T>(), datapar_member_type<T>>(
            [&](auto i) { return static_cast<T>(mem[i]); });
    }

    // load without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(const T *mem, F f, type_tag<T>) noexcept
    {
        return detail::load32(mem, f);
    }

    // convert from an AVX load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U)> = nullarg) noexcept
    {
        return convert<datapar_member_type<U>, datapar_member_type<T>>(load32(mem, f));
    }

    // convert from an SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U) * 2> = nullarg) noexcept
    {
        return convert<sse_datapar_member_type<U>, datapar_member_type<T>>(
            load16(mem, f));
    }

    // convert from a half SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U) * 4> = nullarg) noexcept
    {
        return convert<sse_datapar_member_type<U>, datapar_member_type<T>>(load8(mem, f));
    }

    // convert from a 1/4th SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U) * 8> = nullarg) noexcept
    {
        return convert<sse_datapar_member_type<U>, datapar_member_type<T>>(load4(mem, f));
    }

    // convert from an AVX512/2-AVX load{{{3
    template <class T>
    using avx512_member_type =
        typename traits<T, datapar_abi::avx512>::datapar_member_type;

    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) * 2 == sizeof(U)> = nullarg) noexcept
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
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) * 4 == sizeof(U)> = nullarg) noexcept
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
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) * 8 == sizeof(U)> = nullarg) noexcept
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
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL masked_load(datapar_member_type<T> &merge,
                                                  mask<T> k, const U *mem, F) noexcept
    {
        // TODO: implement with V(P)MASKMOV if AVX(2) is available
        execute_n_times<size<T>()>([&](auto i) {
            if (k.d.m(i)) {
                merge.set(i, static_cast<T>(mem[i]));
            }
        });
    }

    // store {{{2
    // store to long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(datapar_member_type<T> v, long double *mem, F,
                                            type_tag<T>) noexcept
    {
        // alignment F doesn't matter
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // store without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL store(datapar_member_type<T> v, T *mem, F f,
                                            type_tag<T>) noexcept
    {
        store32(v, mem, f);
    }

    // convert and 32-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 8> = nullarg) noexcept
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 64-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 4> = nullarg) noexcept
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 128-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U) * 2> = nullarg) noexcept
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 256-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) == sizeof(U)> = nullarg) noexcept
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 512-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) * 2 == sizeof(U)> = nullarg) noexcept
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 1024-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) * 4 == sizeof(U)> = nullarg) noexcept
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // convert and 2048-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(datapar_member_type<T> v, U *mem, F, type_tag<T>,
          enable_if<sizeof(T) * 8 == sizeof(U)> = nullarg) noexcept
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // masked store {{{2
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL masked_store(datapar_member_type<T> v,
                                                   long double *mem, F,
                                                   mask<T> k) noexcept
    {
        // no SSE support for long double
        execute_n_times<size<T>()>([&](auto i) {
            if (k.d.m(i)) {
                mem[i] = v.m(i);
            }
        });
    }
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL masked_store(datapar_member_type<T> v, U *mem, F,
                                                   mask<T> k) noexcept
    {
        //TODO: detail::masked_store(mem, v.v(), k.d.v(), f);
        execute_n_times<size<T>()>([&](auto i) {
            if (k.d.m(i)) {
                mem[i] = static_cast<T>(v.m(i));
            }
        });
    }

    // negation {{{2
    template <class T> static Vc_INTRINSIC mask<T> Vc_VDECL negate(datapar<T> x) noexcept
    {
#if defined Vc_GCC && defined Vc_USE_BUILTIN_VECTOR_TYPES
        return {private_init, !x.d.builtin()};
#else
        return equal_to(x, datapar<T>(0));
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
    static Vc_INTRINSIC datapar<T_> min(datapar<T_> a, datapar<T_> b)                    \
    {                                                                                    \
        return {private_init, _mm256_min_##suffix_(data(a), data(b))};                   \
    }                                                                                    \
    static Vc_INTRINSIC datapar<T_> max(datapar<T_> a, datapar<T_> b)                    \
    {                                                                                    \
        return {private_init, _mm256_max_##suffix_(data(a), data(b))};                   \
    }                                                                                    \
    static_assert(true, "")
    Vc_MINMAX_(double, pd);
    Vc_MINMAX_( float, ps);
    Vc_MINMAX_(   int, epi32);
    Vc_MINMAX_(  uint, epu32);
    Vc_MINMAX_( short, epi16);
    Vc_MINMAX_(ushort, epu16);
    Vc_MINMAX_( schar, epi8);
    Vc_MINMAX_( uchar, epu8);
#ifdef Vc_HAVE_AVX512VL
    Vc_MINMAX_( llong, epi64);
    Vc_MINMAX_(ullong, epu64);
#elif defined Vc_HAVE_AVX2
    static Vc_INTRINSIC datapar<llong> min(datapar<llong> a, datapar<llong> b)
    {
        auto x = data(a), y = data(b);
        return {private_init, _mm256_blendv_epi8(x, y, _mm256_cmpgt_epi64(x, y))};
    }
    static Vc_INTRINSIC datapar<llong> max(datapar<llong> a, datapar<llong> b)
    {
        auto x = data(a), y = data(b);
        return {private_init, _mm256_blendv_epi8(y, x, _mm256_cmpgt_epi64(x, y))};
    }
    static Vc_INTRINSIC datapar<ullong> min(datapar<ullong> a, datapar<ullong> b)
    {
        auto x = data(a), y = data(b);
        return {private_init, _mm256_blendv_epi8(x, y, cmpgt(x, y))};
    }
    static Vc_INTRINSIC datapar<ullong> max(datapar<ullong> a, datapar<ullong> b)
    {
        auto x = data(a), y = data(b);
        return {private_init, _mm256_blendv_epi8(y, x, cmpgt(x, y))};
    }
#endif
#undef Vc_MINMAX_

#if defined Vc_HAVE_AVX2
    static Vc_INTRINSIC datapar<long> min(datapar<long> a, datapar<long> b)
    {
        return datapar<long>{data(min(datapar<equal_int_type_t<long>>(data(a)),
                                      datapar<equal_int_type_t<long>>(data(b))))};
    }
    static Vc_INTRINSIC datapar<long> max(datapar<long> a, datapar<long> b)
    {
        return datapar<long>{data(max(datapar<equal_int_type_t<long>>(data(a)),
                                      datapar<equal_int_type_t<long>>(data(b))))};
    }

    static Vc_INTRINSIC datapar<ulong> min(datapar<ulong> a, datapar<ulong> b)
    {
        return datapar<ulong>{data(min(datapar<equal_int_type_t<ulong>>(data(a)),
                                       datapar<equal_int_type_t<ulong>>(data(b))))};
    }
    static Vc_INTRINSIC datapar<ulong> max(datapar<ulong> a, datapar<ulong> b)
    {
        return datapar<ulong>{data(max(datapar<equal_int_type_t<ulong>>(data(a)),
                                       datapar<equal_int_type_t<ulong>>(data(b))))};
    }
#endif  // Vc_HAVE_AVX2

    template <class T>
    static Vc_INTRINSIC std::pair<datapar<T>, datapar<T>> minmax(datapar<T> a,
                                                                 datapar<T> b)
    {
        return {min(a, b), max(a, b)};
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
    static Vc_INTRINSIC mask<double> Vc_VDECL equal_to    (datapar<double> x, datapar<double> y) { return {private_init, _mm256_cmp_pd(x.d, y.d, _CMP_EQ_OQ)}; }
    static Vc_INTRINSIC mask<double> Vc_VDECL not_equal_to(datapar<double> x, datapar<double> y) { return {private_init, _mm256_cmp_pd(x.d, y.d, _CMP_NEQ_UQ)}; }
    static Vc_INTRINSIC mask<double> Vc_VDECL less        (datapar<double> x, datapar<double> y) { return {private_init, _mm256_cmp_pd(x.d, y.d, _CMP_LT_OS)}; }
    static Vc_INTRINSIC mask<double> Vc_VDECL less_equal  (datapar<double> x, datapar<double> y) { return {private_init, _mm256_cmp_pd(x.d, y.d, _CMP_LE_OS)}; }
    static Vc_INTRINSIC mask< float> Vc_VDECL equal_to    (datapar< float> x, datapar< float> y) { return {private_init, _mm256_cmp_ps(x.d, y.d, _CMP_EQ_OQ)}; }
    static Vc_INTRINSIC mask< float> Vc_VDECL not_equal_to(datapar< float> x, datapar< float> y) { return {private_init, _mm256_cmp_ps(x.d, y.d, _CMP_NEQ_UQ)}; }
    static Vc_INTRINSIC mask< float> Vc_VDECL less        (datapar< float> x, datapar< float> y) { return {private_init, _mm256_cmp_ps(x.d, y.d, _CMP_LT_OS)}; }
    static Vc_INTRINSIC mask< float> Vc_VDECL less_equal  (datapar< float> x, datapar< float> y) { return {private_init, _mm256_cmp_ps(x.d, y.d, _CMP_LE_OS)}; }

#ifdef Vc_HAVE_FULL_AVX_ABI
    static Vc_INTRINSIC mask< llong> Vc_VDECL equal_to(datapar< llong> x, datapar< llong> y) { return {private_init, _mm256_cmpeq_epi64(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ullong> Vc_VDECL equal_to(datapar<ullong> x, datapar<ullong> y) { return {private_init, _mm256_cmpeq_epi64(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  long> Vc_VDECL equal_to(datapar<  long> x, datapar<  long> y) { return {private_init, sizeof(long) == 8 ? _mm256_cmpeq_epi64(x.d, y.d) : _mm256_cmpeq_epi32(x.d, y.d)}; }
    static Vc_INTRINSIC mask< ulong> Vc_VDECL equal_to(datapar< ulong> x, datapar< ulong> y) { return {private_init, sizeof(long) == 8 ? _mm256_cmpeq_epi64(x.d, y.d) : _mm256_cmpeq_epi32(x.d, y.d)}; }
    static Vc_INTRINSIC mask<   int> Vc_VDECL equal_to(datapar<   int> x, datapar<   int> y) { return {private_init, _mm256_cmpeq_epi32(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  uint> Vc_VDECL equal_to(datapar<  uint> x, datapar<  uint> y) { return {private_init, _mm256_cmpeq_epi32(x.d, y.d)}; }
    static Vc_INTRINSIC mask< short> Vc_VDECL equal_to(datapar< short> x, datapar< short> y) { return {private_init, _mm256_cmpeq_epi16(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ushort> Vc_VDECL equal_to(datapar<ushort> x, datapar<ushort> y) { return {private_init, _mm256_cmpeq_epi16(x.d, y.d)}; }
    static Vc_INTRINSIC mask< schar> Vc_VDECL equal_to(datapar< schar> x, datapar< schar> y) { return {private_init, _mm256_cmpeq_epi8(x.d, y.d)}; }
    static Vc_INTRINSIC mask< uchar> Vc_VDECL equal_to(datapar< uchar> x, datapar< uchar> y) { return {private_init, _mm256_cmpeq_epi8(x.d, y.d)}; }

    static Vc_INTRINSIC mask< llong> Vc_VDECL not_equal_to(datapar< llong> x, datapar< llong> y) { return {private_init, detail::not_(_mm256_cmpeq_epi64(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ullong> Vc_VDECL not_equal_to(datapar<ullong> x, datapar<ullong> y) { return {private_init, detail::not_(_mm256_cmpeq_epi64(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  long> Vc_VDECL not_equal_to(datapar<  long> x, datapar<  long> y) { return {private_init, detail::not_(sizeof(long) == 8 ? _mm256_cmpeq_epi64(x.d, y.d) : _mm256_cmpeq_epi32(x.d, y.d))}; }
    static Vc_INTRINSIC mask< ulong> Vc_VDECL not_equal_to(datapar< ulong> x, datapar< ulong> y) { return {private_init, detail::not_(sizeof(long) == 8 ? _mm256_cmpeq_epi64(x.d, y.d) : _mm256_cmpeq_epi32(x.d, y.d))}; }
    static Vc_INTRINSIC mask<   int> Vc_VDECL not_equal_to(datapar<   int> x, datapar<   int> y) { return {private_init, detail::not_(_mm256_cmpeq_epi32(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  uint> Vc_VDECL not_equal_to(datapar<  uint> x, datapar<  uint> y) { return {private_init, detail::not_(_mm256_cmpeq_epi32(x.d, y.d))}; }
    static Vc_INTRINSIC mask< short> Vc_VDECL not_equal_to(datapar< short> x, datapar< short> y) { return {private_init, detail::not_(_mm256_cmpeq_epi16(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ushort> Vc_VDECL not_equal_to(datapar<ushort> x, datapar<ushort> y) { return {private_init, detail::not_(_mm256_cmpeq_epi16(x.d, y.d))}; }
    static Vc_INTRINSIC mask< schar> Vc_VDECL not_equal_to(datapar< schar> x, datapar< schar> y) { return {private_init, detail::not_(_mm256_cmpeq_epi8(x.d, y.d))}; }
    static Vc_INTRINSIC mask< uchar> Vc_VDECL not_equal_to(datapar< uchar> x, datapar< uchar> y) { return {private_init, detail::not_(_mm256_cmpeq_epi8(x.d, y.d))}; }

    static Vc_INTRINSIC mask< llong> Vc_VDECL less(datapar< llong> x, datapar< llong> y) { return {private_init, _mm256_cmpgt_epi64(y.d, x.d)}; }
    static Vc_INTRINSIC mask<ullong> Vc_VDECL less(datapar<ullong> x, datapar<ullong> y) { return {private_init, cmpgt(y.d, x.d)}; }
    static Vc_INTRINSIC mask<  long> Vc_VDECL less(datapar<  long> x, datapar<  long> y) { return {private_init, sizeof(long) == 8 ? _mm256_cmpgt_epi64(y.d, x.d) : _mm256_cmpgt_epi32(y.d, x.d)}; }
    static Vc_INTRINSIC mask< ulong> Vc_VDECL less(datapar< ulong> x, datapar< ulong> y) { return {private_init, cmpgt(y.d, x.d)}; }
    static Vc_INTRINSIC mask<   int> Vc_VDECL less(datapar<   int> x, datapar<   int> y) { return {private_init, _mm256_cmpgt_epi32(y.d, x.d)}; }
    static Vc_INTRINSIC mask<  uint> Vc_VDECL less(datapar<  uint> x, datapar<  uint> y) { return {private_init, cmpgt(y.d, x.d)}; }
    static Vc_INTRINSIC mask< short> Vc_VDECL less(datapar< short> x, datapar< short> y) { return {private_init, _mm256_cmpgt_epi16(y.d, x.d)}; }
    static Vc_INTRINSIC mask<ushort> Vc_VDECL less(datapar<ushort> x, datapar<ushort> y) { return {private_init, cmpgt(y.d, x.d)}; }
    static Vc_INTRINSIC mask< schar> Vc_VDECL less(datapar< schar> x, datapar< schar> y) { return {private_init, _mm256_cmpgt_epi8 (y.d, x.d)}; }
    static Vc_INTRINSIC mask< uchar> Vc_VDECL less(datapar< uchar> x, datapar< uchar> y) { return {private_init, cmpgt(y.d, x.d)}; }

    static Vc_INTRINSIC mask< llong> Vc_VDECL less_equal(datapar< llong> x, datapar< llong> y) { return {private_init, detail::not_(_mm256_cmpgt_epi64(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ullong> Vc_VDECL less_equal(datapar<ullong> x, datapar<ullong> y) { return {private_init, detail::not_(cmpgt(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  long> Vc_VDECL less_equal(datapar<  long> x, datapar<  long> y) { return {private_init, detail::not_(sizeof(long) == 8 ? _mm256_cmpgt_epi64(x.d, y.d) : _mm256_cmpgt_epi32(x.d, y.d))}; }
    static Vc_INTRINSIC mask< ulong> Vc_VDECL less_equal(datapar< ulong> x, datapar< ulong> y) { return {private_init, detail::not_(cmpgt(x.d, y.d))}; }
    static Vc_INTRINSIC mask<   int> Vc_VDECL less_equal(datapar<   int> x, datapar<   int> y) { return {private_init, detail::not_(_mm256_cmpgt_epi32(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  uint> Vc_VDECL less_equal(datapar<  uint> x, datapar<  uint> y) { return {private_init, detail::not_(cmpgt(x.d, y.d))}; }
    static Vc_INTRINSIC mask< short> Vc_VDECL less_equal(datapar< short> x, datapar< short> y) { return {private_init, detail::not_(_mm256_cmpgt_epi16(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ushort> Vc_VDECL less_equal(datapar<ushort> x, datapar<ushort> y) { return {private_init, detail::not_(cmpgt(x.d, y.d))}; }
    static Vc_INTRINSIC mask< schar> Vc_VDECL less_equal(datapar< schar> x, datapar< schar> y) { return {private_init, detail::not_(_mm256_cmpgt_epi8 (x.d, y.d))}; }
    static Vc_INTRINSIC mask< uchar> Vc_VDECL less_equal(datapar< uchar> x, datapar< uchar> y) { return {private_init, detail::not_(cmpgt (x.d, y.d))}; }
#endif
#endif

    // smart_reference access {{{2
    template <class T, class A>
    static Vc_INTRINSIC T Vc_VDECL get(Vc::datapar<T, A> v, int i) noexcept
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
    template <class T> static constexpr size_t size() { return datapar_size_v<T, abi>; }
    template <class T> using mask_member_type = avx_mask_member_type<T>;
    template <class T> using mask = Vc::mask<T, datapar_abi::avx>;
    template <class T> using mask_bool = MaskBool<sizeof(T)>;
    template <size_t N> using size_tag = std::integral_constant<size_t, N>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    template <typename T> static Vc_INTRINSIC auto broadcast(bool x, type_tag<T>) noexcept
    {
        return detail::broadcast32(T(mask_bool<T>{x}));
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

    // masked load {{{2
    template <class T, class F, class SizeTag>
    static Vc_INTRINSIC void Vc_VDECL masked_load(mask_member_type<T> &merge,
                                                  mask_member_type<T> mask,
                                                  const bool *mem, F, SizeTag) noexcept
    {
        for (std::size_t i = 0; i < size<T>(); ++i) {
            if (mask.m(i)) {
                merge.set(i, mask_bool<T>{mem[i]});
            }
        }
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
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F f,
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
    static Vc_INTRINSIC void Vc_VDECL store(mask_member_type<T> v, bool *mem, F f,
                                            size_tag<32>) noexcept
    {
        const auto bools = detail::and_(one32(uchar()), v.v());
        store32(bools, mem, f);
    }

    // masked store {{{2
    template <class T, class F, class SizeTag>
    static Vc_INTRINSIC void Vc_VDECL masked_store(mask_member_type<T> v, bool *mem, F,
                                                   mask_member_type<T> k,
                                                   SizeTag) noexcept
    {
        for (std::size_t i = 0; i < size<T>(); ++i) {
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
    template <class T> static constexpr size_t size() { return M<T>::size(); }
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
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k);
    return 0 != detail::testc(d, detail::allone_poly);
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL any_of(mask<T, datapar_abi::avx> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k);
    return 0 == detail::testz(d, d);
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL none_of(mask<T, datapar_abi::avx> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k);
    return 0 != detail::testz(d, d);
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL some_of(mask<T, datapar_abi::avx> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k);
    return 0 != detail::testnzc(d, detail::allone_poly);
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL popcount(mask<T, datapar_abi::avx> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k);
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
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k);
    return detail::firstbit(detail::mask_to_int<k.size()>(d));
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL find_last_set(mask<T, datapar_abi::avx> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::avx>::mask_cast_type>(k);
    if (k.size() == 16) {
        return detail::lastbit(detail::mask_to_int<32>(d)) / 2;
    }
    return detail::lastbit(detail::mask_to_int<k.size()>(d));
}

Vc_VERSIONED_NAMESPACE_END
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
            return false;
        }
    }
};
// }}}1
}  // namespace std
#endif  // Vc_HAVE_AVX_ABI

#endif  // Vc_HAVE_SSE
#endif  // VC_DATAPAR_AVX_H_

// vim: foldmethod=marker
