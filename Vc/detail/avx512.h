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
#ifdef Vc_HAVE_SSE
#include "storage.h"
#include "x86/intrinsics.h"
#include "x86/convert.h"
#include "x86/compares.h"

// clang 3.9 doesn't have the _MM_CMPINT_XX constants defined {{{
#ifndef _MM_CMPINT_EQ
#define _MM_CMPINT_EQ 0x0
#endif
#ifndef _MM_CMPINT_LT
#define _MM_CMPINT_LT 0x1
#endif
#ifndef _MM_CMPINT_LE
#define _MM_CMPINT_LE 0x2
#endif
#ifndef _MM_CMPINT_UNUSED
#define _MM_CMPINT_UNUSED 0x3
#endif
#ifndef _MM_CMPINT_NE
#define _MM_CMPINT_NE 0x4
#endif
#ifndef _MM_CMPINT_NLT
#define _MM_CMPINT_NLT 0x5
#endif
#ifndef _MM_CMPINT_GE
#define _MM_CMPINT_GE 0x5
#endif
#ifndef _MM_CMPINT_NLE
#define _MM_CMPINT_NLE 0x6
#endif
#ifndef _MM_CMPINT_GT
#define _MM_CMPINT_GT 0x6
#endif /*}}}*/

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
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
template <size_t N> using avx512_mask_member_type_n = Storage<bool, N>;

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
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#ifdef Vc_HAVE_AVX512_ABI
Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// datapar impl {{{1
struct avx512_datapar_impl : public generic_datapar_impl<avx512_datapar_impl> {
    // member types {{{2
    using abi = datapar_abi::avx512;
    template <class T> static constexpr size_t size() { return datapar_size_v<T, abi>; }
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
        return _mm512_set1_epi64(x);
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

    // load {{{2
    // from long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC datapar_member_type<T> load(const long double *mem, F,
                                                    type_tag<T>) noexcept
    {
#if defined Vc_GCC && Vc_GCC < 0x60000
        // GCC 5.4 ICEs with:
        // datapar/detail.h|74 col 1| error: unrecognizable insn:
        // ||  }
        // ||  ^
        // || (insn 43 42 44 2 (set (reg:V16SF 121 [ D.470376 ])
        // ||         (vec_merge:V16SI (reg:V16SF 144)
        // ||             (reg:V16SF 121 [ D.470376 ])
        // ||             (reg:HI 145))) datapar/storage.h:277 -1
        // ||      (nil))
        // datapar/detail.h|74 col 1| internal compiler error: in extract_insn, at recog.c:2343
        alignas(sizeof(T) * size<T>()) T tmp[size<T>()];
        for (size_t i = 0; i < size<T>(); ++i) {
            tmp[i] = static_cast<T>(mem[i]);
        }
        return load(&tmp[0], flags::vector_aligned, type_tag<T>());
#else
        return generate_from_n_evaluations<size<T>(), datapar_member_type<T>>(
            [&](auto i) { return static_cast<T>(mem[i]); });
#endif
    }

    // load without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(const T *mem, F f, type_tag<T>) noexcept
    {
        return detail::load64(mem, f);
    }

    // convert from an AVX512 load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U)> = nullarg) noexcept
    {
        return convert<datapar_member_type<U>, datapar_member_type<T>>(load64(mem, f));
    }

    // convert from an AVX load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U) * 2> = nullarg) noexcept
    {
        return convert<avx_datapar_member_type<U>, datapar_member_type<T>>(load32(mem, f));
    }

    // convert from an SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U) * 4> = nullarg) noexcept
    {
        return convert<sse_datapar_member_type<U>, datapar_member_type<T>>(load16(mem, f));
    }

    // convert from a half SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U) * 8> = nullarg) noexcept
    {
        return convert<sse_datapar_member_type<U>, datapar_member_type<T>>(load8(mem, f));
    }

    // convert from a 2-AVX512 load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) * 2 == sizeof(U)> = nullarg) noexcept
    {
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
            load64(mem, f), load64(mem + size<U>(), f));
    }

    // convert from a 4-AVX512 load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) * 4 == sizeof(U)> = nullarg) noexcept
    {
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
            load64(mem, f), load64(mem + size<U>(), f), load64(mem + 2 * size<U>(), f),
            load64(mem + 3 * size<U>(), f));
    }

    // convert from a 8-AVX512 load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) * 8 == sizeof(U)> = nullarg) noexcept
    {
        return convert<datapar_member_type<U>, datapar_member_type<T>>(
            load64(mem, f), load64(mem + size<U>(), f), load64(mem + 2 * size<U>(), f),
            load64(mem + 3 * size<U>(), f), load64(mem + 4 * size<U>(), f),
            load64(mem + 5 * size<U>(), f), load64(mem + 6 * size<U>(), f),
            load64(mem + 7 * size<U>(), f));
    }

    // masked load {{{2
    template <class T, class U, class F>
    static Vc_INTRINSIC void masked_load(datapar_member_type<T> &merge, mask<T> k,
                                         const U *mem, F) noexcept
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) {
            if (k.d.m(i)) {
                merge.set(i, static_cast<T>(mem[i]));
            }
        });
    }

    // store {{{2
    // store to long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC void store(datapar_member_type<T> v, long double *mem, F,
                                   type_tag<T>) noexcept
    {
        // alignment F doesn't matter
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }

    // store without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC void store(datapar_member_type<T> v, T *mem, F f,
                                   type_tag<T>) noexcept
    {
        store64(v, mem, f);
    }

    // convert and 64-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(
        datapar_member_type<T> v, U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U) * 8> = nullarg) noexcept
    {
        store8(convert<datapar_member_type<T>, sse_datapar_member_type<U>>(v), mem, f);
    }

    // convert and 128-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(
        datapar_member_type<T> v, U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U) * 4> = nullarg) noexcept
    {
        store16(convert<datapar_member_type<T>, sse_datapar_member_type<U>>(v), mem, f);
    }

    // convert and 256-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(
        datapar_member_type<T> v, U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U) * 2> = nullarg) noexcept
    {
        store32(convert<datapar_member_type<T>, avx_datapar_member_type<U>>(v), mem, f);
    }

    // convert and 512-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(datapar_member_type<T> v, U *mem, F f, type_tag<T>,
                                   enable_if<sizeof(T) == sizeof(U)> = nullarg) noexcept
    {
        store64(convert<datapar_member_type<T>, datapar_member_type<U>>(v), mem, f);
    }

    // convert and 1024-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(
        datapar_member_type<T> v, U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) * 2 == sizeof(U)> = nullarg) noexcept
    {
        store64(convert<avx_datapar_member_type<T>, datapar_member_type<U>>(lo256(v)), mem, f);
        store64(convert<avx_datapar_member_type<T>, datapar_member_type<U>>(hi256(v)),
                mem + size<U>(), f);
    }

    // convert and 2048-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(
        datapar_member_type<T> v, U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) * 4 == sizeof(U)> = nullarg) noexcept
    {
        store64(convert<sse_datapar_member_type<T>, datapar_member_type<U>>(lo128(v)), mem, f);
        store64(
            convert<sse_datapar_member_type<T>, datapar_member_type<U>>(extract128<1>(v)),
            mem + size<U>(), f);
        store64(
            convert<sse_datapar_member_type<T>, datapar_member_type<U>>(extract128<2>(v)),
            mem + 2 * size<U>(), f);
        store64(
            convert<sse_datapar_member_type<T>, datapar_member_type<U>>(extract128<3>(v)),
            mem + 3 * size<U>(), f);
    }

    // convert and 4096-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(
        datapar_member_type<T> v, U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) * 8 == sizeof(U)> = nullarg) noexcept
    {
        store64(convert<sse_datapar_member_type<T>, datapar_member_type<U>>(lo128(v)), mem, f);
        store64(convert<sse_datapar_member_type<T>, datapar_member_type<U>>(x86::shift_right<8>(lo128(v))), mem + 1 * size<U>(), f);
        store64(convert<sse_datapar_member_type<T>, datapar_member_type<U>>(extract128<1>(v)), mem + 2 * size<U>(), f);
        store64(convert<sse_datapar_member_type<T>, datapar_member_type<U>>(x86::shift_right<8>(extract128<1>(v))), mem + 3 * size<U>(), f);
        store64(convert<sse_datapar_member_type<T>, datapar_member_type<U>>(extract128<2>(v)), mem + 4 * size<U>(), f);
        store64(convert<sse_datapar_member_type<T>, datapar_member_type<U>>(x86::shift_right<8>(extract128<2>(v))), mem + 5 * size<U>(), f);
        store64(convert<sse_datapar_member_type<T>, datapar_member_type<U>>(extract128<3>(v)), mem + 6 * size<U>(), f);
        store64(convert<sse_datapar_member_type<T>, datapar_member_type<U>>(x86::shift_right<8>(extract128<3>(v))), mem + 7 * size<U>(), f);
    }

    // masked store {{{2
    template <class T, class F>
    static Vc_INTRINSIC void masked_store(datapar_member_type<T> v, long double *mem, F,
                                          mask<T> k) noexcept
    {
        // no support for long double
        execute_n_times<size<T>()>([&](auto i) {
            if (k.d.m(i)) {
                mem[i] = v.m(i);
            }
        });
    }
    template <class T, class U, class F>
    static Vc_INTRINSIC void masked_store(datapar_member_type<T> v, U *mem, F,
                                          mask<T> k) noexcept
    {
        //TODO
        execute_n_times<size<T>()>([&](auto i) {
            if (k.d.m(i)) {
                mem[i] = static_cast<T>(v.m(i));
            }
        });
    }

    // negation {{{2
    template <class T> static Vc_INTRINSIC mask<T> negate(datapar<T> x) noexcept
    {
        return equal_to(x, datapar<T>(0));
    }

    // reductions {{{2
    template <class T, class BinaryOperation, size_t N>
    static Vc_INTRINSIC T Vc_VDECL reduce(size_tag<N>, datapar<T> x,
                                          BinaryOperation &binary_op)
    {
        using V = Vc::datapar<T, datapar_abi::avx>;
        return avx_datapar_impl::reduce(size_tag<N / 2>(),
                                        binary_op(V(lo256(data(x))), V(hi256(data(x)))),
                                        binary_op);
    }

    // min, max {{{2
#define Vc_MINMAX_(T_, suffix_)                                                          \
    static Vc_INTRINSIC datapar<T_> min(datapar<T_> a, datapar<T_> b)                    \
    {                                                                                    \
        return {private_init, _mm512_min_##suffix_(data(a), data(b))};                   \
    }                                                                                    \
    static Vc_INTRINSIC datapar<T_> max(datapar<T_> a, datapar<T_> b)                    \
    {                                                                                    \
        return {private_init, _mm512_max_##suffix_(data(a), data(b))};                   \
    }                                                                                    \
    static_assert(true, "")
    Vc_MINMAX_(double, pd);
    Vc_MINMAX_( float, ps);
    Vc_MINMAX_( llong, epi64);
    Vc_MINMAX_(ullong, epu64);
    Vc_MINMAX_(   int, epi32);
    Vc_MINMAX_(  uint, epu32);
    Vc_MINMAX_( short, epi16);
    Vc_MINMAX_(ushort, epu16);
    Vc_MINMAX_( schar, epi8);
    Vc_MINMAX_( uchar, epu8);
#undef Vc_MINMAX_
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

    template <class T>
    static Vc_INTRINSIC std::pair<datapar<T>, datapar<T>> minmax(datapar<T> a,
                                                                 datapar<T> b)
    {
        return {min(a, b), max(a, b)};
    }


    // compares {{{2
#if 0  // defined Vc_USE_BUILTIN_VECTOR_TYPES
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
    static Vc_INTRINSIC mask<double> equal_to    (datapar<double> x, datapar<double> y) { return {private_init, _mm512_cmp_pd_mask(x.d, y.d, _MM_CMPINT_EQ)}; }
    static Vc_INTRINSIC mask<double> not_equal_to(datapar<double> x, datapar<double> y) { return {private_init, _mm512_cmp_pd_mask(x.d, y.d, _MM_CMPINT_NE)}; }
    static Vc_INTRINSIC mask<double> less        (datapar<double> x, datapar<double> y) { return {private_init, _mm512_cmp_pd_mask(x.d, y.d, _MM_CMPINT_LT)}; }
    static Vc_INTRINSIC mask<double> less_equal  (datapar<double> x, datapar<double> y) { return {private_init, _mm512_cmp_pd_mask(x.d, y.d, _MM_CMPINT_LE)}; }
    static Vc_INTRINSIC mask< float> equal_to    (datapar< float> x, datapar< float> y) { return {private_init, _mm512_cmp_ps_mask(x.d, y.d, _MM_CMPINT_EQ)}; }
    static Vc_INTRINSIC mask< float> not_equal_to(datapar< float> x, datapar< float> y) { return {private_init, _mm512_cmp_ps_mask(x.d, y.d, _MM_CMPINT_NE)}; }
    static Vc_INTRINSIC mask< float> less        (datapar< float> x, datapar< float> y) { return {private_init, _mm512_cmp_ps_mask(x.d, y.d, _MM_CMPINT_LT)}; }
    static Vc_INTRINSIC mask< float> less_equal  (datapar< float> x, datapar< float> y) { return {private_init, _mm512_cmp_ps_mask(x.d, y.d, _MM_CMPINT_LE)}; }

    static Vc_INTRINSIC mask< llong> equal_to(datapar< llong> x, datapar< llong> y) { return {private_init, _mm512_cmpeq_epi64_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ullong> equal_to(datapar<ullong> x, datapar<ullong> y) { return {private_init, _mm512_cmpeq_epi64_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  long> equal_to(datapar<  long> x, datapar<  long> y) { return {private_init, detail::cmpeq_long_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< ulong> equal_to(datapar< ulong> x, datapar< ulong> y) { return {private_init, detail::cmpeq_long_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<   int> equal_to(datapar<   int> x, datapar<   int> y) { return {private_init, _mm512_cmpeq_epi32_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  uint> equal_to(datapar<  uint> x, datapar<  uint> y) { return {private_init, _mm512_cmpeq_epi32_mask(x.d, y.d)}; }

    static Vc_INTRINSIC mask< llong> not_equal_to(datapar< llong> x, datapar< llong> y) { return {private_init, detail::not_(_mm512_cmpeq_epi64_mask(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ullong> not_equal_to(datapar<ullong> x, datapar<ullong> y) { return {private_init, detail::not_(_mm512_cmpeq_epi64_mask(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  long> not_equal_to(datapar<  long> x, datapar<  long> y) { return {private_init, detail::not_(detail::cmpeq_long_mask(x.d, y.d))}; }
    static Vc_INTRINSIC mask< ulong> not_equal_to(datapar< ulong> x, datapar< ulong> y) { return {private_init, detail::not_(detail::cmpeq_long_mask(x.d, y.d))}; }
    static Vc_INTRINSIC mask<   int> not_equal_to(datapar<   int> x, datapar<   int> y) { return {private_init, detail::not_(_mm512_cmpeq_epi32_mask(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  uint> not_equal_to(datapar<  uint> x, datapar<  uint> y) { return {private_init, detail::not_(_mm512_cmpeq_epi32_mask(x.d, y.d))}; }

    static Vc_INTRINSIC mask< llong> less(datapar< llong> x, datapar< llong> y) { return {private_init, _mm512_cmplt_epi64_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ullong> less(datapar<ullong> x, datapar<ullong> y) { return {private_init, _mm512_cmplt_epu64_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  long> less(datapar<  long> x, datapar<  long> y) { return {private_init, detail::cmplt_long_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< ulong> less(datapar< ulong> x, datapar< ulong> y) { return {private_init, detail::cmplt_ulong_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<   int> less(datapar<   int> x, datapar<   int> y) { return {private_init, _mm512_cmplt_epi32_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  uint> less(datapar<  uint> x, datapar<  uint> y) { return {private_init, _mm512_cmplt_epu32_mask(x.d, y.d)}; }

    static Vc_INTRINSIC mask< llong> less_equal(datapar< llong> x, datapar< llong> y) { return {private_init, _mm512_cmple_epi64_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ullong> less_equal(datapar<ullong> x, datapar<ullong> y) { return {private_init, _mm512_cmple_epu64_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  long> less_equal(datapar<  long> x, datapar<  long> y) { return {private_init, detail::cmple_long_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< ulong> less_equal(datapar< ulong> x, datapar< ulong> y) { return {private_init, detail::cmple_ulong_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<   int> less_equal(datapar<   int> x, datapar<   int> y) { return {private_init, _mm512_cmple_epi32_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  uint> less_equal(datapar<  uint> x, datapar<  uint> y) { return {private_init, _mm512_cmple_epu32_mask(x.d, y.d)}; }

#ifdef Vc_HAVE_FULL_AVX512_ABI
    static Vc_INTRINSIC mask< short> equal_to(datapar< short> x, datapar< short> y) { return {private_init, _mm512_cmpeq_epi16_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ushort> equal_to(datapar<ushort> x, datapar<ushort> y) { return {private_init, _mm512_cmpeq_epi16_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< schar> equal_to(datapar< schar> x, datapar< schar> y) { return {private_init, _mm512_cmpeq_epi8_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< uchar> equal_to(datapar< uchar> x, datapar< uchar> y) { return {private_init, _mm512_cmpeq_epi8_mask(x.d, y.d)}; }

    static Vc_INTRINSIC mask< short> not_equal_to(datapar< short> x, datapar< short> y) { return {private_init, detail::not_(_mm512_cmpeq_epi16_mask(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ushort> not_equal_to(datapar<ushort> x, datapar<ushort> y) { return {private_init, detail::not_(_mm512_cmpeq_epi16_mask(x.d, y.d))}; }
    static Vc_INTRINSIC mask< schar> not_equal_to(datapar< schar> x, datapar< schar> y) { return {private_init, detail::not_(_mm512_cmpeq_epi8_mask(x.d, y.d))}; }
    static Vc_INTRINSIC mask< uchar> not_equal_to(datapar< uchar> x, datapar< uchar> y) { return {private_init, detail::not_(_mm512_cmpeq_epi8_mask(x.d, y.d))}; }

    static Vc_INTRINSIC mask< short> less(datapar< short> x, datapar< short> y) { return {private_init, _mm512_cmplt_epi16_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ushort> less(datapar<ushort> x, datapar<ushort> y) { return {private_init, _mm512_cmplt_epu16_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< schar> less(datapar< schar> x, datapar< schar> y) { return {private_init, _mm512_cmplt_epi8_mask (x.d, y.d)}; }
    static Vc_INTRINSIC mask< uchar> less(datapar< uchar> x, datapar< uchar> y) { return {private_init, _mm512_cmplt_epu8_mask (x.d, y.d)}; }

    static Vc_INTRINSIC mask< short> less_equal(datapar< short> x, datapar< short> y) { return {private_init, _mm512_cmple_epi16_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ushort> less_equal(datapar<ushort> x, datapar<ushort> y) { return {private_init, _mm512_cmple_epu16_mask(x.d, y.d)}; }
    static Vc_INTRINSIC mask< schar> less_equal(datapar< schar> x, datapar< schar> y) { return {private_init, _mm512_cmple_epi8_mask (x.d, y.d)}; }
    static Vc_INTRINSIC mask< uchar> less_equal(datapar< uchar> x, datapar< uchar> y) { return {private_init, _mm512_cmple_epu8_mask (x.d, y.d)}; }
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
    template <class T> static constexpr size_t size() { return datapar_size_v<T, abi>; }
    template <size_t N> using mask_member_type = avx512_mask_member_type_n<N>;
    template <class T> using mask = Vc::mask<T, abi>;
    template <class T> using mask_bool = MaskBool<sizeof(T)>;
    template <size_t N> using size_tag = std::integral_constant<size_t, N>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    static Vc_INTRINSIC __mmask8 broadcast_impl(bool x, size_tag<8>) noexcept
    {
        return static_cast<__mmask8>(x) * ~__mmask8();
    }
    static Vc_INTRINSIC __mmask16 broadcast_impl(bool x, size_tag<16>) noexcept
    {
        return static_cast<__mmask16>(x) * ~__mmask16();
    }
    static Vc_INTRINSIC __mmask32 broadcast_impl(bool x, size_tag<32>) noexcept
    {
        return static_cast<__mmask32>(x) * ~__mmask32();
    }
    static Vc_INTRINSIC __mmask64 broadcast_impl(bool x, size_tag<64>) noexcept
    {
        return static_cast<__mmask64>(x) * ~__mmask64();
    }
    template <typename T> static Vc_INTRINSIC auto broadcast(bool x, type_tag<T>) noexcept
    {
        return broadcast_impl(x, size_tag<size<T>()>());
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
        return _mm512_test_epi64_mask(b, b);
#endif  // Vc_HAVE_AVX512BW
    }
    template <class F>
    static Vc_INTRINSIC __mmask16 load(const bool *mem, F f, size_tag<16>) noexcept
    {
        const auto a = load16(mem, f);
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        return _mm_test_epi8_mask(a, a);
#else
        const auto b = _mm512_cvtepi8_epi32(a);
        return _mm512_test_epi32_mask(b, b);
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
        return _mm512_test_epi32_mask(a, a) | (_mm512_test_epi32_mask(b, b) << 16);
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
        return _mm512_test_epi32_mask(a, a) | (_mm512_test_epi32_mask(b, b) << 16) |
               (_mm512_test_epi32_mask(b, b) << 32) | (_mm512_test_epi32_mask(b, b) << 48);
#endif  // Vc_HAVE_AVX512BW
    }

    // masked load {{{2
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
    template <class F>
    static Vc_INTRINSIC void masked_load(mask_member_type<8> &merge,
                                         mask_member_type<8> mask, const bool *mem, F,
                                             size_tag<8>) noexcept
    {
        const auto a = _mm_mask_loadu_epi8(zero<__m128i>(), mask.v(), mem);
        merge = (merge & ~mask) | _mm_test_epi8_mask(a, a);
    }

    template <class F>
    static Vc_INTRINSIC void masked_load(mask_member_type<16> &merge,
                                         mask_member_type<16> mask, const bool *mem, F,
                                         size_tag<16>) noexcept
    {
        const auto a = _mm_mask_loadu_epi8(zero<__m128i>(), mask.v(), mem);
        merge = (merge & ~mask) | _mm_test_epi8_mask(a, a);
    }

    template <class F>
    static Vc_INTRINSIC void masked_load(mask_member_type<32> &merge,
                                         mask_member_type<32> mask, const bool *mem, F,
                                         size_tag<32>) noexcept
    {
        const auto a = _mm256_mask_loadu_epi8(zero<__m256i>(), mask.v(), mem);
        merge = (merge & ~mask) | _mm256_test_epi8_mask(a, a);
    }

    template <class F>
    static Vc_INTRINSIC void masked_load(mask_member_type<64> &merge,
                                         mask_member_type<64> mask, const bool *mem, F,
                                         size_tag<64>) noexcept
    {
        const auto a = _mm512_mask_loadu_epi8(zero<__m512i>(), mask.v(), mem);
        merge = (merge & ~mask) | _mm512_test_epi8_mask(a, a);
    }

#else
    template <class F, size_t N>
    static Vc_INTRINSIC void masked_load(mask_member_type<N> &merge,
                                         mask_member_type<N> mask, const bool *mem, F,
                                         size_tag<N>) noexcept
    {
        for (std::size_t i = 0; i < N; ++i) {
            if (mask[i]) {
                merge.set(i, mem[i]);
            }
        }
    }
#endif

    // store {{{2
    template <class F>
    static constexpr void store(mask_member_type<8> v, bool *mem, F, size_tag<8>) noexcept
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        _mm_storel_epi64(reinterpret_cast<__m128i *>(mem),
                         and_(one16(uchar()), _mm_movm_epi8(v.v())));
#elif defined Vc_HAVE_AVX512DQ
        _mm_storel_epi64(
            reinterpret_cast<__m128i *>(mem),
            _mm512_cvtepi64_epi8(_mm512_srli_epi64(_mm512_movm_epi64(v.v()), 63)));
#elif defined Vc_IS_AMD64
        *reinterpret_cast<may_alias<ullong> *>(mem) =
            _pdep_u64(v.v(), 0x0101010101010101ULL);
#else
        *reinterpret_cast<may_alias<uint> *>(mem) =
            _pdep_u32(v.v(), 0x01010101U);
        *reinterpret_cast<may_alias<uint> *>(mem + 4) =
            _pdep_u32(v.v() >> 4, 0x01010101U);
#endif
    }
    template <class F>
    static constexpr void store(mask_member_type<16> v, bool *mem, F f, size_tag<16>) noexcept
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        _mm_store_si128(reinterpret_cast<__m128i *>(mem),
                        and_(one16(uchar()), _mm_movm_epi8(v.v())));
        unused(f);
#elif defined Vc_HAVE_AVX512DQ
        store16(_mm512_cvtepi32_epi8(_mm512_srli_epi32(_mm512_movm_epi32(v.v()), 31)),
                mem, f);
#elif defined Vc_IS_AMD64
        *reinterpret_cast<may_alias<ullong> *>(mem) =
            _pdep_u64(v.v(), 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 8) =
            _pdep_u64(v.v() >> 8, 0x0101010101010101ULL);
        unused(f);
#else
        execute_n_times<4>([&](auto i) {
            using namespace Vc::detail;
            constexpr uint offset = 4u * i;
            *reinterpret_cast<may_alias<uint> *>(mem + offset) =
                _pdep_u32(v.v() >> offset, 0x01010101U);
        });
        unused(f);
#endif
    }
    template <class F>
    static constexpr void store(mask_member_type<32> v, bool *mem, F, size_tag<32>) noexcept
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        _mm256_store_si256(reinterpret_cast<__m256i *>(mem),
                           and_(one32(uchar()), _mm256_movm_epi8(v.v())));
#elif defined Vc_HAVE_AVX512DQ
        store32(
            concat(_mm512_cvtepi32_epi8(_mm512_srli_epi32(_mm512_movm_epi32(v.v()), 31)),
                   _mm512_cvtepi32_epi8(
                       _mm512_srli_epi32(_mm512_movm_epi32(v.v() >> 16), 31))),
            mem, f);
#elif defined Vc_IS_AMD64
        *reinterpret_cast<may_alias<ullong> *>(mem) =
            _pdep_u64(v.v(), 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 8) =
            _pdep_u64(v.v() >> 8, 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 16) =
            _pdep_u64(v.v() >> 16, 0x0101010101010101ULL);
        *reinterpret_cast<may_alias<ullong> *>(mem + 24) =
            _pdep_u64(v.v() >> 24, 0x0101010101010101ULL);
#else
        execute_n_times<8>([&](auto i) {
            using namespace Vc::detail;
            constexpr uint offset = 4u * i;
            *reinterpret_cast<may_alias<uint> *>(mem + offset) =
                _pdep_u32(v.v() >> offset, 0x01010101U);
        });
#endif
    }
    template <class F>
    static constexpr void store(mask_member_type<64> v, bool *mem, F f,
                                size_tag<64>) noexcept
    {
#if defined Vc_HAVE_AVX512BW
        store64(and_(one64(uchar()), _mm512_movm_epi8(v.v())), mem, f);
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
#elif defined Vc_IS_AMD64
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
        unused(f);
#else
        execute_n_times<16>([&](auto i) {
            using namespace Vc::detail;
            constexpr uint offset = 4u * i;
            *reinterpret_cast<may_alias<uint> *>(mem + offset) =
                _pdep_u32(v.v() >> offset, 0x01010101U);
        });
        unused(f);
#endif
    }

    // masked store {{{2
#if defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL
    template <class F>
    static Vc_INTRINSIC void masked_store(mask_member_type<8> v, bool *mem, F,
                                          mask_member_type<8> k, size_tag<8>) noexcept
    {
        _mm_mask_storeu_epi8(mem, k.v(), and_(one16(uchar()), _mm_movm_epi8(v.v())));
    }

    template <class F>
    static Vc_INTRINSIC void masked_store(mask_member_type<16> v, bool *mem, F,
                                          mask_member_type<16> k, size_tag<16>) noexcept
    {
        _mm_mask_storeu_epi8(mem, k.v(), and_(one16(uchar()), _mm_movm_epi8(v.v())));
    }

    template <class F>
    static Vc_INTRINSIC void masked_store(mask_member_type<32> v, bool *mem, F,
                                          mask_member_type<32> k, size_tag<32>) noexcept
    {
        _mm256_mask_storeu_epi8(mem, k.v(), and_(one32(uchar()), _mm256_movm_epi8(v.v())));
    }

    template <class F>
    static Vc_INTRINSIC void masked_store(mask_member_type<64> v, bool *mem, F,
                                          mask_member_type<64> k, size_tag<64>) noexcept
    {
        _mm512_mask_storeu_epi8(mem, k.v(), and_(one64(uchar()), _mm512_movm_epi8(v.v())));
    }

#else   // defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL
    template <class F, size_t N>
    static Vc_INTRINSIC void masked_store(mask_member_type<N> v, bool *mem, F,
                                          mask_member_type<N> k, size_tag<N>) noexcept
    {
        for (std::size_t i = 0; i < N; ++i) {
            if (k[i]) {
                mem[i] = v.m(i);
            }
        }
    }
#endif  // defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL

    // negation {{{2
    template <class T, class SizeTag>
    static Vc_INTRINSIC T negate(const T &x, SizeTag) noexcept
    {
        return ~x.v();
    }

    // logical and bitwise operators {{{2
    template <class T>
    static Vc_INTRINSIC mask<T> logical_and(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, mask_member_type<size<T>()>(x.d & y.d)};
    }

    template <class T>
    static Vc_INTRINSIC mask<T> logical_or(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, mask_member_type<size<T>()>(x.d | y.d)};
    }

    template <class T>
    static Vc_INTRINSIC mask<T> bit_and(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, mask_member_type<size<T>()>(x.d & y.d)};
    }

    template <class T>
    static Vc_INTRINSIC mask<T> bit_or(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, mask_member_type<size<T>()>(x.d | y.d)};
    }

    template <class T>
    static Vc_INTRINSIC mask<T> bit_xor(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, mask_member_type<size<T>()>(x.d ^ y.d)};
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
    template <class T> static constexpr size_t size() { return M<T>::size(); }
};
// }}}1
}  // namespace detail

// where implementation {{{1
#define Vc_MASKED_CASSIGN_SPECIALIZATION(TYPE_, TYPE_SUFFIX_, OP_, OP_NAME_)             \
    template <>                                                                          \
    inline void Vc_VDECL masked_cassign<OP_, TYPE_, datapar_abi::avx512, 1>(             \
        mask<TYPE_, datapar_abi::avx512> k, datapar<TYPE_, datapar_abi::avx512> & lhs,   \
        const datapar<TYPE_, datapar_abi::avx512> &rhs)                                  \
    {                                                                                    \
        const auto kv = detail::data(k);                                                 \
        const auto lv = detail::data(lhs);                                               \
        const auto rv = detail::data(rhs);                                               \
        lhs = datapar<TYPE_, datapar_abi::avx512>(                                       \
            _mm512_mask_##OP_NAME_##_##TYPE_SUFFIX_(lv, kv, lv, rv));                    \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON

Vc_MASKED_CASSIGN_SPECIALIZATION(        double,  pd  , std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(         float,  ps  , std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: llong, epi64, std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail::ullong, epi64, std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(          long, epi64, std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: ulong, epi64, std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(           int, epi32, std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail::  uint, epi32, std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(         short, epi16, std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail::ushort, epi16, std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: schar, epi8 , std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: uchar, epi8 , std::plus, add);

Vc_MASKED_CASSIGN_SPECIALIZATION(        double,  pd  , std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(         float,  ps  , std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: llong, epi64, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail::ullong, epi64, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(          long, epi64, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: ulong, epi64, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(           int, epi32, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail::  uint, epi32, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(         short, epi16, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail::ushort, epi16, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: schar, epi8 , std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: uchar, epi8 , std::minus, sub);

// }}}1

// [mask.reductions] {{{
template <class T> Vc_ALWAYS_INLINE bool all_of(mask<T, datapar_abi::avx512> k)
{
    const auto v = detail::data(k);
    return detail::x86::testallset(v);
}

template <class T> Vc_ALWAYS_INLINE bool any_of(mask<T, datapar_abi::avx512> k)
{
    const auto v = detail::data(k);
    return v != 0U;
}

template <class T> Vc_ALWAYS_INLINE bool none_of(mask<T, datapar_abi::avx512> k)
{
    const auto v = detail::data(k);
    return v == 0U;
}

template <class T> Vc_ALWAYS_INLINE bool some_of(mask<T, datapar_abi::avx512> k)
{
    const auto v = detail::data(k);
    return v != 0 && !all_of(k);
}

template <class T> Vc_ALWAYS_INLINE int popcount(mask<T, datapar_abi::avx512> k)
{
    const auto v = detail::data(k);
    switch (k.size()) {
    case  8: return detail::popcnt8(v);
    case 16: return detail::popcnt16(v);
    case 32: return detail::popcnt32(v);
    case 64: return detail::popcnt64(v);
    default: Vc_UNREACHABLE();
    }
}

template <class T> Vc_ALWAYS_INLINE int find_first_set(mask<T, datapar_abi::avx512> k)
{
    const auto v = detail::data(k);
    return _tzcnt_u32(v);
}

Vc_ALWAYS_INLINE int find_first_set(mask<signed char, datapar_abi::avx512> k)
{
    const __mmask64 v = detail::data(k);
    return detail::firstbit(v);
}
Vc_ALWAYS_INLINE int find_first_set(mask<unsigned char, datapar_abi::avx512> k)
{
    const __mmask64 v = detail::data(k);
    return detail::firstbit(v);
}

template <class T> Vc_ALWAYS_INLINE int find_last_set(mask<T, datapar_abi::avx512> k)
{
    return 31 - _lzcnt_u32(detail::data(k));
}

Vc_ALWAYS_INLINE int find_last_set(mask<signed char, datapar_abi::avx512> k)
{
    const __mmask64 v = detail::data(k);
    return detail::lastbit(v);
}

Vc_ALWAYS_INLINE int find_last_set(mask<unsigned char, datapar_abi::avx512> k)
{
    const __mmask64 v = detail::data(k);
    return detail::lastbit(v);
}

Vc_VERSIONED_NAMESPACE_END
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
        return Vc::detail::data(x) == Vc::detail::data(y);
    }
};
// }}}1
}  // namespace std
#endif  // Vc_HAVE_AVX512_ABI

#endif  // Vc_HAVE_SSE
#endif  // VC_DATAPAR_AVX512_H_

// vim: foldmethod=marker
