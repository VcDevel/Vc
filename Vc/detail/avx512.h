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

#ifndef VC_SIMD_AVX512_H_
#define VC_SIMD_AVX512_H_

#include "macros.h"
#ifdef Vc_HAVE_AVX512_ABI
#include "avx.h"
#include "storage.h"
#include "simd_tuple.h"
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
// simd impl {{{1
struct avx512_simd_impl : public generic_simd_impl<avx512_simd_impl> {
    // member types {{{2
    using abi = simd_abi::Avx512;
    template <class T> static constexpr size_t size() { return simd_size_v<T, abi>; }
    template <class T> using simd_member_type = avx512_simd_member_type<T>;
    template <class T> using intrinsic_type = intrinsic_type_t<T, 64 / sizeof(T)>;
    template <class T> using mask_member_type = avx512_mask_member_type<T>;
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
    static Vc_INTRINSIC simd_member_type<T> load(const T *mem, F f, type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        return detail::load64(mem, f);
    }

    // convert from an AVX512 load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<simd_member_type<T>, simd_member_type<U>>(load64(mem, f));
    }

    // convert from an AVX load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U) * 2> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<simd_member_type<T>, avx_simd_member_type<U>>(load32(mem, f));
    }

    // convert from an SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U) * 4> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<simd_member_type<T>, sse_simd_member_type<U>>(load16(mem, f));
    }

    // convert from a half SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) == sizeof(U) * 8> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<simd_member_type<T>, sse_simd_member_type<U>>(load8(mem, f));
    }

    // convert from a 2-AVX512 load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) * 2 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<simd_member_type<T>, simd_member_type<U>>(
            load64(mem, f), load64(mem + size<U>(), f));
    }

    // convert from a 4-AVX512 load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) * 4 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<simd_member_type<T>, simd_member_type<U>>(
            load64(mem, f), load64(mem + size<U>(), f), load64(mem + 2 * size<U>(), f),
            load64(mem + 3 * size<U>(), f));
    }

    // convert from a 8-AVX512 load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const U *mem, F f, type_tag<T>,
        enable_if<sizeof(T) * 8 == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<simd_member_type<T>, simd_member_type<U>>(
            load64(mem, f), load64(mem + size<U>(), f), load64(mem + 2 * size<U>(), f),
            load64(mem + 3 * size<U>(), f), load64(mem + 4 * size<U>(), f),
            load64(mem + 5 * size<U>(), f), load64(mem + 6 * size<U>(), f),
            load64(mem + 7 * size<U>(), f));
    }

    // masked load {{{2
    template <class T, class U, class... Abis, size_t... Indexes>
    static Vc_INTRINSIC simd_member_type<T> convert_helper(
        const simd_tuple<U, Abis...> &uncvted, std::index_sequence<Indexes...>)
    {
        return x86::convert<simd_member_type<T>>(detail::get<Indexes>(uncvted)...);
    }
    template <class T, class U, class F>
    static Vc_INTRINSIC void masked_load(simd_member_type<T> &merge, mask_member_type<T> k,
                                         const U *mem, F) Vc_NOEXCEPT_OR_IN_TEST
    {
        if constexpr(sizeof(U) > 8) {
            // no SIMD support, use a scalar loop
            bit_iteration(k.d, [&](auto i) { merge.set(i, static_cast<T>(mem[i])); });
        } else {
            static_assert(!std::is_same<T, U>::value, "");
            using fixed_traits = detail::traits<U, simd_abi::fixed_size<size<T>()>>;
            using fixed_impl = typename fixed_traits::simd_impl_type;
            typename fixed_traits::simd_member_type uncvted{};
            fixed_impl::masked_load(uncvted, static_cast<ullong>(k), mem, F());
            masked_assign(k, merge,
                          convert_helper<T>(
                              uncvted, std::make_index_sequence<uncvted.tuple_size>()));
        }
    }

    // non-converting masked loads
    template <class T, class F>
    static Vc_INTRINSIC void masked_load(simd_member_type<T> &merge, mask_member_type<T> k,
                                         const T *mem, F) Vc_NOEXCEPT_OR_IN_TEST
    {
        if constexpr (have_avx512bw && sizeof(T) == 1) {
            merge = _mm512_mask_loadu_epi8(merge, k, mem);
        } else if constexpr (have_avx512bw && sizeof(T) == 2) {
            merge = _mm512_mask_loadu_epi16(merge, k, mem);
        } else if constexpr (sizeof(T) == 4 && std::is_integral_v<T>) {
            merge = _mm512_mask_loadu_epi32(merge, k, mem);
        } else if constexpr (sizeof(T) == 4 && std::is_floating_point_v<T>) {
            merge = _mm512_mask_loadu_ps(merge, k, mem);
        } else if constexpr (sizeof(T) == 8 && std::is_integral_v<T>) {
            merge = _mm512_mask_loadu_epi64(merge, k, mem);
        } else if constexpr (sizeof(T) == 8 && std::is_floating_point_v<T>) {
            merge = _mm512_mask_loadu_pd(merge, k, mem);
        } else {
            execute_n_times<size<T>()>([&](auto i) {
                if (k[i]) {
                    merge.set(i, static_cast<T>(mem[i]));
                }
            });
        }
    }

    // store {{{2
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(simd_member_type<T> v, U *mem, F,
                                   type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        if constexpr (std::is_same_v<T, U>) {
            store64(v, mem, F());
        } else if constexpr (sizeof(U) <= 8) {  // make sure to skip long double
            if constexpr (sizeof(T) == sizeof(U) * 8) {
                store8(convert<sse_simd_member_type<U>>(v), mem, F());
            } else if constexpr (sizeof(T) == sizeof(U) * 4) {
                store16(convert<sse_simd_member_type<U>>(v), mem, F());
            } else if constexpr (sizeof(T) == sizeof(U) * 2) {
                store32(convert<avx_simd_member_type<U>>(v), mem, F());
            } else if constexpr (sizeof(T) == sizeof(U)) {
                store64(convert<simd_member_type<U>>(v), mem, F());
            } else if constexpr (sizeof(T) * 2 == sizeof(U)) {
                store64(convert<simd_member_type<U>>(lo256(v)), mem, F());
                store64(convert<simd_member_type<U>>(hi256(v)), mem + size<U>(), F());
            } else if constexpr (sizeof(T) * 4 == sizeof(U)) {
                store64(convert<simd_member_type<U>>(lo128(v)), mem, F());
                store64(convert<simd_member_type<U>>(extract128<1>(v)), mem + size<U>(),
                        F());
                store64(convert<simd_member_type<U>>(extract128<2>(v)),
                        mem + 2 * size<U>(), F());
                store64(convert<simd_member_type<U>>(extract128<3>(v)),
                        mem + 3 * size<U>(), F());
            } else if constexpr (sizeof(T) * 8 == sizeof(U)) {
                const std::array<simd_member_type<U>, 8> converted = x86::convert_all<simd_member_type<U>>(v);
                store64(converted[0], mem + 0 * size<U>(), F());
                store64(converted[1], mem + 1 * size<U>(), F());
                store64(converted[2], mem + 2 * size<U>(), F());
                store64(converted[3], mem + 3 * size<U>(), F());
                store64(converted[4], mem + 4 * size<U>(), F());
                store64(converted[5], mem + 5 * size<U>(), F());
                store64(converted[6], mem + 6 * size<U>(), F());
                store64(converted[7], mem + 7 * size<U>(), F());
            } else {
                static_assert(!std::is_same_v<T, T>, "this should be unreachable");
            }
        } else {
            execute_n_times<size<T>()>([&](auto i) { mem[i] = v[i]; });
        }
    }

    // masked store {{{2
    template <class T, class F>
    static Vc_INTRINSIC void masked_store(simd_member_type<T> v, long double *mem, F,
                                          mask_member_type<T> k) Vc_NOEXCEPT_OR_IN_TEST
    {
        // no support for long double
        execute_n_times<size<T>()>([&](auto i) {
            if (k[i]) {
                mem[i] = v[i];
            }
        });
    }

    template <class T, class U, class F>
    static Vc_INTRINSIC void masked_store(simd_member_type<T> v, U *mem, F,
                                          mask_member_type<T> k) Vc_NOEXCEPT_OR_IN_TEST
    {
        constexpr bool truncate = std::is_integral_v<T> && std::is_integral_v<U> && sizeof(T) > sizeof(U);
        using V = simd_member_type<U>;
        using M = mask_member_type<U>;

        if constexpr (std::is_same_v<T, U> ||
                      (std::is_integral_v<T> && std::is_integral_v<U> &&
                       sizeof(T) == sizeof(U))) {
            // bitwise or no conversion, reinterpret:
            x86::maskstore(storage_bitcast<U>(v), mem, F(), k);
        } else if constexpr(truncate && sizeof(T) == 8) {
            if constexpr (sizeof(U) == 4) {
                _mm512_mask_cvtepi64_storeu_epi32(mem, k, v);
            } else if constexpr (sizeof(U) == 2) {
                _mm512_mask_cvtepi64_storeu_epi16(mem, k, v);
            } else if constexpr (sizeof(U) == 1) {
                _mm512_mask_cvtepi64_storeu_epi8(mem, k, v);
            }
        } else if constexpr (truncate && sizeof(T) == 4) {
            if constexpr (sizeof(U) == 2) {
                _mm512_mask_cvtepi32_storeu_epi16(mem, k, v);
            } else if constexpr (sizeof(U) == 1) {
                _mm512_mask_cvtepi32_storeu_epi8(mem, k, v);
            }
        } else if constexpr (truncate && have_avx512bw && sizeof(T) == 2) {
            _mm512_mask_cvtepi16_storeu_epi8(mem, k, v);
        } else if constexpr (sizeof(T) == sizeof(U)) {
            x86::maskstore(convert<V>(v), mem, F(), M(k.d));
        } else if constexpr (sizeof(T) * 2 == sizeof(U)) {
            const std::array<V, 2> converted = convert_all<V>(v);
            x86::maskstore(converted[0], mem, F(), M(k >> 0));
            x86::maskstore(converted[1], mem + V::width, F(), M(k >> V::width));
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
            x86::maskstore(convert<V>(v), mem, F(), k.d);
        } else {
            static_assert(!std::is_same_v<T, T>, "this should be unreachable");
            execute_n_times<size<T>()>([&](auto i) {
                if (k[i]) {
                    mem[i] = static_cast<T>(v[i]);
                }
            });
        }
    }

    // negation {{{2
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> negate(simd_member_type<T> x) noexcept
    {
        return equal_to(x, simd_member_type<T>());
    }

    // reductions {{{2
    template <class T, class BinaryOperation, size_t N>
    static Vc_INTRINSIC T Vc_VDECL reduce(size_tag<N>, simd<T> x,
                                          BinaryOperation &binary_op)
    {
        using V = Vc::simd<T, simd_abi::Avx>;
        return avx_simd_impl::reduce(size_tag<N / 2>(),
                                     binary_op(V(detail::private_init, lo256(data(x))),
                                               V(detail::private_init, hi256(data(x)))),
                                     binary_op);
    }

    // compares {{{2
    // we cannot use built in compares since they return __m512i instead of real bitmasks
    static Vc_INTRINSIC mask_member_type<double> equal_to    (simd_member_type<double> x, simd_member_type<double> y) { return _mm512_cmp_pd_mask(x, y, _MM_CMPINT_EQ); }
    static Vc_INTRINSIC mask_member_type<double> not_equal_to(simd_member_type<double> x, simd_member_type<double> y) { return _mm512_cmp_pd_mask(x, y, _MM_CMPINT_NE); }
    static Vc_INTRINSIC mask_member_type<double> less        (simd_member_type<double> x, simd_member_type<double> y) { return _mm512_cmp_pd_mask(x, y, _MM_CMPINT_LT); }
    static Vc_INTRINSIC mask_member_type<double> less_equal  (simd_member_type<double> x, simd_member_type<double> y) { return _mm512_cmp_pd_mask(x, y, _MM_CMPINT_LE); }
    static Vc_INTRINSIC mask_member_type< float> equal_to    (simd_member_type< float> x, simd_member_type< float> y) { return _mm512_cmp_ps_mask(x, y, _MM_CMPINT_EQ); }
    static Vc_INTRINSIC mask_member_type< float> not_equal_to(simd_member_type< float> x, simd_member_type< float> y) { return _mm512_cmp_ps_mask(x, y, _MM_CMPINT_NE); }
    static Vc_INTRINSIC mask_member_type< float> less        (simd_member_type< float> x, simd_member_type< float> y) { return _mm512_cmp_ps_mask(x, y, _MM_CMPINT_LT); }
    static Vc_INTRINSIC mask_member_type< float> less_equal  (simd_member_type< float> x, simd_member_type< float> y) { return _mm512_cmp_ps_mask(x, y, _MM_CMPINT_LE); }

    static Vc_INTRINSIC mask_member_type< llong> equal_to(simd_member_type< llong> x, simd_member_type< llong> y) { return _mm512_cmpeq_epi64_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<ullong> equal_to(simd_member_type<ullong> x, simd_member_type<ullong> y) { return _mm512_cmpeq_epi64_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<  long> equal_to(simd_member_type<  long> x, simd_member_type<  long> y) { return detail::cmpeq_long_mask(x, y); }
    static Vc_INTRINSIC mask_member_type< ulong> equal_to(simd_member_type< ulong> x, simd_member_type< ulong> y) { return detail::cmpeq_long_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<   int> equal_to(simd_member_type<   int> x, simd_member_type<   int> y) { return _mm512_cmpeq_epi32_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<  uint> equal_to(simd_member_type<  uint> x, simd_member_type<  uint> y) { return _mm512_cmpeq_epi32_mask(x, y); }

    static Vc_INTRINSIC mask_member_type< llong> not_equal_to(simd_member_type< llong> x, simd_member_type< llong> y) { return ~_mm512_cmpeq_epi64_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<ullong> not_equal_to(simd_member_type<ullong> x, simd_member_type<ullong> y) { return ~_mm512_cmpeq_epi64_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<  long> not_equal_to(simd_member_type<  long> x, simd_member_type<  long> y) { return ~detail::cmpeq_long_mask(x, y); }
    static Vc_INTRINSIC mask_member_type< ulong> not_equal_to(simd_member_type< ulong> x, simd_member_type< ulong> y) { return ~detail::cmpeq_long_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<   int> not_equal_to(simd_member_type<   int> x, simd_member_type<   int> y) { return ~_mm512_cmpeq_epi32_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<  uint> not_equal_to(simd_member_type<  uint> x, simd_member_type<  uint> y) { return ~_mm512_cmpeq_epi32_mask(x, y); }

    static Vc_INTRINSIC mask_member_type< llong> less(simd_member_type< llong> x, simd_member_type< llong> y) { return _mm512_cmplt_epi64_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<ullong> less(simd_member_type<ullong> x, simd_member_type<ullong> y) { return _mm512_cmplt_epu64_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<  long> less(simd_member_type<  long> x, simd_member_type<  long> y) { return detail::cmplt_long_mask(x, y); }
    static Vc_INTRINSIC mask_member_type< ulong> less(simd_member_type< ulong> x, simd_member_type< ulong> y) { return detail::cmplt_ulong_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<   int> less(simd_member_type<   int> x, simd_member_type<   int> y) { return _mm512_cmplt_epi32_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<  uint> less(simd_member_type<  uint> x, simd_member_type<  uint> y) { return _mm512_cmplt_epu32_mask(x, y); }

    static Vc_INTRINSIC mask_member_type< llong> less_equal(simd_member_type< llong> x, simd_member_type< llong> y) { return _mm512_cmple_epi64_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<ullong> less_equal(simd_member_type<ullong> x, simd_member_type<ullong> y) { return _mm512_cmple_epu64_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<  long> less_equal(simd_member_type<  long> x, simd_member_type<  long> y) { return detail::cmple_long_mask(x, y); }
    static Vc_INTRINSIC mask_member_type< ulong> less_equal(simd_member_type< ulong> x, simd_member_type< ulong> y) { return detail::cmple_ulong_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<   int> less_equal(simd_member_type<   int> x, simd_member_type<   int> y) { return _mm512_cmple_epi32_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<  uint> less_equal(simd_member_type<  uint> x, simd_member_type<  uint> y) { return _mm512_cmple_epu32_mask(x, y); }

#ifdef Vc_HAVE_FULL_AVX512_ABI
    static_assert(std::is_signed_v<char>);
    static Vc_INTRINSIC mask_member_type< short> equal_to(simd_member_type< short> x, simd_member_type< short> y) { return _mm512_cmpeq_epi16_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<ushort> equal_to(simd_member_type<ushort> x, simd_member_type<ushort> y) { return _mm512_cmpeq_epi16_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<  char> equal_to(simd_member_type<  char> x, simd_member_type<  char> y) { return _mm512_cmpeq_epi8_mask(x, y); }
    static Vc_INTRINSIC mask_member_type< schar> equal_to(simd_member_type< schar> x, simd_member_type< schar> y) { return _mm512_cmpeq_epi8_mask(x, y); }
    static Vc_INTRINSIC mask_member_type< uchar> equal_to(simd_member_type< uchar> x, simd_member_type< uchar> y) { return _mm512_cmpeq_epi8_mask(x, y); }

    static Vc_INTRINSIC mask_member_type< short> not_equal_to(simd_member_type< short> x, simd_member_type< short> y) { return ~_mm512_cmpeq_epi16_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<ushort> not_equal_to(simd_member_type<ushort> x, simd_member_type<ushort> y) { return ~_mm512_cmpeq_epi16_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<  char> not_equal_to(simd_member_type<  char> x, simd_member_type<  char> y) { return ~_mm512_cmpeq_epi8_mask(x, y); }
    static Vc_INTRINSIC mask_member_type< schar> not_equal_to(simd_member_type< schar> x, simd_member_type< schar> y) { return ~_mm512_cmpeq_epi8_mask(x, y); }
    static Vc_INTRINSIC mask_member_type< uchar> not_equal_to(simd_member_type< uchar> x, simd_member_type< uchar> y) { return ~_mm512_cmpeq_epi8_mask(x, y); }

    static Vc_INTRINSIC mask_member_type< short> less(simd_member_type< short> x, simd_member_type< short> y) { return _mm512_cmplt_epi16_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<ushort> less(simd_member_type<ushort> x, simd_member_type<ushort> y) { return _mm512_cmplt_epu16_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<  char> less(simd_member_type<  char> x, simd_member_type<  char> y) { return _mm512_cmplt_epi8_mask (x, y); }
    static Vc_INTRINSIC mask_member_type< schar> less(simd_member_type< schar> x, simd_member_type< schar> y) { return _mm512_cmplt_epi8_mask (x, y); }
    static Vc_INTRINSIC mask_member_type< uchar> less(simd_member_type< uchar> x, simd_member_type< uchar> y) { return _mm512_cmplt_epu8_mask (x, y); }

    static Vc_INTRINSIC mask_member_type< short> less_equal(simd_member_type< short> x, simd_member_type< short> y) { return _mm512_cmple_epi16_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<ushort> less_equal(simd_member_type<ushort> x, simd_member_type<ushort> y) { return _mm512_cmple_epu16_mask(x, y); }
    static Vc_INTRINSIC mask_member_type<  char> less_equal(simd_member_type<  char> x, simd_member_type<  char> y) { return _mm512_cmple_epi8_mask (x, y); }
    static Vc_INTRINSIC mask_member_type< schar> less_equal(simd_member_type< schar> x, simd_member_type< schar> y) { return _mm512_cmple_epi8_mask (x, y); }
    static Vc_INTRINSIC mask_member_type< uchar> less_equal(simd_member_type< uchar> x, simd_member_type< uchar> y) { return _mm512_cmple_epu8_mask (x, y); }
#endif  // Vc_HAVE_FULL_AVX512_ABI

    // math {{{2
    // sqrt {{{3
    static Vc_INTRINSIC simd_member_type<float> sqrt(simd_member_type<float> x)
    {
        return _mm512_sqrt_ps(x);
    }
    static Vc_INTRINSIC simd_member_type<double> sqrt(simd_member_type<double> x)
    {
        return _mm512_sqrt_pd(x);
    }

    // logb {{{3
    static Vc_INTRINSIC Vc_CONST simd_member_type<float> logb_positive(simd_member_type<float> v)
    {
        return _mm512_getexp_ps(v);
    }
    static Vc_INTRINSIC Vc_CONST simd_member_type<double> logb_positive(simd_member_type<double> v)
    {
        return _mm512_getexp_pd(v);
    }

    static Vc_INTRINSIC Vc_CONST simd_member_type<float> logb(simd_member_type<float> v)
    {
        return _mm512_fixupimm_ps(logb_positive(abs(v)), v, broadcast64(0x00550433),
                                  0x00);
    }
    static Vc_INTRINSIC Vc_CONST simd_member_type<double> logb(simd_member_type<double> v)
    {
        return _mm512_fixupimm_pd(logb_positive(abs(v)), v, broadcast64(0x00550433),
                                  0x00);
    }

    // trunc {{{3
    static Vc_INTRINSIC simd_member_type<float> trunc(simd_member_type<float> x)
    {
        return _mm512_roundscale_round_ps(x, 0x03, _MM_FROUND_CUR_DIRECTION);
    }
    static Vc_INTRINSIC simd_member_type<double> trunc(simd_member_type<double> x)
    {
        return _mm512_roundscale_round_pd(x, 0x03, _MM_FROUND_CUR_DIRECTION);
    }

    // floor {{{3
    static Vc_INTRINSIC simd_member_type<float> floor(simd_member_type<float> x)
    {
        return _mm512_roundscale_round_ps(x, 0x01, _MM_FROUND_CUR_DIRECTION);
    }
    static Vc_INTRINSIC simd_member_type<double> floor(simd_member_type<double> x)
    {
        return _mm512_roundscale_round_pd(x, 0x01, _MM_FROUND_CUR_DIRECTION);
    }

    // ceil {{{3
    static Vc_INTRINSIC simd_member_type<float> ceil(simd_member_type<float> x)
    {
        return _mm512_roundscale_round_ps(x, 0x02, _MM_FROUND_CUR_DIRECTION);
    }
    static Vc_INTRINSIC simd_member_type<double> ceil(simd_member_type<double> x)
    {
        return _mm512_roundscale_round_pd(x, 0x02, _MM_FROUND_CUR_DIRECTION);
    }

    // frexp {{{3
    /**
     * splits \p v into exponent and mantissa, the sign is kept with the mantissa
     *
     * The return value will be in the range [0.5, 1.0[
     * The \p e value will be an integer defining the power-of-two exponent
     */
    static inline simd_member_type<double> frexp(simd_member_type<double> v,
                                                 avx_simd_member_type<int> &exp)
    {
        if (Vc_IS_LIKELY(detail::testallset(isnonzerovalue(v)))) {
            exp = _mm256_add_epi32(broadcast32(1),
                                   _mm512_cvttpd_epi32(_mm512_getexp_pd(v)));
            return _mm512_getmant_pd(v, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
        }
        exp = _mm256_add_epi32(
            broadcast32(1), _mm512_mask_cvttpd_epi32(broadcast32(-1), isnonzerovalue(v),
                                                     _mm512_getexp_pd(v)));
        return _mm512_mask_getmant_pd(v, isnonzerovalue(v), v, _MM_MANT_NORM_p5_1,
                                      _MM_MANT_SIGN_src);
    }
    static Vc_ALWAYS_INLINE simd_member_type<double> frexp(
        simd_member_type<double> v, simd_tuple<int, simd_abi::Avx> &exp)
    {
        return frexp(v, exp.first);
    }

    static inline simd_member_type<float> frexp(simd_member_type<float> v,
                                                simd_member_type<int> &exp)
    {
        if (Vc_IS_LIKELY(detail::testallset(isnonzerovalue(v)))) {
            exp = _mm512_add_epi32(broadcast64(1),
                                   _mm512_cvttps_epi32(_mm512_getexp_ps(v)));
            return _mm512_getmant_ps(v, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
        }
        exp = _mm512_mask_add_epi32(_mm512_setzero_si512(), isnonzerovalue(v), broadcast64(1),
                                    _mm512_cvttps_epi32(_mm512_getexp_ps(v)));
        return _mm512_mask_getmant_ps(v, isnonzerovalue(v), v, _MM_MANT_NORM_p5_1,
                                      _MM_MANT_SIGN_src);
    }
    static Vc_ALWAYS_INLINE simd_member_type<float> frexp(
        simd_member_type<float> v, simd_tuple<int, simd_abi::Avx512> &exp)
    {
        return frexp(v, exp.first);
    }

    // isfinite {{{3
    static Vc_INTRINSIC mask_member_type<float> isfinite(simd_member_type<float> x)
    {
        return _mm512_cmp_ps_mask(x, _mm512_mul_ps(_mm512_setzero_ps(), x), _CMP_ORD_Q);
    }
    static Vc_INTRINSIC mask_member_type<double> isfinite(simd_member_type<double> x)
    {
        return _mm512_cmp_pd_mask(x, _mm512_mul_pd(_mm512_setzero_pd(), x), _CMP_ORD_Q);
    }

    // isinf {{{3
    static Vc_INTRINSIC mask_member_type<float> isinf(simd_member_type<float> x)
    {
#if defined Vc_HAVE_AVX512DQ
        return _mm512_fpclass_ps_mask(x, 0x08) | _mm512_fpclass_ps_mask(x, 0x10);
#else
        return _mm512_cmp_epi32_mask(
            _mm512_castps_si512(abs(x)), broadcast64(0x7f800000u), _CMP_EQ_OQ);
#endif
    }
    static Vc_INTRINSIC mask_member_type<double> isinf(simd_member_type<double> x)
    {
#if defined Vc_HAVE_AVX512DQ
        return _mm512_fpclass_pd_mask(x, 0x08) | _mm512_fpclass_pd_mask(x, 0x10);
#else
        return _mm512_cmp_epi64_mask(_mm512_castpd_si512(abs(x)),
                                     broadcast64(0x7ff0000000000000ull), _CMP_EQ_OQ);
#endif
    }

    // isnan {{{3
    static Vc_INTRINSIC mask_member_type<float> isnan(simd_member_type<float> x)
    {
        return _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
    }
    static Vc_INTRINSIC mask_member_type<double> isnan(simd_member_type<double> x)
    {
        return _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);
    }

    // isnonzerovalue (isnormal | is subnormal == !isinf & !isnan & !is zero) {{{3
    static Vc_INTRINSIC mask_member_type<float> isnonzerovalue(simd_member_type<float> x)
    {
        return _mm512_cmp_ps_mask(
            _mm512_mul_ps(broadcast64(std::numeric_limits<float>::infinity()),
                          x),                                    // NaN if x == 0 / NaN
            _mm512_mul_ps(_mm512_setzero_ps(), x), _CMP_ORD_Q);  // NaN if x == inf / NaN
    }
    static Vc_INTRINSIC mask_member_type<double> isnonzerovalue(
        simd_member_type<double> x)
    {
        return _mm512_cmp_pd_mask(
            _mm512_mul_pd(broadcast64(std::numeric_limits<double>::infinity()),
                          x),                                    // NaN if x == 0
            _mm512_mul_pd(_mm512_setzero_pd(), x), _CMP_ORD_Q);  // NaN if x == inf
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
        return isnonzerovalue(
            z_f32(and_(x, intrin_cast<__m512>(broadcast64(0x7f800000u)))));
    }
    static Vc_INTRINSIC mask_member_type<double> isnormal(simd_member_type<double> x)
    {
        return isnonzerovalue(
            z_f64(and_(x, intrin_cast<__m512d>(broadcast64(0x7ff0'0000'0000'0000ull)))));
    }

    // signbit {{{3
    static Vc_INTRINSIC mask_member_type<float> signbit(simd_member_type<float> x)
    {
#ifdef Vc_HAVE_AVX512DQ
        return _mm512_movepi32_mask(_mm512_castps_si512(x));
#else
        const auto signbit = broadcast64(0x80000000u);
        return _mm512_cmpeq_epi32_mask(and_(to_m512i(x), signbit), signbit);
#endif
    }
    static Vc_INTRINSIC mask_member_type<double> signbit(simd_member_type<double> x)
    {
#ifdef Vc_HAVE_AVX512DQ
        return _mm512_movepi64_mask(_mm512_castpd_si512(x));
#else
        const auto signbit = broadcast64(0x8000000000000000ull);
        return _mm512_cmpeq_epi64_mask(and_(to_m512i(x), signbit), signbit);
#endif
    }

    // isunordered {{{3
    static Vc_INTRINSIC mask_member_type<float> isunordered(simd_member_type<float> x,
                                                            simd_member_type<float> y)
    {
        return _mm512_cmp_ps_mask(x, y, _CMP_UNORD_Q);
    }
    static Vc_INTRINSIC mask_member_type<double> isunordered(simd_member_type<double> x,
                                                             simd_member_type<double> y)
    {
        return _mm512_cmp_pd_mask(x, y, _CMP_UNORD_Q);
    }

    // fpclassify {{{3
    static Vc_INTRINSIC simd_tuple<int, simd_abi::Avx512> fpclassify(
        simd_member_type<float> x)
    {
        auto &&b = [](int y) { return broadcast64(y); };
        return {_mm512_mask_mov_epi32(
            _mm512_mask_mov_epi32(
                _mm512_mask_mov_epi32(b(FP_NORMAL), isnan(x), b(FP_NAN)), isinf(x),
                b(FP_INFINITE)),
            _mm512_cmp_ps_mask(abs(x), broadcast64(std::numeric_limits<float>::min()),
                               _CMP_LT_OS),
            _mm512_mask_mov_epi32(b(FP_SUBNORMAL),
                                  _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_EQ_OQ),
                                  b(FP_ZERO)))};
    }
    static Vc_INTRINSIC simd_tuple<int, simd_abi::Avx> fpclassify(
        simd_member_type<double> x)
    {
#ifdef Vc_HAVE_AVX512VL
        auto &&b = [](int y) { return broadcast32(y); };
        return {_mm256_mask_mov_epi32(
            _mm256_mask_mov_epi32(
                _mm256_mask_mov_epi32(b(FP_NORMAL), isnan(x), b(FP_NAN)), isinf(x),
                b(FP_INFINITE)),
            _mm512_cmp_pd_mask(abs(x), broadcast64(std::numeric_limits<double>::min()),
                               _CMP_LT_OS),
            _mm256_mask_mov_epi32(b(FP_SUBNORMAL),
                                  _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                                  b(FP_ZERO)))};
#else   // Vc_HAVE_AVX512VL
        auto &&b = [](int y) { return broadcast64(y); };
        return {lo256(_mm512_mask_mov_epi32(
            _mm512_mask_mov_epi32(
                _mm512_mask_mov_epi32(b(FP_NORMAL), isnan(x), b(FP_NAN)), isinf(x),
                b(FP_INFINITE)),
            _mm512_cmp_pd_mask(abs(x), broadcast64(std::numeric_limits<double>::min()),
                               _CMP_LT_OS),
            _mm512_mask_mov_epi32(b(FP_SUBNORMAL),
                                  _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                                  b(FP_ZERO))))};
#endif  // Vc_HAVE_AVX512VL
    }

    // smart_reference access {{{2
    template <class T> static Vc_INTRINSIC T get(simd_member_type<T> v, int i) noexcept
    {
        return v[i];
    }
    template <class T, class U>
    static Vc_INTRINSIC void set(simd_member_type<T> &v, int i, U &&x) noexcept
    {
        v.set(i, std::forward<U>(x));
    }
    // }}}2
};

// simd_mask impl {{{1
struct avx512_mask_impl
    : public generic_mask_impl<simd_abi::Avx512, avx512_mask_member_type> {
    // member types {{{2
    using abi = simd_abi::Avx512;
    template <class T> static constexpr size_t size() { return simd_size_v<T, abi>; }
    template <size_t N> using mask_member_type = avx512_mask_member_type_n<N>;
    template <class T> using simd_mask = Vc::simd_mask<T, abi>;
    template <class T> using mask_bool = MaskBool<sizeof(T)>;
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;

    // to_bitset {{{2
    template <size_t N>
    static Vc_INTRINSIC std::bitset<N> to_bitset(mask_member_type<N> v) noexcept
    {
        return v.intrin();
    }

    // from_bitset{{{2
    template <size_t N, class T>
    static Vc_INTRINSIC mask_member_type<N> from_bitset(std::bitset<N> bits, type_tag<T>)
    {
        return bits.to_ullong();
    }

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
                                         mask_member_type<8> mask, const bool *mem,
                                         F) noexcept
    {
        const auto a = _mm_mask_loadu_epi8(zero<__m128i>(), mask, mem);
        merge = (merge & ~mask) | _mm_test_epi8_mask(a, a);
    }

    template <class F>
    static Vc_INTRINSIC void masked_load(mask_member_type<16> &merge,
                                         mask_member_type<16> mask, const bool *mem,
                                         F) noexcept
    {
        const auto a = _mm_mask_loadu_epi8(zero<__m128i>(), mask, mem);
        merge = (merge & ~mask) | _mm_test_epi8_mask(a, a);
    }

    template <class F>
    static Vc_INTRINSIC void masked_load(mask_member_type<32> &merge,
                                         mask_member_type<32> mask, const bool *mem,
                                         F) noexcept
    {
        const auto a = _mm256_mask_loadu_epi8(zero<__m256i>(), mask, mem);
        merge = (merge & ~mask) | _mm256_test_epi8_mask(a, a);
    }

    template <class F>
    static Vc_INTRINSIC void masked_load(mask_member_type<64> &merge,
                                         mask_member_type<64> mask, const bool *mem,
                                         F) noexcept
    {
        const auto a = _mm512_mask_loadu_epi8(zero<__m512i>(), mask, mem);
        merge = (merge & ~mask) | _mm512_test_epi8_mask(a, a);
    }

#else
    template <size_t N, class F>
    static Vc_INTRINSIC void masked_load(mask_member_type<N> &merge,
                                         const mask_member_type<N> mask, const bool *mem,
                                         F) noexcept
    {
        detail::execute_n_times<N>([&](auto i) {
            if (mask[i]) {
                merge.set(i, mem[i]);
            }
        });
    }
#endif

    // store {{{2
    template <class F>
    static Vc_INTRINSIC void store(mask_member_type<8> v, bool *mem, F f,
                                   size_tag<8>) noexcept
    {
        x86::store8(
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
            _mm_maskz_set1_epi8(v, 1),
#elif defined __x86_64__
            make_storage<ullong>(_pdep_u64(v, 0x0101010101010101ULL), 0ull),
#else
            make_storage<uint>(_pdep_u32(v, 0x01010101U), _pdep_u32(v >> 4, 0x01010101U)),
#endif
            mem, f);
    }
    template <class F>
    static Vc_INTRINSIC void store(mask_member_type<16> v, bool *mem, F f,
                                   size_tag<16>) noexcept
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        x86::store16(_mm_maskz_set1_epi8(v, 1), mem, f);
#else
        _mm512_mask_cvtepi32_storeu_epi8(mem, ~__mmask16(),
                                         _mm512_maskz_set1_epi32(v, 1));
        unused(f);
#endif
    }
#ifdef Vc_HAVE_AVX512BW
    template <class F>
    static Vc_INTRINSIC void store(mask_member_type<32> v, bool *mem, F f,
                                   size_tag<32>) noexcept
    {
#if defined Vc_HAVE_AVX512VL
        x86::store32(_mm256_maskz_set1_epi8(v, 1), mem, f);
#else
        x86::store32(lo256(_mm512_maskz_set1_epi8(v, 1)), mem, f);
#endif
    }
    template <class F>
    static Vc_INTRINSIC void store(mask_member_type<64> v, bool *mem, F f,
                                   size_tag<64>) noexcept
    {
        x86::store64(_mm512_maskz_set1_epi8(v, 1), mem, f);
    }
#endif  // Vc_HAVE_AVX512BW

    // masked store {{{2
    template <class F>
    static Vc_INTRINSIC void masked_store(mask_member_type<8> v, bool *mem, F,
                                          mask_member_type<8> k) noexcept
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        _mm_mask_cvtepi16_storeu_epi8(mem, k, _mm_maskz_set1_epi16(v, 1));
#elif defined Vc_HAVE_AVX512VL
        _mm256_mask_cvtepi32_storeu_epi8(mem, k, _mm256_maskz_set1_epi32(v, 1));
#else
        // we rely on k < 0x100:
        _mm512_mask_cvtepi32_storeu_epi8(mem, k, _mm512_maskz_set1_epi32(v, 1));
#endif
    }

    template <class F>
    static Vc_INTRINSIC void masked_store(mask_member_type<16> v, bool *mem, F,
                                          mask_member_type<16> k) noexcept
    {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
        _mm_mask_storeu_epi8(mem, k, _mm_maskz_set1_epi8(v, 1));
#else
        _mm512_mask_cvtepi32_storeu_epi8(mem, k, _mm512_maskz_set1_epi32(v, 1));
#endif
    }

#ifdef Vc_HAVE_AVX512BW
    template <class F>
    static Vc_INTRINSIC void masked_store(mask_member_type<32> v, bool *mem, F,
                                          mask_member_type<32> k) noexcept
    {
#if defined Vc_HAVE_AVX512VL
        _mm256_mask_storeu_epi8(mem, k, _mm256_maskz_set1_epi8(v, 1));
#else
        _mm256_mask_storeu_epi8(mem, k, lo256(_mm512_maskz_set1_epi8(v, 1)));
#endif
    }

    template <class F>
    static Vc_INTRINSIC void masked_store(mask_member_type<64> v, bool *mem, F,
                                          mask_member_type<64> k) noexcept
    {
        _mm512_mask_storeu_epi8(mem, k, _mm512_maskz_set1_epi8(v, 1));
    }
#endif  // Vc_HAVE_AVX512BW

    // negation {{{2
    template <class T, class SizeTag>
    static Vc_INTRINSIC T negate(const T &x, SizeTag) noexcept
    {
        return ~x.d;
    }

    // logical and bitwise operators {{{2
    template <class T>
    static Vc_INTRINSIC simd_mask<T> logical_and(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, mask_member_type<size<T>()>(x.d & y.d)};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> logical_or(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, mask_member_type<size<T>()>(x.d | y.d)};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> bit_and(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, mask_member_type<size<T>()>(x.d & y.d)};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> bit_or(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, mask_member_type<size<T>()>(x.d | y.d)};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> bit_xor(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, mask_member_type<size<T>()>(x.d ^ y.d)};
    }

    // smart_reference access {{{2
    template <size_t N>
    static Vc_INTRINSIC bool get(mask_member_type<N> k, int i) noexcept
    {
        return k[i];
    }
    template <size_t N>
    static Vc_INTRINSIC void set(mask_member_type<N> &k, int i, bool x) noexcept
    {
        k.set(i, x);
    }
    // }}}2
};

// simd_converter Avx512 -> scalar {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::Avx512, To, simd_abi::scalar> {
    using Arg = avx512_simd_member_type<From>;

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
// simd_converter scalar -> Avx512 {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::scalar, To, simd_abi::Avx512> {
    using R = avx512_simd_member_type<To>;

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
                              From x12, From x13, From x14, From x15, From x16, From x17,
                              From x18, From x19, From x20, From x21, From x22, From x23,
                              From x24, From x25, From x26, From x27, From x28, From x29,
                              From x30, From x31)
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
        r.set(16, static_cast<To>(x16));
        r.set(17, static_cast<To>(x17));
        r.set(18, static_cast<To>(x18));
        r.set(19, static_cast<To>(x19));
        r.set(20, static_cast<To>(x20));
        r.set(21, static_cast<To>(x21));
        r.set(22, static_cast<To>(x22));
        r.set(23, static_cast<To>(x23));
        r.set(24, static_cast<To>(x24));
        r.set(25, static_cast<To>(x25));
        r.set(26, static_cast<To>(x26));
        r.set(27, static_cast<To>(x27));
        r.set(28, static_cast<To>(x28));
        r.set(29, static_cast<To>(x29));
        r.set(30, static_cast<To>(x30));
        r.set(31, static_cast<To>(x31));
        return r;
    }
    Vc_INTRINSIC R operator()(From x0, From x1, From x2, From x3, From x4, From x5,
                              From x6, From x7, From x8, From x9, From x10, From x11,
                              From x12, From x13, From x14, From x15, From x16, From x17,
                              From x18, From x19, From x20, From x21, From x22, From x23,
                              From x24, From x25, From x26, From x27, From x28, From x29,
                              From x30, From x31, From x32, From x33, From x34, From x35,
                              From x36, From x37, From x38, From x39, From x40, From x41,
                              From x42, From x43, From x44, From x45, From x46, From x47,
                              From x48, From x49, From x50, From x51, From x52, From x53,
                              From x54, From x55, From x56, From x57, From x58, From x59,
                              From x60, From x61, From x62, From x63)
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
                 static_cast<To>(x30), static_cast<To>(x31), static_cast<To>(x32),
                 static_cast<To>(x33), static_cast<To>(x34), static_cast<To>(x35),
                 static_cast<To>(x36), static_cast<To>(x37), static_cast<To>(x38),
                 static_cast<To>(x39), static_cast<To>(x40), static_cast<To>(x41),
                 static_cast<To>(x42), static_cast<To>(x43), static_cast<To>(x44),
                 static_cast<To>(x45), static_cast<To>(x46), static_cast<To>(x47),
                 static_cast<To>(x48), static_cast<To>(x49), static_cast<To>(x50),
                 static_cast<To>(x51), static_cast<To>(x52), static_cast<To>(x53),
                 static_cast<To>(x54), static_cast<To>(x55), static_cast<To>(x56),
                 static_cast<To>(x57), static_cast<To>(x58), static_cast<To>(x59),
                 static_cast<To>(x60), static_cast<To>(x61), static_cast<To>(x62),
                 static_cast<To>(x63));
    }
};

// }}}1
// simd_converter Sse -> Avx512 {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::Sse, To, simd_abi::Avx512> {
    using Arg = sse_simd_member_type<From>;

    Vc_INTRINSIC auto operator()(Arg a)
    {
        return x86::convert_all<avx512_simd_member_type<To>>(a);
    }
    Vc_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(2 * sizeof(From) >= sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b);
    }
    Vc_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 1 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b, c, d);
    }
    Vc_INTRINSIC avx512_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(x0, x1, x2, x3, x4, x5, x6,
                                                           x7);
    }
    Vc_INTRINSIC avx512_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7,
                                                     Arg x8, Arg x9, Arg x10, Arg x11,
                                                     Arg x12, Arg x13, Arg x14, Arg x15)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
    }
    Vc_INTRINSIC avx512_simd_member_type<To> operator()(
        Arg x0, Arg x1, Arg x2, Arg x3, Arg x4, Arg x5, Arg x6, Arg x7, Arg x8, Arg x9,
        Arg x10, Arg x11, Arg x12, Arg x13, Arg x14, Arg x15, Arg x16, Arg x17, Arg x18,
        Arg x19, Arg x20, Arg x21, Arg x22, Arg x23, Arg x24, Arg x25, Arg x26, Arg x27,
        Arg x28, Arg x29, Arg x30, Arg x31)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16,
            x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31);
    }
};

// }}}1
// simd_converter Avx512 -> Sse {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::Avx512, To, simd_abi::Sse> {
    using Arg = avx512_simd_member_type<From>;

    Vc_INTRINSIC auto operator()(Arg a)
    {
        return x86::convert_all<sse_simd_member_type<To>>(a);
    }
    Vc_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<sse_simd_member_type<To>>(a, b);
    }
};

// }}}1
// simd_converter Avx -> Avx512 {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::Avx, To, simd_abi::Avx512> {
    using Arg = avx_simd_member_type<From>;

    Vc_INTRINSIC auto operator()(Arg a)
    {
        return x86::convert_all<avx512_simd_member_type<To>>(a);
    }
    Vc_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 1 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b);
    }
    Vc_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b, c, d);
    }
    Vc_INTRINSIC avx512_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(x0, x1, x2, x3, x4, x5, x6,
                                                           x7);
    }
    Vc_INTRINSIC avx512_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7,
                                                     Arg x8, Arg x9, Arg x10, Arg x11,
                                                     Arg x12, Arg x13, Arg x14, Arg x15)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
    }
};

// }}}1
// simd_converter Avx512 -> Avx {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::Avx512, To, simd_abi::Avx> {
    using Arg = avx512_simd_member_type<From>;

    Vc_INTRINSIC auto operator()(Arg a)
    {
        return x86::convert_all<avx_simd_member_type<To>>(a);
    }
    Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<avx_simd_member_type<To>>(a, b);
    }
    Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<avx_simd_member_type<To>>(a, b, c, d);
    }
};

// }}}1
// simd_converter Avx512 -> Avx512 {{{1
template <class T> struct simd_converter<T, simd_abi::Avx512, T, simd_abi::Avx512> {
    using Arg = avx512_simd_member_type<T>;
    Vc_INTRINSIC const Arg &operator()(const Arg &x) { return x; }
};

template <class From, class To>
struct simd_converter<From, simd_abi::Avx512, To, simd_abi::Avx512> {
    using Arg = avx512_simd_member_type<From>;

    Vc_INTRINSIC auto operator()(Arg a)
    {
        return x86::convert_all<avx512_simd_member_type<To>>(a);
    }
    Vc_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b);
    }
    Vc_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b, c, d);
    }
    Vc_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d, Arg e,
                                                     Arg f, Arg g, Arg h)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b, c, d, e, f, g, h);
    }
};

// split_to_array {{{1
template <class T> struct split_to_array<simd<T, simd_abi::Avx>, 2> {
    using V = simd<T, simd_abi::Avx>;
    std::array<V, 2> operator()(simd<T, simd_abi::Avx512> x, std::index_sequence<0, 1>)
    {
        const auto xx = detail::data(x);
        return {V(detail::private_init, lo256(xx)), V(detail::private_init, hi256(xx))};
    }
};

template <class T> struct split_to_array<simd<T, simd_abi::Sse>, 4> {
    using V = simd<T, simd_abi::Sse>;
    std::array<V, 4> operator()(simd<T, simd_abi::Avx512> x,
                                std::index_sequence<0, 1, 2, 3>)
    {
        const auto xx = detail::data(x);
        return {V(detail::private_init, lo128(xx)),
                V(detail::private_init, extract128<1>(xx)),
                V(detail::private_init, extract128<2>(xx)),
                V(detail::private_init, extract128<3>(xx))};
    }
};

// split_to_tuple {{{1
template <class T>
struct split_to_tuple<std::tuple<simd<T, simd_abi::Avx>, simd<T, simd_abi::Avx>>,
                      simd_abi::Avx512> {
    using V = simd<T, simd_abi::Avx>;
    std::tuple<V, V> operator()(simd<T, simd_abi::Avx512> x)
    {
        const auto xx = detail::data(x);
        return {V(detail::private_init, lo256(xx)), V(detail::private_init, hi256(xx))};
    }
};

template <class T>
struct split_to_tuple<std::tuple<simd<T, simd_abi::Sse>, simd<T, simd_abi::Sse>,
                                 simd<T, simd_abi::Sse>, simd<T, simd_abi::Sse>>,
                      simd_abi::Avx512> {
    using V = simd<T, simd_abi::Sse>;
    std::tuple<V, V, V, V> operator()(simd<T, simd_abi::Avx512> x)
    {
        const auto xx = detail::data(x);
        return {V(detail::private_init, lo128(xx)),
                V(detail::private_init, extract128<1>(xx)),
                V(detail::private_init, extract128<2>(xx)),
                V(detail::private_init, extract128<3>(xx))};
    }
};

template <class T>
struct split_to_tuple<
    std::tuple<simd<T, simd_abi::Avx>, simd<T, simd_abi::Sse>, simd<T, simd_abi::Sse>>,
    simd_abi::Avx512> {
    using V0 = simd<T, simd_abi::Avx>;
    using V1 = simd<T, simd_abi::Sse>;
    std::tuple<V0, V1, V1> operator()(simd<T, simd_abi::Avx512> x)
    {
        const auto xx = detail::data(x);
        return {V0(detail::private_init, lo256(xx)),
                V1(detail::private_init, extract128<2>(xx)),
                V1(detail::private_init, extract128<3>(xx))};
    }
};

template <class T>
struct split_to_tuple<
    std::tuple<simd<T, simd_abi::Sse>, simd<T, simd_abi::Sse>, simd<T, simd_abi::Avx>>,
    simd_abi::Avx512> {
    using V0 = simd<T, simd_abi::Sse>;
    using V1 = simd<T, simd_abi::Avx>;
    std::tuple<V0, V0, V1> operator()(simd<T, simd_abi::Avx512> x)
    {
        const auto xx = detail::data(x);
        return {V0(detail::private_init, lo128(xx)),
                V0(detail::private_init, extract128<1>(xx)),
                V1(detail::private_init, hi256(xx))};
    }
};

template <class T>
struct split_to_tuple<
    std::tuple<simd<T, simd_abi::Sse>, simd<T, simd_abi::Avx>, simd<T, simd_abi::Sse>>,
    simd_abi::Avx512> {
    using V0 = simd<T, simd_abi::Sse>;
    using V1 = simd<T, simd_abi::Avx>;
    std::tuple<V0, V1, V0> operator()(simd<T, simd_abi::Avx512> x)
    {
        const auto xx = detail::data(x);
        return {V0(detail::private_init, lo128(xx)),
                V1(detail::private_init, extract256_center(xx)),
                V0(detail::private_init, extract128<3>(xx))};
    }
};

// }}}1
// generic_simd_impl::masked_cassign specializations {{{1
#define Vc_MASKED_CASSIGN_SPECIALIZATION(TYPE_, TYPE_SUFFIX_, OP_, OP_NAME_)             \
    template <>                                                                          \
    template <>                                                                          \
    Vc_INTRINSIC void Vc_VDECL                                                           \
    generic_simd_impl<avx512_simd_impl>::masked_cassign<OP_, TYPE_, bool,          \
                                                              64 / sizeof(TYPE_)>(       \
        const Storage<bool, 64 / sizeof(TYPE_)> k,                                       \
        Storage<TYPE_, 64 / sizeof(TYPE_)> &lhs,                                         \
        const detail::id<Storage<TYPE_, 64 / sizeof(TYPE_)>> rhs)                        \
    {                                                                                    \
        lhs = _mm512_mask_##OP_NAME_##_##TYPE_SUFFIX_(lhs, k, lhs, rhs);                 \
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
#ifdef Vc_HAVE_FULL_AVX512_ABI
Vc_MASKED_CASSIGN_SPECIALIZATION(         short, epi16, std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail::ushort, epi16, std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: schar, epi8 , std::plus, add);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: uchar, epi8 , std::plus, add);
#endif  // Vc_HAVE_FULL_AVX512_ABI

Vc_MASKED_CASSIGN_SPECIALIZATION(        double,  pd  , std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(         float,  ps  , std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: llong, epi64, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail::ullong, epi64, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(          long, epi64, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: ulong, epi64, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(           int, epi32, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail::  uint, epi32, std::minus, sub);
#ifdef Vc_HAVE_FULL_AVX512_ABI
Vc_MASKED_CASSIGN_SPECIALIZATION(         short, epi16, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail::ushort, epi16, std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: schar, epi8 , std::minus, sub);
Vc_MASKED_CASSIGN_SPECIALIZATION(detail:: uchar, epi8 , std::minus, sub);
#endif  // Vc_HAVE_FULL_AVX512_ABI
#undef Vc_MASKED_CASSIGN_SPECIALIZATION

// }}}1
}  // namespace detail

// [simd_mask.reductions] {{{
template <class T> Vc_ALWAYS_INLINE bool all_of(simd_mask<T, simd_abi::Avx512> k)
{
    const auto v = detail::data(k);
    return detail::x86::testallset(v);
}

template <class T> Vc_ALWAYS_INLINE bool any_of(simd_mask<T, simd_abi::Avx512> k)
{
    const auto v = detail::data(k);
    return v != 0U;
}

template <class T> Vc_ALWAYS_INLINE bool none_of(simd_mask<T, simd_abi::Avx512> k)
{
    const auto v = detail::data(k);
    return v == 0U;
}

template <class T> Vc_ALWAYS_INLINE bool some_of(simd_mask<T, simd_abi::Avx512> k)
{
    const auto v = detail::data(k);
    return v != 0 && !all_of(k);
}

template <class T> Vc_ALWAYS_INLINE int popcount(simd_mask<T, simd_abi::Avx512> k)
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

template <class T> Vc_ALWAYS_INLINE int find_first_set(simd_mask<T, simd_abi::Avx512> k)
{
    const auto v = detail::data(k);
    return _tzcnt_u32(v);
}

#ifdef Vc_HAVE_FULL_AVX512_ABI
Vc_ALWAYS_INLINE int find_first_set(simd_mask<signed char, simd_abi::Avx512> k)
{
    const __mmask64 v = detail::data(k);
    return detail::firstbit(v);
}
Vc_ALWAYS_INLINE int find_first_set(simd_mask<unsigned char, simd_abi::Avx512> k)
{
    const __mmask64 v = detail::data(k);
    return detail::firstbit(v);
}
#endif  // Vc_HAVE_FULL_AVX512_ABI

template <class T> Vc_ALWAYS_INLINE int find_last_set(simd_mask<T, simd_abi::Avx512> k)
{
    return 31 - _lzcnt_u32(detail::data(k));
}

#ifdef Vc_HAVE_FULL_AVX512_ABI
Vc_ALWAYS_INLINE int find_last_set(simd_mask<signed char, simd_abi::Avx512> k)
{
    const __mmask64 v = detail::data(k);
    return detail::lastbit(v);
}

Vc_ALWAYS_INLINE int find_last_set(simd_mask<unsigned char, simd_abi::Avx512> k)
{
    const __mmask64 v = detail::data(k);
    return detail::lastbit(v);
}
#endif  // Vc_HAVE_FULL_AVX512_ABI

// }}}
Vc_VERSIONED_NAMESPACE_END

#endif  // Vc_HAVE_AVX512_ABI
#endif  // VC_SIMD_AVX512_H_

// vim: foldmethod=marker
