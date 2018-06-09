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

#include "macros.h"
#ifdef Vc_HAVE_AVX_ABI
#include "sse.h"
#include "storage.h"
#include "concepts.h"
#include "x86/intrinsics.h"
#include "x86/convert.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
struct avx_mask_impl : generic_mask_impl<simd_abi::__avx> {
};

constexpr struct {
    template <class T> operator T() const { return detail::allone<T>(); }
} allone_poly = {};
}  // namespace detail

// [simd_mask.reductions] {{{
template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL all_of(simd_mask<T, simd_abi::__avx> k)
{
    const auto d = detail::data(k);
    return 0 != detail::testc(d, detail::allone_poly);
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL any_of(simd_mask<T, simd_abi::__avx> k)
{
    const auto d = detail::data(k);
    return 0 == detail::testz(d, d);
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL none_of(simd_mask<T, simd_abi::__avx> k)
{
    const auto d = detail::data(k);
    return 0 != detail::testz(d, d);
}

template <class T> Vc_ALWAYS_INLINE bool Vc_VDECL some_of(simd_mask<T, simd_abi::__avx> k)
{
    const auto d = detail::data(k);
    return 0 != detail::testnzc(d, detail::allone_poly);
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL popcount(simd_mask<T, simd_abi::__avx> k)
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

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL find_first_set(simd_mask<T, simd_abi::__avx> k)
{
    const auto d = detail::data(k);
    return detail::firstbit(detail::mask_to_int<k.size()>(d));
}

template <class T> Vc_ALWAYS_INLINE int Vc_VDECL find_last_set(simd_mask<T, simd_abi::__avx> k)
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
struct avx_simd_impl : public generic_simd_impl<avx_simd_impl, simd_abi::__avx> {
    // member types {{{2
    using abi = simd_abi::__avx;
    template <class T> static constexpr size_t size() { return simd_size_v<T, abi>; }
    template <class T> using simd_member_type = avx_simd_member_type<T>;
    template <class T> using intrinsic_type = typename simd_member_type<T>::register_type;
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
        return detail::load32(mem, f);
    }

    // convert from an AVX load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const convertible_memory<U, sizeof(T), T> *mem, F f, type_tag<T>,
        tag<1> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<simd_member_type<T>, simd_member_type<U>>(load32(mem, f));
    }

    // convert from an SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 2, T> *mem, F f, type_tag<T>,
        tag<2> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<simd_member_type<T>, sse_simd_member_type<U>>(load16(mem, f));
    }

    // convert from a half SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 4, T> *mem, F f, type_tag<T>,
        tag<3> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<simd_member_type<T>, sse_simd_member_type<U>>(load8(mem, f));
    }

    // convert from a 1/4th SSE load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) / 8, T> *mem, F f, type_tag<T>,
        tag<4> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        return convert<simd_member_type<T>, sse_simd_member_type<U>>(load4(mem, f));
    }

    // convert from an AVX512/2-AVX load{{{3
    template <class T> using avx512_member_type = avx512_simd_member_type<T>;

    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) * 2, T> *mem, F f, type_tag<T>,
        tag<5> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        if constexpr (have_avx512f && (sizeof(U) >= 4 || have_avx512bw)) {
            return convert<simd_member_type<T>, avx512_member_type<U>>(load64(mem, f));
        } else if constexpr (have_avx2 || std::is_floating_point_v<U>) {
            return convert<simd_member_type<T>, simd_member_type<U>>(
                load32(mem, f), load32(mem + size<U>(), f));
        } else {
            static_assert(!have_avx2 && sizeof(U) == 8 && std::is_integral_v<U> &&
                          size<T>() == 8);
            return convert<simd_member_type<T>, storage16_t<U>>(
                load16(mem, f), load16(mem + 2, f), load16(mem + 4, f),
                load16(mem + 6, f));
        }
    }

    // convert from an 2-AVX512/4-AVX load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) * 4, T> *mem, F f, type_tag<T>,
        tag<6> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        if constexpr (have_avx512f && (sizeof(U) >= 4 || have_avx512bw)) {
            using LoadT = avx512_member_type<U>;
            constexpr auto N = LoadT::width;
            return convert<simd_member_type<T>, LoadT>(load64(mem, f),
                                                       load64(mem + N, f));
        } else {
            return convert<simd_member_type<T>, simd_member_type<U>>(
                load32(mem, f), load32(mem + size<U>(), f),
                load32(mem + 2 * size<U>(), f), load32(mem + 3 * size<U>(), f));
        }
    }

    // convert from a 4-AVX512/8-AVX load{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(
        const convertible_memory<U, sizeof(T) * 8, T> *mem, F f, type_tag<T>,
        tag<7> = {}) Vc_NOEXCEPT_OR_IN_TEST
    {
        if constexpr (have_avx512f && (sizeof(U) >= 4 || have_avx512bw)) {
            using LoadT = avx512_member_type<U>;
            constexpr auto N = LoadT::width;
            return convert<simd_member_type<T>, LoadT>(load64(mem, f), load64(mem + N, f),
                                                       load64(mem + 2 * N, f),
                                                       load64(mem + 3 * N, f));
        } else {
            using LoadT = simd_member_type<U>;
            constexpr auto N = LoadT::width;
            return convert<simd_member_type<T>, simd_member_type<U>>(
                load32(mem, f), load32(mem + N, f), load32(mem + 2 * N, f),
                load32(mem + 3 * N, f), load32(mem + 4 * N, f), load32(mem + 5 * N, f),
                load32(mem + 6 * N, f), load32(mem + 7 * N, f));
        }
    }

    // masked load {{{2
    template <class T, class U, class F>
    static inline void masked_load(simd_member_type<T> &merge, mask_member_type<T> k,
                                   const U *mem, F) Vc_NOEXCEPT_OR_IN_TEST
    {
        if constexpr (have_avx512bw_vl && sizeof(T) == 1 && std::is_same_v<T, U>) {
            merge = _mm256_mask_loadu_epi8(merge, _mm256_movemask_epi8(k), mem);
        } else if constexpr (have_avx512bw_vl && sizeof(T) == 2 && std::is_same_v<T, U>) {
            merge = _mm256_mask_loadu_epi16(merge, x86::movemask_epi16(k), mem);
        } else if constexpr (have_avx2 && sizeof(T) == 4 && std::is_same_v<T, U> &&
                             std::is_integral_v<U>) {
            merge = (~k.d & merge.d) | builtin_cast<T>(_mm256_maskload_epi32(
                                           reinterpret_cast<const int *>(mem), k));
        } else if constexpr (have_avx && sizeof(T) == 4 && std::is_same_v<T, U>) {
            merge = to_storage(
                or_(andnot_(k.d, merge.d),
                    _mm256_maskload_ps(reinterpret_cast<const float *>(mem), to_m256i(k))));
        } else if constexpr (have_avx2 && sizeof(T) == 8 && std::is_same_v<T, U> &&
                             std::is_integral_v<U>) {
            merge = (~k.d & merge.d) | builtin_cast<T>(_mm256_maskload_epi64(
                                           reinterpret_cast<const llong *>(mem), k));
        } else if constexpr (have_avx && sizeof(T) == 8 && std::is_same_v<T, U>) {
            merge = to_storage(
                or_(andnot_(k.d, merge.d),
                    _mm256_maskload_pd(reinterpret_cast<const double *>(mem), to_m256i(k))));
        } else {
            detail::bit_iteration(mask_to_int<size<T>()>(k),
                                  [&](auto i) { merge.set(i, static_cast<T>(mem[i])); });
        }
    }

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
        store32(v, mem, f);
    }

    // convert and 32-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL store(
        simd_member_type<T> v, U *mem, F, type_tag<T>,
        std::enable_if_t<sizeof(T) == sizeof(U) * 8, detail::nullarg_t> = detail::nullarg)
        Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v[i]; });
    }

    // convert and 64-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL store(
        simd_member_type<T> v, U *mem, F, type_tag<T>,
        std::enable_if_t<sizeof(T) == sizeof(U) * 4, detail::nullarg_t> = detail::nullarg)
        Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v[i]; });
    }

    // convert and 128-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL store(
        simd_member_type<T> v, U *mem, F, type_tag<T>,
        std::enable_if_t<sizeof(T) == sizeof(U) * 2, detail::nullarg_t> = detail::nullarg)
        Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v[i]; });
    }

    // convert and 256-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL
    store(simd_member_type<T> v, U *mem, F, type_tag<T>,
          std::enable_if_t<sizeof(T) == sizeof(U), detail::nullarg_t> = detail::nullarg)
        Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v[i]; });
    }

    // convert and 512-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL store(
        simd_member_type<T> v, U *mem, F, type_tag<T>,
        std::enable_if_t<sizeof(T) * 2 == sizeof(U), detail::nullarg_t> = detail::nullarg)
        Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v[i]; });
    }

    // convert and 1024-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL store(
        simd_member_type<T> v, U *mem, F, type_tag<T>,
        std::enable_if_t<sizeof(T) * 4 == sizeof(U), detail::nullarg_t> = detail::nullarg)
        Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v[i]; });
    }

    // convert and 2048-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL store(
        simd_member_type<T> v, U *mem, F, type_tag<T>,
        std::enable_if_t<sizeof(T) * 8 == sizeof(U), detail::nullarg_t> = detail::nullarg)
        Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v[i]; });
    }

    // masked store {{{2
    template <class T, class F>
    static Vc_INTRINSIC void Vc_VDECL
    masked_store(simd_member_type<T> v, long double *mem, F,
                 mask_member_type<T> k) Vc_NOEXCEPT_OR_IN_TEST
    {
        // no SSE support for long double
        execute_n_times<size<T>()>([&](auto i) {
            if (k[i]) {
                mem[i] = v[i];
            }
        });
    }
    template <class T, class U, class F>
    static Vc_INTRINSIC void Vc_VDECL masked_store(
        simd_member_type<T> v, U *mem, F, mask_member_type<T> k) Vc_NOEXCEPT_OR_IN_TEST
    {
        //TODO: detail::masked_store(mem, v.intrin(), k.d.intrin(), f);
        execute_n_times<size<T>()>([&](auto i) {
            if (k[i]) {
                mem[i] = static_cast<T>(v[i]);
            }
        });
    }

    // }}}2
    };

    // simd_converter __avx -> scalar {{{1
    template <class From, class To>
    struct simd_converter<From, simd_abi::__avx, To, simd_abi::scalar> {
        using Arg = avx_simd_member_type<From>;

        Vc_INTRINSIC std::array<To, Arg::width> operator()(Arg a)
        {
            return impl(std::make_index_sequence<Arg::width>(), a);
        }

        template <size_t... Indexes>
        Vc_INTRINSIC std::array<To, Arg::width> impl(std::index_sequence<Indexes...>,
                                                      Arg a)
        {
            return {static_cast<To>(a[Indexes])...};
        }
    };

    // }}}1
    // simd_converter scalar -> __avx {{{1
    template <class From, class To>
    struct simd_converter<From, simd_abi::scalar, To, simd_abi::__avx> {
        using R = avx_simd_member_type<To>;
        template <class... More> constexpr Vc_INTRINSIC R operator()(From a, More... b)
        {
            static_assert(sizeof...(More) + 1 == R::width);
            static_assert(std::conjunction_v<std::is_same<From, More>...>);
            return builtin_type32_t<To>{static_cast<To>(a), static_cast<To>(b)...};
        }
    };

    // }}}1
    // simd_converter __sse -> __avx {{{1
    template <class From, class To>
    struct simd_converter<From, simd_abi::__sse, To, simd_abi::__avx> {
        using Arg = sse_simd_member_type<From>;

        Vc_INTRINSIC auto operator()(Arg a)
        {
            return x86::convert_all<avx_simd_member_type<To>>(a);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b)
        {
            static_assert(sizeof(From) >= 1 * sizeof(To), "");
            return x86::convert<avx_simd_member_type<To>>(a, b);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
        {
            static_assert(sizeof(From) >= 2 * sizeof(To), "");
            return x86::convert<avx_simd_member_type<To>>(a, b, c, d);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                         Arg x4, Arg x5, Arg x6, Arg x7)
        {
            static_assert(sizeof(From) >= 4 * sizeof(To), "");
            return x86::convert<avx_simd_member_type<To>>(x0, x1, x2, x3, x4, x5, x6,
                                                               x7);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                         Arg x4, Arg x5, Arg x6, Arg x7,
                                                         Arg x8, Arg x9, Arg x10, Arg x11,
                                                         Arg x12, Arg x13, Arg x14,
                                                         Arg x15)
        {
            static_assert(sizeof(From) >= 8 * sizeof(To), "");
            return x86::convert<avx_simd_member_type<To>>(
                x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
        }
    };

    // }}}1
    // simd_converter __avx -> __sse {{{1
    template <class From, class To>
    struct simd_converter<From, simd_abi::__avx, To, simd_abi::__sse> {
        using Arg = avx_simd_member_type<From>;

        Vc_INTRINSIC auto operator()(Arg a)
        {
            return x86::convert_all<sse_simd_member_type<To>>(a);
        }
        Vc_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b)
        {
            static_assert(sizeof(From) >= 4 * sizeof(To), "");
            return x86::convert<sse_simd_member_type<To>>(a, b);
        }
        Vc_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
        {
            static_assert(sizeof(From) >= 8 * sizeof(To), "");
            return x86::convert<sse_simd_member_type<To>>(a, b, c, d);
        }
    };

    // }}}1
    // simd_converter __avx -> __avx {{{1
    template <class T> struct simd_converter<T, simd_abi::__avx, T, simd_abi::__avx> {
        using Arg = avx_simd_member_type<T>;
        Vc_INTRINSIC const Arg &operator()(const Arg &x) { return x; }
    };

    template <class From, class To>
    struct simd_converter<From, simd_abi::__avx, To, simd_abi::__avx> {
        using Arg = avx_simd_member_type<From>;

        Vc_INTRINSIC auto operator()(Arg a)
        {
            return x86::convert_all<avx_simd_member_type<To>>(a);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b)
        {
            static_assert(sizeof(From) >= 2 * sizeof(To), "");
            return x86::convert<avx_simd_member_type<To>>(a, b);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
        {
            static_assert(sizeof(From) >= 4 * sizeof(To), "");
            return x86::convert<avx_simd_member_type<To>>(a, b, c, d);
        }
        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d,
                                                         Arg e, Arg f, Arg g, Arg h)
        {
            static_assert(sizeof(From) >= 8 * sizeof(To), "");
            return x86::convert<avx_simd_member_type<To>>(a, b, c, d, e, f, g, h);
        }
    };

    // }}}1
    }  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // Vc_HAVE_AVX_ABI
#endif  // VC_SIMD_AVX_H_

// vim: foldmethod=marker
