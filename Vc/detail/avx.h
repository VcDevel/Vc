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
struct avx_mask_impl : generic_mask_impl<simd_abi::__avx> {};

// simd impl {{{1
struct avx_simd_impl : public generic_simd_impl<simd_abi::__avx> {
};

    // simd_converter __avx -> scalar {{{1
    template <class From, class To>
    struct simd_converter<From, simd_abi::__avx, To, simd_abi::scalar> {
        using Arg = avx_simd_member_type<From>;

        Vc_INTRINSIC std::array<To, Arg::width> all(Arg a)
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

        Vc_INTRINSIC auto all(Arg a)
        {
            return x86::convert_all<builtin_type32_t<To>>(a);
        }

        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a)
        {
            return x86::convert<builtin_type32_t<To>>(a);
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

        Vc_INTRINSIC auto all(Arg a)
        {
            return x86::convert_all<builtin_type16_t<To>>(a);
        }

        Vc_INTRINSIC sse_simd_member_type<To> operator()(Arg a)
        {
            return x86::convert<builtin_type16_t<To>>(a);
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

        Vc_INTRINSIC auto all(Arg a)
        {
            return x86::convert_all<builtin_type32_t<To>>(a);
        }

        Vc_INTRINSIC avx_simd_member_type<To> operator()(Arg a)
        {
            return x86::convert<builtin_type32_t<To>>(a);
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
