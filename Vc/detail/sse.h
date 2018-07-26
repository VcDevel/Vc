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
#include "genericimpl.h"
#include "simd_tuple.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
struct sse_mask_impl : generic_mask_impl<simd_abi::__sse> {};

// simd impl {{{1
struct sse_simd_impl : public generic_simd_impl<simd_abi::__sse> {
    // member types {{{2
    using abi = simd_abi::__sse;
    template <class T> static constexpr size_t size() { return simd_size_v<T, abi>; }
    template <class T> using simd_member_type = sse_simd_member_type<T>;
    template <class T> using intrinsic_type = typename simd_member_type<T>::register_type;
    template <class T> using mask_member_type = sse_mask_member_type<T>;
    template <class T> using simd = Vc::simd<T, abi>;
    template <class T> using simd_mask = Vc::simd_mask<T, abi>;
    // }}}2
};

// simd_converter __sse -> scalar {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::__sse, To, simd_abi::scalar> {
    using Arg = sse_simd_member_type<From>;

    Vc_INTRINSIC std::array<To, Arg::width> all(Arg a)
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
// simd_converter scalar -> __sse {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::scalar, To, simd_abi::__sse> {
    using R = sse_simd_member_type<To>;
    template <class... More> constexpr Vc_INTRINSIC R operator()(From a, More... b)
    {
        static_assert(sizeof...(More) + 1 == R::width);
        static_assert(std::conjunction_v<std::is_same<From, More>...>);
        return builtin_type16_t<To>{static_cast<To>(a), static_cast<To>(b)...};
    }
};

// }}}1
// simd_converter __sse -> __sse {{{1
template <class T> struct simd_converter<T, simd_abi::__sse, T, simd_abi::__sse> {
    using Arg = sse_simd_member_type<T>;
    Vc_INTRINSIC const Arg &operator()(const Arg &x) { return x; }
};

template <class From, class To>
struct simd_converter<From, simd_abi::__sse, To, simd_abi::__sse> {
    using Arg = sse_simd_member_type<From>;

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
