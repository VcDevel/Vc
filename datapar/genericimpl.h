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

#ifndef VC_DATAPAR_GENERICIMPL_H_
#define VC_DATAPAR_GENERICIMPL_H_

#include "detail.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
// datapar impl {{{1
template <class Derived> struct generic_datapar_impl {
    // adjust_for_long{{{2
    template <size_t Size>
    static Vc_INTRINSIC Storage<equal_int_type_t<long>, Size> adjust_for_long(
        Storage<long, Size> x)
    {
        return {x.v()};
    }
    template <size_t Size>
    static Vc_INTRINSIC Storage<equal_int_type_t<ulong>, Size> adjust_for_long(
        Storage<ulong, Size> x)
    {
        return {x.v()};
    }
    template <class T, size_t Size>
    static Vc_INTRINSIC const Storage<T, Size> &adjust_for_long(const Storage<T, Size> &x)
    {
        return x;
    }

    template <class T, class A, class U>
    static Vc_INTRINSIC Vc::datapar<T, A> make_datapar(const U &x)
    {
        using traits = typename Vc::datapar<T, A>::traits;
        using V = typename traits::datapar_member_type;
        return {private_init, static_cast<V>(x)};
    }

    // unary minus {{{2
    template <class T, class A>
    static Vc_INTRINSIC Vc::datapar<T, A> unary_minus(const Vc::datapar<T, A> &x) noexcept
    {
        using detail::x86::unary_minus;
        return make_datapar<T, A>(unary_minus(adjust_for_long(detail::data(x))));
    }

    // arithmetic operators {{{2
#define Vc_ARITHMETIC_OP_(name_)                                                         \
    template <class T, class A>                                                          \
    static Vc_INTRINSIC datapar<T, A> name_(datapar<T, A> x, datapar<T, A> y)            \
    {                                                                                    \
        return make_datapar<T, A>(                                                       \
            detail::name_(adjust_for_long(x.d), adjust_for_long(y.d)));                  \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON

    Vc_ARITHMETIC_OP_(plus);
    Vc_ARITHMETIC_OP_(minus);
    Vc_ARITHMETIC_OP_(multiplies);
    Vc_ARITHMETIC_OP_(divides);
#undef Vc_ARITHMETIC_OP_

};
//}}}1
}  // namespace detail

// where implementation {{{1
template <class T, class A>
inline void Vc_VDECL masked_assign(mask<T, A> k, datapar<T, A> &lhs,
                                   const datapar<T, A> &rhs)
{
    lhs = static_cast<datapar<T, A>>(
        detail::x86::blend(detail::data(k), detail::data(lhs), detail::data(rhs)));
}

template <class T, class A, class U>
inline enable_if<std::is_convertible<U, datapar<T, A>>::value, void> masked_assign(
    const mask<T, A> &k, datapar<T, A> &lhs, const U &rhs)
{
    masked_assign(k, lhs, datapar<T, A>(rhs));
}

// special case for long double because it falls back to using fixed_size<1>
template <class A>
inline void Vc_VDECL masked_assign(const mask<long double, A> &k,
                                   datapar<long double, A> &lhs,
                                   const datapar<long double, A> &rhs)
{
    if (k[0]) {
        lhs[0] = rhs[0];
    }
}

// special case for long double & fixed_size to disambiguate with the above
template <int N>
inline void Vc_VDECL
masked_assign(const mask<long double, datapar_abi::fixed_size<N>> &k,
              datapar<long double, datapar_abi::fixed_size<N>> &lhs,
              const datapar<long double, datapar_abi::fixed_size<N>> &rhs)
{
    masked_assign<long double, N>(k, lhs, rhs);
}

//}}}1
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_GENERICIMPL_H_
