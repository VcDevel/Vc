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
    // member types {{{2
    template <size_t N> using size_tag = std::integral_constant<size_t, N>;

    // adjust_for_long{{{2
    template <size_t Size>
    static Vc_INTRINSIC Storage<equal_int_type_t<long>, Size> Vc_VDECL
    adjust_for_long(Storage<long, Size> x)
    {
        return {x.v()};
    }
    template <size_t Size>
    static Vc_INTRINSIC Storage<equal_int_type_t<ulong>, Size> Vc_VDECL
    adjust_for_long(Storage<ulong, Size> x)
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

    // complement {{{2
    template <class T, class A>
    static Vc_INTRINSIC Vc::datapar<T, A> complement(const Vc::datapar<T, A> &x) noexcept
    {
        using detail::x86::complement;
        return make_datapar<T, A>(complement(adjust_for_long(detail::data(x))));
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
    static Vc_INTRINSIC datapar<T, A> Vc_VDECL name_(datapar<T, A> x, datapar<T, A> y)   \
    {                                                                                    \
        return make_datapar<T, A>(                                                       \
            detail::name_(adjust_for_long(x.d), adjust_for_long(y.d)));                  \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON

    Vc_ARITHMETIC_OP_(plus);
    Vc_ARITHMETIC_OP_(minus);
    Vc_ARITHMETIC_OP_(multiplies);
    Vc_ARITHMETIC_OP_(divides);
    Vc_ARITHMETIC_OP_(modulus);
    Vc_ARITHMETIC_OP_(bit_and);
    Vc_ARITHMETIC_OP_(bit_or);
    Vc_ARITHMETIC_OP_(bit_xor);
    Vc_ARITHMETIC_OP_(bit_shift_left);
    Vc_ARITHMETIC_OP_(bit_shift_right);
#undef Vc_ARITHMETIC_OP_

    // increment & decrement{{{2
    template <class T, size_t N> static Vc_INTRINSIC void increment(Storage<T, N> &x)
    {
        x = detail::plus(x, Storage<T, N>(Derived::broadcast(T(1), size_tag<N>())));
    }
    template <size_t N> static Vc_INTRINSIC void increment(Storage<long, N> &x)
    {
        x = detail::plus(adjust_for_long(x), Storage<equal_int_type_t<long>, N>(
                                                 Derived::broadcast(1L, size_tag<N>())));
    }
    template <size_t N> static Vc_INTRINSIC void increment(Storage<ulong, N> &x)
    {
        x = detail::plus(adjust_for_long(x), Storage<equal_int_type_t<ulong>, N>(
                                                 Derived::broadcast(1L, size_tag<N>())));
    }

    template <class T, size_t N> static Vc_INTRINSIC void decrement(Storage<T, N> &x)
    {
        x = detail::minus(x, Storage<T, N>(Derived::broadcast(T(1), size_tag<N>())));
    }
    template <size_t N> static Vc_INTRINSIC void decrement(Storage<long, N> &x)
    {
        x = detail::minus(adjust_for_long(x), Storage<equal_int_type_t<long>, N>(
                                                  Derived::broadcast(1L, size_tag<N>())));
    }
    template <size_t N> static Vc_INTRINSIC void decrement(Storage<ulong, N> &x)
    {
        x = detail::minus(adjust_for_long(x), Storage<equal_int_type_t<ulong>, N>(
                                                  Derived::broadcast(1L, size_tag<N>())));
    }
};
//}}}1
}  // namespace detail

// where implementation {{{1
template <class T, class A>
inline void Vc_VDECL masked_assign(mask<T, A> k, datapar<T, A> &lhs,
                                   const detail::id<datapar<T, A>> &rhs)
{
    lhs = static_cast<datapar<T, A>>(
        detail::x86::blend(detail::data(k), detail::data(lhs), detail::data(rhs)));
}

template <class T, class A>
inline void Vc_VDECL masked_assign(mask<T, A> k, mask<T, A> &lhs,
                                   const detail::id<mask<T, A>> &rhs)
{
    lhs = static_cast<mask<T, A>>(
        detail::x86::blend(detail::data(k), detail::data(lhs), detail::data(rhs)));
}

template <template <typename> class Op, typename T, class A,
          int = 1  // the int parameter is used to disambiguate the function template
                   // specialization for the avx512 ABI
          >
inline void Vc_VDECL masked_cassign(mask<T, A> k, datapar<T, A> &lhs,
                                    const datapar<T, A> &rhs)
{
    lhs = static_cast<datapar<T, A>>(detail::x86::blend(
        detail::data(k), detail::data(lhs), detail::data(Op<void>{}(lhs, rhs))));
}

template <template <typename> class Op, typename T, class A, class U>
inline enable_if<std::is_convertible<U, datapar<T, A>>::value, void> Vc_VDECL
masked_cassign(mask<T, A> k, datapar<T, A> &lhs, const U &rhs)
{
    masked_cassign<Op>(k, lhs, datapar<T, A>(rhs));
}

template <template <typename> class Op, typename T, class A,
          int = 1  // the int parameter is used to disambiguate the function template
                   // specialization for the avx512 ABI
          >
inline datapar<T, A> Vc_VDECL masked_unary(mask<T, A> k, datapar<T, A> v)
{
    Op<datapar<T, A>> op;
    return static_cast<datapar<T, A>>(
        detail::x86::blend(detail::data(k), detail::data(v), detail::data(op(v))));
}

//}}}1
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_GENERICIMPL_H_
