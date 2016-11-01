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

#ifndef VC_DATAPAR_RETURNTYPE_H_
#define VC_DATAPAR_RETURNTYPE_H_

#include "synopsis.h"

namespace Vc_VERSIONED_NAMESPACE::detail {
// common<A, B>{{{1
template <class A, class B, bool arithmetic_A = std::is_arithmetic<A>::value,
          bool arithmetic_B = std::is_arithmetic<B>::value,
          bool any_float = std::disjunction<std::is_floating_point<A>,
                                            std::is_floating_point<B>>::value>
struct common;

// - A if A equals B
template <class A> struct common<A, A, true, true, true> {
    using type = A;
};
template <class A> struct common<A, A, true, true, false> {
    using type = A;
};

// - Otherwise, A if B is not arithmetic
template <class A, class B, bool arithmetic_A, bool any_float>
struct common<A, B, arithmetic_A, false, any_float> {
    using type = A;
};

// - Otherwise, B if A is not arithmetic
template <class A, class B, bool any_float> struct common<A, B, false, true, any_float> {
    using type = B;
};

// - Otherwise, decltype(A() + B()) if any of A or B is floating point
template <class A, class B> struct common<A, B, true, true, true> {
    using type = decltype(A() + B());
};

// - Otherwise, the integer conversion rank determines the result type
template <typename T> struct make_signed_or_bool : public std::make_signed<T> {
};
template <> struct make_signed_or_bool<bool> {
    using type = bool;
};
template <typename TT, typename UU> struct greater_integer_conversion_rank {
    template <typename A>
    using fix_sign =
        typename std::conditional<(std::is_unsigned<TT>::value ||
                                   std::is_unsigned<UU>::value),
                                  typename std::make_unsigned<A>::type, A>::type;
    using T = typename make_signed_or_bool<TT>::type;
    using U = typename make_signed_or_bool<UU>::type;
    template <typename Test, typename Otherwise>
    using c = typename std::conditional<std::is_same<T, Test>::value ||
                                            std::is_same<U, Test>::value,
                                        Test, Otherwise>::type;

    using type = fix_sign<c<llong, c<long, c<int, c<short, c<schar, void>>>>>>;
};
template <class A, class B> struct common<A, B, true, true, false> {
    using C = greater_integer_conversion_rank<A, B>;
    using type = std::conditional_t<
        (sizeof(A) > sizeof(B)), A,  // - Otherwise, A if sizeof(A) > sizeof(B)
        std::conditional_t<
            (sizeof(A) < sizeof(B)),
            B,  // - Otherwise, B if sizeof(A) < sizeof(B)
            std::conditional_t<std::is_signed<A>::value == std::is_signed<B>::value,
                               typename C::type,
                               std::make_unsigned_t<typename C::type>>>>;
};

// commonabi(V0, V1, T){{{1
template <class V0, class V1, class T, class A0 = typename V0::abi_type,
          class A1 = typename V1::abi_type>
struct commonabi;

// - V0::abi_type if V0::abi_type equals V1::abi_type
template <class V0, class V1, class T, class Abi> struct commonabi<V0, V1, T, Abi, Abi> {
    using type = Abi;
};

// - Otherwise, abi_for_size_t<T, V0::size()> if both V0 and V1 are implicitly convertible to
//   datapar<T, abi_for_size_t<T, V0::size()>>
template <class V0, class V1, class T, class A0, class A1> struct commonabi {
    using other_abi = abi_for_size_t<T, V0::size()>;
    using other_datapar = datapar<T, other_abi>;
    using type = std::conditional_t<
        std::conjunction<std::is_convertible<V0, other_datapar>,
                         std::is_convertible<V1, other_datapar>>::value,
        other_abi, datapar_abi::fixed_size<V0::size()>>;
};

// return_type_impl{{{1
template <class T, class V, class A>
std::enable_if_t<!std::is_same<T, datapar<V, A>>::value, datapar<V, A>>
    deduce_implicit_datapar_conversion(datapar<V, A>);

template <class T> T deduce_implicit_datapar_conversion(T);
template <class T> void deduce_implicit_datapar_conversion(...);

template <class L, class R, bool = std::is_arithmetic<R>::value,
          bool = std::is_convertible<R, int>::value>
struct return_type_impl3 {
    // type is datapar<V, A> if V and A can be deduced,
    // otherwise if R is convertible to L, type is L,
    // otherwise 
    using type = decltype(deduce_implicit_datapar_conversion<L>(declval<R>()));
};

template <class L, class R, bool Convertible>
struct return_type_impl3<L, R, true, Convertible> {
    using T = typename L::value_type;
    using common_value_type = typename common<T, R>::type;
    using RV = datapar<R, datapar_abi::fixed_size<L::size()>>;
    using type =
        datapar<common_value_type, typename commonabi<L, RV, common_value_type>::type>;
};

template <class L, class R> struct return_type_impl3<L, R, false, true> {
    using T = typename L::value_type;
    using common_value_type = typename common<T, R>::type;
    using RV = datapar<int, datapar_abi::fixed_size<L::size()>>;
    using type =
        datapar<common_value_type, typename commonabi<L, RV, common_value_type>::type>;
};

template <class L, class R, bool = is_datapar_v<R>,  // is_datapar<L> is implicit
          bool = std::is_integral<typename L::value_type>::value>
struct return_type_impl2 : public return_type_impl3<L, R> {
};

template <class L, class R, bool Integral>
struct return_type_impl2<L, R, true, Integral> {
    using common_value_type =
        typename common<typename L::value_type, typename R::value_type>::type;
    using type =
        datapar<common_value_type, typename commonabi<L, R, common_value_type>::type>;
};

template <class L> struct return_type_impl2<L, int, false, true> {
    using type = L;
};

template <class L> struct return_type_impl2<L, uint, false, true> {
    using LT = typename L::value_type;
    using A = typename L::abi_type;
    using type = std::conditional_t<(sizeof(LT) <= sizeof(int)),
                                    datapar<std::make_unsigned_t<LT>, A>, L>;
};

template <class L, class R> struct return_type_impl : public return_type_impl2<L, R> {
    static_assert(
        is_datapar_v<L>,
        "return_type_impl requires the first template argument to be a datapar type");
};

// shortcut for the simple (and common) case
template <class L> struct return_type_impl<L, L> {
    using type = L;
};

// mask_return_type_impl{{{1
template <class T, class A> struct mask_return_type_impl<T, A, T, A> {
    using type = mask<T, A>;
};
template <class T0, class A0, class T1, class A1> struct mask_return_type_impl {
    using CommonT = typename common<T0, T1>::type;
    using type =
        mask<CommonT,
             typename commonabi<datapar<T0, A0>, datapar<T1, A1>, CommonT>::type>;
};

//}}}2
}  // namespace Vc_VERSIONED_NAMESPACE::detail

#endif  // VC_DATAPAR_RETURNTYPE_H_
