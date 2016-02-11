/*  This file is part of the Vc library. {{{
Copyright © 2012-2016 Matthias Kretz <kretz@kde.org>
All rights reserved.

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

#ifndef COMMON_OPERATORS_H_
#define COMMON_OPERATORS_H_
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Detail
{
template <typename T, typename Abi, typename U>
enable_if<!std::is_same<T, U>::value, U> is_convertible_to_any_vector(Vector<U, Abi>);
template <typename T, typename Abi> T is_convertible_to_any_vector(Vector<T, Abi>);
template <typename T, typename Abi> void is_convertible_to_any_vector(...);

template <typename T, typename U, bool = std::is_integral<T>::value,
          bool = std::is_integral<U>::value>
struct FundamentalReturnType;
template <typename T, typename U> struct FundamentalReturnType<T, U, false, false> {
    using type = typename std::conditional<
        std::is_arithmetic<U>::value,
        typename std::conditional<(sizeof(T) < sizeof(U)), U, T>::type,
        // U is not arithmetic, e.g. an enum or a type with e.g. operator int()
        T>::type;
};
template <typename T, typename U> struct FundamentalReturnType<T, U, true, false> {
    using type = typename std::conditional<
        std::is_arithmetic<U>::value, U,
        // U is not arithmetic, e.g. an enum or a type with e.g. operator int()
        T>::type;
};
template <typename T, typename U> struct FundamentalReturnType<T, U, false, true> {
    using type = T;
};

template <typename T, typename U, typename Test, typename Otherwise>
using larger_conditional = typename std::conditional<
    std::is_same<T, Test>::value || std::is_same<U, Test>::value, Test, Otherwise>::type;

template <typename T, typename U>
using larger_type = larger_conditional<
    T, U, long long,
    larger_conditional<
        T, U, long,
        larger_conditional<
            T, U, int, larger_conditional<T, U, short,
                                          larger_conditional<T, U, signed char, void>>>>>;

template <typename T> struct my_make_signed : public std::make_signed<T> {
};
template <> struct my_make_signed<bool> {
    using type = bool;
};

template <typename T, typename U> struct FundamentalReturnType<T, U, true, true> {
    using type = typename std::conditional<
        (sizeof(T) > sizeof(U)), T,
        typename std::conditional<
            (sizeof(T) < sizeof(U)), U,
            typename std::conditional<
                (std::is_unsigned<T>::value || std::is_unsigned<U>::value),
                typename std::make_unsigned<
                    larger_type<typename my_make_signed<T>::type,
                                typename my_make_signed<U>::type>>::type,
                larger_type<typename my_make_signed<T>::type,
                            typename my_make_signed<U>::type>>::type>::type>::type;
};
static_assert(std::is_same<long, typename FundamentalReturnType<int, long>::type>::value, "");

template <typename V, typename T, bool, typename, bool> struct ReturnTypeImpl {
    // no type => SFINAE
};
template <typename T, typename U, typename Abi, typename Deduced>
struct ReturnTypeImpl<Vector<T, Abi>, Vector<U, Abi>, false, Deduced, false> {
    using type = Vector<typename FundamentalReturnType<T, U>::type, Abi>;
};
template <typename T, typename Abi>
struct ReturnTypeImpl<Vector<T, Abi>, int, true, T, true> {
    using type = Vector<T, Abi>;
};
template <typename T, typename Abi>
struct ReturnTypeImpl<Vector<T, Abi>, unsigned int, true, T, true> {
    using type = Vector<typename std::make_unsigned<T>::type, Abi>;
};
template <typename T, typename U, typename Abi, bool Integral>
struct ReturnTypeImpl<Vector<T, Abi>, U, true, T, Integral> {
    using type = Vector<typename FundamentalReturnType<T, U>::type, Abi>;
};
template <typename T, typename U, typename Abi, bool Integral>
struct ReturnTypeImpl<Vector<T, Abi>, U, false, void, Integral> {
    // no type => SFINAE
};
template <typename T, typename U, typename Abi, typename V, bool Integral>
struct ReturnTypeImpl<Vector<T, Abi>, U, false, V, Integral> {
    using type = Vector<typename FundamentalReturnType<T, V>::type, Abi>;
};
template <typename V, typename T>
using ReturnType = typename ReturnTypeImpl<
    V, T, std::is_arithmetic<T>::value || std::is_convertible<T, int>::value,
    decltype(is_convertible_to_any_vector<typename V::value_type, typename V::abi>(
        std::declval<const T &>())),
    std::is_integral<typename V::value_type>::value>::type;

template <typename T> struct is_a_type : public std::true_type {
};
}  // namespace Detail

#define Vc_GENERIC_OPERATOR(op)                                                          \
    template <typename T, typename Abi, typename U>                                      \
    Vc_ALWAYS_INLINE enable_if<                                                          \
        Detail::is_a_type<decltype(                                                      \
            std::declval<typename Detail::ReturnType<Vector<T, Abi>, U>::EntryType>()    \
                op std::declval<typename Detail::ReturnType<Vector<T, Abi>,              \
                                                            U>::EntryType>())>::value && \
            std::is_convertible<Vector<T, Abi>,                                          \
                                Detail::ReturnType<Vector<T, Abi>, U>>::value &&         \
            std::is_convertible<U, Detail::ReturnType<Vector<T, Abi>, U>>::value,        \
        Detail::ReturnType<Vector<T, Abi>, U>>                                           \
    operator op(Vector<T, Abi> x, const U &y)                                            \
    {                                                                                    \
        using V = Detail::ReturnType<Vector<T, Abi>, U>;                                 \
        return Detail::operator op(V(x), V(y));                                          \
    }                                                                                    \
    template <typename T, typename Abi, typename U>                                      \
    Vc_ALWAYS_INLINE enable_if<                                                          \
        Detail::is_a_type<decltype(                                                      \
            std::declval<typename Detail::ReturnType<Vector<T, Abi>, U>::EntryType>()    \
                op std::declval<typename Detail::ReturnType<Vector<T, Abi>,              \
                                                            U>::EntryType>())>::value && \
            !Traits::is_simd_vector_internal<U>::value &&                                \
            std::is_convertible<Vector<T, Abi>,                                          \
                                Detail::ReturnType<Vector<T, Abi>, U>>::value &&         \
            std::is_convertible<U, Detail::ReturnType<Vector<T, Abi>, U>>::value,        \
        Detail::ReturnType<Vector<T, Abi>, U>>                                           \
    operator op(const U &x, Vector<T, Abi> y)                                            \
    {                                                                                    \
        using V = Detail::ReturnType<Vector<T, Abi>, U>;                                 \
        return Detail::operator op(V(x), V(y));                                          \
    }

#define Vc_LOGICAL_OPERATOR(op)                                                          \
    template <typename T, typename Abi>                                                  \
    Vc_ALWAYS_INLINE typename Vector<T, Abi>::Mask operator op(Vector<T, Abi> x,         \
                                                               Vector<T, Abi> y)         \
    {                                                                                    \
        return !!x op !!y;                                                               \
    }                                                                                    \
    template <typename T, typename Abi, typename U>                                      \
    Vc_ALWAYS_INLINE                                                                     \
        enable_if<std::is_convertible<Vector<T, Abi>, Vector<U, Abi>>::value &&          \
                      std::is_convertible<Vector<U, Abi>, Vector<T, Abi>>::value,        \
                  typename Detail::ReturnType<Vector<T, Abi>, Vector<U, Abi>>::Mask>     \
        operator op(Vector<T, Abi> x, Vector<U, Abi> y)                                  \
    {                                                                                    \
        return !!x op !!y;                                                               \
    }                                                                                    \
    template <typename T, typename Abi, typename U>                                      \
    Vc_ALWAYS_INLINE                                                                     \
        enable_if<std::is_same<bool, decltype(!std::declval<const U &>())>::value,       \
                  typename Vector<T, Abi>::Mask>                                         \
        operator op(Vector<T, Abi> x, const U &y)                                        \
    {                                                                                    \
        using M = typename Vector<T, Abi>::Mask;                                         \
        return !!x op M(!!y);                                                            \
    }                                                                                    \
    template <typename T, typename Abi, typename U>                                      \
    Vc_ALWAYS_INLINE                                                                     \
        enable_if<std::is_same<bool, decltype(!std::declval<const U &>())>::value,       \
                  typename Vector<T, Abi>::Mask>                                         \
        operator op(const U &x, Vector<T, Abi> y)                                        \
    {                                                                                    \
        using M = typename Vector<T, Abi>::Mask;                                         \
        return M(!!x) op !!y;                                                            \
    }

#define Vc_COMPARE_OPERATOR(op)                                                          \
    template <typename T, typename Abi, typename U>                                      \
    Vc_ALWAYS_INLINE enable_if<                                                          \
        !std::is_same<Vector<T, Abi>, U>::value &&                                       \
            std::is_convertible<Vector<T, Abi>,                                          \
                                Detail::ReturnType<Vector<T, Abi>, U>>::value &&         \
            std::is_convertible<U, Detail::ReturnType<Vector<T, Abi>, U>>::value,        \
        typename Detail::ReturnType<Vector<T, Abi>, U>::Mask>                            \
    operator op(Vector<T, Abi> x, const U &y)                                            \
    {                                                                                    \
        using V = Detail::ReturnType<Vector<T, Abi>, U>;                                 \
        return V(x) op V(y);                                                             \
    }                                                                                    \
    template <typename T, typename Abi, typename U>                                      \
    Vc_ALWAYS_INLINE enable_if<                                                          \
        !Traits::is_simd_vector_internal<U>::value &&                                    \
            std::is_convertible<Vector<T, Abi>,                                          \
                                Detail::ReturnType<Vector<T, Abi>, U>>::value &&         \
            std::is_convertible<U, Detail::ReturnType<Vector<T, Abi>, U>>::value,        \
        typename Detail::ReturnType<Vector<T, Abi>, U>::Mask>                            \
    operator op(const U &x, Vector<T, Abi> y)                                            \
    {                                                                                    \
        using V = Detail::ReturnType<Vector<T, Abi>, U>;                                 \
        return V(x) op V(y);                                                             \
    }

Vc_ALL_LOGICAL    (Vc_LOGICAL_OPERATOR)
Vc_ALL_BINARY     (Vc_GENERIC_OPERATOR)
Vc_ALL_ARITHMETICS(Vc_GENERIC_OPERATOR)
Vc_ALL_COMPARES   (Vc_COMPARE_OPERATOR)

#undef Vc_LOGICAL_OPERATOR
#undef Vc_GENERIC_OPERATOR
#undef Vc_COMPARE_OPERATOR
#undef Vc_INVALID_OPERATOR

}  // namespace Vc
#endif  // COMMON_OPERATORS_H_
