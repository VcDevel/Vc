/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

}}}*/

#ifndef VC_COMMON_SIMD_ARRAY_DATA_H
#define VC_COMMON_SIMD_ARRAY_DATA_H

#include "subscript.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{

template<typename V, std::size_t N> struct ArrayData;

namespace Operations/*{{{*/
{
struct tag {};
#define Vc_DEFINE_OPERATION(name__)                                                                \
    struct name__ : public tag                                                                     \
    {                                                                                              \
        template <typename V, typename... Args>                                                    \
        Vc_INTRINSIC void operator()(V &v, Args &&... args)                                        \
        {                                                                                          \
            v.name__(std::forward<Args>(args)...);                                                 \
        }                                                                                          \
    }
Vc_DEFINE_OPERATION(gather);
Vc_DEFINE_OPERATION(scatter);
Vc_DEFINE_OPERATION(load);
Vc_DEFINE_OPERATION(store);
Vc_DEFINE_OPERATION(setZero);
Vc_DEFINE_OPERATION(setZeroInverted);
Vc_DEFINE_OPERATION(assign);
#undef Vc_DEFINE_OPERATION
#define Vc_DEFINE_OPERATION(name__, code__)                                              \
    struct name__ : public tag                                                           \
    {                                                                                    \
        template <typename V> Vc_INTRINSIC void operator()(V & v) { code__; }            \
    }
Vc_DEFINE_OPERATION(increment, ++v);
Vc_DEFINE_OPERATION(decrement, --v);
Vc_DEFINE_OPERATION(random, v = V::Random());
#undef Vc_DEFINE_OPERATION
#define Vc_DEFINE_OPERATION(name__, code__)                                                        \
    struct name__ : public tag                                                                     \
    {                                                                                              \
        template <typename V, typename... Args>                                                    \
        Vc_INTRINSIC void operator()(V &v, Args &&... args)                                        \
        {                                                                                          \
            code__;                                                                                \
        }                                                                                          \
    }
Vc_DEFINE_OPERATION(Abs, v = abs(std::forward<Args>(args)...));
Vc_DEFINE_OPERATION(Isnan, v = isnan(std::forward<Args>(args)...));
Vc_DEFINE_OPERATION(Frexp, v = frexp(std::forward<Args>(args)...));
Vc_DEFINE_OPERATION(Ldexp, v = ldexp(std::forward<Args>(args)...));
#undef Vc_DEFINE_OPERATION
template<typename T> using is_operation = std::is_base_of<tag, T>;
}  // namespace Operations }}}
/*select_best_vector_type{{{*/
namespace internal
{
/**
 * \internal
 * AVX::Vector<T> with T int, uint, short, or ushort is either two SSE::Vector<T> or the same as
 * SSE::Vector<T>. Thus we can skip AVX::Vector<T> for integral types altogether.
 */
template <typename T> struct never_best_vector_type : public std::false_type {};

// the AVX namespace only exists in AVX compilations, otherwise it's AVX2 - which is fine
#if defined(VC_IMPL_AVX) && !defined(VC_IMPL_AVX2)
template <typename T> struct never_best_vector_type<AVX::Vector<T>> : public std::is_integral<T> {};
#endif
}  // namespace internal

/**
 * \internal
 * Selects the best SIMD type out of a typelist to store N scalar values.
 */
template<std::size_t N, typename... Typelist> struct select_best_vector_type_impl;

template<std::size_t N, typename T> struct select_best_vector_type_impl<N, T>
{
    using type = T;
};
template<std::size_t N, typename T, typename... Typelist> struct select_best_vector_type_impl<N, T, Typelist...>
{
    using type = typename std::conditional<(N < T::Size || internal::never_best_vector_type<T>::value),
                                           typename select_best_vector_type_impl<N, Typelist...>::type,
                                           T>::type;
};
template <typename T, std::size_t N>
using select_best_vector_type =
    typename select_best_vector_type_impl<N,
#ifdef VC_IMPL_AVX2
                                          Vc::AVX2::Vector<T>,
                                          Vc::SSE::Vector<T>,
                                          Vc::Scalar::Vector<T>
#elif defined(VC_IMPL_AVX)
                                          Vc::AVX::Vector<T>,
                                          Vc::SSE::Vector<T>,
                                          Vc::Scalar::Vector<T>
#elif defined(VC_IMPL_Scalar)
                                          Vc::Scalar::Vector<T>
#elif defined(VC_IMPL_SSE)
                                          Vc::SSE::Vector<T>,
                                          Vc::Scalar::Vector<T>
#elif defined(VC_IMPL_MIC)
                                          Vc::MIC::Vector<T>,
                                          Vc::Scalar::Vector<T>
#endif
                                          >::type;
//}}}

/**
 * \internal
 * Helper type to statically communicate segmentation of one vector register into 2^n parts
 * (Pieces).
 */
template <typename T_, std::size_t Pieces_, std::size_t Index_> struct Segment/*{{{*/
{
    static_assert(Index_ < Pieces_, "You found a bug in Vc. Please report.");

    using type = T_;
    using type_decayed = typename std::decay<type>::type;
    static constexpr std::size_t Pieces = Pieces_;
    static constexpr std::size_t Index = Index_;

    type data;

    static constexpr std::size_t EntryOffset = Index * type_decayed::Size / Pieces;

    decltype(std::declval<type>()[0]) operator[](size_t i) { return data[i + EntryOffset]; }
    decltype(std::declval<type>()[0]) operator[](size_t i) const { return data[i + EntryOffset]; }
};/*}}}*/

template <typename T, std::size_t Offset> struct AddOffset
{
    constexpr AddOffset() = default;
};

/**
 * \internal
 * Helper type with static functions to generically adjust arguments for the data0 and data1
 * members.
 */
template <std::size_t secondOffset> struct Split/*{{{*/
{
    template <typename Op = void, typename U>
    static Vc_ALWAYS_INLINE U
        lo(U &&x,
           enable_if<(!Traits::is_simd_vector<U>::value && !Traits::is_simd_mask<U>::value &&
                      !std::is_pointer<Traits::decay<U>>::value)> = nullarg)
    {
        return std::forward<U>(x);
    }
    template <typename Op = void, typename U>
    static Vc_ALWAYS_INLINE U
        hi(U &&x,
           enable_if<(!Traits::is_simd_vector<U>::value && !Traits::is_simd_mask<U>::value &&
                      !std::is_pointer<Traits::decay<U>>::value)> = nullarg)
    {
        return std::forward<U>(x);
    }

    template <typename U>
    static Vc_INTRINSIC const U *lo(Operations::gather, const U *ptr)
    {
        return ptr;
    }
    template <typename U>
    static Vc_INTRINSIC const U *hi(Operations::gather, const U *ptr)
    {
        return ptr + secondOffset;
    }

    // generic pointer arguments (simdarray pointers below)
    template <typename Op, typename U> static Vc_ALWAYS_INLINE U *lo(U *ptr) { return ptr; }
    template <typename Op, typename U>
    static Vc_ALWAYS_INLINE U *hi(
        U *ptr,
        typename std::enable_if<std::is_same<Op, Operations::gather>::value ||
                                std::is_same<Op, Operations::scatter>::value>::type = nullptr)
    {
        return ptr;
    }
    template <typename Op, typename U>
    static Vc_ALWAYS_INLINE U *hi(
        U *ptr,
        typename std::enable_if<!std::is_same<Op, Operations::gather>::value &&
                                !std::is_same<Op, Operations::scatter>::value>::type = nullptr)
    {
        return ptr + secondOffset;
    }

    static constexpr AddOffset<VectorSpecialInitializerIndexesFromZero::IEnum, secondOffset> hi(
        VectorSpecialInitializerIndexesFromZero::IEnum)
    {
        return {};
    }
    template <std::size_t Offset>
    static constexpr AddOffset<VectorSpecialInitializerIndexesFromZero::IEnum,
                               Offset + secondOffset>
        hi(AddOffset<VectorSpecialInitializerIndexesFromZero::IEnum, Offset>)
    {
        return {};
    }

    // split composite simdarray
    template <typename U, std::size_t N, typename V, std::size_t M>
    static Vc_INTRINSIC auto lo(const simdarray<U, N, V, M> &x) -> decltype(internal_data0(x))
    {
        return internal_data0(x);
    }
    template <typename U, std::size_t N, typename V, std::size_t M>
    static Vc_INTRINSIC auto hi(const simdarray<U, N, V, M> &x) -> decltype(internal_data1(x))
    {
        return internal_data1(x);
    }
    template <typename U, std::size_t N, typename V, std::size_t M>
    static Vc_INTRINSIC auto lo(simdarray<U, N, V, M> *x) -> decltype(&internal_data0(*x))
    {
        return &internal_data0(*x);
    }
    template <typename U, std::size_t N, typename V, std::size_t M>
    static Vc_INTRINSIC auto hi(simdarray<U, N, V, M> *x) -> decltype(&internal_data1(*x))
    {
        return &internal_data1(*x);
    }

    template <typename U, std::size_t N, typename V>
    static Vc_INTRINSIC Segment<V, 2, 0> lo(const simdarray<U, N, V, N> &x)
    {
        return {internal_data(x)};
    }
    template <typename U, std::size_t N, typename V>
    static Vc_INTRINSIC Segment<V, 2, 1> hi(const simdarray<U, N, V, N> &x)
    {
        return {internal_data(x)};
    }
    template <typename U, std::size_t N, typename V>
    static Vc_INTRINSIC Segment<V *, 2, 0> lo(const simdarray<U, N, V, N> *x)
    {
        return {&internal_data(*x)};
    }
    template <typename U, std::size_t N, typename V>
    static Vc_INTRINSIC Segment<V *, 2, 1> hi(const simdarray<U, N, V, N> *x)
    {
        return {&internal_data(*x)};
    }

    // split composite simd_mask_array
    template <typename U, std::size_t N, typename V, std::size_t M>
    static Vc_INTRINSIC auto lo(const simd_mask_array<U, N, V, M> &x) -> decltype(internal_data0(x))
    {
        return internal_data0(x);
    }
    template <typename U, std::size_t N, typename V, std::size_t M>
    static Vc_INTRINSIC auto hi(const simd_mask_array<U, N, V, M> &x) -> decltype(internal_data1(x))
    {
        return internal_data1(x);
    }

    template <typename U, std::size_t N, typename V>
    static Vc_INTRINSIC Segment<typename simd_mask_array<U, N, V, N>::mask_type, 2, 0> lo(
        const simd_mask_array<U, N, V, N> &x)
    {
        return {internal_data(x)};
    }
    template <typename U, std::size_t N, typename V>
    static Vc_INTRINSIC Segment<typename simd_mask_array<U, N, V, N>::mask_type, 2, 1> hi(
        const simd_mask_array<U, N, V, N> &x)
    {
        return {internal_data(x)};
    }

    // split Vector<T> and Mask<T>
    template <typename T>
    static constexpr bool is_vector_or_mask(){
        return (Traits::is_simd_vector<T>::value && !Traits::is_simdarray<T>::value) ||
               (Traits::is_simd_mask<T>::value && !Traits::is_simd_mask_array<T>::value);
    }
    template <typename V>
    static Vc_INTRINSIC Segment<V, 2, 0> lo(V &&x, enable_if<is_vector_or_mask<V>()> = nullarg)
    {
        return {std::forward<V>(x)};
    }
    template <typename V>
    static Vc_INTRINSIC Segment<V, 2, 1> hi(V &&x, enable_if<is_vector_or_mask<V>()> = nullarg)
    {
        return {std::forward<V>(x)};
    }

    // generically split Segments
    template <typename V, std::size_t Pieces, std::size_t Index>
    static Vc_INTRINSIC Segment<V, 2 * Pieces, Index *Pieces + 0> lo(Segment<V, Pieces, Index> &&x)
    {
        return {x.data};
    }
    template <typename V, std::size_t Pieces, std::size_t Index>
    static Vc_INTRINSIC Segment<V, 2 * Pieces, Index *Pieces + 1> hi(Segment<V, Pieces, Index> &&x)
    {
        return {x.data};
    }
};/*}}}*/

template <typename Op, typename U> static Vc_INTRINSIC U actual_value(Op, U &&x)
{
  return std::forward<U>(x);
}
template <typename Op, typename U, std::size_t M, typename V>
static Vc_INTRINSIC const V &actual_value(Op, const simdarray<U, M, V, M> &x)
{
  return internal_data(x);
}
template <typename Op, typename U, std::size_t M, typename V>
static Vc_INTRINSIC const V &actual_value(Op, simdarray<U, M, V, M> &&x)
{
  return internal_data(x);
}
template <typename Op, typename U, std::size_t M, typename V>
static Vc_INTRINSIC V *actual_value(Op, simdarray<U, M, V, M> *x)
{
  return &internal_data(*x);
}
template <typename Op, typename U, std::size_t M, typename V>
static Vc_INTRINSIC const typename V::Mask &actual_value(Op, const simd_mask_array<U, M, V, M> &x)
{
  return internal_data(x);
}
template <typename Op, typename U, std::size_t M, typename V>
static Vc_INTRINSIC const typename V::Mask &actual_value(Op, simd_mask_array<U, M, V, M> &&x)
{
  return internal_data(x);
}
template <typename Op, typename U, std::size_t M, typename V>
static Vc_INTRINSIC typename V::Mask *actual_value(Op, simd_mask_array<U, M, V, M> *x)
{
  return &internal_data(*x);
}

}  // namespace Common
}  // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_SIMD_ARRAY_DATA_H

// vim: foldmethod=marker
