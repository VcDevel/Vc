/*  This file is part of the Vc library. {{{
Copyright © 2013-2015 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_COMMON_SIMDARRAYHELPER_H_
#define VC_COMMON_SIMDARRAYHELPER_H_

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{

/// \addtogroup SimdArray
/// @{

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

/** \internal
  Template class that is used to attach an offset value to an existing type. It is used
  for IndexesFromZero construction in SimdArray. The \c data1 constructor needs to know
  that the IndexesFromZero constructor requires an offset so that the whole data is
  constructed as a correct sequence from `0` to `Size - 1`.

  \tparam T The original type that needs the offset attached.
  \tparam Offset An integral value that determines the offset in the complete SimdArray.
 */
template <typename T, std::size_t Offset> struct AddOffset
{
    constexpr AddOffset() = default;
};

/** \internal
  Helper type with static functions to generically adjust arguments for the \c data0 and
  \c data1 members of SimdArray and SimdMaskArray.

  \tparam secondOffset The offset in number of elements that \c data1 has in the SimdArray
                       / SimdMaskArray. This is essentially equal to the number of
                       elements in \c data0.
 */
template <std::size_t secondOffset> class Split/*{{{*/
{
    static constexpr AddOffset<VectorSpecialInitializerIndexesFromZero::IEnum, secondOffset> hiImpl(
        VectorSpecialInitializerIndexesFromZero::IEnum)
    {
        return {};
    }
    template <std::size_t Offset>
    static constexpr AddOffset<VectorSpecialInitializerIndexesFromZero::IEnum,
                               Offset + secondOffset>
        hiImpl(AddOffset<VectorSpecialInitializerIndexesFromZero::IEnum, Offset>)
    {
        return {};
    }

    // split composite SimdArray
    template <typename U, std::size_t N, typename V, std::size_t M>
    static Vc_INTRINSIC auto loImpl(const SimdArray<U, N, V, M> &x) -> decltype(internal_data0(x))
    {
        return internal_data0(x);
    }
    template <typename U, std::size_t N, typename V, std::size_t M>
    static Vc_INTRINSIC auto hiImpl(const SimdArray<U, N, V, M> &x) -> decltype(internal_data1(x))
    {
        return internal_data1(x);
    }
    template <typename U, std::size_t N, typename V, std::size_t M>
    static Vc_INTRINSIC auto loImpl(SimdArray<U, N, V, M> *x) -> decltype(&internal_data0(*x))
    {
        return &internal_data0(*x);
    }
    template <typename U, std::size_t N, typename V, std::size_t M>
    static Vc_INTRINSIC auto hiImpl(SimdArray<U, N, V, M> *x) -> decltype(&internal_data1(*x))
    {
        return &internal_data1(*x);
    }

    template <typename U, std::size_t N, typename V>
    static Vc_INTRINSIC Segment<V, 2, 0> loImpl(const SimdArray<U, N, V, N> &x)
    {
        return {internal_data(x)};
    }
    template <typename U, std::size_t N, typename V>
    static Vc_INTRINSIC Segment<V, 2, 1> hiImpl(const SimdArray<U, N, V, N> &x)
    {
        return {internal_data(x)};
    }
    template <typename U, std::size_t N, typename V>
    static Vc_INTRINSIC Segment<V *, 2, 0> loImpl(const SimdArray<U, N, V, N> *x)
    {
        return {&internal_data(*x)};
    }
    template <typename U, std::size_t N, typename V>
    static Vc_INTRINSIC Segment<V *, 2, 1> hiImpl(const SimdArray<U, N, V, N> *x)
    {
        return {&internal_data(*x)};
    }

    // split composite SimdMaskArray
    template <typename U, std::size_t N, typename V, std::size_t M>
    static Vc_INTRINSIC auto loImpl(const SimdMaskArray<U, N, V, M> &x) -> decltype(internal_data0(x))
    {
        return internal_data0(x);
    }
    template <typename U, std::size_t N, typename V, std::size_t M>
    static Vc_INTRINSIC auto hiImpl(const SimdMaskArray<U, N, V, M> &x) -> decltype(internal_data1(x))
    {
        return internal_data1(x);
    }

    template <typename U, std::size_t N, typename V>
    static Vc_INTRINSIC Segment<typename SimdMaskArray<U, N, V, N>::mask_type, 2, 0> loImpl(
        const SimdMaskArray<U, N, V, N> &x)
    {
        return {internal_data(x)};
    }
    template <typename U, std::size_t N, typename V>
    static Vc_INTRINSIC Segment<typename SimdMaskArray<U, N, V, N>::mask_type, 2, 1> hiImpl(
        const SimdMaskArray<U, N, V, N> &x)
    {
        return {internal_data(x)};
    }

    // split Vector<T> and Mask<T>
    template <typename T>
    static constexpr bool is_vector_or_mask(){
        return (Traits::is_simd_vector<T>::value && !Traits::isSimdArray<T>::value) ||
               (Traits::is_simd_mask<T>::value && !Traits::isSimdMaskArray<T>::value);
    }
    template <typename V>
    static Vc_INTRINSIC Segment<V, 2, 0> loImpl(V &&x, enable_if<is_vector_or_mask<V>()> = nullarg)
    {
        return {std::forward<V>(x)};
    }
    template <typename V>
    static Vc_INTRINSIC Segment<V, 2, 1> hiImpl(V &&x, enable_if<is_vector_or_mask<V>()> = nullarg)
    {
        return {std::forward<V>(x)};
    }

    // generically split Segments
    template <typename V, std::size_t Pieces, std::size_t Index>
    static Vc_INTRINSIC Segment<V, 2 * Pieces, 2 * Index> loImpl(
        const Segment<V, Pieces, Index> &x)
    {
        return {x.data};
    }
    template <typename V, std::size_t Pieces, std::size_t Index>
    static Vc_INTRINSIC Segment<V, 2 * Pieces, 2 * Index + 1> hiImpl(
        const Segment<V, Pieces, Index> &x)
    {
        return {x.data};
    }

    /** \internal
     * \name Checks for existence of \c loImpl / \c hiImpl
     */
    //@{
    template <typename T, typename = decltype(loImpl(std::declval<T>()))>
    static std::true_type have_lo_impl(int);
    template <typename T> static std::false_type have_lo_impl(float);
    template <typename T> static constexpr bool have_lo_impl()
    {
        return decltype(have_lo_impl<T>(1))::value;
    }

    template <typename T, typename = decltype(hiImpl(std::declval<T>()))>
    static std::true_type have_hi_impl(int);
    template <typename T> static std::false_type have_hi_impl(float);
    template <typename T> static constexpr bool have_hi_impl()
    {
        return decltype(have_hi_impl<T>(1))::value;
    }
    //@}

public:
    /** \internal
     * \name with Operations tag
     *
     * These functions don't overload on the data parameter. The first parameter (the tag) clearly
     * identifies the intended function.
     */
    //@{
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
    template <typename U, typename = enable_if<!std::is_pointer<U>::value>>
    static Vc_ALWAYS_INLINE decltype(loImpl(std::declval<U>()))
        lo(Operations::gather, U &&x)
    {
        return loImpl(std::forward<U>(x));
    }
    template <typename U, typename = enable_if<!std::is_pointer<U>::value>>
    static Vc_ALWAYS_INLINE decltype(hiImpl(std::declval<U>()))
        hi(Operations::gather, U &&x)
    {
        return hiImpl(std::forward<U>(x));
    }
    template <typename U>
    static Vc_INTRINSIC const U *lo(Operations::scatter, const U *ptr)
    {
        return ptr;
    }
    template <typename U>
    static Vc_INTRINSIC const U *hi(Operations::scatter, const U *ptr)
    {
        return ptr + secondOffset;
    }
    //@}

    /** \internal
      \name without Operations tag

      These functions are not clearly tagged as to where they are used and therefore
      behave differently depending on the type of the parameter. Different behavior is
      implemented via overloads of \c loImpl and \c hiImpl. They are not overloads of \c
      lo and \c hi directly because it's hard to compete against a universal reference
      (i.e. an overload for `int` requires overloads for `int &`, `const int &`, and `int
      &&`. If one of them were missing `U &&` would win in overload resolution).
     */
    //@{
    template <typename U>
    static Vc_ALWAYS_INLINE decltype(loImpl(std::declval<U>())) lo(U &&x)
    {
        return loImpl(std::forward<U>(x));
    }
    template <typename U>
    static Vc_ALWAYS_INLINE decltype(hiImpl(std::declval<U>())) hi(U &&x)
    {
        return hiImpl(std::forward<U>(x));
    }

    template <typename U>
    static Vc_ALWAYS_INLINE enable_if<!have_lo_impl<U>(), U> lo(U &&x)
    {
        return std::forward<U>(x);
    }
    template <typename U>
    static Vc_ALWAYS_INLINE enable_if<!have_hi_impl<U>(), U> hi(U &&x)
    {
        return std::forward<U>(x);
    }
    //@}
};/*}}}*/

template <typename Op, typename U> static Vc_INTRINSIC U actual_value(Op, U &&x)
{
  return std::forward<U>(x);
}
template <typename Op, typename U, std::size_t M, typename V>
static Vc_INTRINSIC const V &actual_value(Op, const SimdArray<U, M, V, M> &x)
{
  return internal_data(x);
}
template <typename Op, typename U, std::size_t M, typename V>
static Vc_INTRINSIC const V &actual_value(Op, SimdArray<U, M, V, M> &&x)
{
  return internal_data(x);
}
template <typename Op, typename U, std::size_t M, typename V>
static Vc_INTRINSIC V *actual_value(Op, SimdArray<U, M, V, M> *x)
{
  return &internal_data(*x);
}
template <typename Op, typename U, std::size_t M, typename V>
static Vc_INTRINSIC const typename V::Mask &actual_value(Op, const SimdMaskArray<U, M, V, M> &x)
{
  return internal_data(x);
}
template <typename Op, typename U, std::size_t M, typename V>
static Vc_INTRINSIC const typename V::Mask &actual_value(Op, SimdMaskArray<U, M, V, M> &&x)
{
  return internal_data(x);
}
template <typename Op, typename U, std::size_t M, typename V>
static Vc_INTRINSIC typename V::Mask *actual_value(Op, SimdMaskArray<U, M, V, M> *x)
{
  return &internal_data(*x);
}

/// @}

}  // namespace Common
}  // namespace Vc

#include "undomacros.h"

#endif  // VC_COMMON_SIMDARRAYHELPER_H_

// vim: foldmethod=marker
