/*  This file is part of the Vc library. {{{
Copyright © 2013-2014 Matthias Kretz <kretz@kde.org>
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
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_COMMON_SUBSCRIPT_H
#define VC_COMMON_SUBSCRIPT_H

#include <initializer_list>
#include <type_traits>
#include <vector>
#include "types.h"
#include "macros.h"
#include <assert.h>

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{
// AdaptSubscriptOperator {{{1
template <typename Base> class AdaptSubscriptOperator : public Base
{
public:
    // perfect forward all Base constructors
    template <typename... Args>
    Vc_ALWAYS_INLINE AdaptSubscriptOperator(Args &&... arguments)
        : Base(std::forward<Args>(arguments)...)
    {
    }

    // perfect forward all Base constructors
    template <typename T>
    Vc_ALWAYS_INLINE AdaptSubscriptOperator(std::initializer_list<T> l)
        : Base(l)
    {
    }

    // explicitly enable Base::operator[] because the following would hide it
    using Base::operator[];

    /// \internal forward to non-member subscript_operator function
    template <typename I,
              typename = enable_if<
                  !std::is_arithmetic<typename std::decay<I>::type>::value>  // arithmetic types
                                                                             // should always use
                                                                             // Base::operator[] and
                                                                             // never match this one
              >
    Vc_ALWAYS_INLINE auto operator[](I &&arg__) -> decltype(subscript_operator(*this, std::forward<I>(arg__)))
    {
        return subscript_operator(*this, std::forward<I>(arg__));
    }

    // const overload of the above
    template <typename I,
              typename = enable_if<!std::is_arithmetic<typename std::decay<I>::type>::value>>
    Vc_ALWAYS_INLINE auto operator[](I &&arg__) const -> decltype(subscript_operator(*this, std::forward<I>(arg__)))
    {
        return subscript_operator(*this, std::forward<I>(arg__));
    }
};
// Fraction {{{1
template <std::size_t Numerator, std::size_t Denominator> struct Fraction
{
    static constexpr std::size_t value()
    {
        return Numerator / Denominator;
    }
    template <std::size_t Numerator2, std::size_t Denominator2>
    using Multiply = Fraction<Numerator2 *Numerator, Denominator2 *Denominator>;

    template <typename T,
              typename = enable_if<Traits::has_multiply_operator<T, decltype(value())>::value>>
    static Vc_ALWAYS_INLINE typename std::decay<T>::type apply(T &&x)
    {
        auto tmp = std::forward<T>(x) * value();
        VC_ASSERT(tmp / value() == x);
        return std::move(tmp);
    }

    template <typename T,
              typename = enable_if<!Traits::has_multiply_operator<T, decltype(value())>::value>>
    static Vc_ALWAYS_INLINE T apply(T x)
    {
        for (size_t i = 0; i < x.size(); ++i) {
            VC_ASSERT(x[i] * value() / value() == x[i]);
            x[i] *= value();
        }
        return x;
    }

    template <typename T,
              typename U,
              typename = enable_if<Traits::has_multiply_operator<T, decltype(value())>::value &&
                                       Traits::has_addition_operator<T, U>::value>>
    static Vc_ALWAYS_INLINE typename std::decay<T>::type applyAndAdd(T &&x, U &&y)
    {
        return std::forward<T>(x) * value() + std::forward<U>(y);
    }

    template <typename T,
              typename U,
              typename = enable_if<!(Traits::has_multiply_operator<T &, decltype(value())>::value &&
                                         Traits::has_addition_operator<T &, decltype(std::declval<U>()[0])>::value) &&
                                   Traits::has_subscript_operator<U>::value>>
    static Vc_ALWAYS_INLINE T applyAndAdd(T x, U &&y)
    {
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = x[i] * value() + y[i];
        }
        return x;
    }

    template <typename T, typename U>
    static Vc_ALWAYS_INLINE enable_if<!(Traits::has_multiply_operator<T &, decltype(value())>::
                                            value &&Traits::has_addition_operator<T &, U>::value) &&
                                          !Traits::has_subscript_operator<U>::value,
                                      T> applyAndAdd(T x, U &&y)
    {
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = x[i] * value() + y;
        }
        return x;
    }
};

template <> struct Fraction<1, 1>
{
    static constexpr std::size_t value()
    {
        return 1;
    }
    template <std::size_t Numerator2, std::size_t Denominator2>
    using Multiply = Fraction<Numerator2, Denominator2>;

    template <typename T> static Vc_ALWAYS_INLINE T apply(T &&x)
    {
        return std::forward<T>(x);
    }

    template <typename T, typename U>
    static Vc_ALWAYS_INLINE decltype(std::declval<T>() + std::declval<U>())  // this return type
                                                                             // leads to a
                                                                             // substitution failure
                                                                             // if operator+ does
                                                                             // not exist
        applyAndAdd(T &&x, U &&y)
    {
        return std::forward<T>(x) + std::forward<U>(y);
    }

    template <typename T, typename U> static Vc_ALWAYS_INLINE T applyAndAdd(T x, U &&y)
    {
        auto yIt = begin(y);
        for (auto &entry : x) {
            entry += *yIt;
            ++yIt;
        }
        return x;
    }
};
// IndexVectorSizeMatches {{{1
template <std::size_t MinSize,
          typename IndexT,
          bool = Traits::is_simd_vector<IndexT>::value>
struct IndexVectorSizeMatches
    : public std::true_type  // you might expect this should be false_type here, but the point is
                             // that IndexT is a type where the size is not known at compile time.
                             // Thus it may be good but we cannot know from the type. The only check
                             // we could do is a runtime check, but the type is fine.
{
};

template <std::size_t MinSize, typename V>
struct IndexVectorSizeMatches<MinSize,
                              V,
                              true> : public std::integral_constant<bool, (MinSize <= V::Size)>
{
};

template <std::size_t MinSize, typename T, std::size_t ArraySize>
struct IndexVectorSizeMatches<MinSize,
                              T[ArraySize],
                              false> : public std::integral_constant<bool, (MinSize <= ArraySize)>
{
};

template <std::size_t MinSize, typename T, std::size_t ArraySize>
struct IndexVectorSizeMatches<MinSize,
                              std::array<T, ArraySize>,
                              false> : public std::integral_constant<bool, (MinSize <= ArraySize)>
{
};

template <std::size_t MinSize, typename T, std::size_t ArraySize>
struct IndexVectorSizeMatches<MinSize,
                              Vc::array<T, ArraySize>,
                              false> : public std::integral_constant<bool, (MinSize <= ArraySize)>
{
};
// SubscriptOperation {{{1
template <typename T, typename IndexVector, typename Scale = Fraction<1, 1>>
class SubscriptOperation
{
    const IndexVector m_indexes;
    T *const m_address;
    using ScalarType = typename std::decay<T>::type;

    using IndexVectorScaled = std::vector<unsigned int>;// typename std::conditional<
        //(sizeof(std::declval<const IndexVector &>()[0]) < sizeof(int)),
        //typename std::conditional<
            //Traits::is_simd_vector<IndexVector>::value,
            //Vc::simdarray<unsigned int, IndexVector::Size>,
            //std::vector<unsigned int>/*>::type*/,
        //IndexVector>::type;

    IndexVectorScaled convertedIndexes() const
    {
        IndexVectorScaled r(begin(m_indexes), end(m_indexes));
        return r;
    }

public:
    constexpr Vc_ALWAYS_INLINE SubscriptOperation(T *address, const IndexVector &indexes)
        : m_indexes(indexes), m_address(address)
    {
    }

    Vc_ALWAYS_INLINE GatherArguments<T, IndexVectorScaled> gatherArguments() const
    {
        static_assert(std::is_arithmetic<ScalarType>::value,
                      "Incorrect type for a SIMD vector gather. Must be an arithmetic type.");
        return {Scale::apply(convertedIndexes()), m_address};
    }

    Vc_ALWAYS_INLINE ScatterArguments<T, IndexVectorScaled> scatterArguments() const
    {
        static_assert(std::is_arithmetic<ScalarType>::value,
                      "Incorrect type for a SIMD vector scatter. Must be an arithmetic type.");
        return {Scale::apply(convertedIndexes()), m_address};
    }

    template <typename V,
              typename = enable_if<(std::is_arithmetic<ScalarType>::value &&Traits::is_simd_vector<
                  V>::value &&IndexVectorSizeMatches<V::Size, IndexVector>::value)>>
    Vc_ALWAYS_INLINE operator V() const
    {
        static_assert(std::is_arithmetic<ScalarType>::value,
                      "Incorrect type for a SIMD vector gather. Must be an arithmetic type.");
        const IndexVectorScaled indexes = Scale::apply(convertedIndexes());
        return V(m_address, &indexes[0]);
    }

    template <typename V,
              typename = enable_if<(std::is_arithmetic<ScalarType>::value &&Traits::is_simd_vector<
                  V>::value &&IndexVectorSizeMatches<V::Size, IndexVector>::value)>>
    Vc_ALWAYS_INLINE SubscriptOperation &operator=(const V &rhs)
    {
        static_assert(std::is_arithmetic<ScalarType>::value,
                      "Incorrect type for a SIMD vector scatter. Must be an arithmetic type.");
        const IndexVectorScaled indexes = Scale::apply(convertedIndexes());
        rhs.scatter(m_address, &indexes[0]);
        return *this;
    }

    // precondition: m_address points to a struct/class/union
    template <typename U,
              typename S,  // S must be equal to T. Still we require this template parameter -
                           // otherwise instantiation of SubscriptOperation would only be valid for
                           // structs/unions.
              typename = enable_if<
                  std::is_same<S, T>::value &&(std::is_class<T>::value || std::is_union<T>::value)>>
    Vc_ALWAYS_INLINE auto operator[](U S::*member)
        -> SubscriptOperation<
              typename std::remove_reference<decltype(m_address->*member)>::type,
              IndexVector,
              typename Scale::template Multiply<
                  sizeof(S),
                  sizeof(m_address->*member)>  // By passing the scale factor as a fraction of
                                               // integers in the template arguments the value does
                                               // not lose information if the division yields a
                                               // non-integral value. This could happen e.g. for a
                                               // struct of struct (S2 { S1, char }, with sizeof(S1)
                                               // = 16, sizeof(S2) = 20. Then scale would be 20/16)
              >
    {
        // TODO: check whether scale really works for unions correctly
        return {&(m_address->*member), m_indexes};
    }

    /*
     * The following functions allow subscripting of nested arrays. But
     * there are two cases of containers and only one that we want to support:
     * 1. actual arrays (e.g. T[N] or std::array<T, N>)
     * 2. dynamically allocated vectors (e.g. std::vector<T>)
     *
     * For (1.) the offset calculation is straightforward.
     * For (2.) the m_address pointer points to memory where pointers are
     * stored to the actual data. Meaning the data can be scattered
     * freely in memory (and far away from what m_address points to). Supporting this leads to
     * serious trouble with the pointer (it does not really point to the start of a memory
     * region anymore) and inefficient code. The user is better off to write a loop that assigns the
     * scalars to the vector object sequentially.
     */

    // precondition: m_address points to a type that implements the subscript operator
    template <typename U = T>  // U is only required to delay name lookup to the 2nd phase (on use).
                               // This is necessary because m_address[0][index] is only a correct
                               // expression if has_subscript_operator<T>::value is true.
    Vc_ALWAYS_INLINE auto operator[](
        enable_if<
#ifndef VC_IMPROVE_ERROR_MESSAGES
            Traits::has_no_allocated_data<T>::value &&Traits::has_subscript_operator<T>::value &&
#endif
                std::is_same<T, U>::value,
            std::size_t> index)
        -> SubscriptOperation<
              typename std::remove_reference<decltype(m_address[0][index])>::type,
              IndexVector,
              typename Scale::template Multiply<sizeof(T), sizeof(m_address[0][index])>>
    {
        static_assert(Traits::has_subscript_operator<T>::value,
                      "The subscript operator was called on a type that does not implement it.\n");
        static_assert(Traits::has_no_allocated_data<T>::value,
                      "Invalid container type in gather/scatter operation.\nYou may only use "
                      "nested containers that store the data inside the object (such as builtin "
                      "arrays or std::array) but not containers that store data in allocated "
                      "memory (such as std::vector).\nSince this feature cannot be queried "
                      "generically at compile time you need to spezialize the "
                      "Vc::Traits::has_no_allocated_data_impl<T> type-trait for custom types that "
                      "meet the requirements.\n");
        return {&(m_address[0][index]), m_indexes};
    }

    // precondition: m_address points to a type that implements the subscript operator
    template <typename IT>
    Vc_ALWAYS_INLINE enable_if<
#ifndef VC_IMPROVE_ERROR_MESSAGES
        Traits::has_no_allocated_data<T>::value &&Traits::has_subscript_operator<T>::value &&
#endif
            Traits::has_subscript_operator<IT>::value,
        SubscriptOperation<
            typename std::remove_reference<
                decltype(m_address[0][std::declval<const IT &>()[0]]  // std::declval<IT>()[0] could
                                                                      // be replaced with 0 if it
                         // were not for two-phase lookup. We need to make the
                         // m_address[0][0] expression dependent on IT
                         )>::type,
            IndexVectorScaled,
            Fraction<1, 1>  // reset Scale to 1 since it is applied below
            >>
        operator[](const IT &index)
    {
        static_assert(Traits::has_subscript_operator<T>::value,
                      "The subscript operator was called on a type that does not implement it.\n");
        static_assert(Traits::has_no_allocated_data<T>::value,
                      "Invalid container type in gather/scatter operation.\nYou may only use "
                      "nested containers that store the data inside the object (such as builtin "
                      "arrays or std::array) but not containers that store data in allocated "
                      "memory (such as std::vector).\nSince this feature cannot be queried "
                      "generically at compile time you need to spezialize the "
                      "Vc::Traits::has_no_allocated_data_impl<T> type-trait for custom types that "
                      "meet the requirements.\n");
        using ScaleHere = typename Scale::template Multiply<sizeof(T), sizeof(m_address[0][0])>;
        return {&(m_address[0][0]), ScaleHere::applyAndAdd(convertedIndexes(), index)};
    }
};
// subscript_operator {{{1
template <
    typename Container,
    typename IndexVector,
    typename = enable_if<
        Traits::has_subscript_operator<IndexVector>::value  // The index vector must provide [] for
                                                            // the implementations of gather/scatter
        &&Traits::has_contiguous_storage<Container>::value  // Container must use contiguous
                                                            // storage, otherwise the index vector
        // cannot be used as memory offsets, which is required for efficient
        // gather/scatter implementations
        &&std::is_lvalue_reference<decltype(*begin(std::declval<
            Container>()))>::value  // dereferencing the begin iterator must yield an lvalue
                                    // reference (const or non-const). Otherwise it is not possible
                                    // to determine a pointer to the data storage (see above).
        >>
Vc_ALWAYS_INLINE SubscriptOperation<
    typename std::remove_reference<decltype(*begin(std::declval<Container>()))>::
        type,  // the type of the first value in the container is what the internal array pointer
               // has to point to. But if the subscript operator of the container returns a
               // reference we need to drop that part because it's useless information for us. But
               // const and volatile, as well as array rank/extent are interesting and need not be
               // dropped.
    typename std::remove_const<typename std::remove_reference<
        IndexVector>::type>::type  // keep volatile and possibly the array extent, but the const and
                                   // & parts of the type need to be removed because
                                   // SubscriptOperation explicitly adds them for its member type
    > subscript_operator(Container &&c, IndexVector &&indexes)
{
    VC_ASSERT(std::addressof(*begin(c)) + 1 ==
              std::addressof(*(begin(c) + 1)));  // runtime assertion for contiguous storage, this
                                                 // requires a RandomAccessIterator - but that
                                                 // should be given for a container with contiguous
                                                 // storage
    return {std::addressof(*begin(c)), std::forward<IndexVector>(indexes)};
}


static_assert(Traits::has_subscript_operator<SubscriptOperation<int[100][100], const int *>, int>::value, "");
static_assert(Traits::has_subscript_operator<SubscriptOperation<int[100][100], const int *>, int[4]>::value, "");
static_assert(!Traits::has_subscript_operator<SubscriptOperation<std::vector<int>, const int *>, int[4]>::value, "");

/**
 * \internal
 * Implement subscripts of std::initializer_list. This function must be in the global scope
 * because Container arguments may be in any scope. The other argument is in std scope.
 *
 * -----
 * std::initializer_list does not have constexpr member functions in C++11, but from C++14 onwards
 * the world is a happier place. :)
 */
template <typename Container, typename I>
Vc_ALWAYS_INLINE Vc::Common::SubscriptOperation<
    typename std::remove_reference<decltype(std::declval<Container>()[0])>::type,
    const std::initializer_list<I> &> subscript_operator(Container &&vec,
                                                   const std::initializer_list<I> &indexes)
{
    return {&vec[0], indexes};
}
//}}}1

}  // namespace Common

using Common::subscript_operator;

namespace Traits
{
template <typename T, typename IndexVector, typename Scale>
struct is_subscript_operation_internal<Common::SubscriptOperation<T, IndexVector, Scale>> : public std::true_type
{
};
}  // namespace Traits

namespace Scalar
{
    using Common::subscript_operator;
}  // namespace
namespace SSE
{
    using Common::subscript_operator;
}  // namespace
namespace AVX
{
    using Common::subscript_operator;
}  // namespace
namespace AVX2
{
    using Common::subscript_operator;
}  // namespace
namespace MIC
{
    using Common::subscript_operator;
}  // namespace

}  // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_SUBSCRIPT_H

// vim: foldmethod=marker
