/*  This file is part of the Vc library. {{{
Copyright © 2014 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_COMMON_SIMDIZE_H_
#define VC_COMMON_SIMDIZE_H_

#include <tuple>
#include <array>

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
/**\internal
 * This namespace contains all the required code for implementing simdize<T>. None of this
 * code should be directly accessed by users, though the unit test for simdize<T>
 * certainly may look into some of the details if necessary.
 */
namespace SimdizeDetail
{
using std::is_same;
using std::is_base_of;
using std::false_type;
using std::true_type;
using std::iterator_traits;
using std::conditional;
using std::size_t;

/**\internal
 * Typelist is a simple helper class for supporting multiple parameter packs in one class
 * template.
 */
template <typename... Ts> struct Typelist;

/**\internal
 * The Category identifies how the type argument to simdize<T> has to be transformed.
 */
enum class Category {
    /// No transformation
    None = 0x0,
    /// simple Vector<T> transformation
    ArithmeticVectorizable = 0x1,
    /// transform a forward iterator to return vectorized entries
    ForwardIterator = 0x2,
    /// transform a random access iterator to return vectorized entries
    RandomAccessIterator = 0x6,
    /// transform a class template recursively
    ClassTemplate = 0x8
};

/**\internal
 * iteratorCategories<T>(int()) returns whether iterator_traits<T>::iterator_category is a
 * valid type and whether it is derived from RandomAccessIterator or ForwardIterator.
 */
template <typename T,
          typename ItCat = typename iterator_traits<T>::iterator_category>
constexpr Category iteratorCategories(int)
{
    return is_base_of<std::random_access_iterator_tag, ItCat>::value
               ? Category::RandomAccessIterator
               : is_base_of<std::forward_iterator_tag, ItCat>::value
                     ? Category::ForwardIterator
                     : Category::None;
}
/**\internal
 * This overload is selected if T does not work with iterator_traits.
 */
template <typename T> constexpr Category iteratorCategories(...)
{
    return Category::None;
}

/**\internal
 * Simple trait to identify whether a type T is a class template or not.
 */
template <typename T> struct is_class_template : public false_type
{
};
template <template <typename...> class C, typename... Ts>
struct is_class_template<C<Ts...>> : public true_type
{
};

/**\internal
 * Returns the Category for the given type \p T.
 */
template <typename T> constexpr Category typeCategory()
{
    return (is_same<T, bool>::value || is_same<T, short>::value ||
            is_same<T, unsigned short>::value || is_same<T, int>::value ||
            is_same<T, unsigned int>::value || is_same<T, float>::value ||
            is_same<T, double>::value)
               ? Category::ArithmeticVectorizable
               : iteratorCategories<T>(int()) != Category::None
                     ? iteratorCategories<T>(int())
                     : is_class_template<T>::value ? Category::ClassTemplate
                                                   : Category::None;
}
/**\internal
 * The type behind the simdize expression whose member type \c type determines the
 * transformed type.
 *
 * \tparam T The type to be transformed.
 * \tparam N The width the resulting vectorized type should have. A value of 0 lets the
 *           implementation choose the width.
 * \tparam MT The base type to use for mask types. If set to \c void the implementation
 *            chooses the type itself.
 * \tparam Category The type category of \p T. This determines the implementation strategy
 *                  (via template specialization).
 */
template <typename T, size_t N, typename MT, Category = typeCategory<T>()>
struct ReplaceTypes
{
    typedef T type;
};

/**\internal
 * The ReplaceTypes class template is nicer to use as an alias template. This is exported
 * to the outer Vc namespace.
 */
template <typename T, size_t N = 0, typename MT = void>
using simdize = typename SimdizeDetail::ReplaceTypes<T, N, MT>::type;

/**\internal
 * ReplaceTypes specialization for simdizable arithmetic types. This results in either
 * Vector<T> or simdarray<T, N>.
 */
template <typename T, size_t N, typename MT>
struct ReplaceTypes<T, N, MT, Category::ArithmeticVectorizable>
    : public conditional<(N == 0 || Vector<T>::size() == N), Vector<T>, simdarray<T, N>>
{
};

/**\internal
 * ReplaceTypes specialization for bool. This results either in Mask<MT> or
 * simd_mask_array<MT, N>.
 */
template <size_t N, typename MT>
struct ReplaceTypes<bool, N, MT, Category::ArithmeticVectorizable>
    : public conditional<(N == 0 || Mask<MT>::size() == N), Mask<MT>,
                         simd_mask_array<MT, N>>
{
};
/**\internal
 * ReplaceTypes specialization for bool and MT = void. In that case MT is set to float.
 */
template <size_t N>
struct ReplaceTypes<bool, N, void, Category::ArithmeticVectorizable>
    : public ReplaceTypes<bool, N, float, Category::ArithmeticVectorizable>
{
};

/**\internal
 * This type substitutes the first type (\p T) in \p Remaining via simdize<T, N, MT> and
 * appends it to the Typelist in \p Replaced. If \p N = 0, the first simdize expression
 * that yields a vectorized type determines \p N for the subsequent SubstituteOneByOne
 * instances.
 */
template <size_t N, typename MT, typename Replaced, typename... Remaining>
struct SubstituteOneByOne;

/**\internal
 * Template specialization for the case that there is at least one type in \p Remaining.
 * The member type \p type recurses via SubstituteOneByOne.
 */
template <size_t N, typename MT, typename... Replaced, typename T,
          typename... Remaining>
struct SubstituteOneByOne<N, MT, Typelist<Replaced...>, T, Remaining...>
{
private:
    /**
     * If \p U::size() yields a constant expression convertible to size_t then value will
     * be equal to U::size(), 0 otherwise.
     */
    template <typename U, size_t M = U::size()>
    static std::integral_constant<size_t, M> size_or_0(int);
    template <typename U> static std::integral_constant<size_t, 0> size_or_0(...);

    /// The vectorized type for \p T.
    using V = simdize<T, N, MT>;

    /**
     * Determine the new \p N to use for the SubstituteOneByOne expression below. If N is
     * non-zero that value is used. Otherwise size_or_0<V> determines the new value.
     */
    static constexpr auto NewN = N != 0 ? N : decltype(size_or_0<V>(int()))::value;

    /**
     * Determine the new \p MT type to use for the SubstituteOneByOne expression below.
     * This is normally the old \p MT type. However, if N != NewN and MT = void, NewMT is
     * set to either \c float or \p T, depending on whether \p T is \c bool or not.
     */
    typedef typename conditional<
        (N != NewN && is_same<MT, void>::value),
        typename conditional<is_same<T, bool>::value, float, T>::type,
        MT>::type NewMT;

public:
    /// An alias to the type member of the completed recursion over SubstituteOneByOne.
    using type = typename SubstituteOneByOne<NewN, NewMT, Typelist<Replaced..., V>,
                                             Remaining...>::type;
};

/**\internal
 * Template specialization that ends the recursion and determines the return type \p type.
 * The end of the recursion is identified by an empty typelist (i.e. no template
 * parameters) after the Typelist parameter.
 */
template <size_t N_, typename MT, typename Replaced0, typename... Replaced>
struct SubstituteOneByOne<N_, MT, Typelist<Replaced0, Replaced...>>
{
    /// Return type for returning the vector width and list of substituted types
    struct type
    {
        static constexpr auto N = N_;
        /**
         * Alias template to construct a class template instantiation with the replaced
         * types.
         */
        template <template <typename...> class C>
        using Substituted = C<Replaced0, Replaced...>;
        /**
         * Alias template to construct a class template instantiation with only one
         * replaced type but several values. This is a hack for supporting e.g.
         * std::array<T, N>.
         */
        template <typename ValueT, template <typename, ValueT...> class C,
                  ValueT... Values>
        using Substituted1 = C<Replaced0, Values...>;
        template <typename ValueT, template <typename, typename, ValueT...> class C,
                  ValueT... Values>
        using Substituted2 = C<Replaced0, Replaced..., Values...>;
        template <typename ValueT, template <typename, typename, typename, ValueT...>
                                   class C, ValueT... Values>
        using Substituted3 = C<Replaced0, Replaced..., Values...>;
        template <typename ValueT,
                  template <typename, typename, typename, typename, ValueT...> class C,
                  ValueT... Values>
        using Substituted4 = C<Replaced0, Replaced..., Values...>;
        template <typename ValueT,
                  template <typename, typename, typename, typename, typename, ValueT...>
                  class C, ValueT... Values>
        using Substituted5 = C<Replaced0, Replaced..., Values...>;
        template <typename ValueT, template <typename, typename, typename, typename,
                                             typename, typename, ValueT...> class C,
                  ValueT... Values>
        using Substituted6 = C<Replaced0, Replaced..., Values...>;
        template <typename ValueT,
                  template <typename, typename, typename, typename, typename, typename,
                            typename, ValueT...> class C,
                  ValueT... Values>
        using Substituted7 = C<Replaced0, Replaced..., Values...>;
        template <typename ValueT,
                  template <typename, typename, typename, typename, typename, typename,
                            typename, typename, ValueT...> class C,
                  ValueT... Values>
        using Substituted8 = C<Replaced0, Replaced..., Values...>;
    };
};

/**\internal
 * Vectorized class templates are not substituted directly by ReplaceTypes/simdize.
 * Instead the replaced type is used as a base class for an adapter type which enables
 * the addition of extra operations. Specifically the following features are added:
 * \li a constexpr \p size() function, which returns the width of the vectorization. Note
 *     that this may hide a \p size() member in the original class template (e.g. for STL
 *     container classes).
 * \li The member type \p base_type is an alias for the vectorized (i.e. substituted)
 *     class template
 * \li The member type \p scalar_type is an alias for the class template argument
 *     originally passed to the \ref simdize expression.
 *
 * \tparam Scalar
 * \tparam Base
 * \tparam N
 */
template <typename Scalar, typename Base, size_t N> class Adapter;

/**\internal
 * Specialization of ReplaceTypes for class templates (\p C) where each template argument
 * needs to be substituted via SubstituteOneByOne.
 */
template <template <typename...> class C, typename... Ts, size_t N, typename MT>
struct ReplaceTypes<C<Ts...>, N, MT, Category::ClassTemplate>
{
    /// The \p type member of the SubstituteOneByOne instantiation
    using SubstitutionResult =
        typename SubstituteOneByOne<N, MT, Typelist<>, Ts...>::type;
    /**
     * This expression instantiates the class template \p C with the substituted template
     * arguments in the \p Ts parameter pack. The alias \p Vectorized thus is the
     * vectorized equivalent to \p C<Ts...>.
     */
    using Vectorized = typename SubstitutionResult::template Substituted<C>;
    /**
     * The result type of this ReplaceTypes instantiation is set to \p C<Ts...> if no
     * template parameter substitution was done in SubstituteOneByOne. Otherwise, the type
     * aliases an Adapter instantiation.
     */
    using type =
        typename conditional<is_same<C<Ts...>, Vectorized>::value, C<Ts...>,
                             Adapter<C<Ts...>, Vectorized, SubstitutionResult::N>>::type;
};

/**\internal
 * Specialization of the ReplaceTypes class template allowing transformation of class
 * templates with non-type parameters. This is impossible to express with variadic
 * templates and therefore requires a lot of code duplication.
 */
#define Vc_DEFINE_NONTYPE_REPLACETYPES__(ValueType__)                                    \
    template <template <typename, ValueType__...> class C, typename T,                   \
              ValueType__ Value0, ValueType__... Values>                                 \
    struct is_class_template<C<T, Value0, Values...>> : public true_type                 \
    {                                                                                    \
    };                                                                                   \
    template <template <typename, typename, ValueType__...> class C, typename T0,        \
              typename T1, ValueType__ Value0, ValueType__... Values>                    \
    struct is_class_template<C<T0, T1, Value0, Values...>> : public true_type            \
    {                                                                                    \
    };                                                                                   \
    template <template <typename, typename, typename, ValueType__...> class C,           \
              typename T0, typename T1, typename T2, ValueType__ Value0,                 \
              ValueType__... Values>                                                     \
    struct is_class_template<C<T0, T1, T2, Value0, Values...>> : public true_type        \
    {                                                                                    \
    };                                                                                   \
    template <template <typename, typename, typename, typename, ValueType__...> class C, \
              typename T0, typename T1, typename T2, typename T3, ValueType__ Value0,    \
              ValueType__... Values>                                                     \
    struct is_class_template<C<T0, T1, T2, T3, Value0, Values...>> : public true_type    \
    {                                                                                    \
    };                                                                                   \
    template <                                                                           \
        template <typename, typename, typename, typename, typename, ValueType__...>      \
        class C, typename T0, typename T1, typename T2, typename T3, typename T4,        \
        ValueType__ Value0, ValueType__... Values>                                       \
    struct is_class_template<C<T0, T1, T2, T3, T4, Value0, Values...>>                   \
        : public true_type                                                               \
    {                                                                                    \
    };                                                                                   \
    template <template <typename, typename, typename, typename, typename, typename,      \
                        ValueType__...> class C,                                         \
              typename T0, typename T1, typename T2, typename T3, typename T4,           \
              typename T5, ValueType__ Value0, ValueType__... Values>                    \
    struct is_class_template<C<T0, T1, T2, T3, T4, T5, Value0, Values...>>               \
        : public true_type                                                               \
    {                                                                                    \
    };                                                                                   \
    template <template <typename, typename, typename, typename, typename, typename,      \
                        typename, ValueType__...> class C,                               \
              typename T0, typename T1, typename T2, typename T3, typename T4,           \
              typename T5, typename T6, ValueType__ Value0, ValueType__... Values>       \
    struct is_class_template<C<T0, T1, T2, T3, T4, T5, T6, Value0, Values...>>           \
        : public true_type                                                               \
    {                                                                                    \
    };                                                                                   \
    template <template <typename, ValueType__...> class C, typename T0,                  \
              ValueType__ Value0, ValueType__... Values, size_t N, typename MT>          \
    struct ReplaceTypes<C<T0, Value0, Values...>, N, MT, Category::ClassTemplate>        \
    {                                                                                    \
        typedef typename SubstituteOneByOne<N, MT, Typelist<>, T0>::type tmp;            \
        typedef typename tmp::template Substituted1<ValueType__, C, Value0, Values...>   \
            Substituted;                                                                 \
        static constexpr auto NN = tmp::N;                                               \
        typedef typename conditional<                                                    \
            is_same<C<T0, Value0, Values...>, Substituted>::value,                       \
            C<T0, Value0, Values...>,                                                    \
            Adapter<C<T0, Value0, Values...>, Substituted, NN>>::type type;              \
    };                                                                                   \
    template <template <typename, typename, ValueType__...> class C, typename T0,        \
              typename T1, ValueType__ Value0, ValueType__... Values, size_t N,          \
              typename MT>                                                               \
    struct ReplaceTypes<C<T0, T1, Value0, Values...>, N, MT, Category::ClassTemplate>    \
    {                                                                                    \
        typedef typename SubstituteOneByOne<N, MT, Typelist<>, T0, T1>::type tmp;        \
        typedef typename tmp::template Substituted2<ValueType__, C, Value0, Values...>   \
            Substituted;                                                                 \
        static constexpr auto NN = tmp::N;                                               \
        typedef typename conditional<                                                    \
            is_same<C<T0, T1, Value0, Values...>, Substituted>::value,                   \
            C<T0, T1, Value0, Values...>,                                                \
            Adapter<C<T0, T1, Value0, Values...>, Substituted, NN>>::type type;          \
    };                                                                                   \
    template <template <typename, typename, typename, ValueType__...> class C,           \
              typename T0, typename T1, typename T2, ValueType__ Value0,                 \
              ValueType__... Values, size_t N, typename MT>                              \
    struct ReplaceTypes<C<T0, T1, T2, Value0, Values...>, N, MT,                         \
                        Category::ClassTemplate>                                         \
    {                                                                                    \
        typedef typename SubstituteOneByOne<N, MT, Typelist<>, T0, T1, T2>::type tmp;    \
        typedef typename tmp::template Substituted3<ValueType__, C, Value0, Values...>   \
            Substituted;                                                                 \
        static constexpr auto NN = tmp::N;                                               \
        typedef typename conditional<                                                    \
            is_same<C<T0, T1, T2, Value0, Values...>, Substituted>::value,               \
            C<T0, T1, T2, Value0, Values...>,                                            \
            Adapter<C<T0, T1, T2, Value0, Values...>, Substituted, NN>>::type type;      \
    }
Vc_DEFINE_NONTYPE_REPLACETYPES__(bool);
Vc_DEFINE_NONTYPE_REPLACETYPES__(wchar_t);
Vc_DEFINE_NONTYPE_REPLACETYPES__(char);
Vc_DEFINE_NONTYPE_REPLACETYPES__(  signed char);
Vc_DEFINE_NONTYPE_REPLACETYPES__(unsigned char);
Vc_DEFINE_NONTYPE_REPLACETYPES__(  signed short);
Vc_DEFINE_NONTYPE_REPLACETYPES__(unsigned short);
Vc_DEFINE_NONTYPE_REPLACETYPES__(  signed int);
Vc_DEFINE_NONTYPE_REPLACETYPES__(unsigned int);
Vc_DEFINE_NONTYPE_REPLACETYPES__(  signed long);
Vc_DEFINE_NONTYPE_REPLACETYPES__(unsigned long);
Vc_DEFINE_NONTYPE_REPLACETYPES__(  signed long long);
Vc_DEFINE_NONTYPE_REPLACETYPES__(unsigned long long);
#undef Vc_DEFINE_NONTYPE_REPLACETYPES__

#ifdef VC_ICC
template <typename Class, typename... Args>
constexpr bool is_constructible_with_single_brace()
{
    return true;
}
template <typename Class, typename... Args>
constexpr bool is_constructible_with_double_brace()
{
    return false;
}
#else
namespace is_constructible_with_single_brace_impl
{
template <typename T> T create();
template <typename Class, typename... Args,
          typename = decltype((Class{create<Args>()...}))>
std::true_type test(int);
template <typename Class, typename... Args> std::false_type test(...);
}  // namespace is_constructible_with_single_brace_impl

template <typename Class, typename... Args>
constexpr bool is_constructible_with_single_brace()
{
    return decltype(
        is_constructible_with_single_brace_impl::test<Class, Args...>(1))::value;
}
static_assert(
    is_constructible_with_single_brace<std::tuple<int, int, int>, int, int, int>(), "");
static_assert(is_constructible_with_single_brace<std::array<int, 3>, int, int, int>(),
              "");

namespace is_constructible_with_double_brace_impl
{
template <typename T> T create();
template <typename Class, typename... Args,
          typename = decltype(Class{{create<Args>()...}})>
std::true_type test(int);
template <typename Class, typename... Args> std::false_type test(...);
}  // namespace is_constructible_with_double_brace_impl

template <typename Class, typename... Args>
constexpr bool is_constructible_with_double_brace()
{
    return decltype(
        is_constructible_with_double_brace_impl::test<Class, Args...>(1))::value;
}
static_assert(
    !is_constructible_with_double_brace<std::tuple<int, int, int>, int, int, int>(), "");
static_assert(is_constructible_with_double_brace<std::array<int, 3>, int, int, int>(),
              "");
#endif

// see above
template <typename Scalar, typename Base, size_t N> class Adapter : public Base
{
private:
    /// helper for the broadcast ctor below using double braces for Base initialization
    template <std::size_t... Indexes>
    Adapter(
        enable_if<is_constructible_with_double_brace<
                      Base, decltype(get<Indexes>(std::declval<const Scalar &>()))...>(),
                  const Scalar &> x,
        Vc::index_sequence<Indexes...>)
        : Base{{get<Indexes>(x)...}}
    {
    }

    /// helper for the broadcast ctor below using single braces for Base initialization
    template <std::size_t... Indexes>
    Adapter(
        enable_if<
            is_constructible_with_single_brace<
                Base, decltype(get<Indexes>(std::declval<const Scalar &>()))...>() &&
                !is_constructible_with_double_brace<
                    Base, decltype(get<Indexes>(std::declval<const Scalar &>()))...>(),
            const Scalar &> x,
        Vc::index_sequence<Indexes...>)
        : Base{get<Indexes>(x)...}
    {
    }

    /// helper for the broadcast ctor below using parenthesis for Base initialization
    template <std::size_t... Indexes>
    Adapter(
        enable_if<
            !is_constructible_with_single_brace<
                Base, decltype(get<Indexes>(std::declval<const Scalar &>()))...>() &&
                !is_constructible_with_double_brace<
                    Base, decltype(get<Indexes>(std::declval<const Scalar &>()))...>(),
            const Scalar &> x,
        Vc::index_sequence<Indexes...>)
        : Base(get<Indexes>(x)...)
    {
    }

public:
    /// The SIMD vector width of the members.
    static constexpr size_t size() { return N; }

    /// The vectorized base class template instantiation this Adapter class derives from.
    using base_type = Base;
    /// The original non-vectorized class template instantiation that was passed to the
    /// simdize expression.
    using scalar_type = Scalar;

    /// Allow default construction. This is automatically ill-formed if Base() is
    /// ill-formed.
    Adapter() = default;

    /// Broadcast constructor
    template <typename U, size_t TupleSize = determine_tuple_size<Traits::decay<U>>(),
              typename Seq = Vc::make_index_sequence<TupleSize>>
    Adapter(U &&x)
        : Adapter(static_cast<const Scalar &>(x), Seq())
    {
    }

    /// perfect forward all Base constructors
    template <typename A0, typename... Args,
              typename = typename std::enable_if<
                  (sizeof...(Args) > 0 || !std::is_convertible<A0, Scalar>::value)>::type>
    Adapter(A0 &&arg0, Args &&... arguments)
        : Base(std::forward<A0>(arg0), std::forward<Args>(arguments)...)
    {
    }

    /// perfect forward Base constructors that accept an initializer_list
    template <typename T,
              typename = decltype(Base(std::declval<const std::initializer_list<T> &>()))>
    Adapter(const std::initializer_list<T> &l)
        : Base(l)
    {
    }

    /// Overload the new operator to adhere to the alignment requirements which C++11
    /// ignores by default.
    void *operator new(size_t size) { return Vc::Common::aligned_malloc<alignof(Adapter)>(size); }
    void *operator new(size_t, void *p) { return p; }
    void *operator new[](size_t size) { return Vc::Common::aligned_malloc<alignof(Adapter)>(size); }
    void *operator new[](size_t , void *p) { return p; }
    void operator delete(void *ptr, size_t) { Vc::Common::free(ptr); }
    void operator delete(void *, void *) {}
    void operator delete[](void *ptr, size_t) { Vc::Common::free(ptr); }
    void operator delete[](void *, void *) {}
};

/**internal
 * Delete compare operators for simdize<tuple<...>> types because the tuple compares
 * require the compares to be bool based.
 */
template <class... TTypes, class... TTypesV, class... UTypes, class... UTypesV, size_t N>
inline bool operator==(
    const Adapter<std::tuple<TTypes...>, std::tuple<TTypesV...>, N> &t,
    const Adapter<std::tuple<UTypes...>, std::tuple<UTypesV...>, N> &u) = delete;
template <class... TTypes, class... TTypesV, class... UTypes, class... UTypesV, size_t N>
inline bool operator!=(
    const Adapter<std::tuple<TTypes...>, std::tuple<TTypesV...>, N> &t,
    const Adapter<std::tuple<UTypes...>, std::tuple<UTypesV...>, N> &u) = delete;
template <class... TTypes, class... TTypesV, class... UTypes, class... UTypesV, size_t N>
inline bool operator<=(
    const Adapter<std::tuple<TTypes...>, std::tuple<TTypesV...>, N> &t,
    const Adapter<std::tuple<UTypes...>, std::tuple<UTypesV...>, N> &u) = delete;
template <class... TTypes, class... TTypesV, class... UTypes, class... UTypesV, size_t N>
inline bool operator>=(
    const Adapter<std::tuple<TTypes...>, std::tuple<TTypesV...>, N> &t,
    const Adapter<std::tuple<UTypes...>, std::tuple<UTypesV...>, N> &u) = delete;
template <class... TTypes, class... TTypesV, class... UTypes, class... UTypesV, size_t N>
inline bool operator<(
    const Adapter<std::tuple<TTypes...>, std::tuple<TTypesV...>, N> &t,
    const Adapter<std::tuple<UTypes...>, std::tuple<UTypesV...>, N> &u) = delete;
template <class... TTypes, class... TTypesV, class... UTypes, class... UTypesV, size_t N>
inline bool operator>(
    const Adapter<std::tuple<TTypes...>, std::tuple<TTypesV...>, N> &t,
    const Adapter<std::tuple<UTypes...>, std::tuple<UTypesV...>, N> &u) = delete;

}  // namespace SimdizeDetail
}  // namespace Vc

namespace std
{
/**\internal
 * A std::tuple_size specialization for the SimdizeDetail::Adapter class.
 */
template <typename Scalar, typename Base, size_t N>
class tuple_size<Vc::SimdizeDetail::Adapter<Scalar, Base, N>> : public tuple_size<Base>
{
};
/**\internal
 * A std::tuple_element specialization for the SimdizeDetail::Adapter class.
 */
template <size_t I, typename Scalar, typename Base, size_t N>
class tuple_element<I, Vc::SimdizeDetail::Adapter<Scalar, Base, N>>
    : public tuple_element<I, Base>
{
};
// std::get does not need additional work because Vc::Adapter derives from
// C<Ts...> and therefore if get<N>(C<Ts...>) works it works for Adapter as well.

/**\internal
 * A std::allocator specialization for SimdizeDetail::Adapter which uses the Vc::Allocator
 * class to make allocation correctly aligned per default.
 */
template <typename S, typename T, size_t N>
class allocator<Vc::SimdizeDetail::Adapter<S, T, N>>
    : public Vc::Allocator<Vc::SimdizeDetail::Adapter<S, T, N>>
{
public:
    template <typename U> struct rebind
    {
        typedef std::allocator<U> other;
    };
};
}  // namespace std

namespace Vc_VERSIONED_NAMESPACE
{
namespace SimdizeDetail
{
namespace
{
struct Dummy__;
/**\internal
 * Dummy get<N>(x) function to enable compilation of the following code. This code is
 * never meant to be called or used.
 */
template <size_t> Dummy__ get(Dummy__ x);
}  // unnamed namespace

/**\internal
 * Trait determining the number of data members that get<N>(x) can access.
 * The type \p T either has to provide a std::tuple_size specialization or contain a
 * constexpr tuple_size member.
 */
template <typename T, size_t TupleSize = std::tuple_size<T>::value>
constexpr size_t determine_tuple_size()
{
    return TupleSize;
}
template <typename T, size_t TupleSize = T::tuple_size>
constexpr size_t determine_tuple_size(size_t = T::tuple_size)
{
    return TupleSize;
}

/**\internal
 * Since std::decay can ICE GCC (with types that are declared as may_alias), this is used
 * as an alternative approach. Using decltype the template type deduction implements the
 * std::decay behavior.
 */
template <typename T> static inline T decay_workaround(const T &x) { return x; }

/**\internal
 * Generic implementation of assign using the std::tuple get interface.
 */
template <typename S, typename T, size_t N, size_t... Indexes>
inline void assign_impl(Adapter<S, T, N> &a, size_t i, const S &x,
                        Vc::index_sequence<Indexes...>)
{
    const std::tuple<decltype(decay_workaround(get<Indexes>(x)))...> tmp(
        decay_workaround(get<Indexes>(x))...);
    auto &&unused = {(get<Indexes>(a)[i] = get<Indexes>(tmp), 0)...};
    if (&unused == &unused) {}
}

/**
 * Assigns one scalar object \p x to a SIMD slot at offset \p i in the simdized object \p
 * a.
 */
template <typename S, typename T, size_t N>
inline void assign(Adapter<S, T, N> &a, size_t i, const S &x)
{
    assign_impl(a, i, x, Vc::make_index_sequence<determine_tuple_size<T>()>());
}

template <typename S, typename T, size_t N, size_t... Indexes>
inline S extract_impl(const Adapter<S, T, N> &a, size_t i, Vc::index_sequence<Indexes...>)
{
    const std::tuple<decltype(decay_workaround(get<Indexes>(a)[i]))...> tmp(
        decay_workaround(get<Indexes>(a)[i])...);
    return S(get<Indexes>(tmp)...);
}

template <typename S, typename T, size_t N>
inline S extract(const Adapter<S, T, N> &a, size_t i)
{
    return extract_impl(a, i, Vc::make_index_sequence<determine_tuple_size<S>()>());
}

template <typename A> class Scalar
{
    using reference = typename std::add_lvalue_reference<A>::type;
    using S = typename A::scalar_type;
    using IndexSeq = Vc::make_index_sequence<determine_tuple_size<S>()>;

public:
    Scalar(reference aa, size_t ii) : a(aa), i(ii) {}
    void operator=(const S &x) { assign_impl(a, i, x, IndexSeq()); }
    operator S() const { return extract_impl(a, i, IndexSeq()); }

private:
    reference a;
    size_t i;
};

template <typename A> class Interface
{
    using reference = typename std::add_lvalue_reference<A>::type;

public:
    Interface(reference aa) : a(aa) {}

    Scalar<A> operator[](size_t i)
    {
        return {a, i};
    }
    typename A::scalar_type operator[](size_t i) const
    {
        return extract_impl(
            a, i,
            Vc::make_index_sequence<determine_tuple_size<typename A::scalar_type>()>());
    }

private:
    reference a;
};

template <typename S, typename T, size_t N>
Interface<Adapter<S, T, N>> decorate(Adapter<S, T, N> &a)
{
    return {a};
}
template <typename S, typename T, size_t N>
const Interface<const Adapter<S, T, N>> decorate(const Adapter<S, T, N> &a)
{
    return {a};
}

}  // namespace SimdizeDetail

template <typename T, size_t N = 0, typename MT = void>
using simdize = SimdizeDetail::simdize<T, N, MT>;

}  // namespace Vc

#include "undomacros.h"

#endif  // VC_COMMON_SIMDIZE_H_

// vim: foldmethod=marker
