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

#ifndef VC_EXAMPLES_KDTREE_SIMDIZE_H_
#define VC_EXAMPLES_KDTREE_SIMDIZE_H_

#include <tuple>

#include <Vc/Vc>

namespace simdize_internal
{
/**
 * \internal
 * \brief Replace arithmetic type T with Vc::Vector<T> or Mask<T> if T == bool and apply
 *        recursively via the Adapter class if T is a template class.
 *
 * This general fallback case does not replace the type T and returns it unchanged it via
 * the type member.
 *
 * \tparam N  if non-zero requests that number of entries in "Vector<T>", which determines
 *            the choice between Vector<T> and simdarray<T, N>
 * \tparam T0 This identifies the first type that was transformed and is used when bool
 *            needs to be converted to a Vc::Mask type, in which case the underlying type
 *            would be arbitrary. With T0 it can be smarter and choose between Mask<T0>
 *            and simd_mask_array<T0, N>
 * \tparam T  The type to be converted to a Vector or Mask type.
 */
template <std::size_t N, typename T0, typename T,
          bool =  // Flag to easily distinguish types that need more recursion for
                  // transformation (or no transformation at all
          std::is_same<T, bool>::value || std::is_same<T, short>::value ||
          std::is_same<T, unsigned short>::value || std::is_same<T, int>::value ||
          std::is_same<T, unsigned int>::value || std::is_same<T, float>::value ||
          std::is_same<T, double>::value>
struct make_vector_or_simdarray_impl
{
    using type = T;
};

/** \internal
 * Specialization for non-zero \p N and a type \p T that can be used with Vector<T> to
 * create Vector<T> or simdarray<T, N>
 */
template <std::size_t N, typename T0, typename T>
struct make_vector_or_simdarray_impl<N, T0, T, true>
{
    using type = typename std::conditional<N == Vc::Vector<T>::Size, Vc::Vector<T>,
                                           Vc::simdarray<T, N>>::type;
};
/** \internal
 * Specialization for non-zero \p N and bool to create Mask<T0> or simd_mask_array<T0, N>.
 */
template <std::size_t N, typename T0>
struct make_vector_or_simdarray_impl<N, T0, bool, true>
{
    using type = typename std::conditional<N == Vc::Mask<T0>::Size, Vc::Mask<T0>,
                                           Vc::simd_mask_array<T0, N>>::type;
};
/** \internal
 * Specialization for \p N = 0 and bool to create Mask<T0>.
 */
template <typename T0> struct make_vector_or_simdarray_impl<0, T0, bool, true>
{
    using type = Vc::Mask<T0>;
};
/** \internal
 * Specialization for \p N = 0 and bool to create Mask<float> (because no usable T0 is
 * given).
 */
template <> struct make_vector_or_simdarray_impl<0, bool, bool, true>
{
    using type = Vc::Mask<float>;
};
/** \internal
 * Specialization for \p N = 0 and arithmetic \p T to create Vector<T>.
 */
template <typename T> struct make_vector_or_simdarray_impl<0, T, T, true>
{
    using type = Vc::Vector<T>;
};

/** \internal
 * Specialization for a template class argument to recurse.
 */
template <std::size_t N, typename T0, template <typename...> class C, typename T1,
          typename... Ts>
struct make_vector_or_simdarray_impl<N, T0, C<T1, Ts...>, false>;

/** \internal
 * Specialization for a template class argument with template arguments <type, size_t> to
 * recurse.
 */
template <typename T0, template <typename, std::size_t> class C, typename T,
          std::size_t N, std::size_t M>
struct make_vector_or_simdarray_impl<N, T0, C<T, M>, false>;

/** \internal
 * A SIMD Vector type of \p T, either as Vc::simdarray or Vc::Vector, depending on \p N.
 * If Vector<T> has a size equal to N, Vector<T> is used, otherwise simdarray<T, N>.
 */
template <std::size_t N, typename T0, typename T = T0>
using make_vector_or_simdarray = typename make_vector_or_simdarray_impl<N, T0, T>::type;

/** \internal
 * A type trait test for whether a type T supports std::tuple_size and std::tuple_element.
 */
namespace has_tuple_interface_impl
{
template <typename T, int = std::tuple_size<T>::value,
          typename = typename std::tuple_element<0, T>::type>
std::true_type test(int);
template <typename T> std::false_type test(...);
static_assert(decltype(test<std::tuple<int>>(1))::value == true, "");
static_assert(decltype(test<std::array<int, 3>>(1))::value == true, "");
static_assert(decltype(test<std::allocator<int>>(1))::value == false, "");
}  // namespace has_tuple_interface_impl

/** \internal
 * \brief An adapter for making a simdized template class easier to use.
 *
 * \tparam T The type to simdize and adapt.
 * \tparam N The requested SIMD vector size (0 means it is determined by the first type).
 * \tparam HasTupleInterface Type information about \p T, whether it implements the
 *                           std::tuple_size and std::tuple_element helpers. The flag is
 *                           used to specialize tuple_size and tuple_element only for
 *                           those types that do.
 */
template <typename T, std::size_t N,
          bool HasTupleInterface = decltype(has_tuple_interface_impl::test<T>(1))::value>
struct Adapter;

// dummy get<N>(...)
namespace
{
struct Dummy__;
template <std::size_t> Dummy__ get(Dummy__ x);
}  // unnamed namespace

template <template <typename, std::size_t> class C, typename T0, std::size_t N,
          std::size_t M, bool HasTupleInterface>
class Adapter<C<T0, M>, N, HasTupleInterface>
    : public C<make_vector_or_simdarray<N, T0>, M>
{
public:
    using ScalarBase = C<T0, M>;
    using VectorBase = C<make_vector_or_simdarray<N, T0>, M>;

    using FirstVectorType = make_vector_or_simdarray<N, T0>;
    using VectorTypesTuple = std::tuple<FirstVectorType>;

    static constexpr std::size_t Size = FirstVectorType::Size;
    static constexpr std::size_t size() { return Size; }

    template <std::size_t... Indexes>
    Adapter(const ScalarBase &x, Vc::index_sequence<Indexes...>)
        : VectorBase{get<Indexes>(x)...}
    {
    }

    template <typename U, typename S = std::tuple_size<typename std::decay<U>::type>,
              typename Seq = Vc::make_index_sequence<S::value>>
    Adapter(U &&x)
        : Adapter(static_cast<const ScalarBase &>(x), Seq())
    {
    }

    Adapter() = default;

    // perfect forward all Base constructors
    template <
        typename A0, typename... Args,
        typename = typename std::enable_if<
            (sizeof...(Args) > 0 || !std::is_convertible<A0, ScalarBase>::value)>::type>
    Adapter(A0 &&arg0, Args &&... arguments)
        : VectorBase(std::forward<A0>(arg0), std::forward<Args>(arguments)...)
    {
    }

    // perfect forward Base constructors that accept an initializer_list
    template <typename T> Adapter(const std::initializer_list<T> &l) : VectorBase(l) {}

    void *operator new(size_t size) { return Vc::Common::aligned_malloc<alignof(Adapter)>(size); }
    void *operator new(size_t, void *p) { return p; }
    void *operator new[](size_t size) { return Vc::Common::aligned_malloc<alignof(Adapter)>(size); }
    void *operator new[](size_t , void *p) { return p; }
    void operator delete(void *ptr, size_t) { Vc::Common::free(ptr); }
    void operator delete(void *, void *) {}
    void operator delete[](void *ptr, size_t) { Vc::Common::free(ptr); }
    void operator delete[](void *, void *) {}
};

template <std::size_t N, template <typename...> class C, typename T0, typename... Ts>
using make_adapter_base_type = C<
    make_vector_or_simdarray<N, T0>,
    make_vector_or_simdarray<
        Vc::Traits::simd_vector_size<make_vector_or_simdarray<N, T0>>::value, T0, Ts>...>;

template <template <typename...> class C, typename T0, typename... Ts, std::size_t N,
          bool HasTupleInterface>
class Adapter<C<T0, Ts...>, N, HasTupleInterface>
    : public make_adapter_base_type<N, C, T0, Ts...>
{
public:
    using ScalarBase = C<T0, Ts...>;
    using VectorBase = make_adapter_base_type<N, C, T0, Ts...>;

    using FirstVectorType = make_vector_or_simdarray<N, T0>;
    using VectorTypesTuple = std::tuple<
        FirstVectorType, make_vector_or_simdarray<FirstVectorType::Size, T0, Ts>...>;

    static constexpr std::size_t Size = FirstVectorType::Size;
    static constexpr std::size_t size() { return Size; }

    template <std::size_t... Indexes>
    Adapter(const ScalarBase &x, Vc::index_sequence<Indexes...>)
        : VectorBase{get<Indexes>(x)...}
    {
    }

    template <typename U, typename S = std::tuple_size<typename std::decay<U>::type>,
              typename Seq = Vc::make_index_sequence<S::value>>
    Adapter(U &&x)
        : Adapter(static_cast<const ScalarBase &>(x), Seq())
    {
    }

    Adapter() = default;

    // perfect forward all Base constructors
    template <
        typename A0, typename... Args,
        typename = typename std::enable_if<
            (sizeof...(Args) > 0 || !std::is_convertible<A0, ScalarBase>::value)>::type>
    Adapter(A0 &&arg0, Args &&... arguments)
        : VectorBase(std::forward<A0>(arg0), std::forward<Args>(arguments)...)
    {
    }

    // perfect forward Base constructors that accept an initializer_list
    template <typename T> Adapter(const std::initializer_list<T> &l) : VectorBase(l) {}

    void *operator new(size_t size) { return Vc::Common::aligned_malloc<alignof(Adapter)>(size); }
    void *operator new(size_t, void *p) { return p; }
    void *operator new[](size_t size) { return Vc::Common::aligned_malloc<alignof(Adapter)>(size); }
    void *operator new[](size_t , void *p) { return p; }
    void operator delete(void *ptr, size_t) { Vc::Common::free(ptr); }
    void operator delete(void *, void *) {}
    void operator delete[](void *ptr, size_t) { Vc::Common::free(ptr); }
    void operator delete[](void *, void *) {}
};

template <std::size_t N, typename T0, template <typename...> class C, typename T1,
          typename... Ts>
struct make_vector_or_simdarray_impl<N, T0, C<T1, Ts...>, false>
{
private:
    using base = make_adapter_base_type<N, C, T1, Ts...>;
    using CC = C<T1, Ts...>;

public:
    using type = typename std::conditional<std::is_same<base, CC>::value, CC,
                                           Adapter<CC, N>>::type;
};

template <typename T0, template <typename, std::size_t> class C, typename T,
          std::size_t N, std::size_t M>
struct make_vector_or_simdarray_impl<N, T0, C<T, M>, false>
{
private:
    typedef C<make_vector_or_simdarray<N, T>, M> base;

public:
    using type = typename std::conditional<std::is_same<base, C<T, M>>::value, C<T, M>,
                                           Adapter<C<T, M>, N>>::type;
};

template <typename T, std::size_t N, std::size_t... Indexes>
T simdize_get_impl(const Adapter<T, N> &a, std::size_t i, Vc::index_sequence<Indexes...>)
{
    return T{get<Indexes>(a)[i]...};
}

template <typename T, std::size_t N, std::size_t... Indexes>
inline void simdize_assign_impl(Adapter<T, N> &a, std::size_t i, const T &x,
                                Vc::index_sequence<Indexes...>)
{
    auto &&unused = {(get<Indexes>(a)[i] = x[Indexes])...};
    if (&unused) {}
}
}  // namespace simdize_internal

namespace std
{
// tuple_size
template <template <typename...> class C, typename T0, typename... Ts, std::size_t N>
class tuple_size<simdize_internal::Adapter<C<T0, Ts...>, N, true>>
    : public tuple_size<C<T0, Ts...>>
{
};
// tuple_element
template <std::size_t I, template <typename...> class C, typename T0, typename... Ts,
          std::size_t N>
class tuple_element<I, simdize_internal::Adapter<C<T0, Ts...>, N, true>>
    : public tuple_element<I, typename simdize_internal::Adapter<C<T0, Ts...>, N, true>::VectorBase>
{
};
}  // namespace std

/**
 * \brief Convert the type \p T to a type where, recursively, all arithmetic
 * types are replaced by their SIMD counterparts.
 *
 * Magic.
 *
 * \code
 * using Data = std::tuple<float, int, bool>;
 * Data x;
 * using DataVec = simdize<Data>;
 * Data v = x;
 * // tuple_size<Data>::value == tuple_size<DataVec>
 * get<0>(x) = 1.f;
 * get<0>(v) = Vc::float_v::IndexesFromZero();
 * x = simdize_get(v, 0); // extract one Data object at SIMD index 0.
 * \endcode
 *
 * \tparam T The type to recursively translate to SIMD types.
 *
 * \tparam N Optionally, you can request the number of values in the vector types. This can lead to
 *           make code more efficient for some specific platforms and badly optimized for others.
 *           Thus, setting this value is a portability issue for performance. A possible way out is
 *           to base the value on a native SIMD vector size, such as float_v::Size.
 */
template <typename T, std::size_t N = 0> using simdize = simdize_internal::make_vector_or_simdarray<N, T>;

template <typename T, std::size_t N>
T simdize_get(const simdize_internal::Adapter<T, N> &a, std::size_t i)
{
    return simdize_internal::simdize_get_impl(
        a, i, Vc::make_index_sequence<std::tuple_size<T>::value>());
}

template <typename T, std::size_t N>
inline void simdize_assign(simdize_internal::Adapter<T, N> &a, std::size_t i, const T &x)
{
    simdize_internal::simdize_assign_impl(
        a, i, x, Vc::make_index_sequence<std::tuple_size<T>::value>());
}


#endif  // VC_EXAMPLES_KDTREE_SIMDIZE_H_

// vim: foldmethod=marker
