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
template <std::size_t N,  // if non-zero requests that number of entries in "Vector<T>",
                          // which determines the choice between Vector<T> and
                          // simdarray<T, N>
          typename T0,  // This identifies the first type that was transformed and is used
                        // when bool needs to be converted to a Vc::Mask type, in which
                        // case the underlying type would be arbitrary. With T0 it can be
                        // smarter and choose between Mask<T0> and simd_mask_array<T0, N>
          typename T,   // The type to be converted to a Vector or Mask type.
          bool = std::is_same<T, bool>::value || std::is_same<T, short>::value ||
                 std::is_same<T, unsigned short>::value || std::is_same<T, int>::value ||
                 std::is_same<T, unsigned int>::value || std::is_same<T, float>::value ||
                 std::is_same<T, double>::value
                 // Flag to easily distinguish types that
                 // need more recursion for transformation
                 // (or no transformation at all
          >
struct make_vector_or_simdarray_impl
{
    // fallback captures everything that isn't converted
    using type = T;
};

template <std::size_t N, typename T0, typename T>
struct make_vector_or_simdarray_impl<N, T0, T, true>
{
    using type = typename std::conditional<N == Vc::Vector<T>::Size, Vc::Vector<T>,
                                           Vc::simdarray<T, N>>::type;
};
template <std::size_t N, typename T0>
struct make_vector_or_simdarray_impl<N, T0, bool, true>
{
    using type = typename std::conditional<N == Vc::Mask<T0>::Size, Vc::Mask<T0>,
                                           Vc::simd_mask_array<T0, N>>::type;
};
template <typename T0> struct make_vector_or_simdarray_impl<0, T0, bool, true>
{
    using type = Vc::Mask<T0>;
};
template <> struct make_vector_or_simdarray_impl<0, bool, bool, true>
{
    using type = Vc::Mask<float>;
};
template <typename T> struct make_vector_or_simdarray_impl<0, T, T, true>
{
    using type = Vc::Vector<T>;
};

template <std::size_t N, typename T0, template <typename...> class C, typename... Ts>
struct make_vector_or_simdarray_impl<N, T0, C<Ts...>, false>;

template <typename T0, template <typename, std::size_t> class C, typename T,
          std::size_t N, std::size_t M>
struct make_vector_or_simdarray_impl<N, T0, C<T, M>, false>;

/** \internal
 * A SIMD Vector type of \p T, either as Vc::simdarray or Vc::Vector, depending on \p N.
 * If Vector<T> has a size equal to N, Vector<T> is used, otherwise simdarray<T, N>.
 */
template <std::size_t N, typename T0, typename T = T0>
using make_vector_or_simdarray = typename make_vector_or_simdarray_impl<N, T0, T>::type;

template <typename T, std::size_t N = 0> struct Adapter;

template <template <typename, std::size_t> class C, typename T0, std::size_t N,
          std::size_t M>
class Adapter<C<T0, M>, N> : public C<make_vector_or_simdarray<N, T0>, M>
{
    using Scalar = C<T0, M>;
    using Vector = C<make_vector_or_simdarray<N, T0>, M>;

public:
    using FirstVectorType = make_vector_or_simdarray<N, T0>;
    using VectorTypesTuple = std::tuple<FirstVectorType>;

    static constexpr std::size_t Size = FirstVectorType::Size;
    static constexpr std::size_t size() { return Size; }

    // perfect forward all Base constructors
    template <typename... Args>
    Adapter(Args &&... arguments)
        : Vector(std::forward<Args>(arguments)...)
    {
    }

    // perfect forward Base constructors that accept an initializer_list
    template <typename T> Adapter(const std::initializer_list<T> &l) : Vector(l) {}

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

template <template <typename...> class C, typename T0, typename... Ts, std::size_t N>
class Adapter<C<T0, Ts...>, N> : public make_adapter_base_type<N, C, T0, Ts...>
{
    using Scalar = C<T0, Ts...>;
    using Vector = make_adapter_base_type<N, C, T0, Ts...>;

public:
    using FirstVectorType = make_vector_or_simdarray<N, T0>;
    using VectorTypesTuple = std::tuple<
        FirstVectorType, make_vector_or_simdarray<FirstVectorType::Size, T0, Ts>...>;

    static constexpr std::size_t Size = FirstVectorType::Size;
    static constexpr std::size_t size() { return Size; }

    // perfect forward all Base constructors
    template <typename... Args>
    Adapter(Args &&... arguments)
        : Vector(std::forward<Args>(arguments)...)
    {
    }

    // perfect forward Base constructors that accept an initializer_list
    template <typename T> Adapter(const std::initializer_list<T> &l) : Vector(l) {}

    void *operator new(size_t size) { return Vc::Common::aligned_malloc<alignof(Adapter)>(size); }
    void *operator new(size_t, void *p) { return p; }
    void *operator new[](size_t size) { return Vc::Common::aligned_malloc<alignof(Adapter)>(size); }
    void *operator new[](size_t , void *p) { return p; }
    void operator delete(void *ptr, size_t) { Vc::Common::free(ptr); }
    void operator delete(void *, void *) {}
    void operator delete[](void *ptr, size_t) { Vc::Common::free(ptr); }
    void operator delete[](void *, void *) {}
};

template <std::size_t N, typename T0, template <typename...> class C, typename... Ts>
struct make_vector_or_simdarray_impl<N, T0, C<Ts...>, false>
{
private:
    typedef make_adapter_base_type<N, C, Ts...> base;

public:
    using type = typename std::conditional<std::is_same<base, C<Ts...>>::value, C<Ts...>,
                                           Adapter<C<Ts...>, N>>::type;
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

}  // namespace simdize_internal

template <typename T> using simdize = simdize_internal::make_vector_or_simdarray<0, T>;

#endif  // VC_EXAMPLES_KDTREE_SIMDIZE_H_

// vim: foldmethod=marker
