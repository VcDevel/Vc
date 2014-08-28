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
template <std::size_t N, typename T, bool = std::is_arithmetic<T>::value>
struct make_vector_or_simdarray_impl;

template <std::size_t N, typename T> struct make_vector_or_simdarray_impl<N, T, true>
{
    using type = typename std::conditional<N == Vc::Vector<T>::Size, Vc::Vector<T>,
                                           Vc::simdarray<T, N>>::type;
};
template <typename T> struct make_vector_or_simdarray_impl<0, T, true>
{
    using type = Vc::Vector<T>;
};

template <std::size_t N, template <typename...> class C, typename... Ts>
struct make_vector_or_simdarray_impl<N, C<Ts...>, false>;

template <template <typename, std::size_t> class C, typename T0, std::size_t N,
          std::size_t M>
struct make_vector_or_simdarray_impl<N, C<T0, M>, false>;

/** \internal
 * A SIMD Vector type of \p T, either as Vc::simdarray or Vc::Vector, depending on \p N.
 * If Vector<T> has a size equal to N, Vector<T> is used, otherwise simdarray<T, N>.
 */
template <std::size_t N, typename T>
using make_vector_or_simdarray = typename make_vector_or_simdarray_impl<N, T>::type;

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

template <template <typename...> class C, typename T0, typename... Ts, std::size_t N>
class Adapter<C<T0, Ts...>, N>
    : public C<make_vector_or_simdarray<N, T0>, make_vector_or_simdarray<N, Ts>...>
{
    using Scalar = C<T0, Ts...>;
    using Vector = C<make_vector_or_simdarray<N, T0>, make_vector_or_simdarray<N, Ts>...>;

public:
    using FirstVectorType = make_vector_or_simdarray<N, T0>;
    using VectorTypesTuple =
        std::tuple<FirstVectorType, make_vector_or_simdarray<N, Ts>...>;

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

template <std::size_t N, template <typename...> class C, typename... Ts>
struct make_vector_or_simdarray_impl<N, C<Ts...>, false>
{
    using type = Adapter<C<Ts...>, N>;
};

template <template <typename, std::size_t> class C, typename T0, std::size_t N,
          std::size_t M>
struct make_vector_or_simdarray_impl<N, C<T0, M>, false>
{
    using type = Adapter<C<T0, M>, N>;
};

}  // namespace simdize_internal

template <typename T> using simdize = simdize_internal::Adapter<T>;

inline void f()
{
    using namespace std;
    using namespace Vc;
    using std::array;
    static_assert(is_convertible<simdize<array<float, 3>>, array<float_v, 3>>::value, "");
    static_assert(is_convertible<array<float_v, 3>, simdize<array<float, 3>>>::value, "");
    static_assert(is_convertible<simdize<tuple<float>>, tuple<float_v>>::value, "");
    static_assert(is_convertible<tuple<float_v>, simdize<tuple<float>>>::value, "");
    static_assert(is_convertible<simdize<tuple<array<float, 3>>>, tuple<array<float_v, 3>>>::value, "");
    static_assert(is_convertible<tuple<array<float_v, 3>>, simdize<tuple<array<float, 3>>>>::value, "");
    static_assert(is_convertible<simdize<array<tuple<float>, 3>>, array<simdize<tuple<float>>, 3>>::value, "");
    static_assert(is_convertible<array<tuple<float_v>, 3>, simdize<array<tuple<float>, 3>>>::value, "");
    static_assert(is_convertible<vector<tuple<float_v>>, simdize<vector<tuple<float>>>>::value, "");
    static_assert(is_convertible<
                  tuple<float_v, array<pair<float_v, simdarray<double, 8>>, 3>>,
                  simdize<tuple<float, array<pair<float, double>, 3>>>
                  >::value, "");
}

#endif  // VC_EXAMPLES_KDTREE_SIMDIZE_H_

// vim: foldmethod=marker
