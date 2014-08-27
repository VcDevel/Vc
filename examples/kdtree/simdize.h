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

/** \internal
 * A SIMD Vector type of \p T, either as Vc::simdarray or Vc::Vector, depending on the
 * size of Vector<T0>. If Vector<T> has equal size then Vector<T> is used, otherwise
 * simdarray<T, Vector<T0>::Size>.
 */
template <typename T0, typename T>
using make_vector_or_simdarray =
    typename std::conditional<Vc::Vector<T0>::Size == Vc::Vector<T>::Size, Vc::Vector<T>,
                              Vc::simdarray<T, Vc::Vector<T0>::Size>>::type;

template <typename T> struct Adapter;
template <template <typename...> class C, typename T0, typename... Ts>
class Adapter<C<T0, Ts...>>
    : public C<Vc::Vector<T0>, make_vector_or_simdarray<T0, Ts>...>
{
    using Scalar = C<T0, Ts...>;
    using Vector = C<Vc::Vector<T0>, make_vector_or_simdarray<T0, Ts>...>;

public:
    using VectorTypesTuple =
        std::tuple<Vc::Vector<T0>, make_vector_or_simdarray<T0, Ts>...>;
    using FirstVectorType = Vc::Vector<T0>;

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

}  // namespace simdize_internal

template <typename T> using simdize = simdize_internal::Adapter<T>;

#endif  // VC_EXAMPLES_KDTREE_SIMDIZE_H_

// vim: foldmethod=marker
