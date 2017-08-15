/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DETAIL_SPLIT_H_
#define VC_DETAIL_SPLIT_H_

#include "macros.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
template <class V, size_t Parts, class T, class A, size_t... Indexes>
std::array<V, Parts> split_to_array(const simd<T, A> &x,
                                    std::index_sequence<Indexes...>)
{
    // this could be much simpler:
    //
    // return {V([&](auto i) { return x[i + Indexes * V::size()]; })...};
    //
    // Sadly GCC has a bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=47226. The
    // following works around it by placing the pack outside of the code section of the
    // lambda:
    return {[](size_t j, const simd<T, A> &y) {
        return V([&](auto i) { return y[i + j * V::size()]; });
    }(Indexes, x)...};
}
}  // namespace detail

template <class V, class T, class A, size_t Parts = simd_size_v<T, A> / V::size()>
std::enable_if_t<(is_simd<V>::value && simd_size_v<T, A> == Parts * V::size()),
                 std::array<V, Parts>>
split(const simd<T, A> &x)
{
    return detail::split_to_array<V, Parts>(x, std::make_index_sequence<Parts>());
}

#if defined __cpp_fold_expressions && defined Vc_EXPERIMENTAL
template <size_t... Sizes, class T, class A>
std::enable_if_t<((Sizes + ...) == simd<T, A>::size()),
                 std::tuple<simd<T, abi_for_size_t<T, Sizes>>...>>
split(const simd<T, A> &x)
{
    std::tuple<simd<T, abi_for_size_t<T, Sizes>>...> tup;
    size_t offset = 0;
    detail::execute_n_times<sizeof...(Sizes)>([&](auto i) {
        auto &v_i = std::get<i>(tup);
        constexpr size_t N = std::decay_t<decltype(v_i)>::size();
        detail::execute_n_times<N>([&](auto j) { v_i[j] = x[j + offset]; });
        offset += N;
    });
    return tup;
}

namespace detail
{
template <class T, class...> struct typelist
{
    using first_type = T;
};

template <size_t N, class T, class List,
          bool = (N < simd_size_v<T, typename List::first_type>)>
struct subscript_in_pack;

template <size_t N, class T, class A, class... As>
struct subscript_in_pack<N, T, detail::typelist<A, As...>, true> {
    static Vc_INTRINSIC T get(const simd<T, A> &x, const simd<T, As> &...)
    {
        return x[N];
    }
};
template <size_t N, class T, class A, class... As>
struct subscript_in_pack<N, T, detail::typelist<A, As...>, false> {
    static Vc_INTRINSIC T get(const simd<T, A> &, const simd<T, As> &... xs)
    {
        return subscript_in_pack<N - simd<T, A>::size(), T,
                                 detail::typelist<As...>>::get(xs...);
    }
};
}  // namespace detail

template <class T, class... As>
simd<T, abi_for_size_t<T, (simd_size_v<T, As> + ...)>> concat(
    const simd<T, As> &... xs)
{
    return simd<T, abi_for_size_t<T, (simd_size_v<T, As> + ...)>>([&](auto i) {
        return detail::subscript_in_pack<i, T, detail::typelist<As...>>::get(xs...);
    });
}
#endif  // defined __cpp_fold_expressions && defined Vc_EXPERIMENTAL
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_SPLIT_H_

// vim: foldmethod=marker
