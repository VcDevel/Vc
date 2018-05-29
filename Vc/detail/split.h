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
#include "type_traits.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
template <int Index, int Parts, class T, size_t N>
Vc_INTRINSIC auto extract_part(Storage<T, N> x)
{
    if constexpr (have_sse) {
        return detail::x86::extract_part<Index, Parts>(x);
    } else {
        assert_unreachable<T>();
    }
}

template <int Index, int Parts, class T, class A0, class... As>
Vc_INTRINSIC auto extract_part(const simd_tuple<T, A0, As...> &x)
{
    // worst cases:
    // (a) 4, 4, 4 => 3, 3, 3, 3 (Parts = 4)
    // (b) 2, 2, 2 => 3, 3       (Parts = 2)
    // (c) 4, 2 => 2, 2, 2       (Parts = 3)
    using Tuple = simd_tuple<T, A0, As...>;
    static_assert(Index < Parts && Index >= 0 && Parts >= 1);
    constexpr size_t N = Tuple::size();
    static_assert(N >= Parts && N % Parts == 0);
    constexpr size_t values_per_part = N / Parts;
    if constexpr (Parts == 1) {
        if constexpr (Tuple::tuple_size == 1) {
            return x.first;
        } else {
            return x;
        }
    } else if constexpr (simd_size_v<T, A0> % values_per_part != 0) {
        // nasty case: The requested partition does not match the partition of the
        // simd_tuple. Fall back to construction via scalar copies.
#ifdef Vc_USE_ALIASING_LOADS
        const detail::may_alias<T> *const element_ptr =
            reinterpret_cast<const detail::may_alias<T> *>(&x) + Index * values_per_part;
        return detail::data(simd<T, simd_abi::deduce_t<T, values_per_part>>(
            [&](auto i) { return element_ptr[i]; }));
#else
        constexpr size_t offset = Index * values_per_part;
        return detail::data(simd<T, simd_abi::deduce_t<T, values_per_part>>([&](auto i) {
            constexpr detail::size_constant<i + offset> k;
            return x[k];
        }));
#endif
    } else if constexpr (values_per_part * Index >= simd_size_v<T, A0>) {  // recurse
        constexpr int parts_in_first = simd_size_v<T, A0> / values_per_part;
        return extract_part<Index - parts_in_first, Parts - parts_in_first>(x.second);
    } else {  // at this point we know that all of the return values are in x.first
        static_assert(values_per_part * (1 + Index) <= simd_size_v<T, A0>);
        if constexpr (simd_size_v<T, A0> == values_per_part) {
            return x.first;
        } else {
            return detail::extract_part<Index, simd_size_v<T, A0> / values_per_part>(
                x.first);
        }
    }
}
}  // namespace detail

template <class V, class A,
          size_t Parts = simd_size_v<typename V::value_type, A> / V::size()>
inline std::enable_if_t<(is_simd<V>::value &&
                         simd_size_v<typename V::value_type, A> == Parts * V::size()),
                        std::array<V, Parts>>
split(const simd<typename V::value_type, A> &x)
{
    using T = typename V::value_type;
    if constexpr (Parts == 1) {
        return {simd_cast<V>(x)};
    } else if constexpr (detail::is_fixed_size_abi_v<A> &&
                         (std::is_same_v<typename V::abi_type, simd_abi::scalar> ||
                          (detail::is_fixed_size_abi_v<typename V::abi_type> &&
                           sizeof(V) == sizeof(T) * V::size()  // V doesn't have padding
                           ))) {
        // fixed_size -> fixed_size (w/o padding) or scalar
#ifdef Vc_USE_ALIASING_LOADS
        const detail::may_alias<T> *const element_ptr =
            reinterpret_cast<const detail::may_alias<T> *>(&detail::data(x));
        return detail::generate_from_n_evaluations<Parts, std::array<V, Parts>>(
            [&](auto i) { return V(element_ptr + i * V::size(), vector_aligned); });
#else
        const auto &xx = detail::data(x);
        return detail::generate_from_n_evaluations<Parts, std::array<V, Parts>>(
            [&](auto i) {
                constexpr size_t offset = decltype(i)::value * V::size();
                return V([&](auto j) {
                    constexpr detail::size_constant<j + offset> k;
                    return xx[k];
                });
            });
#endif
    } else if constexpr (std::is_same_v<typename V::abi_type, simd_abi::scalar>) {
        // normally memcpy should work here as well
        return detail::generate_from_n_evaluations<Parts, std::array<V, Parts>>(
            [&](auto i) { return x[i]; });
    } else {
        return detail::generate_from_n_evaluations<Parts, std::array<V, Parts>>(
            [&](auto i) {
                return V(detail::private_init,
                         detail::extract_part<i, Parts>(detail::data(x)));
            });
    }
}

template <class V, class A,
          size_t Parts = simd_size_v<typename V::simd_type::value_type, A> / V::size()>
std::enable_if_t<(is_simd_mask_v<V> &&
                  simd_size_v<typename V::simd_type::value_type, A> == Parts * V::size()),
                 std::array<V, Parts>>
split(const simd_mask<typename V::simd_type::value_type, A> &x)
{
    if constexpr (std::is_same_v<A, typename V::abi_type>) {
        return {x};
    } else if constexpr (Parts == 1) {
        return {static_simd_cast<V>(x)};
    } else if constexpr (Parts == 2) {
        return {V{detail::private_init, [&](size_t i) { return x[i]; }},
                V{detail::private_init, [&](size_t i) { return x[i + V::size()]; }}};
    } else if constexpr (Parts == 3) {
        return {V{detail::private_init, [&](size_t i) { return x[i]; }},
                V{detail::private_init, [&](size_t i) { return x[i + V::size()]; }},
                V{detail::private_init, [&](size_t i) { return x[i + 2 * V::size()]; }}};
    } else if constexpr (Parts == 4) {
        return {V{detail::private_init, [&](size_t i) { return x[i]; }},
                V{detail::private_init, [&](size_t i) { return x[i + V::size()]; }},
                V{detail::private_init, [&](size_t i) { return x[i + 2 * V::size()]; }},
                V{detail::private_init, [&](size_t i) { return x[i + 3 * V::size()]; }}};
    } else {
        detail::assert_unreachable<V>();
    }
}

template <size_t... Sizes, class T, class A,
          class = std::enable_if_t<((Sizes + ...) == simd<T, A>::size())>>
inline std::tuple<simd<T, simd_abi::deduce_t<T, Sizes>>...> split(const simd<T, A> &);

namespace detail
{
template <size_t V0, size_t... Values> struct size_list {
    static constexpr size_t size = sizeof...(Values) + 1;

    template <size_t I> static constexpr size_t at(size_constant<I> = {})
    {
        if constexpr (I == 0) {
            return V0;
        } else {
            return size_list<Values...>::template at<I - 1>();
        }
    }

    template <size_t I> static constexpr auto before(size_constant<I> = {})
    {
        if constexpr (I == 0) {
            return size_constant<0>();
        } else {
            return size_constant<V0 + size_list<Values...>::template before<I - 1>()>();
        }
    }

    template <size_t N> static constexpr auto pop_front(size_constant<N> = {})
    {
        if constexpr (N == 0) {
            return size_list();
        } else {
            return size_list<Values...>::template pop_front<N-1>();
        }
    }
};

template <class T, size_t N> using deduced_simd = simd<T, simd_abi::deduce_t<T, N>>;
template <class T, size_t N>
using deduced_simd_mask = simd_mask<T, simd_abi::deduce_t<T, N>>;

template <class T, size_t N> inline Storage<T, N / 2> extract_center(Storage<T, N> x) {
    if constexpr (have_avx512f && sizeof(x) == 64) {
        if constexpr(std::is_integral_v<T>) {
            return _mm512_castsi512_si256(
                _mm512_shuffle_i32x4(x, x, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40));
        } else if constexpr (sizeof(T) == 4) {
            return _mm512_castps512_ps256(
                _mm512_shuffle_f32x4(x, x, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40));
        } else if constexpr (sizeof(T) == 8) {
            return _mm512_castpd512_pd256(
                _mm512_shuffle_f64x2(x, x, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40));
        } else {
            assert_unreachable<T>();
        }
    } else {
        assert_unreachable<T>();
    }
}
template <class T, class A>
inline Storage<T, simd_size_v<T, A>> extract_center(const simd_tuple<T, A, A> &x)
{
    return detail::concat(detail::extract<1, 2>(x.first.d),
                          detail::extract<0, 2>(x.second.first.d));
}
template <class T, class A>
inline Storage<T, simd_size_v<T, A> / 2> extract_center(const simd_tuple<T, A> &x)
{
    return detail::extract_center(x.first);
}

template <size_t... Sizes, class T, class... As>
auto split_wrapper(size_list<Sizes...>, const simd_tuple<T, As...> &x)
{
    return Vc::split<Sizes...>(
        fixed_size_simd<T, simd_tuple<T, As...>::size_v>(private_init, x));
}
}  // namespace detail

template <size_t... Sizes, class T, class A,
          class = std::enable_if_t<((Sizes + ...) == simd<T, A>::size())>>
Vc_ALWAYS_INLINE std::tuple<simd<T, simd_abi::deduce_t<T, Sizes>>...> split(
    const simd<T, A> &x)
{
    using SL = detail::size_list<Sizes...>;
    using Tuple = std::tuple<detail::deduced_simd<T, Sizes>...>;
    constexpr size_t N = simd_size_v<T, A>;
    constexpr size_t N0 = SL::template at<0>();
    using V = detail::deduced_simd<T, N0>;

    if constexpr (N == N0) {
        static_assert(sizeof...(Sizes) == 1);
        return {simd_cast<V>(x)};
    } else if constexpr (detail::is_fixed_size_abi_v<A> &&
                         detail::fixed_size_storage<T, N>::first_size_v == N0) {
        // if the first part of the simd_tuple input matches the first output vector
        // in the std::tuple, extract it and recurse
        static_assert(!detail::is_fixed_size_abi_v<typename V::abi_type>,
                      "How can <T, N> be a single simd_tuple entry but a fixed_size_simd "
                      "when deduced?");
        const detail::fixed_size_storage<T, N> &xx = detail::data(x);
        return std::tuple_cat(
            std::make_tuple(V(detail::private_init, xx.first)),
            detail::split_wrapper(SL::template pop_front<1>(), xx.second));
    } else if constexpr ((!std::is_same_v<simd_abi::scalar,
                                          simd_abi::deduce_t<T, Sizes>> &&
                          ...) &&
                         (!detail::is_fixed_size_abi_v<simd_abi::deduce_t<T, Sizes>> &&
                          ...)) {
        if constexpr (((Sizes * 2 == N)&&...)) {
            return {{detail::private_init, detail::extract_part<0, 2>(detail::data(x))},
                    {detail::private_init, detail::extract_part<1, 2>(detail::data(x))}};
        } else if constexpr (std::is_same_v<detail::size_list<Sizes...>,
                                            detail::size_list<N / 3, N / 3, N / 3>>) {
            return {{detail::private_init, detail::extract_part<0, 3>(detail::data(x))},
                    {detail::private_init, detail::extract_part<1, 3>(detail::data(x))},
                    {detail::private_init, detail::extract_part<2, 3>(detail::data(x))}};
        } else if constexpr (std::is_same_v<detail::size_list<Sizes...>,
                                            detail::size_list<2 * N / 3, N / 3>>) {
            return {{detail::private_init,
                     detail::concat(detail::extract_part<0, 3>(detail::data(x)),
                                    detail::extract_part<1, 3>(detail::data(x)))},
                    {detail::private_init, detail::extract_part<2, 3>(detail::data(x))}};
        } else if constexpr (std::is_same_v<detail::size_list<Sizes...>,
                                            detail::size_list<N / 3, 2 * N / 3>>) {
            return {{detail::private_init, detail::extract_part<0, 3>(detail::data(x))},
                    {detail::private_init,
                     detail::concat(detail::extract_part<1, 3>(detail::data(x)),
                                    detail::extract_part<2, 3>(detail::data(x)))}};
        } else if constexpr (std::is_same_v<detail::size_list<Sizes...>,
                                            detail::size_list<N / 2, N / 4, N / 4>>) {
            return {{detail::private_init, detail::extract_part<0, 2>(detail::data(x))},
                    {detail::private_init, detail::extract_part<2, 4>(detail::data(x))},
                    {detail::private_init, detail::extract_part<3, 4>(detail::data(x))}};
        } else if constexpr (std::is_same_v<detail::size_list<Sizes...>,
                                            detail::size_list<N / 4, N / 4, N / 2>>) {
            return {{detail::private_init, detail::extract_part<0, 4>(detail::data(x))},
                    {detail::private_init, detail::extract_part<1, 4>(detail::data(x))},
                    {detail::private_init, detail::extract_part<1, 2>(detail::data(x))}};
        } else if constexpr (std::is_same_v<detail::size_list<Sizes...>,
                                            detail::size_list<N / 4, N / 2, N / 4>>) {
            return {
                {detail::private_init, detail::extract_part<0, 4>(detail::data(x))},
                {detail::private_init, detail::extract_center(detail::data(x))},
                {detail::private_init, detail::extract_part<3, 4>(detail::data(x))}};
        } else if constexpr (((Sizes * 4 == N) && ...)) {
            return {{detail::private_init, detail::extract_part<0, 4>(detail::data(x))},
                    {detail::private_init, detail::extract_part<1, 4>(detail::data(x))},
                    {detail::private_init, detail::extract_part<2, 4>(detail::data(x))},
                    {detail::private_init, detail::extract_part<3, 4>(detail::data(x))}};
        //} else if constexpr (detail::is_fixed_size_abi_v<A>) {
        } else {
            detail::assert_unreachable<T>();
        }
    } else {
#ifdef Vc_USE_ALIASING_LOADS
        const detail::may_alias<T> *const element_ptr =
            reinterpret_cast<const detail::may_alias<T> *>(&x);
        return detail::generate_from_n_evaluations<sizeof...(Sizes), Tuple>([&](auto i) {
            using Vi = detail::deduced_simd<T, SL::at(i)>;
            constexpr size_t offset = SL::before(i);
            constexpr size_t base_align = alignof(simd<T, A>);
            constexpr size_t a = base_align - ((offset * sizeof(T)) % base_align);
            constexpr size_t b = ((a - 1) & a) ^ a;
            constexpr size_t alignment = b == 0 ? a : b;
            return Vi(element_ptr + offset, overaligned<alignment>);
        });
#else
        return detail::generate_from_n_evaluations<sizeof...(Sizes), Tuple>([&](auto i) {
            using Vi = detail::deduced_simd<T, SL::at(i)>;
            const auto &xx = detail::data(x);
            using Offset = decltype(SL::before(i));
            return Vi([&](auto j) {
                constexpr detail::size_constant<Offset::value + j> k;
                return xx[k];
            });
        });
#endif
    }
}

namespace detail
{
template <size_t I, class T, class A, class... As>
constexpr Vc_INTRINSIC T subscript_in_pack(const simd<T, A> &x, const simd<T, As> &... xs)
{
    if constexpr (I < simd_size_v<T, A>) {
        return x[I];
    } else {
        return subscript_in_pack<I - simd_size_v<T, A>>(xs...);
    }
}
}  // namespace detail

template <class T, class... As>
simd<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>> concat(
    const simd<T, As> &... xs)
{
    return simd<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>(
        [&](auto i) { return detail::subscript_in_pack<i>(xs...); });
}
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_SPLIT_H_

// vim: foldmethod=marker
