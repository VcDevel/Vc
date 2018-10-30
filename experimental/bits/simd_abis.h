#ifndef _GLIBCXX_EXPERIMENTAL_SIMD_ABIS_H_
#define _GLIBCXX_EXPERIMENTAL_SIMD_ABIS_H_

#pragma GCC system_header

#if __cplusplus >= 201703L

#include "simd.h"
#include <array>
#include <cmath>
#include <cstdlib>

#include "simd_debug.h"
_GLIBCXX_SIMD_BEGIN_NAMESPACE
namespace detail
{
// subscript_read/_write {{{1
template <class T> T subscript_read(Vectorizable<T> x, size_t) noexcept { return x; }
template <class T>
void subscript_write(Vectorizable<T> &x, size_t, detail::id<T> y) noexcept
{
    x = y;
}

template <class T> typename T::value_type subscript_read(const T &x, size_t i) noexcept
{
    return x[i];
}
template <class T> void subscript_write(T &x, size_t i, typename T::value_type y) noexcept
{
    return x.set(i, y);
}

// tuple_element {{{1
template <size_t I, class T> struct tuple_element;
template <class T, class A0, class... As>
struct tuple_element<0, simd_tuple<T, A0, As...>> {
    using type = std::experimental::simd<T, A0>;
};
template <size_t I, class T, class A0, class... As>
struct tuple_element<I, simd_tuple<T, A0, As...>> {
    using type = typename tuple_element<I - 1, simd_tuple<T, As...>>::type;
};
template <size_t I, class T> using tuple_element_t = typename tuple_element<I, T>::type;

// tuple_concat {{{1
template <class T, class... A1s>
_GLIBCXX_SIMD_INTRINSIC constexpr simd_tuple<T, A1s...> tuple_concat(
    const simd_tuple<T>, const simd_tuple<T, A1s...> right)
{
    return right;
}

template <class T, class A00, class... A0s, class A10, class... A1s>
_GLIBCXX_SIMD_INTRINSIC constexpr simd_tuple<T, A00, A0s..., A10, A1s...> tuple_concat(
    const simd_tuple<T, A00, A0s...> left, const simd_tuple<T, A10, A1s...> right)
{
    return {left.first, tuple_concat(left.second, right)};
}

template <class T, class A00, class... A0s>
_GLIBCXX_SIMD_INTRINSIC constexpr simd_tuple<T, A00, A0s...> tuple_concat(
    const simd_tuple<T, A00, A0s...> left, const simd_tuple<T>)
{
    return left;
}

template <class T, class A10, class... A1s>
_GLIBCXX_SIMD_INTRINSIC constexpr simd_tuple<T, simd_abi::scalar, A10, A1s...> tuple_concat(
    const T left, const simd_tuple<T, A10, A1s...> right)
{
    return {left, right};
}

// tuple_pop_front {{{1
template <class T>
_GLIBCXX_SIMD_INTRINSIC constexpr const T &tuple_pop_front(size_constant<0>, const T &x)
{
    return x;
}

template <class T> _GLIBCXX_SIMD_INTRINSIC constexpr T &tuple_pop_front(size_constant<0>, T &x)
{
    return x;
}

template <size_t K, class T>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto &tuple_pop_front(size_constant<K>, const T &x)
{
    return tuple_pop_front(size_constant<K - 1>(), x.second);
}

template <size_t K, class T>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &tuple_pop_front(size_constant<K>, T &x)
{
    return tuple_pop_front(size_constant<K - 1>(), x.second);
}

// tuple_front {{{1
template <size_t K, class T, class A0, class... As>
_GLIBCXX_SIMD_INTRINSIC constexpr auto tuple_front(const simd_tuple<T, A0, As...> &x)
{
    if constexpr (K == 0) {
        return simd_tuple<T>();
    } else if constexpr (K == 1) {
        return simd_tuple<T, A0>{x.first};
    } else {
        return tuple_concat(simd_tuple<T, A0>{x.first}, tuple_front<K - 1>(x.second));
    }
}

// get_simd<N> {{{1
namespace simd_tuple_impl
{
namespace as_simd
{
struct yes {};
struct no {};
}
template <class T, class A0, class... Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr simd<T, A0> get_impl(as_simd::yes,
                                            const simd_tuple<T, A0, Abis...> &t,
                                            size_constant<0>)
{
    return {private_init, t.first};
}
template <class T, class A0, class... Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto &get_impl(as_simd::no,
                                            const simd_tuple<T, A0, Abis...> &t,
                                            size_constant<0>)
{
    return t.first;
}
template <class T, class A0, class... Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &get_impl(as_simd::no, simd_tuple<T, A0, Abis...> &t,
                                      size_constant<0>)
{
    return t.first;
}

template <class R, size_t N, class T, class... Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto get_impl(R, const simd_tuple<T, Abis...> &t, size_constant<N>)
{
    return get_impl(R(), t.second, size_constant<N - 1>());
}
template <size_t N, class T, class... Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &get_impl(as_simd::no, simd_tuple<T, Abis...> &t,
                                      size_constant<N>)
{
    return get_impl(as_simd::no(), t.second, size_constant<N - 1>());
}
}  // namespace simd_tuple_impl

template <size_t N, class T, class... Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto get_simd(const simd_tuple<T, Abis...> &t)
{
    return simd_tuple_impl::get_impl(simd_tuple_impl::as_simd::yes(), t,
                                     size_constant<N>());
}

template <size_t N, class T, class... Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto get(const simd_tuple<T, Abis...> &t)
{
    return simd_tuple_impl::get_impl(simd_tuple_impl::as_simd::no(), t,
                                     size_constant<N>());
}

template <size_t N, class T, class... Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &get(simd_tuple<T, Abis...> &t)
{
    return simd_tuple_impl::get_impl(simd_tuple_impl::as_simd::no(), t,
                                     size_constant<N>());
}

// foo {{{1
template <size_t LeftN, class RightT> struct foo;
template <class RightT> struct foo<0, RightT> : public size_constant<0> {
};
template <size_t LeftN, class RightT>
struct foo
    : public size_constant<
          1 + foo<LeftN - RightT::first_size_v, typename RightT::second_type>::value> {
};

template <size_t LeftN, class RightT, bool = (RightT::first_size_v < LeftN)>
struct how_many_to_extract;

template <size_t LeftN, class RightT> struct how_many_to_extract<LeftN, RightT, true> {
    static constexpr std::make_index_sequence<
        1 + foo<LeftN - RightT::first_size_v, typename RightT::second_type>::value>
    tag()
    {
        return {};
    }
};

template <class T, size_t Offset, size_t Length, bool Done, class IndexSeq>
struct chunked {
};
template <size_t LeftN, class RightT> struct how_many_to_extract<LeftN, RightT, false> {
    static_assert(LeftN != RightT::first_size_v, "");
    static constexpr chunked<typename RightT::first_type, 0, LeftN, false,
                             std::make_index_sequence<LeftN>>
    tag()
    {
        return {};
    }
};

// tuple_element_meta {{{1
template <class T, class Abi, size_t Offset>
struct tuple_element_meta : public Abi::simd_impl_type {
    using value_type = T;
    using abi_type = Abi;
    using traits = detail::traits<T, Abi>;
    using maskimpl = typename traits::mask_impl_type;
    using member_type = typename traits::simd_member_type;
    using mask_member_type = typename traits::mask_member_type;
    using simd_type = std::experimental::simd<T, Abi>;
    static constexpr size_t offset = Offset;
    static constexpr size_t size() { return simd_size<T, Abi>::value; }
    static constexpr maskimpl simd_mask = {};

    template <size_t N>
    _GLIBCXX_SIMD_INTRINSIC static mask_member_type make_mask(std::bitset<N> bits)
    {
        constexpr T *type_tag = nullptr;
        return maskimpl::from_bitset(std::bitset<size()>((bits >> Offset).to_ullong()),
                                     type_tag);
    }

    _GLIBCXX_SIMD_INTRINSIC static ullong mask_to_shifted_ullong(mask_member_type k)
    {
        return detail::to_bitset(k).to_ullong() << Offset;
    }
};

template <size_t Offset, class T, class Abi, class... As>
tuple_element_meta<T, Abi, Offset> make_meta(const simd_tuple<T, Abi, As...> &)
{
    return {};
}

// simd_tuple specializations {{{1
// empty {{{2
template <class T> struct simd_tuple<T> {
    using value_type = T;
    static constexpr size_t tuple_size = 0;
    static constexpr size_t size() { return 0; }
};

// 1 member {{{2
template <class T, class Abi0> struct simd_tuple<T, Abi0> {
    using value_type = T;
    using first_type = typename detail::traits<T, Abi0>::simd_member_type;
    using second_type = simd_tuple<T>;
    using first_abi = Abi0;
    static constexpr size_t tuple_size = 1;
    static constexpr size_t size() { return simd_size_v<T, Abi0>; }
    static constexpr size_t first_size_v = simd_size_v<T, Abi0>;
    alignas(sizeof(first_type)) first_type first;
    static constexpr second_type second = {};

    template <size_t Offset = 0, class F>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple generate(F &&gen, detail::size_constant<Offset> = {})
    {
        return {gen(tuple_element_meta<T, Abi0, Offset>())};
    }

    template <size_t Offset = 0, class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC simd_tuple apply_wrapped(F &&fun, const More &... more) const
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        return {fun(make_meta<Offset>(*this), first, more.first...)};
    }

    template <class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC friend simd_tuple simd_tuple_apply(F &&fun, const simd_tuple &x,
                                                    More &&... more)
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        return simd_tuple::apply_impl(
            bool_constant<all<is_equal<size_t, first_size_v,
                                       std::decay_t<More>::first_size_v>...>::value>(),
            std::forward<F>(fun), x, std::forward<More>(more)...);
    }

private:
    template <class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple
    apply_impl(true_type,  // first_size_v is equal for all arguments
               F &&fun, const simd_tuple &x, More &&... more)
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("more.first = ", more.first..., "more = ", more...);
        return {fun(tuple_element_meta<T, Abi0, 0>(), x.first, more.first...)};
    }

    template <class F, class More>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple apply_impl(false_type,  // at least one argument in
                                                           // More has different
                                                           // first_size_v, x has only one
                                                           // member, so More has 2 or
                                                           // more
                                              F &&fun, const simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("y = ", y);
        return apply_impl(std::make_index_sequence<std::decay_t<More>::tuple_size>(),
                          std::forward<F>(fun), x, std::forward<More>(y));
    }

    template <class F, class More, size_t... Indexes>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple apply_impl(std::index_sequence<Indexes...>, F &&fun,
                                              const simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("y = ", y);
        auto tmp = std::experimental::concat(detail::get_simd<Indexes>(y)...);
        const auto first = fun(tuple_element_meta<T, Abi0, 0>(), x.first, tmp);
        if constexpr (std::is_lvalue_reference<More>::value &&
                      !std::is_const<More>::value) {
            // if y is non-const lvalue ref, assume write back is necessary
            const auto tup =
                std::experimental::split<tuple_element_t<Indexes, std::decay_t<More>>::size()...>(tmp);
            auto &&ignore = {
                (get<Indexes>(y) = detail::data(std::get<Indexes>(tup)), 0)...};
            detail::unused(ignore);
        }
        return {first};
    }

public:
    // apply_impl2 can only be called from a 2-element simd_tuple
    template <class Tuple, size_t Offset, class F2>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple extract(
        size_constant<Offset>, size_constant<std::decay_t<Tuple>::first_size_v - Offset>,
        Tuple &&tup, F2 &&fun2)
    {
        static_assert(Offset > 0, "");
        auto splitted =
            split<Offset, std::decay_t<Tuple>::first_size_v - Offset>(get_simd<0>(tup));
        simd_tuple r = fun2(detail::data(std::get<1>(splitted)));
        // if tup is non-const lvalue ref, write get<0>(splitted) back
        tup.first = detail::data(concat(std::get<0>(splitted), std::get<1>(splitted)));
        return r;
    }

    template <class F, class More, class U, size_t Length, size_t... Indexes>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple
    apply_impl2(chunked<U, std::decay_t<More>::first_size_v, Length, true,
                        std::index_sequence<Indexes...>>,
                F &&fun, const simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        return simd_tuple_apply(std::forward<F>(fun), x, y.second);
    }

    template <class F, class More, class U, size_t Offset, size_t Length,
              size_t... Indexes>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple
    apply_impl2(chunked<U, Offset, Length, false, std::index_sequence<Indexes...>>,
                F &&fun, const simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        static_assert(Offset < std::decay_t<More>::first_size_v, "");
        static_assert(Offset > 0, "");
        return extract(size_constant<Offset>(), size_constant<Length>(), y,
                       [&](auto &&yy) -> simd_tuple {
                           return {fun(tuple_element_meta<T, Abi0, 0>(), x.first, yy)};
                       });
    }

    template <class R = T, class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC fixed_size_storage<R, size()> apply_r(F &&fun,
                                                       const More &... more) const
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        return {fun(tuple_element_meta<T, Abi0, 0>(), first, more.first...)};
    }

    template <class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC friend std::bitset<size()> test(F &&fun, const simd_tuple &x,
                                                 const More &... more)
    {
        return detail::to_bitset(
            fun(tuple_element_meta<T, Abi0, 0>(), x.first, more.first...));
    }

    T operator[](size_t i) const noexcept { return subscript_read(first, i); }
    void set(size_t i, T val) noexcept { subscript_write(first, i, val); }
};

// 2 or more {{{2
template <class T, class Abi0, class... Abis> struct simd_tuple<T, Abi0, Abis...> {
    using value_type = T;
    using first_type = typename detail::traits<T, Abi0>::simd_member_type;
    using first_abi = Abi0;
    using second_type = simd_tuple<T, Abis...>;
    static constexpr size_t tuple_size = sizeof...(Abis) + 1;
    static constexpr size_t size() { return simd_size_v<T, Abi0> + second_type::size(); }
    static constexpr size_t first_size_v = simd_size_v<T, Abi0>;
    static constexpr size_t alignment =
        std::clamp(next_power_of_2(sizeof(T) * size()), size_t(16), size_t(256));
    alignas(alignment) first_type first;
    second_type second;

    template <size_t Offset = 0, class F>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple generate(F &&gen, detail::size_constant<Offset> = {})
    {
        return {gen(tuple_element_meta<T, Abi0, Offset>()),
                second_type::generate(
                    std::forward<F>(gen),
                    detail::size_constant<Offset + simd_size_v<T, Abi0>>())};
    }

    template <size_t Offset = 0, class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC simd_tuple apply_wrapped(F &&fun, const More &... more) const
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        return {fun(make_meta<Offset>(*this), first, more.first...),
                second.template apply_wrapped<Offset + simd_size_v<T, Abi0>>(
                    std::forward<F>(fun), more.second...)};
    }

    template <class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC friend simd_tuple simd_tuple_apply(F &&fun, const simd_tuple &x,
                                                    More &&... more)
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("more = ", more...);
        return simd_tuple::apply_impl(
            bool_constant<all<is_equal<size_t, first_size_v,
                                       std::decay_t<More>::first_size_v>...>::value>(),
            std::forward<F>(fun), x, std::forward<More>(more)...);
    }

private:
    template <class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple
    apply_impl(true_type,  // first_size_v is equal for all arguments
               F &&fun, const simd_tuple &x, More &&... more)
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        return {fun(tuple_element_meta<T, Abi0, 0>(), x.first, more.first...),
                simd_tuple_apply(std::forward<F>(fun), x.second, more.second...)};
    }

    template <class F, class More>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple
    apply_impl(false_type,  // at least one argument in More has different first_size_v
               F &&fun, const simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("y = ", y);
        return apply_impl2(how_many_to_extract<first_size_v, std::decay_t<More>>::tag(),
                           std::forward<F>(fun), x, y);
    }

    template <class F, class More, size_t... Indexes>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple apply_impl2(std::index_sequence<Indexes...>, F &&fun,
                                               const simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("y = ", y);
        auto tmp = std::experimental::concat(detail::get_simd<Indexes>(y)...);
        const auto first = fun(tuple_element_meta<T, Abi0, 0>(), x.first, tmp);
        if constexpr (std::is_lvalue_reference<More>::value &&
                      !std::is_const<More>::value) {
            // if y is non-const lvalue ref, assume write back is necessary
            const auto tup =
                std::experimental::split<tuple_element_t<Indexes, std::decay_t<More>>::size()...>(tmp);
            [](std::initializer_list<int>) {
            }({(get<Indexes>(y) = detail::data(std::get<Indexes>(tup)), 0)...});
        }
        return {first, simd_tuple_apply(
                           std::forward<F>(fun), x.second,
                           tuple_pop_front(size_constant<sizeof...(Indexes)>(), y))};
    }

public:
    template <class F, class More, class U, size_t Length, size_t... Indexes>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple
    apply_impl2(chunked<U, std::decay_t<More>::first_size_v, Length, true,
                        std::index_sequence<Indexes...>>,
                F &&fun, const simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        return simd_tuple_apply(std::forward<F>(fun), x, y.second);
    }

    template <class Tuple, size_t Length, class F2>
    _GLIBCXX_SIMD_INTRINSIC static auto extract(size_constant<0>, size_constant<Length>, Tuple &&tup,
                                     F2 &&fun2)
    {
        auto splitted =
            split<Length, std::decay_t<Tuple>::first_size_v - Length>(get_simd<0>(tup));
        auto r = fun2(detail::data(std::get<0>(splitted)));
        // if tup is non-const lvalue ref, write get<0>(splitted) back
        tup.first = detail::data(concat(std::get<0>(splitted), std::get<1>(splitted)));
        return r;
    }

    template <class Tuple, size_t Offset, class F2>
    _GLIBCXX_SIMD_INTRINSIC static auto extract(
        size_constant<Offset>, size_constant<std::decay_t<Tuple>::first_size_v - Offset>,
        Tuple &&tup, F2 &&fun2)
    {
        auto splitted =
            split<Offset, std::decay_t<Tuple>::first_size_v - Offset>(get_simd<0>(tup));
        auto r = fun2(detail::data(std::get<1>(splitted)));
        // if tup is non-const lvalue ref, write get<0>(splitted) back
        tup.first = detail::data(concat(std::get<0>(splitted), std::get<1>(splitted)));
        return r;
    }

    template <
        class Tuple, size_t Offset, size_t Length, class F2,
        class = std::enable_if_t<(Offset + Length < std::decay_t<Tuple>::first_size_v)>>
    _GLIBCXX_SIMD_INTRINSIC static auto extract(size_constant<Offset>, size_constant<Length>,
                                     Tuple &&tup, F2 &&fun2)
    {
        static_assert(Offset + Length < std::decay_t<Tuple>::first_size_v, "");
        auto splitted =
            split<Offset, Length, std::decay_t<Tuple>::first_size_v - Offset - Length>(
                get_simd<0>(tup));
        auto r = fun2(detail::data(std::get<1>(splitted)));
        // if tup is non-const lvalue ref, write get<0>(splitted) back
        tup.first = detail::data(
            concat(std::get<0>(splitted), std::get<1>(splitted), std::get<2>(splitted)));
        return r;
    }

    template <class F, class More, class U, size_t Offset, size_t Length,
              size_t... Indexes>
    _GLIBCXX_SIMD_INTRINSIC static simd_tuple
    apply_impl2(chunked<U, Offset, Length, false, std::index_sequence<Indexes...>>,
                F &&fun, const simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        static_assert(Offset < std::decay_t<More>::first_size_v, "");
        return {extract(size_constant<Offset>(), size_constant<Length>(), y,
                        [&](auto &&yy) {
                            return fun(tuple_element_meta<T, Abi0, 0>(), x.first, yy);
                        }),
                second_type::apply_impl2(
                    chunked<U, Offset + Length, Length,
                            Offset + Length == std::decay_t<More>::first_size_v,
                            std::index_sequence<Indexes...>>(),
                    std::forward<F>(fun), x.second, y)};
    }

    template <class R = T, class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC auto apply_r(F &&fun, const More &... more) const
    {
        _GLIBCXX_SIMD_DEBUG(simd_tuple);
        return detail::tuple_concat<R>(
            fun(tuple_element_meta<T, Abi0, 0>(), first, more.first...),
            second.template apply_r<R>(std::forward<F>(fun), more.second...));
    }

    template <class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC friend std::bitset<size()> test(F &&fun, const simd_tuple &x,
                                                 const More &... more)
    {
        return detail::to_bitset(
                   fun(tuple_element_meta<T, Abi0, 0>(), x.first, more.first...))
                   .to_ullong() |
               (test(fun, x.second, more.second...).to_ullong() << simd_size_v<T, Abi0>);
    }

    template <class U, U I>
    _GLIBCXX_SIMD_INTRINSIC constexpr T operator[](std::integral_constant<U, I>) const noexcept
    {
        if constexpr (I < simd_size_v<T, Abi0>) {
            return subscript_read(first, I);
        } else {
            return second[std::integral_constant<U, I - simd_size_v<T, Abi0>>()];
        }
    }

    T operator[](size_t i) const noexcept
    {
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        return reinterpret_cast<const may_alias<T> *>(this)[i];
#else
        return i < simd_size_v<T, Abi0> ? subscript_read(first, i)
                                        : second[i - simd_size_v<T, Abi0>];
#endif
    }
    void set(size_t i, T val) noexcept
    {
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        reinterpret_cast<may_alias<T> *>(this)[i] = val;
#else
        if (i < simd_size_v<T, Abi0>) {
            subscript_write(first, i, val);
        } else {
            second.set(i - simd_size_v<T, Abi0>, val);
        }
#endif
    }
};

// make_tuple {{{1
template <class T, class A0>
_GLIBCXX_SIMD_INTRINSIC simd_tuple<T, A0> make_tuple(std::experimental::simd<T, A0> x0)
{
    return {detail::data(x0)};
}
template <class T, class A0, class... As>
_GLIBCXX_SIMD_INTRINSIC simd_tuple<T, A0, As...> make_tuple(const std::experimental::simd<T, A0> &x0,
                                                    const std::experimental::simd<T, As> &... xs)
{
    return {detail::data(x0), make_tuple(xs...)};
}

template <class T, class A0>
_GLIBCXX_SIMD_INTRINSIC simd_tuple<T, A0> make_tuple(
    const typename detail::traits<T, A0>::simd_member_type &arg0)
{
    return {arg0};
}

template <class T, class A0, class A1, class... Abis>
_GLIBCXX_SIMD_INTRINSIC simd_tuple<T, A0, A1, Abis...> make_tuple(
    const typename detail::traits<T, A0>::simd_member_type &arg0,
    const typename detail::traits<T, A1>::simd_member_type &arg1,
    const typename detail::traits<T, Abis>::simd_member_type &... args)
{
    return {arg0, make_tuple<T, A1, Abis...>(arg1, args...)};
}

// to_tuple {{{1
template <size_t, class T> using to_tuple_helper = T;
template <class T, class A0, size_t... Indexes>
_GLIBCXX_SIMD_INTRINSIC simd_tuple<T, to_tuple_helper<Indexes, A0>...> to_tuple_impl(
    std::index_sequence<Indexes...>,
    const std::array<detail::builtin_type_t<T, simd_size_v<T, A0>>, sizeof...(Indexes)>
        &args)
{
    return make_tuple<T, to_tuple_helper<Indexes, A0>...>(args[Indexes]...);
}

template <class T, class A0, size_t N>
_GLIBCXX_SIMD_INTRINSIC auto to_tuple(
    const std::array<detail::builtin_type_t<T, simd_size_v<T, A0>>, N> &args)
{
    return to_tuple_impl<T, A0>(std::make_index_sequence<N>(), args);
}

// optimize_tuple {{{1
template <class T> _GLIBCXX_SIMD_INTRINSIC simd_tuple<T> optimize_tuple(const simd_tuple<T>)
{
    return {};
}

template <class T, class A>
_GLIBCXX_SIMD_INTRINSIC const simd_tuple<T, A> &optimize_tuple(const simd_tuple<T, A> &x)
{
    return x;
}

template <class T, class A0, class A1, class... Abis,
          class R = fixed_size_storage<T, simd_tuple<T, A0, A1, Abis...>::size()>>
_GLIBCXX_SIMD_INTRINSIC R optimize_tuple(const simd_tuple<T, A0, A1, Abis...> &x)
{
    using Tup = simd_tuple<T, A0, A1, Abis...>;
    if constexpr (R::first_size_v == simd_size_v<T, A0>) {
        return tuple_concat(simd_tuple<T, typename R::first_abi>{x.first},
                            optimize_tuple(x.second));
    } else if constexpr (R::first_size_v == simd_size_v<T, A0> + simd_size_v<T, A1>) {
        return tuple_concat(simd_tuple<T, typename R::first_abi>{detail::data(
                                std::experimental::concat(get_simd<0>(x), get_simd<1>(x)))},
                            optimize_tuple(x.second.second));
    } else if constexpr (sizeof...(Abis) >= 2) {
        if constexpr (R::first_size_v == tuple_element_t<0, Tup>::size() +
                                             tuple_element_t<1, Tup>::size() +
                                             tuple_element_t<2, Tup>::size() +
                                             tuple_element_t<3, Tup>::size()) {
            return tuple_concat(
                simd_tuple<T, typename R::first_abi>{detail::data(concat(
                    get_simd<0>(x), get_simd<1>(x), get_simd<2>(x), get_simd<3>(x)))},
                optimize_tuple(x.second.second.second.second));
        }
    } else {
        return x;
    }
}

// number_of_preceding_elements {{{1
template <size_t I, class T> struct number_of_preceding_elements;
template <class T, class A0, class... As>
struct number_of_preceding_elements<0, simd_tuple<T, A0, As...>>
    : public size_constant<0> {
};
template <size_t I, class T, class A0, class... As>
struct number_of_preceding_elements<I, simd_tuple<T, A0, As...>>
    : public std::integral_constant<
          size_t,
          simd<T, A0>::size() +
              number_of_preceding_elements<I - 1, simd_tuple<T, As...>>::value> {
};

// for_each(const simd_tuple &, Fun) {{{1
template <size_t Offset = 0, class T, class A0, class F>
_GLIBCXX_SIMD_INTRINSIC void for_each(const simd_tuple<T, A0> &t_, F &&fun_)
{
    std::forward<F>(fun_)(make_meta<Offset>(t_), t_.first);
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
_GLIBCXX_SIMD_INTRINSIC void for_each(const simd_tuple<T, A0, A1, As...> &t_, F &&fun_)
{
    fun_(make_meta<Offset>(t_), t_.first);
    for_each<Offset + simd_size<T, A0>::value>(t_.second, std::forward<F>(fun_));
}

// for_each(simd_tuple &, Fun) {{{1
template <size_t Offset = 0, class T, class A0, class F>
_GLIBCXX_SIMD_INTRINSIC void for_each(simd_tuple<T, A0> &t_, F &&fun_)
{
    std::forward<F>(fun_)(make_meta<Offset>(t_), t_.first);
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
_GLIBCXX_SIMD_INTRINSIC void for_each(simd_tuple<T, A0, A1, As...> &t_, F &&fun_)
{
    fun_(make_meta<Offset>(t_), t_.first);
    for_each<Offset + simd_size<T, A0>::value>(t_.second, std::forward<F>(fun_));
}

// for_each(simd_tuple &, const simd_tuple &, Fun) {{{1
template <size_t Offset = 0, class T, class A0, class F>
_GLIBCXX_SIMD_INTRINSIC void for_each(simd_tuple<T, A0> &a_, const simd_tuple<T, A0> &b_,
                           F &&fun_)
{
    std::forward<F>(fun_)(make_meta<Offset>(a_), a_.first, b_.first);
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
_GLIBCXX_SIMD_INTRINSIC void for_each(simd_tuple<T, A0, A1, As...> & a_,
                           const simd_tuple<T, A0, A1, As...> &b_, F &&fun_)
{
    fun_(make_meta<Offset>(a_), a_.first, b_.first);
    for_each<Offset + simd_size<T, A0>::value>(a_.second, b_.second,
                                                  std::forward<F>(fun_));
}

// for_each(const simd_tuple &, const simd_tuple &, Fun) {{{1
template <size_t Offset = 0, class T, class A0, class F>
_GLIBCXX_SIMD_INTRINSIC void for_each(const simd_tuple<T, A0> &a_, const simd_tuple<T, A0> &b_,
                           F &&fun_)
{
    std::forward<F>(fun_)(make_meta<Offset>(a_), a_.first, b_.first);
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
_GLIBCXX_SIMD_INTRINSIC void for_each(const simd_tuple<T, A0, A1, As...> &a_,
                           const simd_tuple<T, A0, A1, As...> &b_, F &&fun_)
{
    fun_(make_meta<Offset>(a_), a_.first, b_.first);
    for_each<Offset + simd_size<T, A0>::value>(a_.second, b_.second,
                                                  std::forward<F>(fun_));
}

// }}}1
#if defined _GLIBCXX_SIMD_HAVE_SSE || defined _GLIBCXX_SIMD_HAVE_MMX
namespace x86
{
// missing intrinsics {{{
#if defined _GLIBCXX_SIMD_GCC && _GLIBCXX_SIMD_GCC < 0x80000
_GLIBCXX_SIMD_INTRINSIC void _mm_mask_cvtepi16_storeu_epi8(void *p, __mmask8 k, __m128i x)
{
    asm("vpmovwb %0,(%2)%{%1%}" :: "x"(x), "k"(k), "g"(p) : "k0");
}
_GLIBCXX_SIMD_INTRINSIC void _mm256_mask_cvtepi16_storeu_epi8(void *p, __mmask16 k, __m256i x)
{
    asm("vpmovwb %0,(%2)%{%1%}" :: "x"(x), "k"(k), "g"(p) : "k0");
}
_GLIBCXX_SIMD_INTRINSIC void _mm512_mask_cvtepi16_storeu_epi8(void *p, __mmask32 k, __m512i x)
{
    asm("vpmovwb %0,(%2)%{%1%}" :: "x"(x), "k"(k), "g"(p) : "k0");
}
#endif

// }}}
// set16/32/64{{{1
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m128 set(float x0, float x1, float x2, float x3)
{
    return _mm_set_ps(x3, x2, x1, x0);
}
#ifdef _GLIBCXX_SIMD_HAVE_SSE2
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m128d set(double x0, double x1) { return _mm_set_pd(x1, x0); }

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m128i set(llong x0, llong x1) { return _mm_set_epi64x(x1, x0); }
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m128i set(ullong x0, ullong x1) { return _mm_set_epi64x(x1, x0); }

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m128i set(int x0, int x1, int x2, int x3)
{
    return _mm_set_epi32(x3, x2, x1, x0);
}
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m128i set(uint x0, uint x1, uint x2, uint x3)
{
    return _mm_set_epi32(x3, x2, x1, x0);
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m128i set(short x0, short x1, short x2, short x3, short x4,
                                  short x5, short x6, short x7)
{
    return _mm_set_epi16(x7, x6, x5, x4, x3, x2, x1, x0);
}
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m128i set(ushort x0, ushort x1, ushort x2, ushort x3, ushort x4,
                                  ushort x5, ushort x6, ushort x7)
{
    return _mm_set_epi16(x7, x6, x5, x4, x3, x2, x1, x0);
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m128i set(schar x0, schar x1, schar x2, schar x3, schar x4,
                                  schar x5, schar x6, schar x7, schar x8, schar x9,
                                  schar x10, schar x11, schar x12, schar x13, schar x14,
                                  schar x15)
{
    return _mm_set_epi8(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1,
                         x0);
}
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m128i set(uchar x0, uchar x1, uchar x2, uchar x3, uchar x4,
                                  uchar x5, uchar x6, uchar x7, uchar x8, uchar x9,
                                  uchar x10, uchar x11, uchar x12, uchar x13, uchar x14,
                                  uchar x15)
{
    return _mm_set_epi8(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1,
                         x0);
}
#endif  // _GLIBCXX_SIMD_HAVE_SSE2

#ifdef _GLIBCXX_SIMD_HAVE_AVX
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m256 set(float x0, float x1, float x2, float x3, float x4,
                                 float x5, float x6, float x7)
{
    return _mm256_set_ps(x7, x6, x5, x4, x3, x2, x1, x0);
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m256d set(double x0, double x1, double x2, double x3)
{
    return _mm256_set_pd(x3, x2, x1, x0);
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m256i set(llong x0, llong x1, llong x2, llong x3)
{
    return _mm256_set_epi64x(x3, x2, x1, x0);
}
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m256i set(ullong x0, ullong x1, ullong x2, ullong x3)
{
    return _mm256_set_epi64x(x3, x2, x1, x0);
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m256i set(int x0, int x1, int x2, int x3, int x4, int x5, int x6,
                                  int x7)
{
    return _mm256_set_epi32(x7, x6, x5, x4, x3, x2, x1, x0);
}
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m256i set(uint x0, uint x1, uint x2, uint x3, uint x4, uint x5,
                                  uint x6, uint x7)
{
    return _mm256_set_epi32(x7, x6, x5, x4, x3, x2, x1, x0);
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m256i set(short x0, short x1, short x2, short x3, short x4,
                                  short x5, short x6, short x7, short x8, short x9,
                                  short x10, short x11, short x12, short x13, short x14,
                                  short x15)
{
    return _mm256_set_epi16(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2,
                            x1, x0);
}
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m256i set(ushort x0, ushort x1, ushort x2, ushort x3, ushort x4,
                                  ushort x5, ushort x6, ushort x7, ushort x8, ushort x9,
                                  ushort x10, ushort x11, ushort x12, ushort x13,
                                  ushort x14, ushort x15)
{
    return _mm256_set_epi16(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2,
                            x1, x0);
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m256i set(schar x0, schar x1, schar x2, schar x3, schar x4,
                                  schar x5, schar x6, schar x7, schar x8, schar x9,
                                  schar x10, schar x11, schar x12, schar x13, schar x14,
                                  schar x15, schar x16, schar x17, schar x18, schar x19,
                                  schar x20, schar x21, schar x22, schar x23, schar x24,
                                  schar x25, schar x26, schar x27, schar x28, schar x29,
                                  schar x30, schar x31)
{
    return _mm256_set_epi8(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21, x20,
                           x19, x18, x17, x16, x15, x14, x13, x12, x11, x10, x9, x8, x7,
                           x6, x5, x4, x3, x2, x1, x0);
}
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m256i set(uchar x0, uchar x1, uchar x2, uchar x3, uchar x4,
                                  uchar x5, uchar x6, uchar x7, uchar x8, uchar x9,
                                  uchar x10, uchar x11, uchar x12, uchar x13, uchar x14,
                                  uchar x15, uchar x16, uchar x17, uchar x18, uchar x19,
                                  uchar x20, uchar x21, uchar x22, uchar x23, uchar x24,
                                  uchar x25, uchar x26, uchar x27, uchar x28, uchar x29,
                                  uchar x30, uchar x31)
{
    return _mm256_set_epi8(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21, x20,
                           x19, x18, x17, x16, x15, x14, x13, x12, x11, x10, x9, x8, x7,
                           x6, x5, x4, x3, x2, x1, x0);
}
#endif  // _GLIBCXX_SIMD_HAVE_AVX

#ifdef _GLIBCXX_SIMD_HAVE_AVX512F
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m512d set(double x0, double x1, double x2, double x3, double x4,
                                  double x5, double x6, double x7)
{
    return _mm512_set_pd(x7, x6, x5, x4, x3, x2, x1, x0);
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m512 set(float x0, float x1, float x2, float x3, float x4,
                                 float x5, float x6, float x7, float x8, float x9,
                                 float x10, float x11, float x12, float x13, float x14,
                                 float x15)
{
    return _mm512_set_ps(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1,
                         x0);
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m512i set(llong x0, llong x1, llong x2, llong x3, llong x4,
                                  llong x5, llong x6, llong x7)
{
    return _mm512_set_epi64(x7, x6, x5, x4, x3, x2, x1, x0);
}
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m512i set(ullong x0, ullong x1, ullong x2, ullong x3, ullong x4,
                                  ullong x5, ullong x6, ullong x7)
{
    return _mm512_set_epi64(x7, x6, x5, x4, x3, x2, x1, x0);
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m512i set(int x0, int x1, int x2, int x3, int x4, int x5, int x6,
                                  int x7, int x8, int x9, int x10, int x11, int x12,
                                  int x13, int x14, int x15)
{
    return _mm512_set_epi32(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2,
                            x1, x0);
}
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m512i set(uint x0, uint x1, uint x2, uint x3, uint x4, uint x5,
                                  uint x6, uint x7, uint x8, uint x9, uint x10, uint x11,
                                  uint x12, uint x13, uint x14, uint x15)
{
    return _mm512_set_epi32(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2,
                            x1, x0);
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m512i set(short x0, short x1, short x2, short x3, short x4,
                                  short x5, short x6, short x7, short x8, short x9,
                                  short x10, short x11, short x12, short x13, short x14,
                                  short x15, short x16, short x17, short x18, short x19,
                                  short x20, short x21, short x22, short x23, short x24,
                                  short x25, short x26, short x27, short x28, short x29,
                                  short x30, short x31)
{
    return concat(_mm256_set_epi16(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4,
                                   x3, x2, x1, x0),
                  _mm256_set_epi16(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21,
                                   x20, x19, x18, x17, x16));
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m512i set(ushort x0, ushort x1, ushort x2, ushort x3, ushort x4,
                                  ushort x5, ushort x6, ushort x7, ushort x8, ushort x9,
                                  ushort x10, ushort x11, ushort x12, ushort x13, ushort x14,
                                  ushort x15, ushort x16, ushort x17, ushort x18, ushort x19,
                                  ushort x20, ushort x21, ushort x22, ushort x23, ushort x24,
                                  ushort x25, ushort x26, ushort x27, ushort x28, ushort x29,
                                  ushort x30, ushort x31)
{
    return concat(_mm256_set_epi16(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4,
                                   x3, x2, x1, x0),
                  _mm256_set_epi16(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21,
                                   x20, x19, x18, x17, x16));
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m512i
set(schar x0, schar x1, schar x2, schar x3, schar x4, schar x5, schar x6, schar x7,
    schar x8, schar x9, schar x10, schar x11, schar x12, schar x13, schar x14, schar x15,
    schar x16, schar x17, schar x18, schar x19, schar x20, schar x21, schar x22,
    schar x23, schar x24, schar x25, schar x26, schar x27, schar x28, schar x29,
    schar x30, schar x31, schar x32, schar x33, schar x34, schar x35, schar x36,
    schar x37, schar x38, schar x39, schar x40, schar x41, schar x42, schar x43,
    schar x44, schar x45, schar x46, schar x47, schar x48, schar x49, schar x50,
    schar x51, schar x52, schar x53, schar x54, schar x55, schar x56, schar x57,
    schar x58, schar x59, schar x60, schar x61, schar x62, schar x63)
{
    return concat(_mm256_set_epi8(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21,
                                  x20, x19, x18, x17, x16, x15, x14, x13, x12, x11, x10,
                                  x9, x8, x7, x6, x5, x4, x3, x2, x1, x0),
                  _mm256_set_epi8(x63, x62, x61, x60, x59, x58, x57, x56, x55, x54, x53,
                                  x52, x51, x50, x49, x48, x47, x46, x45, x44, x43, x42,
                                  x41, x40, x39, x38, x37, x36, x35, x34, x33, x32));
}

_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST __m512i
set(uchar x0, uchar x1, uchar x2, uchar x3, uchar x4, uchar x5, uchar x6, uchar x7,
    uchar x8, uchar x9, uchar x10, uchar x11, uchar x12, uchar x13, uchar x14, uchar x15,
    uchar x16, uchar x17, uchar x18, uchar x19, uchar x20, uchar x21, uchar x22,
    uchar x23, uchar x24, uchar x25, uchar x26, uchar x27, uchar x28, uchar x29,
    uchar x30, uchar x31, uchar x32, uchar x33, uchar x34, uchar x35, uchar x36,
    uchar x37, uchar x38, uchar x39, uchar x40, uchar x41, uchar x42, uchar x43,
    uchar x44, uchar x45, uchar x46, uchar x47, uchar x48, uchar x49, uchar x50,
    uchar x51, uchar x52, uchar x53, uchar x54, uchar x55, uchar x56, uchar x57,
    uchar x58, uchar x59, uchar x60, uchar x61, uchar x62, uchar x63)
{
    return concat(_mm256_set_epi8(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21,
                                  x20, x19, x18, x17, x16, x15, x14, x13, x12, x11, x10,
                                  x9, x8, x7, x6, x5, x4, x3, x2, x1, x0),
                  _mm256_set_epi8(x63, x62, x61, x60, x59, x58, x57, x56, x55, x54, x53,
                                  x52, x51, x50, x49, x48, x47, x46, x45, x44, x43, x42,
                                  x41, x40, x39, x38, x37, x36, x35, x34, x33, x32));
}

#endif  // _GLIBCXX_SIMD_HAVE_AVX512F

// generic forward for (u)long to (u)int or (u)llong
template <typename... Ts> _GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST auto set(Ts... args)
{
    return set(static_cast<equal_int_type_t<Ts>>(args)...);
}

// blend{{{1
template <class K, class V0, class V1>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST auto blend(K mask, V0 at0, V1 at1)
{
    using V = V0;
    if constexpr (!std::is_same_v<V0, V1>) {
        static_assert(sizeof(V0) == sizeof(V1));
        if constexpr (is_builtin_vector_v<V0> && !is_builtin_vector_v<V1>) {
            return blend(mask, at0, reinterpret_cast<V0>(at1.d));
        } else if constexpr (!is_builtin_vector_v<V0> && is_builtin_vector_v<V1>) {
            return blend(mask, reinterpret_cast<V1>(at0.d), at1);
        } else {
            assert_unreachable<K>();
        }
    } else if constexpr (sizeof(V) < 16) {
        static_assert(sizeof(K) == sizeof(V0) && sizeof(V0) == sizeof(V1));
        return (mask & at1) | (~mask & at0);
    } else if constexpr (!is_builtin_vector_v<V>) {
        return blend(mask, at0.d, at1.d);
    } else if constexpr (sizeof(K) < 16) {
        using T = typename builtin_traits<V>::value_type;
        if constexpr (sizeof(V) == 16 && have_avx512bw_vl && sizeof(T) <= 2) {
            if constexpr (sizeof(T) == 1) {
                return _mm_mask_mov_epi8(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (sizeof(T) == 2) {
                return _mm_mask_mov_epi16(to_intrin(at0), mask, to_intrin(at1));
            }
        } else if constexpr (sizeof(V) == 16 && have_avx512vl && sizeof(T) > 2) {
            if constexpr (std::is_integral_v<T> && sizeof(T) == 4) {
                return _mm_mask_mov_epi32(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
                return _mm_mask_mov_epi64(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) {
                return _mm_mask_mov_ps(at0, mask, at1);
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 8) {
                return _mm_mask_mov_pd(at0, mask, at1);
            }
        } else if constexpr (sizeof(V) == 16 && have_avx512f && sizeof(T) > 2) {
            if constexpr (std::is_integral_v<T> && sizeof(T) == 4) {
                return lo128(_mm512_mask_mov_epi32(auto_cast(at0), mask, auto_cast(at1)));
            } else if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
                return lo128(_mm512_mask_mov_epi64(auto_cast(at0), mask, auto_cast(at1)));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) {
                return lo128(_mm512_mask_mov_ps(auto_cast(at0), mask, auto_cast(at1)));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 8) {
                return lo128(_mm512_mask_mov_pd(auto_cast(at0), mask, auto_cast(at1)));
            }
        } else if constexpr (sizeof(V) == 32 && have_avx512bw_vl && sizeof(T) <= 2) {
            if constexpr (sizeof(T) == 1) {
                return _mm256_mask_mov_epi8(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (sizeof(T) == 2) {
                return _mm256_mask_mov_epi16(to_intrin(at0), mask, to_intrin(at1));
            }
        } else if constexpr (sizeof(V) == 32 && have_avx512vl && sizeof(T) > 2) {
            if constexpr (std::is_integral_v<T> && sizeof(T) == 4) {
                return _mm256_mask_mov_epi32(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
                return _mm256_mask_mov_epi64(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) {
                return _mm256_mask_mov_ps(at0, mask, at1);
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 8) {
                return _mm256_mask_mov_pd(at0, mask, at1);
            }
        } else if constexpr (sizeof(V) == 32 && have_avx512f && sizeof(T) > 2) {
            if constexpr (std::is_integral_v<T> && sizeof(T) == 4) {
                return lo256(_mm512_mask_mov_epi32(auto_cast(at0), mask, auto_cast(at1)));
            } else if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
                return lo256(_mm512_mask_mov_epi64(auto_cast(at0), mask, auto_cast(at1)));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) {
                return lo256(_mm512_mask_mov_ps(auto_cast(at0), mask, auto_cast(at1)));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 8) {
                return lo256(_mm512_mask_mov_pd(auto_cast(at0), mask, auto_cast(at1)));
            }
        } else if constexpr (sizeof(V) == 64 && have_avx512bw && sizeof(T) <= 2) {
            if constexpr (sizeof(T) == 1) {
                return _mm512_mask_mov_epi8(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (sizeof(T) == 2) {
                return _mm512_mask_mov_epi16(to_intrin(at0), mask, to_intrin(at1));
            }
        } else if constexpr (sizeof(V) == 64 && have_avx512f && sizeof(T) > 2) {
            if constexpr (std::is_integral_v<T> && sizeof(T) == 4) {
                return _mm512_mask_mov_epi32(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
                return _mm512_mask_mov_epi64(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) {
                return _mm512_mask_mov_ps(at0, mask, at1);
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 8) {
                return _mm512_mask_mov_pd(at0, mask, at1);
            }
        } else {
            assert_unreachable<K>();
        }
    } else {
        const V k = auto_cast(mask);
        using T = typename builtin_traits<V>::value_type;
        if constexpr (sizeof(V) == 16 && have_sse4_1) {
            if constexpr (std::is_integral_v<T>) {
                return _mm_blendv_epi8(to_intrin(at0), to_intrin(at1), to_intrin(k));
            } else if constexpr (sizeof(T) == 4) {
                return _mm_blendv_ps(at0, at1, k);
            } else if constexpr (sizeof(T) == 8) {
                return _mm_blendv_pd(at0, at1, k);
            }
        } else if constexpr (sizeof(V) == 32) {
            if constexpr (std::is_integral_v<T>) {
                return _mm256_blendv_epi8(to_intrin(at0), to_intrin(at1), to_intrin(k));
            } else if constexpr (sizeof(T) == 4) {
                return _mm256_blendv_ps(at0, at1, k);
            } else if constexpr (sizeof(T) == 8) {
                return _mm256_blendv_pd(at0, at1, k);
            }
        } else {
            return or_(andnot_(k, at0), and_(k, at1));
        }
    }
}

// fixup_avx_xzyw{{{1
template <class T, class Traits = builtin_traits<T>> _GLIBCXX_SIMD_INTRINSIC T fixup_avx_xzyw(T a)
{
    static_assert(sizeof(T) == 32);
    using V = std::conditional_t<std::is_floating_point_v<typename Traits::value_type>,
                                 __m256d, __m256i>;
    const V x = reinterpret_cast<V>(a);
    return reinterpret_cast<T>(V{x[0], x[2], x[1], x[3]});
}

// shift_right{{{1
template <int n> _GLIBCXX_SIMD_INTRINSIC __m128  shift_right(__m128  v);
template <> _GLIBCXX_SIMD_INTRINSIC __m128  shift_right< 0>(__m128  v) { return v; }
template <> _GLIBCXX_SIMD_INTRINSIC __m128  shift_right<16>(__m128   ) { return _mm_setzero_ps(); }

#ifdef _GLIBCXX_SIMD_HAVE_SSE2
template <int n> _GLIBCXX_SIMD_INTRINSIC __m128  shift_right(__m128  v) { return _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), n)); }
template <int n> _GLIBCXX_SIMD_INTRINSIC __m128d shift_right(__m128d v) { return _mm_castsi128_pd(_mm_srli_si128(_mm_castpd_si128(v), n)); }
template <int n> _GLIBCXX_SIMD_INTRINSIC __m128i shift_right(__m128i v) { return _mm_srli_si128(v, n); }

template <> _GLIBCXX_SIMD_INTRINSIC __m128  shift_right< 8>(__m128  v) { return _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(v), _mm_setzero_pd())); }
template <> _GLIBCXX_SIMD_INTRINSIC __m128d shift_right< 0>(__m128d v) { return v; }
template <> _GLIBCXX_SIMD_INTRINSIC __m128d shift_right< 8>(__m128d v) { return _mm_unpackhi_pd(v, _mm_setzero_pd()); }
template <> _GLIBCXX_SIMD_INTRINSIC __m128d shift_right<16>(__m128d  ) { return _mm_setzero_pd(); }
template <> _GLIBCXX_SIMD_INTRINSIC __m128i shift_right< 0>(__m128i v) { return v; }
template <> _GLIBCXX_SIMD_INTRINSIC __m128i shift_right<16>(__m128i  ) { return _mm_setzero_si128(); }
#endif  // _GLIBCXX_SIMD_HAVE_SSE2

#ifdef _GLIBCXX_SIMD_HAVE_AVX2
template <int n> _GLIBCXX_SIMD_INTRINSIC __m256 shift_right(__m256 v)
{
    __m256i vi = _mm256_castps_si256(v);
    return _mm256_castsi256_ps(
        n < 16 ? _mm256_srli_si256(vi, n)
               : _mm256_srli_si256(_mm256_permute2x128_si256(vi, vi, 0x81), n));
}
template <> _GLIBCXX_SIMD_INTRINSIC __m256 shift_right<0>(__m256 v) { return v; }
template <> _GLIBCXX_SIMD_INTRINSIC __m256 shift_right<16>(__m256 v) { return intrin_cast<__m256>(lo128(v)); }
template <int n> _GLIBCXX_SIMD_INTRINSIC __m256d shift_right(__m256d v)
{
    __m256i vi = _mm256_castpd_si256(v);
    return _mm256_castsi256_pd(
        n < 16 ? _mm256_srli_si256(vi, n)
               : _mm256_srli_si256(_mm256_permute2x128_si256(vi, vi, 0x81), n));
}
template <> _GLIBCXX_SIMD_INTRINSIC __m256d shift_right<0>(__m256d v) { return v; }
template <> _GLIBCXX_SIMD_INTRINSIC __m256d shift_right<16>(__m256d v) { return intrin_cast<__m256d>(lo128(v)); }
template <int n> _GLIBCXX_SIMD_INTRINSIC __m256i shift_right(__m256i v)
{
    return n < 16 ? _mm256_srli_si256(v, n)
                  : _mm256_srli_si256(_mm256_permute2x128_si256(v, v, 0x81), n);
}
template <> _GLIBCXX_SIMD_INTRINSIC __m256i shift_right<0>(__m256i v) { return v; }
template <> _GLIBCXX_SIMD_INTRINSIC __m256i shift_right<16>(__m256i v) { return _mm256_permute2x128_si256(v, v, 0x81); }
#endif

// cmpord{{{1
_GLIBCXX_SIMD_INTRINSIC builtin_type16_t<float> cmpord(builtin_type16_t<float> x,
                                            builtin_type16_t<float> y)
{
    return _mm_cmpord_ps(x, y);
}
_GLIBCXX_SIMD_INTRINSIC builtin_type16_t<double> cmpord(builtin_type16_t<double> x,
                                             builtin_type16_t<double> y)
{
    return _mm_cmpord_pd(x, y);
}

#ifdef _GLIBCXX_SIMD_HAVE_AVX
_GLIBCXX_SIMD_INTRINSIC builtin_type32_t<float> cmpord(builtin_type32_t<float> x,
                                            builtin_type32_t<float> y)
{
    return _mm256_cmp_ps(x, y, _CMP_ORD_Q);
}
_GLIBCXX_SIMD_INTRINSIC builtin_type32_t<double> cmpord(builtin_type32_t<double> x,
                                             builtin_type32_t<double> y)
{
    return _mm256_cmp_pd(x, y, _CMP_ORD_Q);
}
#endif  // _GLIBCXX_SIMD_HAVE_AVX

#ifdef _GLIBCXX_SIMD_HAVE_AVX512F
_GLIBCXX_SIMD_INTRINSIC __mmask16 cmpord(builtin_type64_t<float> x, builtin_type64_t<float> y)
{
    return _mm512_cmp_ps_mask(x, y, _CMP_ORD_Q);
}
_GLIBCXX_SIMD_INTRINSIC __mmask8 cmpord(builtin_type64_t<double> x, builtin_type64_t<double> y)
{
    return _mm512_cmp_pd_mask(x, y, _CMP_ORD_Q);
}
#endif  // _GLIBCXX_SIMD_HAVE_AVX512F

// cmpunord{{{1
_GLIBCXX_SIMD_INTRINSIC builtin_type16_t<float> cmpunord(builtin_type16_t<float> x,
                                              builtin_type16_t<float> y)
{
    return _mm_cmpunord_ps(x, y);
}
_GLIBCXX_SIMD_INTRINSIC builtin_type16_t<double> cmpunord(builtin_type16_t<double> x,
                                               builtin_type16_t<double> y)
{
    return _mm_cmpunord_pd(x, y);
}

#ifdef _GLIBCXX_SIMD_HAVE_AVX
_GLIBCXX_SIMD_INTRINSIC builtin_type32_t<float> cmpunord(builtin_type32_t<float> x,
                                              builtin_type32_t<float> y)
{
    return _mm256_cmp_ps(x, y, _CMP_UNORD_Q);
}
_GLIBCXX_SIMD_INTRINSIC builtin_type32_t<double> cmpunord(builtin_type32_t<double> x,
                                               builtin_type32_t<double> y)
{
    return _mm256_cmp_pd(x, y, _CMP_UNORD_Q);
}
#endif  // _GLIBCXX_SIMD_HAVE_AVX

#ifdef _GLIBCXX_SIMD_HAVE_AVX512F
_GLIBCXX_SIMD_INTRINSIC __mmask16 cmpunord(builtin_type64_t<float> x, builtin_type64_t<float> y)
{
    return _mm512_cmp_ps_mask(x, y, _CMP_UNORD_Q);
}
_GLIBCXX_SIMD_INTRINSIC __mmask8 cmpunord(builtin_type64_t<double> x, builtin_type64_t<double> y)
{
    return _mm512_cmp_pd_mask(x, y, _CMP_UNORD_Q);
}
#endif  // _GLIBCXX_SIMD_HAVE_AVX512F

// }}}
// integer sign-extension {{{
_GLIBCXX_SIMD_INTRINSIC constexpr builtin_type_t<short,  8> sign_extend16(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovsxbw128(x); }
_GLIBCXX_SIMD_INTRINSIC constexpr builtin_type_t<  int,  4> sign_extend32(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovsxbd128(x); }
_GLIBCXX_SIMD_INTRINSIC constexpr builtin_type_t<llong,  2> sign_extend64(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovsxbq128(x); }
_GLIBCXX_SIMD_INTRINSIC constexpr builtin_type_t<  int,  4> sign_extend32(builtin_type_t<short, 8> x) { return __builtin_ia32_pmovsxwd128(x); }
_GLIBCXX_SIMD_INTRINSIC constexpr builtin_type_t<llong,  2> sign_extend64(builtin_type_t<short, 8> x) { return __builtin_ia32_pmovsxwq128(x); }
_GLIBCXX_SIMD_INTRINSIC constexpr builtin_type_t<llong,  2> sign_extend64(builtin_type_t<  int, 4> x) { return __builtin_ia32_pmovsxdq128(x); }

// }}}
// integer zero-extension {{{
_GLIBCXX_SIMD_INTRINSIC constexpr builtin_type_t<short,  8> zero_extend16(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovzxbw128(x); }
_GLIBCXX_SIMD_INTRINSIC constexpr builtin_type_t<  int,  4> zero_extend32(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovzxbd128(x); }
_GLIBCXX_SIMD_INTRINSIC constexpr builtin_type_t<llong,  2> zero_extend64(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovzxbq128(x); }
_GLIBCXX_SIMD_INTRINSIC constexpr builtin_type_t<  int,  4> zero_extend32(builtin_type_t<short, 8> x) { return __builtin_ia32_pmovzxwd128(x); }
_GLIBCXX_SIMD_INTRINSIC constexpr builtin_type_t<llong,  2> zero_extend64(builtin_type_t<short, 8> x) { return __builtin_ia32_pmovzxwq128(x); }
_GLIBCXX_SIMD_INTRINSIC constexpr builtin_type_t<llong,  2> zero_extend64(builtin_type_t<  int, 4> x) { return __builtin_ia32_pmovzxdq128(x); }

// }}}
// non-converting maskstore (SSE-AVX512BWVL) {{{
template <class T, class F>
_GLIBCXX_SIMD_INTRINSIC void maskstore(storage64_t<T> v, T *mem, F,
                            Storage<bool, storage64_t<T>::width> k)
{
    static_assert(sizeof(v) == 64 && have_avx512f);
    if constexpr (have_avx512bw && sizeof(T) == 1) {
        _mm512_mask_storeu_epi8(mem, k, v);
    } else if constexpr (have_avx512bw && sizeof(T) == 2) {
        _mm512_mask_storeu_epi16(mem, k, v);
    } else if constexpr (sizeof(T) == 4) {
        if constexpr (is_aligned_v<F, 64> && std::is_integral_v<T>) {
            _mm512_mask_store_epi32(mem, k, v);
        } else if constexpr (is_aligned_v<F, 64> && std::is_floating_point_v<T>) {
            _mm512_mask_store_ps(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm512_mask_storeu_epi32(mem, k, v);
        } else {
            _mm512_mask_storeu_ps(mem, k, v);
        }
    } else if constexpr (sizeof(T) == 8) {
        if constexpr (is_aligned_v<F, 64> && std::is_integral_v<T>) {
            _mm512_mask_store_epi64(mem, k, v);
        } else if constexpr (is_aligned_v<F, 64> && std::is_floating_point_v<T>) {
            _mm512_mask_store_pd(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm512_mask_storeu_epi64(mem, k, v);
        } else {
            _mm512_mask_storeu_pd(mem, k, v);
        }
    } else {
        constexpr int N = 16 / sizeof(T);
        using M = builtin_type_t<T, N>;
        _mm_maskmoveu_si128(auto_cast(extract<0, 4>(v.d)),
                            auto_cast(convert_mask<M>(k.d)),
                            reinterpret_cast<char *>(mem));
        _mm_maskmoveu_si128(auto_cast(extract<1, 4>(v.d)),
                            auto_cast(convert_mask<M>(k.d >> 1 * N)),
                            reinterpret_cast<char *>(mem) + 1 * 16);
        _mm_maskmoveu_si128(auto_cast(extract<2, 4>(v.d)),
                            auto_cast(convert_mask<M>(k.d >> 2 * N)),
                            reinterpret_cast<char *>(mem) + 2 * 16);
        _mm_maskmoveu_si128(auto_cast(extract<3, 4>(v.d)),
                            auto_cast(convert_mask<M>(k.d >> 3 * N)),
                            reinterpret_cast<char *>(mem) + 3 * 16);
    }
}

template <class T, class F>
_GLIBCXX_SIMD_INTRINSIC void maskstore(storage32_t<T> v, T *mem, F, storage32_t<T> k)
{
    static_assert(sizeof(v) == 32 && have_avx);
    if constexpr (have_avx512bw_vl && sizeof(T) == 1) {
        _mm256_mask_storeu_epi8(mem, _mm256_movepi8_mask(k), v);
    } else if constexpr (have_avx512bw_vl && sizeof(T) == 2) {
        _mm256_mask_storeu_epi16(mem, _mm256_movepi16_mask(k), v);
    } else if constexpr (have_avx2 && sizeof(T) == 4 && std::is_integral_v<T>) {
        _mm256_maskstore_epi32(reinterpret_cast<int *>(mem), k, v);
    } else if constexpr (sizeof(T) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float *>(mem), builtin_cast<llong>(k),
                            builtin_cast<float>(v));
    } else if constexpr (have_avx2 && sizeof(T) == 8 && std::is_integral_v<T>) {
        _mm256_maskstore_epi64(reinterpret_cast<llong *>(mem), k, v);
    } else if constexpr (sizeof(T) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double *>(mem), builtin_cast<llong>(k),
                            builtin_cast<double>(v));
    } else {
        _mm_maskmoveu_si128(builtin_cast<llong>(lo128(v)), builtin_cast<llong>(lo128(k)),
                            reinterpret_cast<char *>(mem));
        _mm_maskmoveu_si128(builtin_cast<llong>(hi128(v)), builtin_cast<llong>(hi128(k)),
                            reinterpret_cast<char *>(mem) + 16);
    }
}

template <class T, class F>
_GLIBCXX_SIMD_INTRINSIC void maskstore(storage32_t<T> v, T *mem, F,
                            Storage<bool, storage32_t<T>::width> k)
{
    static_assert(sizeof(v) == 32 && have_avx512f);
    if constexpr (have_avx512bw_vl && sizeof(T) == 1) {
        _mm256_mask_storeu_epi8(mem, k, v);
    } else if constexpr (have_avx512bw_vl && sizeof(T) == 2) {
        _mm256_mask_storeu_epi16(mem, k, v);
    } else if constexpr (have_avx512vl && sizeof(T) == 4) {
        if constexpr (is_aligned_v<F, 32> && std::is_integral_v<T>) {
            _mm256_mask_store_epi32(mem, k, v);
        } else if constexpr (is_aligned_v<F, 32> && std::is_floating_point_v<T>) {
            _mm256_mask_store_ps(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm256_mask_storeu_epi32(mem, k, v);
        } else {
            _mm256_mask_storeu_ps(mem, k, v);
        }
    } else if constexpr (have_avx512vl && sizeof(T) == 8) {
        if constexpr (is_aligned_v<F, 32> && std::is_integral_v<T>) {
            _mm256_mask_store_epi64(mem, k, v);
        } else if constexpr (is_aligned_v<F, 32> && std::is_floating_point_v<T>) {
            _mm256_mask_store_pd(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm256_mask_storeu_epi64(mem, k, v);
        } else {
            _mm256_mask_storeu_pd(mem, k, v);
        }
    } else if constexpr (have_avx512f && (sizeof(T) >= 4 || have_avx512bw)) {
        // use a 512-bit maskstore, using zero-extension of the bitmask
        maskstore(
            detail::storage64_t<T>(
                detail::intrin_cast<detail::intrinsic_type64_t<T>>(v.d)),
            mem,
            // careful, vector_aligned has a stricter meaning in the 512-bit maskstore:
            std::conditional_t<std::is_same_v<F, vector_aligned_tag>, overaligned_tag<32>,
                               F>(),
            detail::Storage<bool, 64 / sizeof(T)>(k.d));
    } else {
        maskstore(
            v, mem, F(),
            detail::storage32_t<T>(convert_mask<builtin_type_t<T, 32 / sizeof(T)>>(k)));
    }
}

template <class T, class F>
_GLIBCXX_SIMD_INTRINSIC void maskstore(storage16_t<T> v, T *mem, F, storage16_t<T> k)
{
    if constexpr (have_avx512bw_vl && sizeof(T) == 1) {
        _mm_mask_storeu_epi8(mem, _mm_movepi8_mask(k), v);
    } else if constexpr (have_avx512bw_vl && sizeof(T) == 2) {
        _mm_mask_storeu_epi16(mem, _mm_movepi16_mask(k), v);
    } else if constexpr (have_avx2 && sizeof(T) == 4 && std::is_integral_v<T>) {
        _mm_maskstore_epi32(reinterpret_cast<int *>(mem), k, v);
    } else if constexpr (have_avx && sizeof(T) == 4) {
        _mm_maskstore_ps(reinterpret_cast<float *>(mem), builtin_cast<llong>(k),
                         builtin_cast<float>(v));
    } else if constexpr (have_avx2 && sizeof(T) == 8 && std::is_integral_v<T>) {
        _mm_maskstore_epi64(reinterpret_cast<llong *>(mem), k, v);
    } else if constexpr (have_avx && sizeof(T) == 8) {
        _mm_maskstore_pd(reinterpret_cast<double *>(mem), builtin_cast<llong>(k),
                         builtin_cast<double>(v));
    } else if constexpr (have_sse2) {
        _mm_maskmoveu_si128(builtin_cast<llong>(v), builtin_cast<llong>(k),
                            reinterpret_cast<char *>(mem));
    } else {
        execute_n_times<storage16_t<T>::width>([&](auto i) {
            if (k[i]) {
                mem[i] = v[i];
            }
        });
    }
}

template <class T, class F>
_GLIBCXX_SIMD_INTRINSIC void maskstore(storage16_t<T> v, T *mem, F,
                            Storage<bool, storage16_t<T>::width> k)
{
    static_assert(sizeof(v) == 16 && have_avx512f);
    if constexpr (have_avx512bw_vl && sizeof(T) == 1) {
        _mm_mask_storeu_epi8(mem, k, v);
    } else if constexpr (have_avx512bw_vl && sizeof(T) == 2) {
        _mm_mask_storeu_epi16(mem, k, v);
    } else if constexpr (have_avx512vl && sizeof(T) == 4) {
        if constexpr (is_aligned_v<F, 16> && std::is_integral_v<T>) {
            _mm_mask_store_epi32(mem, k, v);
        } else if constexpr (is_aligned_v<F, 16> && std::is_floating_point_v<T>) {
            _mm_mask_store_ps(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm_mask_storeu_epi32(mem, k, v);
        } else {
            _mm_mask_storeu_ps(mem, k, v);
        }
    } else if constexpr (have_avx512vl && sizeof(T) == 8) {
        if constexpr (is_aligned_v<F, 16> && std::is_integral_v<T>) {
            _mm_mask_store_epi64(mem, k, v);
        } else if constexpr (is_aligned_v<F, 16> && std::is_floating_point_v<T>) {
            _mm_mask_store_pd(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm_mask_storeu_epi64(mem, k, v);
        } else {
            _mm_mask_storeu_pd(mem, k, v);
        }
    } else if constexpr (have_avx512f && (sizeof(T) >= 4 || have_avx512bw)) {
        // use a 512-bit maskstore, using zero-extension of the bitmask
        maskstore(
            detail::storage64_t<T>(
                detail::intrin_cast<detail::intrinsic_type64_t<T>>(v.d)),
            mem,
            // careful, vector_aligned has a stricter meaning in the 512-bit maskstore:
            std::conditional_t<std::is_same_v<F, vector_aligned_tag>, overaligned_tag<16>,
                               F>(),
            detail::Storage<bool, 64 / sizeof(T)>(k.d));
    } else {
        maskstore(
            v, mem, F(),
            detail::storage16_t<T>(convert_mask<builtin_type_t<T, 16 / sizeof(T)>>(k)));
    }
}

// }}}
// extract_part {{{1
// identity {{{2
template <class T>
_GLIBCXX_SIMD_INTRINSIC constexpr const storage16_t<T>& extract_part_impl(std::true_type,
                                                               size_constant<0>,
                                                               size_constant<1>,
                                                               const storage16_t<T>& x)
{
    return x;
}

// by 2 and by 4 splits {{{2
template <class T, size_t N, size_t Index, size_t Total>
_GLIBCXX_SIMD_INTRINSIC constexpr Storage<T, N / Total> extract_part_impl(std::true_type,
                                                               size_constant<Index>,
                                                               size_constant<Total>,
                                                               Storage<T, N> x)
{
    return detail::extract<Index, Total>(x.d);
}

// partial SSE (shifts) {{{2
template <class T, size_t Index, size_t Total, size_t N>
_GLIBCXX_SIMD_INTRINSIC storage16_t<T> extract_part_impl(std::false_type, size_constant<Index>,
                                              size_constant<Total>, Storage<T, N> x)
{
    constexpr int split = sizeof(x) / 16;
    constexpr int shift = (sizeof(x) / Total * Index) % 16;
    return x86::shift_right<shift>(
        extract_part_impl<T>(std::true_type(), size_constant<Index * split / Total>(),
                             size_constant<split>(), x));
}

// public interface {{{2
template <size_t Index, size_t Total, class T, size_t N>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST Storage<T, std::max(16 / sizeof(T), N / Total)> extract_part(
    Storage<T, N> x)
{
    constexpr size_t NewN = N / Total;
    static_assert(Total > 1, "Total must be greater than 1");
    static_assert(NewN * Total == N, "N must be divisible by Total");
    return extract_part_impl<T>(
        bool_constant<(sizeof(T) * NewN >= 16)>(),  // dispatch on whether the result is a
                                                    // partial SSE register or larger
        size_constant<Index>(), size_constant<Total>(), x);
}

// }}}1
// extract_part(Storage<bool, N>) {{{
template <size_t Offset, size_t SplitBy, size_t N>
_GLIBCXX_SIMD_INTRINSIC constexpr Storage<bool, N / SplitBy> extract_part(Storage<bool, N> x)
{
    static_assert(SplitBy >= 2 && Offset < SplitBy && Offset >= 0);
    return x.d >> (Offset * N / SplitBy);
}

// }}}
}  // namespace x86
using namespace x86;
#endif  // SSE || MMX
// extract_part(Storage) {{{
template <int Index, int Parts, class T, size_t N>
_GLIBCXX_SIMD_INTRINSIC auto extract_part(Storage<T, N> x)
{
    if constexpr (have_sse) {
        return detail::x86::extract_part<Index, Parts>(x);
    } else {
        assert_unreachable<T>();
    }
}
// }}}
// extract_part(simd_tuple) {{{
template <int Index, int Parts, class T, class A0, class... As>
_GLIBCXX_SIMD_INTRINSIC auto extract_part(const simd_tuple<T, A0, As...> &x)
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
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        const detail::may_alias<T> *const element_ptr =
            reinterpret_cast<const detail::may_alias<T> *>(&x) + Index * values_per_part;
        return detail::data(simd<T, simd_abi::deduce_t<T, values_per_part>>(
            [&](auto i) { return element_ptr[i]; }));
#else
        constexpr size_t offset = Index * values_per_part;
        detail::unused(offset);  // not really
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
// }}}
// to_storage specializations for bitset and __mmask<N> {{{
#ifdef _GLIBCXX_SIMD_HAVE_AVX512_ABI
template <size_t N> class to_storage<std::bitset<N>>
{
    std::bitset<N> d;

public:
    [[deprecated("use convert_mask<To>(bitset)")]]
    constexpr to_storage(std::bitset<N> x) : d(x) {}

    // can convert to larger storage for Abi::is_partial == true
    template <class U, size_t M> constexpr operator Storage<U, M>() const
    {
        static_assert(M >= N);
        return convert_mask<Storage<U, M>>(d);
    }
};

#define _GLIBCXX_SIMD_TO_STORAGE(type_)                                                             \
    template <> class to_storage<type_>                                                  \
    {                                                                                    \
        type_ d;                                                                         \
                                                                                         \
    public:                                                                              \
        [[deprecated("use convert_mask<To>(bitset)")]] constexpr to_storage(type_ x)     \
            : d(x)                                                                       \
        {                                                                                \
        }                                                                                \
                                                                                         \
        template <class U, size_t N> constexpr operator Storage<U, N>() const            \
        {                                                                                \
            static_assert(N >= sizeof(type_) * CHAR_BIT);                                \
            return reinterpret_cast<builtin_type_t<U, N>>(                               \
                convert_mask<Storage<U, N>>(d));                                         \
        }                                                                                \
                                                                                         \
        template <size_t N> constexpr operator Storage<bool, N>() const                  \
        {                                                                                \
            static_assert(                                                               \
                std::is_same_v<type_, typename bool_storage_member_type<N>::type>);      \
            return d;                                                                    \
        }                                                                                \
    }
_GLIBCXX_SIMD_TO_STORAGE(__mmask8);
_GLIBCXX_SIMD_TO_STORAGE(__mmask16);
_GLIBCXX_SIMD_TO_STORAGE(__mmask32);
_GLIBCXX_SIMD_TO_STORAGE(__mmask64);
#undef _GLIBCXX_SIMD_TO_STORAGE
#endif  // _GLIBCXX_SIMD_HAVE_AVX512_ABI

// }}}
#ifdef _GLIBCXX_SIMD_HAVE_SSE
namespace x86
{
// converts_via_decomposition{{{
template <class From, class To, size_t ToSize> struct converts_via_decomposition {
private:
    static constexpr bool i_to_i = std::is_integral_v<From> && std::is_integral_v<To>;
    static constexpr bool f_to_i =
        std::is_floating_point_v<From> && std::is_integral_v<To>;
    static constexpr bool f_to_f =
        std::is_floating_point_v<From> && std::is_floating_point_v<To>;
    static constexpr bool i_to_f =
        std::is_integral_v<From> && std::is_floating_point_v<To>;

    template <size_t A, size_t B>
    static constexpr bool sizes = sizeof(From) == A && sizeof(To) == B;

public:
    static constexpr bool value =
        (i_to_i && sizes<8, 2> && !have_ssse3 && ToSize == 16) ||
        (i_to_i && sizes<8, 1> && !have_avx512f && ToSize == 16) ||
        (f_to_i && sizes<4, 8> && !have_avx512dq) ||
        (f_to_i && sizes<8, 8> && !have_avx512dq) ||
        (f_to_i && sizes<8, 4> && !have_sse4_1 && ToSize == 16) ||
        (i_to_f && sizes<8, 4> && !have_avx512dq && ToSize == 16) ||
        (i_to_f && sizes<8, 8> && !have_avx512dq && ToSize < 64);
};

template <class From, class To, size_t ToSize>
inline constexpr bool converts_via_decomposition_v =
    converts_via_decomposition<From, To, ToSize>::value;

// }}}
// convert_builtin{{{
#ifdef _GLIBCXX_SIMD_USE_BUILTIN_VECTOR_TYPES
template <typename To, typename From, size_t... I>
_GLIBCXX_SIMD_INTRINSIC constexpr To convert_builtin(From v0, std::index_sequence<I...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(v0[I]...);
}

template <typename To, typename From, size_t... I, size_t... Z>
_GLIBCXX_SIMD_INTRINSIC constexpr To convert_builtin_z(From v0, std::index_sequence<I...>,
                                            std::index_sequence<Z...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(v0[I]..., ((void)Z, 0)...);
}

template <typename To, typename From, size_t... I>
_GLIBCXX_SIMD_INTRINSIC constexpr To convert_builtin(From v0, From v1, std::index_sequence<I...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(v0[I]..., v1[I]...);
}

template <typename To, typename From, size_t... I>
_GLIBCXX_SIMD_INTRINSIC constexpr To convert_builtin(From v0, From v1, From v2, From v3,
                                          std::index_sequence<I...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(static_cast<T>(v0[I])..., static_cast<T>(v1[I])...,
                                   static_cast<T>(v2[I])..., static_cast<T>(v3[I])...);
}

template <typename To, typename From, size_t... I>
_GLIBCXX_SIMD_INTRINSIC constexpr To convert_builtin(From v0, From v1, From v2, From v3, From v4,
                                          From v5, From v6, From v7,
                                          std::index_sequence<I...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(static_cast<T>(v0[I])..., static_cast<T>(v1[I])...,
                                   static_cast<T>(v2[I])..., static_cast<T>(v3[I])...,
                                   static_cast<T>(v4[I])..., static_cast<T>(v5[I])...,
                                   static_cast<T>(v6[I])..., static_cast<T>(v7[I])...);
}

template <typename To, typename From, size_t... I0, size_t... I1>
_GLIBCXX_SIMD_INTRINSIC constexpr To convert_builtin(From v0, From v1, std::index_sequence<I0...>,
                                          std::index_sequence<I1...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(static_cast<T>(v0[I0])..., static_cast<T>(v1[I0])...,
                                   (I1, T{})...);
}

template <typename To, typename From, size_t... I0, size_t... I1>
_GLIBCXX_SIMD_INTRINSIC constexpr To convert_builtin(From v0, From v1, From v2, From v3,
                                          std::index_sequence<I0...>,
                                          std::index_sequence<I1...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(static_cast<T>(v0[I0])..., static_cast<T>(v1[I0])...,
                                   static_cast<T>(v2[I0])..., static_cast<T>(v3[I0])...,
                                   (I1, T{})...);
}

template <typename To, typename From, size_t... I0, size_t... I1>
_GLIBCXX_SIMD_INTRINSIC constexpr To convert_builtin(From v0, From v1, From v2, From v3, From v4,
                                          From v5, From v6, From v7,
                                          std::index_sequence<I0...>,
                                          std::index_sequence<I1...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(
        static_cast<T>(v0[I0])..., static_cast<T>(v1[I0])..., static_cast<T>(v2[I0])...,
        static_cast<T>(v3[I0])..., static_cast<T>(v4[I0])..., static_cast<T>(v5[I0])...,
        static_cast<T>(v6[I0])..., static_cast<T>(v7[I0])..., (I1, T{})...);
}
#endif  // _GLIBCXX_SIMD_USE_BUILTIN_VECTOR_TYPES
//}}}

#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85048
#include "simd_x86_conversions.h"
#endif  // _GLIBCXX_SIMD_WORKAROUND_PR85048

// convert from scalars{{{
template <typename To, typename... From>
[[deprecated("use make_storage instead")]]
_GLIBCXX_SIMD_INTRINSIC To convert_to(Vectorizable<From>... scalars)
{
    return x86::set(static_cast<typename To::value_type>(scalars)...);
}
// }}}
// convert function{{{
template <class To, class From> _GLIBCXX_SIMD_INTRINSIC auto convert(From v)
{
    if constexpr (detail::is_builtin_vector_v<From>) {
        using Trait = detail::builtin_traits<From>;
        return convert<To>(Storage<typename Trait::value_type, Trait::width>(v));
    } else if constexpr (detail::is_builtin_vector_v<To>) {
        using Trait = detail::builtin_traits<To>;
        return convert<Storage<typename Trait::value_type, Trait::width>>(v).d;
    } else if constexpr (detail::is_vectorizable_v<To>) {
        return convert<Storage<To, From::width>>(v).d;
    } else {
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85048
        return convert_to<To>(v);
#else
        if constexpr (From::width >= To::width) {
            return convert_builtin<To>(v.d, std::make_index_sequence<To::width>());
        } else {
            return convert_builtin_z<To>(
                v.d, std::make_index_sequence<From::width>(),
                std::make_index_sequence<To::width - From::width>());
        }
#endif
    }
}

template <class To, class From> _GLIBCXX_SIMD_INTRINSIC auto convert(From v0, From v1)
{
    if constexpr (detail::is_vectorizable_v<To>) {
        return convert<Storage<To, From::width * 2>>(v0, v1).d;
    } else if constexpr (detail::is_builtin_vector_v<From>) {
        using Trait = detail::builtin_traits<From>;
        using S = Storage<typename Trait::value_type, Trait::width>;
        return convert<To>(S(v0), S(v1));
    } else if constexpr (std::is_arithmetic_v<From>) {
        using T = typename To::value_type;
        return make_storage<T>(v0, v1);
    } else {
        static_assert(To::width >= 2 * From::width,
                      "convert(v0, v1) requires the input to fit into the output");
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85048
        return convert_to<To>(v0, v1);
#else
        return convert_builtin<To>(
            v0.d, v1.d, std::make_index_sequence<From::width>(),
            std::make_index_sequence<To::width - 2 * From::width>());
#endif
    }
}

template <class To, class From>
_GLIBCXX_SIMD_INTRINSIC auto convert(From v0, From v1, From v2, From v3)
{
    if constexpr (detail::is_vectorizable_v<To>) {
        return convert<Storage<To, From::width * 4>>(v0, v1, v2, v3).d;
    } else if constexpr (detail::is_builtin_vector_v<From>) {
        using Trait = detail::builtin_traits<From>;
        using S = Storage<typename Trait::value_type, Trait::width>;
        return convert<To>(S(v0), S(v1), S(v2), S(v3));
    } else if constexpr (std::is_arithmetic_v<From>) {
        using T = typename To::value_type;
        return make_storage<T>(v0, v1, v2, v3);
    } else {
        static_assert(
            To::width >= 4 * From::width,
            "convert(v0, v1, v2, v3) requires the input to fit into the output");
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85048
        return convert_to<To>(v0, v1, v2, v3);
#else
        return convert_builtin<To>(
            v0.d, v1.d, v2.d, v3.d, std::make_index_sequence<From::width>(),
            std::make_index_sequence<To::width - 4 * From::width>());
#endif
    }
}

template <class To, class From>
_GLIBCXX_SIMD_INTRINSIC To convert(From v0, From v1, From v2, From v3,
                                 From v4, From v5, From v6, From v7)
{
    if constexpr (detail::is_builtin_vector_v<From>) {
        using Trait = detail::builtin_traits<From>;
        using S = Storage<typename Trait::value_type, Trait::width>;
        return convert<To>(S(v0), S(v1), S(v2), S(v3), S(v4), S(v5), S(v6), S(v7));
    } else if constexpr (std::is_arithmetic_v<From>) {
        using T = typename To::value_type;
        return make_storage<T>(v0, v1, v2, v3, v4, v5, v6, v7);
    } else {
        static_assert(To::width >= 8 * From::width,
                      "convert(v0, v1, v2, v3, v4, v5, v6, v7) "
                      "requires the input to fit into the output");
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85048
        return convert_to<To>(v0, v1, v2, v3, v4, v5, v6, v7);
#else
        return convert_builtin<To>(
            v0.d, v1.d, v2.d, v3.d, v4.d, v5.d, v6.d, v7.d,
            std::make_index_sequence<From::width>(),
            std::make_index_sequence<To::width - 8 * From::width>());
#endif
    }
}
// }}}
// convert_all function{{{
template <typename To, typename From> _GLIBCXX_SIMD_INTRINSIC auto convert_all(From v)
{
    static_assert(detail::is_builtin_vector_v<To>);
    if constexpr (detail::is_builtin_vector_v<From>) {
        using Trait = detail::builtin_traits<From>;
        using S = Storage<typename Trait::value_type, Trait::width>;
        return convert_all<To>(S(v));
    } else if constexpr (From::width > builtin_traits<To>::width) {
        constexpr size_t N = From::width / builtin_traits<To>::width;
        return generate_from_n_evaluations<N, std::array<To, N>>([&](auto i) {
            auto part = x86::extract_part<decltype(i)::value, N>(v);
            return convert<To>(part);
        });
    } else {
        return convert<To>(v);
    }
}

// }}}
// plus{{{1
template <class T, size_t N>
_GLIBCXX_SIMD_INTRINSIC constexpr Storage<T, N> plus(Storage<T, N> a, Storage<T, N> b)
{
    return a.d + b.d;
}

// minus{{{1
template <class T, size_t N>
_GLIBCXX_SIMD_INTRINSIC constexpr Storage<T, N> minus(Storage<T, N> a, Storage<T, N> b)
{
    return a.d - b.d;
}

// multiplies{{{1
template <class T, size_t N>
_GLIBCXX_SIMD_INTRINSIC constexpr Storage<T, N> multiplies(Storage<T, N> a, Storage<T, N> b)
{
    if constexpr (sizeof(T) == 1) {
        return builtin_cast<T>(
            ((builtin_cast<short>(a) * builtin_cast<short>(b)) &
             builtin_cast<short>(~builtin_type_t<ushort, N / 2>() >> 8)) |
            (((builtin_cast<short>(a) >> 8) * (builtin_cast<short>(b) >> 8)) << 8));
    }
    return a.d * b.d;
}

// complement{{{1
template <class T, size_t N>
_GLIBCXX_SIMD_INTRINSIC constexpr Storage<T, N> complement(Storage<T, N> v)
{
    return ~v.d;
}

//}}}1
// unary_minus{{{1
// GCC doesn't use the psign instructions, but pxor & psub seem to be just as good a
// choice as pcmpeqd & psign. So meh.
template <class T, size_t N>
_GLIBCXX_SIMD_INTRINSIC constexpr Storage<T, N> unary_minus(Storage<T, N> v)
{
    return -v.d;
}

// abs{{{1
template <class T, size_t N>
_GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC constexpr Storage<T, N> abs(Storage<T, N> v)
{
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85572
    if constexpr (!have_avx512vl && std::is_integral_v<T> && sizeof(T) == 8 && N <= 4) {
        // positive value:
        //   negative == 0
        //   a unchanged after xor
        //   a - 0 -> a
        // negative value:
        //   negative == ~0 == -1
        //   a xor ~0    -> -a - 1
        //   -a - 1 - -1 -> -a
        if constexpr(have_sse4_2) {
            const auto negative = reinterpret_cast<builtin_type_t<T, N>>(v.d < 0);
            return (v.d ^ negative) - negative;
        } else {
            // arithmetic right shift doesn't exist for 64-bit integers, use the following
            // instead:
            // >>63: negative ->  1, positive ->  0
            //  -  : negative -> -1, positive ->  0
            const auto negative = -reinterpret_cast<builtin_type_t<T, N>>(
                reinterpret_cast<builtin_type_t<ullong, N>>(v.d) >> 63);
            return (v.d ^ negative) - negative;
        }
    } else
#endif
        if constexpr (std::is_floating_point_v<T>) {
        // this workaround is only required because __builtin_abs is not a constant
        // expression
        using I = std::make_unsigned_t<int_for_sizeof_t<T>>;
        return builtin_cast<T>(builtin_cast<I>(v.d) & builtin_broadcast<N, I>(~I() >> 1));
    } else {
        return v.d < 0 ? -v.d : v.d;
    }
}

//}}}1
}  // namespace x86
#endif  // _GLIBCXX_SIMD_HAVE_SSE

// interleave (lo/hi/128) {{{
template <class A, class B, class T = std::common_type_t<A, B>,
          class Trait = builtin_traits<T>>
_GLIBCXX_SIMD_INTRINSIC constexpr T interleave_lo(A _a, B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const T a(_a);
    const T b(_b);
    if constexpr (sizeof(T) == 16 && needs_intrinsics) {
        if constexpr (Trait::width == 2) {
            if constexpr (std::is_integral_v<typename Trait::value_type>) {
                return reinterpret_cast<T>(
                    _mm_unpacklo_epi64(builtin_cast<llong>(a), builtin_cast<llong>(b)));
            } else {
                return reinterpret_cast<T>(
                    _mm_unpacklo_pd(builtin_cast<double>(a), builtin_cast<double>(b)));
            }
        } else if constexpr (Trait::width == 4) {
            if constexpr (std::is_integral_v<typename Trait::value_type>) {
                return reinterpret_cast<T>(
                    _mm_unpacklo_epi32(builtin_cast<llong>(a), builtin_cast<llong>(b)));
            } else {
                return reinterpret_cast<T>(
                    _mm_unpacklo_ps(builtin_cast<float>(a), builtin_cast<float>(b)));
            }
        } else if constexpr (Trait::width == 8) {
            return reinterpret_cast<T>(
                _mm_unpacklo_epi16(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        } else if constexpr (Trait::width == 16) {
            return reinterpret_cast<T>(
                _mm_unpacklo_epi8(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        }
    } else if constexpr (Trait::width == 2) {
        return T{a[0], b[0]};
    } else if constexpr (Trait::width == 4) {
        return T{a[0], b[0], a[1], b[1]};
    } else if constexpr (Trait::width == 8) {
        return T{a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]};
    } else if constexpr (Trait::width == 16) {
        return T{a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3],
                 a[4], b[4], a[5], b[5], a[6], b[6], a[7], b[7]};
    } else if constexpr (Trait::width == 32) {
        return T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],
                 a[4],  b[4],  a[5],  b[5],  a[6],  b[6],  a[7],  b[7],
                 a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11],
                 a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15]};
    } else if constexpr (Trait::width == 64) {
        return T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],  a[4],  b[4],
                 a[5],  b[5],  a[6],  b[6],  a[7],  b[7],  a[8],  b[8],  a[9],  b[9],
                 a[10], b[10], a[11], b[11], a[12], b[12], a[13], b[13], a[14], b[14],
                 a[15], b[15], a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19],
                 a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23], a[24], b[24],
                 a[25], b[25], a[26], b[26], a[27], b[27], a[28], b[28], a[29], b[29],
                 a[30], b[30], a[31], b[31]};
    } else {
        assert_unreachable<T>();
    }
}

template <class A, class B, class T = std::common_type_t<A, B>,
          class Trait = builtin_traits<T>>
_GLIBCXX_SIMD_INTRINSIC constexpr T interleave_hi(A _a, B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const T a(_a);
    const T b(_b);
    if constexpr (sizeof(T) == 16 && needs_intrinsics) {
        if constexpr (Trait::width == 2) {
            if constexpr (std::is_integral_v<typename Trait::value_type>) {
                return reinterpret_cast<T>(
                    _mm_unpackhi_epi64(builtin_cast<llong>(a), builtin_cast<llong>(b)));
            } else {
                return reinterpret_cast<T>(
                    _mm_unpackhi_pd(builtin_cast<double>(a), builtin_cast<double>(b)));
            }
        } else if constexpr (Trait::width == 4) {
            if constexpr (std::is_integral_v<typename Trait::value_type>) {
                return reinterpret_cast<T>(
                    _mm_unpackhi_epi32(builtin_cast<llong>(a), builtin_cast<llong>(b)));
            } else {
                return reinterpret_cast<T>(
                    _mm_unpackhi_ps(builtin_cast<float>(a), builtin_cast<float>(b)));
            }
        } else if constexpr (Trait::width == 8) {
            return reinterpret_cast<T>(
                _mm_unpackhi_epi16(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        } else if constexpr (Trait::width == 16) {
            return reinterpret_cast<T>(
                _mm_unpackhi_epi8(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        }
    } else if constexpr (Trait::width == 2) {
        return T{a[1], b[1]};
    } else if constexpr (Trait::width == 4) {
        return T{a[2], b[2], a[3], b[3]};
    } else if constexpr (Trait::width == 8) {
        return T{a[4], b[4], a[5], b[5], a[6], b[6], a[7], b[7]};
    } else if constexpr (Trait::width == 16) {
        return T{a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11],
                 a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15]};
    } else if constexpr (Trait::width == 32) {
        return T{a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19],
                 a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23],
                 a[24], b[24], a[25], b[25], a[26], b[26], a[27], b[27],
                 a[28], b[28], a[29], b[29], a[30], b[30], a[31], b[31]};
    } else if constexpr (Trait::width == 64) {
        return T{a[32], b[32], a[33], b[33], a[34], b[34], a[35], b[35],
                 a[36], b[36], a[37], b[37], a[38], b[38], a[39], b[39],
                 a[40], b[40], a[41], b[41], a[42], b[42], a[43], b[43],
                 a[44], b[44], a[45], b[45], a[46], b[46], a[47], b[47],
                 a[48], b[48], a[49], b[49], a[50], b[50], a[51], b[51],
                 a[52], b[52], a[53], b[53], a[54], b[54], a[55], b[55],
                 a[56], b[56], a[57], b[57], a[58], b[58], a[59], b[59],
                 a[60], b[60], a[61], b[61], a[62], b[62], a[63], b[63]};
    } else {
        assert_unreachable<T>();
    }
}

template <class A, class B, class T = std::common_type_t<A, B>,
          class Trait = builtin_traits<T>>
_GLIBCXX_SIMD_INTRINSIC constexpr T interleave128_lo(A _a, B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const T a(_a);
    const T b(_b);
    if constexpr (sizeof(T) == 16) {
        return interleave_lo(a, b);
    } else if constexpr (sizeof(T) == 32 && needs_intrinsics) {
        if constexpr (Trait::width == 4) {
            return reinterpret_cast<T>(
                _mm256_unpacklo_pd(builtin_cast<double>(a), builtin_cast<double>(b)));
        } else if constexpr (Trait::width == 8) {
            return reinterpret_cast<T>(
                _mm256_unpacklo_ps(builtin_cast<float>(a), builtin_cast<float>(b)));
        } else if constexpr (Trait::width == 16) {
            return reinterpret_cast<T>(
                _mm256_unpacklo_epi16(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        } else if constexpr (Trait::width == 32) {
            return reinterpret_cast<T>(
                _mm256_unpacklo_epi8(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        }
    } else if constexpr (sizeof(T) == 32) {
        if constexpr (Trait::width == 4) {
            return T{a[0], b[0], a[2], b[2]};
        } else if constexpr (Trait::width == 8) {
            return T{a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5]};
        } else if constexpr (Trait::width == 16) {
            return T{a[0], b[0], a[1], b[1], a[2],  b[2],  a[3],  b[3],
                     a[8], b[8], a[9], b[9], a[10], b[10], a[11], b[11]};
        } else if constexpr (Trait::width == 32) {
            return T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],
                     a[4],  b[4],  a[5],  b[5],  a[6],  b[6],  a[7],  b[7],
                     a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19],
                     a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23]};
        } else if constexpr (Trait::width == 64) {
            return T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],  a[4],  b[4],
                     a[5],  b[5],  a[6],  b[6],  a[7],  b[7],  a[8],  b[8],  a[9],  b[9],
                     a[10], b[10], a[11], b[11], a[12], b[12], a[13], b[13], a[14], b[14],
                     a[15], b[15], a[32], b[32], a[33], b[33], a[34], b[34], a[35], b[35],
                     a[36], b[36], a[37], b[37], a[38], b[38], a[39], b[39], a[40], b[40],
                     a[41], b[41], a[42], b[42], a[43], b[43], a[44], b[44], a[45], b[45],
                     a[46], b[46], a[47], b[47]};
        } else {
            assert_unreachable<T>();
        }
    } else if constexpr (sizeof(T) == 64 && needs_intrinsics) {
        if constexpr (Trait::width == 8) {
            return reinterpret_cast<T>(
                _mm512_unpacklo_pd(builtin_cast<double>(a), builtin_cast<double>(b)));
        } else if constexpr (Trait::width == 16) {
            return reinterpret_cast<T>(
                _mm512_unpacklo_ps(builtin_cast<float>(a), builtin_cast<float>(b)));
        } else if constexpr (Trait::width == 32) {
            return reinterpret_cast<T>(
                _mm512_unpacklo_epi16(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        } else if constexpr (Trait::width == 64) {
            return reinterpret_cast<T>(
                _mm512_unpacklo_epi8(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        }
    } else if constexpr (sizeof(T) == 64) {
        if constexpr (Trait::width == 8) {
            return T{a[0], b[0], a[2], b[2], a[4], b[4], a[6], b[6]};
        } else if constexpr (Trait::width == 16) {
            return T{a[0], b[0], a[1], b[1], a[4],  b[4],  a[5],  b[5],
                     a[8], b[8], a[9], b[9], a[12], b[12], a[13], b[13]};
        } else if constexpr (Trait::width == 32) {
            return T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],
                     a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11],
                     a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19],
                     a[24], b[24], a[25], b[25], a[26], b[26], a[27], b[27]};
        } else if constexpr (Trait::width == 64) {
            return T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],  a[4],  b[4],
                     a[5],  b[5],  a[6],  b[6],  a[7],  b[7],  a[16], b[16], a[17], b[17],
                     a[18], b[18], a[19], b[19], a[20], b[20], a[21], b[21], a[22], b[22],
                     a[23], b[23], a[32], b[32], a[33], b[33], a[34], b[34], a[35], b[35],
                     a[36], b[36], a[37], b[37], a[38], b[38], a[39], b[39], a[48], b[48],
                     a[49], b[49], a[50], b[50], a[51], b[51], a[52], b[52], a[53], b[53],
                     a[54], b[54], a[55], b[55]};
        } else {
            assert_unreachable<T>();
        }
    }
}

template <class A, class B, class T = std::common_type_t<A, B>,
          class Trait = builtin_traits<T>>
_GLIBCXX_SIMD_INTRINSIC constexpr T interleave128_hi(A _a, B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const T a(_a);
    const T b(_b);
    if constexpr (sizeof(T) == 16) {
        return interleave_hi(a, b);
    } else if constexpr (sizeof(T) == 32 && needs_intrinsics) {
        if constexpr (Trait::width == 4) {
            return reinterpret_cast<T>(
                _mm256_unpackhi_pd(builtin_cast<double>(a), builtin_cast<double>(b)));
        } else if constexpr (Trait::width == 8) {
            return reinterpret_cast<T>(
                _mm256_unpackhi_ps(builtin_cast<float>(a), builtin_cast<float>(b)));
        } else if constexpr (Trait::width == 16) {
            return reinterpret_cast<T>(
                _mm256_unpackhi_epi16(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        } else if constexpr (Trait::width == 32) {
            return reinterpret_cast<T>(
                _mm256_unpackhi_epi8(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        }
    } else if constexpr (sizeof(T) == 32) {
        if constexpr (Trait::width == 4) {
            return T{a[1], b[1], a[3], b[3]};
        } else if constexpr (Trait::width == 8) {
            return T{a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7]};
        } else if constexpr (Trait::width == 16) {
            return T{a[4],  b[4],  a[5],  b[5],  a[6],  b[6],  a[7],  b[7],
                     a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15]};
        } else if constexpr (Trait::width == 32) {
            return T{a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11],
                     a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15],
                     a[24], b[24], a[25], b[25], a[26], b[26], a[27], b[27],
                     a[28], b[28], a[29], b[29], a[30], b[30], a[31], b[31]};
        } else if constexpr (Trait::width == 64) {
            return T{a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19], a[20], b[20],
                     a[21], b[21], a[22], b[22], a[23], b[23], a[24], b[24], a[25], b[25],
                     a[26], b[26], a[27], b[27], a[28], b[28], a[29], b[29], a[30], b[30],
                     a[31], b[31], a[48], b[48], a[49], b[49], a[50], b[50], a[51], b[51],
                     a[52], b[52], a[53], b[53], a[54], b[54], a[55], b[55], a[56], b[56],
                     a[57], b[57], a[58], b[58], a[59], b[59], a[60], b[60], a[61], b[61],
                     a[62], b[62], a[63], b[63]};
        } else {
            assert_unreachable<T>();
        }
    } else if constexpr (sizeof(T) == 64 && needs_intrinsics) {
        if constexpr (Trait::width == 8) {
            return reinterpret_cast<T>(
                _mm512_unpackhi_pd(builtin_cast<double>(a), builtin_cast<double>(b)));
        } else if constexpr (Trait::width == 16) {
            return reinterpret_cast<T>(
                _mm512_unpackhi_ps(builtin_cast<float>(a), builtin_cast<float>(b)));
        } else if constexpr (Trait::width == 32) {
            return reinterpret_cast<T>(
                _mm512_unpackhi_epi16(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        } else if constexpr (Trait::width == 64) {
            return reinterpret_cast<T>(
                _mm512_unpackhi_epi8(builtin_cast<llong>(a), builtin_cast<llong>(b)));
        }
    } else if constexpr (sizeof(T) == 64) {
        if constexpr (Trait::width == 8) {
            return T{a[1], b[1], a[3], b[3], a[5], b[5], a[7], b[7]};
        } else if constexpr (Trait::width == 16) {
            return T{a[2],  b[2],  a[3],  b[3],  a[6],  b[6],  a[7],  b[7],
                     a[10], b[10], a[11], b[11], a[14], b[14], a[15], b[15]};
        } else if constexpr (Trait::width == 32) {
            return T{a[4],  b[4],  a[5],  b[5],  a[6],  b[6],  a[7],  b[7],
                     a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15],
                     a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23],
                     a[28], b[28], a[29], b[29], a[30], b[30], a[31], b[31]};
        } else if constexpr (Trait::width == 64) {
            return T{a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11], a[12], b[12],
                     a[13], b[13], a[14], b[14], a[15], b[15], a[24], b[24], a[25], b[25],
                     a[26], b[26], a[27], b[27], a[28], b[28], a[29], b[29], a[30], b[30],
                     a[31], b[31], a[40], b[40], a[41], b[41], a[42], b[42], a[43], b[43],
                     a[44], b[44], a[45], b[45], a[46], b[46], a[47], b[47], a[56], b[56],
                     a[57], b[57], a[58], b[58], a[59], b[59], a[60], b[60], a[61], b[61],
                     a[62], b[62], a[63], b[63]};
        } else {
            assert_unreachable<T>();
        }
    }
}

template <class T> struct interleaved_pair {
    T lo, hi;
};

template <class A, class B, class T = std::common_type_t<A, B>,
          class Trait = builtin_traits<T>>
_GLIBCXX_SIMD_INTRINSIC constexpr interleaved_pair<T> interleave(A a, B b)
{
    return {interleave_lo(a, b), interleave_hi(a, b)};
}

template <class A, class B, class T = std::common_type_t<A, B>,
          class Trait = builtin_traits<T>>
_GLIBCXX_SIMD_INTRINSIC constexpr interleaved_pair<T> interleave128(A a, B b)
{
    return {interleave128_lo(a, b), interleave128_hi(a, b)};
}
// }}}
// is_bitset {{{
template <class T> struct is_bitset : std::false_type {};
template <size_t N> struct is_bitset<std::bitset<N>> : std::true_type {};
template <class T> inline constexpr bool is_bitset_v = is_bitset<T>::value;

// }}}
// is_storage {{{
template <class T> struct is_storage : std::false_type {};
template <class T, size_t N> struct is_storage<Storage<T, N>> : std::true_type {};
template <class T> inline constexpr bool is_storage_v = is_storage<T>::value;

// }}}
// convert_mask{{{
template <class To, class From> inline To convert_mask(From k) {
    if constexpr (std::is_same_v<To, From>) {  // also covers bool -> bool
        return k;
    } else if constexpr (std::is_unsigned_v<From> && std::is_unsigned_v<To>) {
        // bits -> bits
        return k;  // zero-extends or truncates
    } else if constexpr (is_bitset_v<From>) {
        // from std::bitset {{{
        static_assert(k.size() <= sizeof(ullong) * CHAR_BIT);
        using T = std::conditional_t<
            (k.size() <= sizeof(ushort) * CHAR_BIT),
            std::conditional_t<(k.size() <= CHAR_BIT), uchar, ushort>,
            std::conditional_t<(k.size() <= sizeof(uint) * CHAR_BIT), uint, ullong>>;
        return convert_mask<To>(static_cast<T>(k.to_ullong()));
        // }}}
    } else if constexpr (is_bitset_v<To>) {
        // to std::bitset {{{
        static_assert(To().size() <= sizeof(ullong) * CHAR_BIT);
        using T = std::conditional_t<
            (To().size() <= sizeof(ushort) * CHAR_BIT),
            std::conditional_t<(To().size() <= CHAR_BIT), uchar, ushort>,
            std::conditional_t<(To().size() <= sizeof(uint) * CHAR_BIT), uint, ullong>>;
        return convert_mask<T>(k);
        // }}}
    } else if constexpr (is_storage_v<From>) {
        return convert_mask<To>(k.d);
    } else if constexpr (is_storage_v<To>) {
        return convert_mask<typename To::register_type>(k);
    } else if constexpr (std::is_unsigned_v<From> && is_builtin_vector_v<To>) {
        // bits -> vector {{{
        using Trait = builtin_traits<To>;
        constexpr size_t N_in = sizeof(From) * CHAR_BIT;
        using ToT = typename Trait::value_type;
        constexpr size_t N_out = Trait::width;
        constexpr size_t N = std::min(N_in, N_out);
        constexpr size_t bytes_per_output_element = sizeof(ToT);
        if constexpr (have_avx512f) {
            if constexpr (bytes_per_output_element == 1 && sizeof(To) == 16) {
                if constexpr (have_avx512bw_vl) {
                    return builtin_cast<ToT>(_mm_movm_epi8(k));
                } else if constexpr (have_avx512bw) {
                    return builtin_cast<ToT>(lo128(_mm512_movm_epi8(k)));
                } else {
                    auto as32bits = _mm512_maskz_mov_epi32(k, ~__m512i());
                    auto as16bits = fixup_avx_xzyw(
                        _mm256_packs_epi32(lo256(as32bits), hi256(as32bits)));
                    return builtin_cast<ToT>(
                        _mm_packs_epi16(lo128(as16bits), hi128(as16bits)));
                }
            } else if constexpr (bytes_per_output_element == 1 && sizeof(To) == 32) {
                if constexpr (have_avx512bw_vl) {
                    return builtin_cast<ToT>(_mm256_movm_epi8(k));
                } else if constexpr (have_avx512bw) {
                    return builtin_cast<ToT>(lo256(_mm512_movm_epi8(k)));
                } else {
                    auto as16bits =  // 0 16 1 17 ... 15 31
                        _mm512_srli_epi32(_mm512_maskz_mov_epi32(k, ~__m512i()), 16) |
                        _mm512_slli_epi32(_mm512_maskz_mov_epi32(k >> 16, ~__m512i()),
                                          16);
                    auto _0_16_1_17 = fixup_avx_xzyw(_mm256_packs_epi16(
                        lo256(as16bits),
                        hi256(as16bits))  // 0 16 1 17 2 18 3 19 8 24 9 25 ...
                    );
                    // deinterleave:
                    return builtin_cast<ToT>(fixup_avx_xzyw(_mm256_shuffle_epi8(
                        _0_16_1_17,  // 0 16 1 17 2 ...
                        _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13,
                                         15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11,
                                         13, 15))));  // 0-7 16-23 8-15 24-31 -> xzyw
                                                      // 0-3  8-11 16-19 24-27
                                                      // 4-7 12-15 20-23 28-31
                }
            } else if constexpr (bytes_per_output_element == 1 && sizeof(To) == 64) {
                return reinterpret_cast<builtin_type_t<schar, 64>>(_mm512_movm_epi8(k));
            } else if constexpr (bytes_per_output_element == 2 && sizeof(To) == 16) {
                if constexpr (have_avx512bw_vl) {
                    return builtin_cast<ToT>(_mm_movm_epi16(k));
                } else if constexpr (have_avx512bw) {
                    return builtin_cast<ToT>(lo128(_mm512_movm_epi16(k)));
                } else {
                    __m256i as32bits;
                    if constexpr (have_avx512vl) {
                        as32bits = _mm256_maskz_mov_epi32(k, ~__m256i());
                    } else {
                        as32bits = lo256(_mm512_maskz_mov_epi32(k, ~__m512i()));
                    }
                    return builtin_cast<ToT>(
                        _mm_packs_epi32(lo128(as32bits), hi128(as32bits)));
                }
            } else if constexpr (bytes_per_output_element == 2 && sizeof(To) == 32) {
                if constexpr (have_avx512bw_vl) {
                    return builtin_cast<ToT>(_mm256_movm_epi16(k));
                } else if constexpr (have_avx512bw) {
                    return builtin_cast<ToT>(lo256(_mm512_movm_epi16(k)));
                } else {
                    auto as32bits = _mm512_maskz_mov_epi32(k, ~__m512i());
                    return builtin_cast<ToT>(fixup_avx_xzyw(
                        _mm256_packs_epi32(lo256(as32bits), hi256(as32bits))));
                }
            } else if constexpr (bytes_per_output_element == 2 && sizeof(To) == 64) {
                return builtin_cast<ToT>(_mm512_movm_epi16(k));
            } else if constexpr (bytes_per_output_element == 4 && sizeof(To) == 16) {
                return builtin_cast<ToT>(
                    have_avx512dq_vl
                        ? _mm_movm_epi32(k)
                        : have_avx512dq
                              ? lo128(_mm512_movm_epi32(k))
                              : have_avx512vl
                                    ? _mm_maskz_mov_epi32(k, ~__m128i())
                                    : lo128(_mm512_maskz_mov_epi32(k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 4 && sizeof(To) == 32) {
                return builtin_cast<ToT>(
                    have_avx512dq_vl
                        ? _mm256_movm_epi32(k)
                        : have_avx512dq
                              ? lo256(_mm512_movm_epi32(k))
                              : have_avx512vl
                                    ? _mm256_maskz_mov_epi32(k, ~__m256i())
                                    : lo256(_mm512_maskz_mov_epi32(k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 4 && sizeof(To) == 64) {
                return builtin_cast<ToT>(have_avx512dq
                                             ? _mm512_movm_epi32(k)
                                             : _mm512_maskz_mov_epi32(k, ~__m512i()));
            } else if constexpr (bytes_per_output_element == 8 && sizeof(To) == 16) {
                return builtin_cast<ToT>(
                    have_avx512dq_vl
                        ? _mm_movm_epi64(k)
                        : have_avx512dq
                              ? lo128(_mm512_movm_epi64(k))
                              : have_avx512vl
                                    ? _mm_maskz_mov_epi64(k, ~__m128i())
                                    : lo128(_mm512_maskz_mov_epi64(k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 8 && sizeof(To) == 32) {
                return builtin_cast<ToT>(
                    have_avx512dq_vl
                        ? _mm256_movm_epi64(k)
                        : have_avx512dq
                              ? lo256(_mm512_movm_epi64(k))
                              : have_avx512vl
                                    ? _mm256_maskz_mov_epi64(k, ~__m256i())
                                    : lo256(_mm512_maskz_mov_epi64(k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 8 && sizeof(To) == 64) {
                return builtin_cast<ToT>(have_avx512dq
                                             ? _mm512_movm_epi64(k)
                                             : _mm512_maskz_mov_epi64(k, ~__m512i()));
            } else {
                assert_unreachable<To>();
            }
        } else if constexpr (have_sse) {
            using U = std::make_unsigned_t<detail::int_for_sizeof_t<ToT>>;
            using V = builtin_type_t<U, N>;  // simd<U, Abi>;
            static_assert(sizeof(V) <= 32);  // can't be AVX512
            constexpr size_t bits_per_element = sizeof(U) * CHAR_BIT;
            if constexpr (!have_avx2 && have_avx && sizeof(V) == 32) {
                if constexpr (N == 8) {
                    return _mm256_cmp_ps(
                        _mm256_and_ps(
                            _mm256_castsi256_ps(_mm256_set1_epi32(k)),
                            _mm256_castsi256_ps(_mm256_setr_epi32(
                                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80))),
                        _mm256_setzero_ps(), _CMP_NEQ_UQ);
                } else if constexpr (N == 4) {
                    return _mm256_cmp_pd(
                        _mm256_and_pd(
                            _mm256_castsi256_pd(_mm256_set1_epi64x(k)),
                            _mm256_castsi256_pd(
                                _mm256_setr_epi64x(0x01, 0x02, 0x04, 0x08))),
                        _mm256_setzero_pd(), _CMP_NEQ_UQ);
                } else {
                    assert_unreachable<To>();
                }
            } else if constexpr (bits_per_element >= N) {
                constexpr auto bitmask = generate_builtin<builtin_type_t<U, N>>(
                    [](auto i) -> U { return 1ull << i; });
                return builtin_cast<ToT>(
                    (builtin_broadcast<N, U>(k) & bitmask) != 0);
            } else if constexpr (sizeof(V) == 16 && sizeof(ToT) == 1 && have_ssse3) {
                const auto bitmask = to_intrin(make_builtin<uchar>(
                    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128));
                return builtin_cast<ToT>(
                    builtin_cast<ToT>(
                        _mm_shuffle_epi8(
                            to_intrin(builtin_type_t<ullong, 2>{k}),
                            _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                                          1)) &
                        bitmask) != 0);
            } else if constexpr (sizeof(V) == 32 && sizeof(ToT) == 1 && have_avx2) {
                const auto bitmask =
                    _mm256_broadcastsi128_si256(to_intrin(make_builtin<uchar>(
                        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128)));
                return builtin_cast<ToT>(
                    builtin_cast<ToT>(_mm256_shuffle_epi8(
                                        _mm256_broadcastsi128_si256(to_intrin(
                                            builtin_type_t<ullong, 2>{k})),
                                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                                                         1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                                         2, 2, 3, 3, 3, 3, 3, 3, 3, 3)) &
                                    bitmask) != 0);
                /* TODO:
                } else if constexpr (sizeof(V) == 32 && sizeof(ToT) == 2 && have_avx2) {
                    constexpr auto bitmask = _mm256_broadcastsi128_si256(
                        _mm_setr_epi8(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000)); return
                builtin_cast<ToT>( _mm256_shuffle_epi8(
                                   _mm256_broadcastsi128_si256(__m128i{k}),
                                   _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)) & bitmask) != 0;
                */
            } else {
                const V tmp = generate_builtin<V>([&](auto i) {
                                  return static_cast<U>(
                                      k >> (bits_per_element * (i / bits_per_element)));
                              }) &
                              generate_builtin<V>([](auto i) {
                                  return static_cast<U>(1ull << (i % bits_per_element));
                              });  // mask bit index
                return builtin_cast<ToT>(tmp != V());
            }
        } else {
            assert_unreachable<To>();
        } // }}}
    } else if constexpr (is_builtin_vector_v<From> && std::is_unsigned_v<To>) {
        // vector -> bits {{{
        using Trait = builtin_traits<From>;
        using T = typename Trait::value_type;
        constexpr size_t FromN = Trait::width;
        constexpr size_t cvt_id = FromN * 10 + sizeof(T);
        constexpr bool have_avx512_int = have_avx512f && std::is_integral_v<T>;
        [[maybe_unused]]  // PR85827
        const auto intrin = to_intrin(k);

             if constexpr (cvt_id == 16'1 && have_avx512bw_vl) { return    _mm_movepi8_mask(intrin); }
        else if constexpr (cvt_id == 16'1 && have_avx512bw   ) { return _mm512_movepi8_mask(zero_extend(intrin)); }
        else if constexpr (cvt_id == 16'1                    ) { return    _mm_movemask_epi8(intrin); }
        else if constexpr (cvt_id == 32'1 && have_avx512bw_vl) { return _mm256_movepi8_mask(intrin); }
        else if constexpr (cvt_id == 32'1 && have_avx512bw   ) { return _mm512_movepi8_mask(zero_extend(intrin)); }
        else if constexpr (cvt_id == 32'1                    ) { return _mm256_movemask_epi8(intrin); }
        else if constexpr (cvt_id == 64'1 && have_avx512bw   ) { return _mm512_movepi8_mask(intrin); }
        else if constexpr (cvt_id ==  8'2 && have_avx512bw_vl) { return    _mm_movepi16_mask(intrin); }
        else if constexpr (cvt_id ==  8'2 && have_avx512bw   ) { return _mm512_movepi16_mask(zero_extend(intrin)); }
        else if constexpr (cvt_id ==  8'2                    ) { return movemask_epi16(intrin); }
        else if constexpr (cvt_id == 16'2 && have_avx512bw_vl) { return _mm256_movepi16_mask(intrin); }
        else if constexpr (cvt_id == 16'2 && have_avx512bw   ) { return _mm512_movepi16_mask(zero_extend(intrin)); }
        else if constexpr (cvt_id == 16'2                    ) { return movemask_epi16(intrin); }
        else if constexpr (cvt_id == 32'2 && have_avx512bw   ) { return _mm512_movepi16_mask(intrin); }
        else if constexpr (cvt_id ==  4'4 && have_avx512dq_vl) { return    _mm_movepi32_mask(builtin_cast<llong>(k)); }
        else if constexpr (cvt_id ==  4'4 && have_avx512dq   ) { return _mm512_movepi32_mask(zero_extend(builtin_cast<llong>(k))); }
        else if constexpr (cvt_id ==  4'4 && have_avx512vl   ) { return    _mm_cmp_epi32_mask(builtin_cast<llong>(k), __m128i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'4 && have_avx512_int ) { return _mm512_cmp_epi32_mask(zero_extend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'4                    ) { return    _mm_movemask_ps(k); }
        else if constexpr (cvt_id ==  8'4 && have_avx512dq_vl) { return _mm256_movepi32_mask(builtin_cast<llong>(k)); }
        else if constexpr (cvt_id ==  8'4 && have_avx512dq   ) { return _mm512_movepi32_mask(zero_extend(builtin_cast<llong>(k))); }
        else if constexpr (cvt_id ==  8'4 && have_avx512vl   ) { return _mm256_cmp_epi32_mask(builtin_cast<llong>(k), __m256i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  8'4 && have_avx512_int ) { return _mm512_cmp_epi32_mask(zero_extend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  8'4                    ) { return _mm256_movemask_ps(k); }
        else if constexpr (cvt_id == 16'4 && have_avx512dq   ) { return _mm512_movepi32_mask(builtin_cast<llong>(k)); }
        else if constexpr (cvt_id == 16'4                    ) { return _mm512_cmp_epi32_mask(builtin_cast<llong>(k), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8 && have_avx512dq_vl) { return    _mm_movepi64_mask(builtin_cast<llong>(k)); }
        else if constexpr (cvt_id ==  2'8 && have_avx512dq   ) { return _mm512_movepi64_mask(zero_extend(builtin_cast<llong>(k))); }
        else if constexpr (cvt_id ==  2'8 && have_avx512vl   ) { return    _mm_cmp_epi64_mask(builtin_cast<llong>(k), __m128i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8 && have_avx512_int ) { return _mm512_cmp_epi64_mask(zero_extend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8                    ) { return    _mm_movemask_pd(k); }
        else if constexpr (cvt_id ==  4'8 && have_avx512dq_vl) { return _mm256_movepi64_mask(builtin_cast<llong>(k)); }
        else if constexpr (cvt_id ==  4'8 && have_avx512dq   ) { return _mm512_movepi64_mask(zero_extend(builtin_cast<llong>(k))); }
        else if constexpr (cvt_id ==  4'8 && have_avx512vl   ) { return _mm256_cmp_epi64_mask(builtin_cast<llong>(k), __m256i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'8 && have_avx512_int ) { return _mm512_cmp_epi64_mask(zero_extend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'8                    ) { return _mm256_movemask_pd(k); }
        else if constexpr (cvt_id ==  8'8 && have_avx512dq   ) { return _mm512_movepi64_mask(builtin_cast<llong>(k)); }
        else if constexpr (cvt_id ==  8'8                    ) { return _mm512_cmp_epi64_mask(builtin_cast<llong>(k), __m512i(), _MM_CMPINT_LT); }
        else { assert_unreachable<To>(); }
        // }}}
    } else if constexpr (is_builtin_vector_v<From> && is_builtin_vector_v<To>) {
        // vector -> vector {{{
        using ToTrait = builtin_traits<To>;
        using FromTrait = builtin_traits<From>;
        using ToT = typename ToTrait::value_type;
        using T = typename FromTrait::value_type;
        constexpr size_t FromN = FromTrait::width;
        constexpr size_t ToN = ToTrait::width;
        constexpr int FromBytes = sizeof(T);
        constexpr int ToBytes = sizeof(ToT);

        if constexpr (FromN == ToN && sizeof(From) == sizeof(To)) {
            // reinterpret the bits
            return reinterpret_cast<To>(k);
        } else if constexpr (sizeof(To) == 16 && sizeof(k) == 16) {
            // SSE -> SSE {{{
            if constexpr (FromBytes == 4 && ToBytes == 8) {
                if constexpr(std::is_integral_v<T>) {
                    return builtin_cast<ToT>(interleave128_lo(k, k));
                } else {
                    return builtin_cast<ToT>(interleave128_lo(k, k));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 8) {
                const auto y = builtin_cast<int>(interleave128_lo(k, k));
                return builtin_cast<ToT>(interleave128_lo(y, y));
            } else if constexpr (FromBytes == 1 && ToBytes == 8) {
                auto y = builtin_cast<short>(interleave128_lo(k, k));
                auto z = builtin_cast<int>(interleave128_lo(y, y));
                return builtin_cast<ToT>(interleave128_lo(z, z));
            } else if constexpr (FromBytes == 8 && ToBytes == 4) {
                if constexpr (std::is_floating_point_v<T>) {
                    return builtin_cast<ToT>(_mm_shuffle_ps(builtin_cast<float>(k), __m128(),
                                                     make_immediate<4>(1, 3, 1, 3)));
                } else {
                    auto y = builtin_cast<llong>(k);
                    return builtin_cast<ToT>(_mm_packs_epi32(y, __m128i()));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 4) {
                return builtin_cast<ToT>(interleave128_lo(k, k));
            } else if constexpr (FromBytes == 1 && ToBytes == 4) {
                const auto y = builtin_cast<short>(interleave128_lo(k, k));
                return builtin_cast<ToT>(interleave128_lo(y, y));
            } else if constexpr (FromBytes == 8 && ToBytes == 2) {
                if constexpr(have_ssse3) {
                    return builtin_cast<ToT>(
                        _mm_shuffle_epi8(builtin_cast<llong>(k),
                                         _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    const auto y = _mm_packs_epi32(builtin_cast<llong>(k), __m128i());
                    return builtin_cast<ToT>(_mm_packs_epi32(y, __m128i()));
                }
            } else if constexpr (FromBytes == 4 && ToBytes == 2) {
                return builtin_cast<ToT>(
                    _mm_packs_epi32(builtin_cast<llong>(k), __m128i()));
            } else if constexpr (FromBytes == 1 && ToBytes == 2) {
                return builtin_cast<ToT>(interleave128_lo(k, k));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {
                if constexpr(have_ssse3) {
                    return builtin_cast<ToT>(
                        _mm_shuffle_epi8(builtin_cast<llong>(k),
                                         _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    auto y = _mm_packs_epi32(builtin_cast<llong>(k), __m128i());
                    y = _mm_packs_epi32(y, __m128i());
                    return builtin_cast<ToT>(_mm_packs_epi16(y, __m128i()));
                }
            } else if constexpr (FromBytes == 4 && ToBytes == 1) {
                if constexpr(have_ssse3) {
                    return builtin_cast<ToT>(
                        _mm_shuffle_epi8(builtin_cast<llong>(k),
                                         _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    const auto y = _mm_packs_epi32(builtin_cast<llong>(k), __m128i());
                    return builtin_cast<ToT>(_mm_packs_epi16(y, __m128i()));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 1) {
                return builtin_cast<ToT>(_mm_packs_epi16(builtin_cast<llong>(k), __m128i()));
            } else {
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(To) == 32 && sizeof(k) == 32) {
            // AVX -> AVX {{{
            if constexpr (FromBytes == ToBytes) {  // keep low 1/2
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            } else if constexpr (FromBytes == ToBytes * 2) {
                const auto y = builtin_cast<llong>(k);
                return builtin_cast<ToT>(
                    _mm256_castsi128_si256(_mm_packs_epi16(lo128(y), hi128(y))));
            } else if constexpr (FromBytes == ToBytes * 4) {
                const auto y = builtin_cast<llong>(k);
                return builtin_cast<ToT>(_mm256_castsi128_si256(
                    _mm_packs_epi16(_mm_packs_epi16(lo128(y), hi128(y)), __m128i())));
            } else if constexpr (FromBytes == ToBytes * 8) {
                const auto y = builtin_cast<llong>(k);
                return builtin_cast<ToT>(_mm256_castsi128_si256(
                    _mm_shuffle_epi8(_mm_packs_epi16(lo128(y), hi128(y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1))));
            } else if constexpr (FromBytes * 2 == ToBytes) {
                auto y = fixup_avx_xzyw(to_intrin(k));
                if constexpr(std::is_floating_point_v<T>) {
                    return builtin_cast<ToT>(_mm256_unpacklo_ps(y, y));
                } else {
                    return builtin_cast<ToT>(_mm256_unpacklo_epi8(y, y));
                }
            } else if constexpr (FromBytes * 4 == ToBytes) {
                auto y = _mm_unpacklo_epi8(lo128(builtin_cast<llong>(k)),
                                           lo128(builtin_cast<llong>(k)));  // drops 3/4 of input
                return builtin_cast<ToT>(
                    concat(_mm_unpacklo_epi16(y, y), _mm_unpackhi_epi16(y, y)));
            } else if constexpr (FromBytes == 1 && ToBytes == 8) {
                auto y = _mm_unpacklo_epi8(lo128(builtin_cast<llong>(k)),
                                           lo128(builtin_cast<llong>(k)));  // drops 3/4 of input
                y = _mm_unpacklo_epi16(y, y);  // drops another 1/2 => 7/8 total
                return builtin_cast<ToT>(
                    concat(_mm_unpacklo_epi32(y, y), _mm_unpackhi_epi32(y, y)));
            } else {
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(To) == 32 && sizeof(k) == 16) {
            // SSE -> AVX {{{
            if constexpr (FromBytes == ToBytes) {
                return builtin_cast<ToT>(
                    intrinsic_type_t<T, 32 / sizeof(T)>(zero_extend(to_intrin(k))));
            } else if constexpr (FromBytes * 2 == ToBytes) {  // keep all
                return builtin_cast<ToT>(concat(_mm_unpacklo_epi8(builtin_cast<llong>(k), builtin_cast<llong>(k)),
                                         _mm_unpackhi_epi8(builtin_cast<llong>(k), builtin_cast<llong>(k))));
            } else if constexpr (FromBytes * 4 == ToBytes) {
                if constexpr (have_avx2) {
                    return builtin_cast<ToT>(_mm256_shuffle_epi8(
                        concat(builtin_cast<llong>(k), builtin_cast<llong>(k)),
                        _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7,
                                         7)));
                } else {
                    return builtin_cast<ToT>(
                        concat(_mm_shuffle_epi8(builtin_cast<llong>(k),
                                                _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2,
                                                              2, 2, 2, 3, 3, 3, 3)),
                               _mm_shuffle_epi8(builtin_cast<llong>(k),
                                                _mm_setr_epi8(4, 4, 4, 4, 5, 5, 5, 5, 6,
                                                              6, 6, 6, 7, 7, 7, 7))));
                }
            } else if constexpr (FromBytes * 8 == ToBytes) {
                if constexpr (have_avx2) {
                    return builtin_cast<ToT>(_mm256_shuffle_epi8(
                        concat(builtin_cast<llong>(k), builtin_cast<llong>(k)),
                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                                         3)));
                } else {
                    return builtin_cast<ToT>(
                        concat(_mm_shuffle_epi8(builtin_cast<llong>(k),
                                                _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                              1, 1, 1, 1, 1, 1, 1)),
                               _mm_shuffle_epi8(builtin_cast<llong>(k),
                                                _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3,
                                                              3, 3, 3, 3, 3, 3, 3))));
                }
            } else if constexpr (FromBytes == ToBytes * 2) {
                return builtin_cast<ToT>(
                    __m256i(zero_extend(_mm_packs_epi16(builtin_cast<llong>(k), __m128i()))));
            } else if constexpr (FromBytes == 8 && ToBytes == 2) {
                return builtin_cast<ToT>(__m256i(zero_extend(
                    _mm_shuffle_epi8(builtin_cast<llong>(k),
                                     _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else if constexpr (FromBytes == 4 && ToBytes == 1) {
                return builtin_cast<ToT>(__m256i(zero_extend(
                    _mm_shuffle_epi8(builtin_cast<llong>(k),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {
                return builtin_cast<ToT>(__m256i(zero_extend(
                    _mm_shuffle_epi8(builtin_cast<llong>(k),
                                     _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else {
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(To) == 16 && sizeof(k) == 32) {
            // AVX -> SSE {{{
            if constexpr (FromBytes == ToBytes) {  // keep low 1/2
                return builtin_cast<ToT>(lo128(k));
            } else if constexpr (FromBytes == ToBytes * 2) {  // keep all
                auto y = builtin_cast<llong>(k);
                return builtin_cast<ToT>(_mm_packs_epi16(lo128(y), hi128(y)));
            } else if constexpr (FromBytes == ToBytes * 4) {  // add 1/2 undef
                auto y = builtin_cast<llong>(k);
                return builtin_cast<ToT>(
                    _mm_packs_epi16(_mm_packs_epi16(lo128(y), hi128(y)), __m128i()));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {  // add 3/4 undef
                auto y = builtin_cast<llong>(k);
                return builtin_cast<ToT>(
                    _mm_shuffle_epi8(_mm_packs_epi16(lo128(y), hi128(y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)));
            } else if constexpr (FromBytes * 2 == ToBytes) {  // keep low 1/4
                auto y = lo128(builtin_cast<llong>(k));
                return builtin_cast<ToT>(_mm_unpacklo_epi8(y, y));
            } else if constexpr (FromBytes * 4 == ToBytes) {  // keep low 1/8
                auto y = lo128(builtin_cast<llong>(k));
                y = _mm_unpacklo_epi8(y, y);
                return builtin_cast<ToT>(_mm_unpacklo_epi8(y, y));
            } else if constexpr (FromBytes * 8 == ToBytes) {  // keep low 1/16
                auto y = lo128(builtin_cast<llong>(k));
                y = _mm_unpacklo_epi8(y, y);
                y = _mm_unpacklo_epi8(y, y);
                return builtin_cast<ToT>(_mm_unpacklo_epi8(y, y));
            } else {
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            }
            // }}}
        }
        // }}}
    } else {
        assert_unreachable<To>();
    }
}

// }}}

template <class Abi> struct simd_math_fallback {  //{{{
    template <class T> simd<T, Abi> __acos(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::acos(x[i]); });
    }

    template <class T> simd<T, Abi> __asin(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::asin(x[i]); });
    }

    template <class T> simd<T, Abi> __atan(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::atan(x[i]); });
    }

    template <class T> simd<T, Abi> __atan2(const simd<T, Abi> &x, const simd<T, Abi> &y)
    {
        return simd<T, Abi>([&](auto i) { return std::atan2(x[i], y[i]); });
    }

    template <class T> simd<T, Abi> __cos(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::cos(x[i]); });
    }

    template <class T> simd<T, Abi> __sin(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::sin(x[i]); });
    }

    template <class T> simd<T, Abi> __tan(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::tan(x[i]); });
    }

    template <class T> simd<T, Abi> __acosh(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::acosh(x[i]); });
    }

    template <class T> simd<T, Abi> __asinh(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::asinh(x[i]); });
    }

    template <class T> simd<T, Abi> __atanh(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::atanh(x[i]); });
    }

    template <class T> simd<T, Abi> __cosh(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::cosh(x[i]); });
    }

    template <class T> simd<T, Abi> __sinh(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::sinh(x[i]); });
    }

    template <class T> simd<T, Abi> __tanh(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::tanh(x[i]); });
    }

    template <class T> simd<T, Abi> __exp(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::exp(x[i]); });
    }

    template <class T> simd<T, Abi> __exp2(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::exp2(x[i]); });
    }

    template <class T> simd<T, Abi> __expm1(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::expm1(x[i]); });
    }

    template <class T>
    simd<T, Abi> __frexp(const simd<T, Abi> &x,
                         fixed_size_simd<int, simd_size_v<T, Abi>> &exp)
    {
        return simd<T, Abi>([&](auto i) {
            int tmp;
            T r = std::frexp(x[i], &tmp);
            exp[i] = tmp;
            return r;
        });
    }

    template <class T>
    simd<T, Abi> __ldexp(const simd<T, Abi> &x,
                         const fixed_size_simd<int, simd_size_v<T, Abi>> &exp)
    {
        return simd<T, Abi>([&](auto i) { return std::ldexp(x[i], exp[i]); });
    }

    template <class T>
    fixed_size_simd<int, simd_size_v<T, Abi>> __ilogb(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::ilogb(x[i]); });
    }

    template <class T> simd<T, Abi> __log(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::log(x[i]); });
    }

    template <class T> simd<T, Abi> __log10(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::log10(x[i]); });
    }

    template <class T> simd<T, Abi> __log1p(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::log1p(x[i]); });
    }

    template <class T> simd<T, Abi> __log2(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::log2(x[i]); });
    }

    template <class T> simd<T, Abi> __logb(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::logb(x[i]); });
    }

    template <class T> simd<T, Abi> __modf(const simd<T, Abi> &x, simd<T, Abi> &y)
    {
        return simd<T, Abi>([&](auto i) {
            T tmp;
            T r = std::modf(x[i], &tmp);
            y[i] = tmp;
            return r;
        });
    }

    template <class T>
    simd<T, Abi> __scalbn(const simd<T, Abi> &x,
                          const fixed_size_simd<int, simd_size_v<T, Abi>> &y)
    {
        return simd<T, Abi>([&](auto i) { return std::scalbn(x[i], y[i]); });
    }

    template <class T>
    simd<T, Abi> __scalbln(const simd<T, Abi> &x,
                           const fixed_size_simd<long, simd_size_v<T, Abi>> &y)
    {
        return simd<T, Abi>([&](auto i) { return std::scalbln(x[i], y[i]); });
    }

    template <class T> simd<T, Abi> __cbrt(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::cbrt(x[i]); });
    }

    template <class T> simd<T, Abi> __abs(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::abs(x[i]); });
    }

    template <class T> simd<T, Abi> __fabs(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::fabs(x[i]); });
    }

    template <class T> simd<T, Abi> __pow(const simd<T, Abi> &x, const simd<T, Abi> &y)
    {
        return simd<T, Abi>([&](auto i) { return std::pow(x[i], y[i]); });
    }

    template <class T> simd<T, Abi> __sqrt(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::sqrt(x[i]); });
    }

    template <class T> simd<T, Abi> __erf(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::erf(x[i]); });
    }

    template <class T> simd<T, Abi> __erfc(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::erfc(x[i]); });
    }

    template <class T> simd<T, Abi> __lgamma(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::lgamma(x[i]); });
    }

    template <class T> simd<T, Abi> __tgamma(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::tgamma(x[i]); });
    }

    template <class T> simd<T, Abi> __ceil(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::ceil(x[i]); });
    }

    template <class T> simd<T, Abi> __floor(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::floor(x[i]); });
    }

    template <class T> simd<T, Abi> __nearbyint(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::nearbyint(x[i]); });
    }

    template <class T> simd<T, Abi> __rint(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::rint(x[i]); });
    }

    template <class T>
    fixed_size_simd<long, simd_size_v<T, Abi>> __lrint(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::lrint(x[i]); });
    }

    template <class T>
    fixed_size_simd<long long, simd_size_v<T, Abi>> __llrint(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::llrint(x[i]); });
    }

    template <class T> simd<T, Abi> __round(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::round(x[i]); });
    }

    template <class T>
    fixed_size_simd<long, simd_size_v<T, Abi>> __lround(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::lround(x[i]); });
    }

    template <class T>
    fixed_size_simd<long long, simd_size_v<T, Abi>> __llround(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::llround(x[i]); });
    }

    template <class T> simd<T, Abi> __trunc(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::trunc(x[i]); });
    }

    template <class T> simd<T, Abi> __fmod(const simd<T, Abi> &x, const simd<T, Abi> &y)
    {
        return simd<T, Abi>([&](auto i) { return std::fmod(x[i], y[i]); });
    }

    template <class T>
    simd<T, Abi> __remainder(const simd<T, Abi> &x, const simd<T, Abi> &y)
    {
        return simd<T, Abi>([&](auto i) { return std::remainder(x[i], y[i]); });
    }

    template <class T>
    simd<T, Abi> __remquo(const simd<T, Abi> &x, const simd<T, Abi> &y,
                          fixed_size_simd<int, simd_size_v<T, Abi>> &z)
    {
        return simd<T, Abi>([&](auto i) {
            int tmp;
            T r = std::remquo(x[i], y[i], &tmp);
            z[i] = tmp;
            return r;
        });
    }

    template <class T>
    simd<T, Abi> __copysign(const simd<T, Abi> &x, const simd<T, Abi> &y)
    {
        return simd<T, Abi>([&](auto i) { return std::copysign(x[i], y[i]); });
    }

    template <class T>
    simd<T, Abi> __nextafter(const simd<T, Abi> &x, const simd<T, Abi> &y)
    {
        return simd<T, Abi>([&](auto i) { return std::nextafter(x[i], y[i]); });
    }

    template <class T> simd<T, Abi> __fdim(const simd<T, Abi> &x, const simd<T, Abi> &y)
    {
        return simd<T, Abi>([&](auto i) { return std::fdim(x[i], y[i]); });
    }

    template <class T> simd<T, Abi> __fmax(const simd<T, Abi> &x, const simd<T, Abi> &y)
    {
        return simd<T, Abi>([&](auto i) { return std::fmax(x[i], y[i]); });
    }

    template <class T> simd<T, Abi> __fmin(const simd<T, Abi> &x, const simd<T, Abi> &y)
    {
        return simd<T, Abi>([&](auto i) { return std::fmin(x[i], y[i]); });
    }

    template <class T>
    simd<T, Abi> __fma(const simd<T, Abi> &x, const simd<T, Abi> &y,
                       const simd<T, Abi> &z)
    {
        return simd<T, Abi>([&](auto i) { return std::fma(x[i], y[i], z[i]); });
    }

    template <class T>
    fixed_size_simd<int, simd_size_v<T, Abi>> __fpclassify(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::fpclassify(x[i]); });
    }

    template <class T> simd_mask<T, Abi> __isfinite(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::isfinite(x[i]); });
    }

    template <class T> simd_mask<T, Abi> __isinf(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::isinf(x[i]); });
    }

    template <class T> simd_mask<T, Abi> __isnan(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::isnan(x[i]); });
    }

    template <class T> simd_mask<T, Abi> __isnormal(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::isnormal(x[i]); });
    }

    template <class T> simd_mask<T, Abi> __signbit(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::signbit(x[i]); });
    }

    template <class T> simd_mask<T, Abi> __isgreater(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::isgreater(x[i]); });
    }

    template <class T> simd_mask<T, Abi> __isgreaterequal(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::isgreaterequal(x[i]); });
    }

    template <class T> simd_mask<T, Abi> __isless(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::isless(x[i]); });
    }

    template <class T> simd_mask<T, Abi> __islessequal(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::islessequal(x[i]); });
    }

    template <class T> simd_mask<T, Abi> __islessgreater(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::islessgreater(x[i]); });
    }

    template <class T> simd_mask<T, Abi> __isunordered(const simd<T, Abi> &x)
    {
        return simd<T, Abi>([&](auto i) { return std::isunordered(x[i]); });
    }
};  // }}}
// scalar_simd_impl {{{
struct scalar_simd_impl : simd_math_fallback<simd_abi::scalar> {
    // member types {{{2
    using abi = std::experimental::simd_abi::scalar;
    using mask_member_type = bool;
    template <class T> using simd_member_type = T;
    template <class T> using simd = std::experimental::simd<T, abi>;
    template <class T> using simd_mask = std::experimental::simd_mask<T, abi>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    template <class T> _GLIBCXX_SIMD_INTRINSIC static constexpr T broadcast(T x) noexcept
    {
        return x;
    }

    // generator {{{2
    template <class F, class T>
    _GLIBCXX_SIMD_INTRINSIC static T generator(F &&gen, type_tag<T>)
    {
        return gen(size_constant<0>());
    }

    // load {{{2
    template <class T, class U, class F>
    static inline T load(const U *mem, F, type_tag<T>) noexcept
    {
        return static_cast<T>(mem[0]);
    }

    // masked load {{{2
    template <class T, class U, class F>
    static inline T masked_load(T merge, bool k, const U *mem, F) noexcept
    {
        if (k) {
            merge = static_cast<T>(mem[0]);
        }
        return merge;
    }

    // store {{{2
    template <class T, class U, class F>
    static inline void store(T v, U *mem, F, type_tag<T>) noexcept
    {
        mem[0] = static_cast<T>(v);
    }

    // masked store {{{2
    template <class T, class U, class F>
    static inline void masked_store(const T v, U *mem, F, const bool k) noexcept
    {
        if (k) {
            mem[0] = v;
        }
    }

    // negation {{{2
    template <class T> static inline bool negate(T x) noexcept { return !x; }

    // reductions {{{2
    template <class T, class BinaryOperation>
    static inline T reduce(const simd<T> &x, BinaryOperation &)
    {
        return x.d;
    }

    // min, max, clamp {{{2
    template <class T> static inline T min(const T a, const T b)
    {
        return std::min(a, b);
    }

    template <class T> static inline T max(const T a, const T b)
    {
        return std::max(a, b);
    }

    // complement {{{2
    template <class T> static inline T complement(T x) noexcept
    {
        return static_cast<T>(~x);
    }

    // unary minus {{{2
    template <class T> static inline T unary_minus(T x) noexcept
    {
        return static_cast<T>(-x);
    }

    // arithmetic operators {{{2
    template <class T> static inline T plus(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) +
                              detail::promote_preserving_unsigned(y));
    }

    template <class T> static inline T minus(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) -
                              detail::promote_preserving_unsigned(y));
    }

    template <class T> static inline T multiplies(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) *
                              detail::promote_preserving_unsigned(y));
    }

    template <class T> static inline T divides(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) /
                              detail::promote_preserving_unsigned(y));
    }

    template <class T> static inline T modulus(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) %
                              detail::promote_preserving_unsigned(y));
    }

    template <class T> static inline T bit_and(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) &
                              detail::promote_preserving_unsigned(y));
    }
    static inline float bit_and(float x, float y)
    {
        static_assert(sizeof(float) == sizeof(uint), "");
        const uint r = reinterpret_cast<const may_alias<uint> &>(x) &
                       reinterpret_cast<const may_alias<uint> &>(y);
        return reinterpret_cast<const may_alias<float> &>(r);
    }
    static inline double bit_and(double x, double y)
    {
        static_assert(sizeof(double) == sizeof(ullong), "");
        const ullong r = reinterpret_cast<const may_alias<ullong> &>(x) &
                         reinterpret_cast<const may_alias<ullong> &>(y);
        return reinterpret_cast<const may_alias<double> &>(r);
    }

    template <class T> static inline T bit_or(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) |
                              detail::promote_preserving_unsigned(y));
    }
    static inline float bit_or(float x, float y)
    {
        static_assert(sizeof(float) == sizeof(uint), "");
        const uint r = reinterpret_cast<const may_alias<uint> &>(x) |
                       reinterpret_cast<const may_alias<uint> &>(y);
        return reinterpret_cast<const may_alias<float> &>(r);
    }
    static inline double bit_or(double x, double y)
    {
        static_assert(sizeof(double) == sizeof(ullong), "");
        const ullong r = reinterpret_cast<const may_alias<ullong> &>(x) |
                         reinterpret_cast<const may_alias<ullong> &>(y);
        return reinterpret_cast<const may_alias<double> &>(r);
    }


    template <class T> static inline T bit_xor(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) ^
                              detail::promote_preserving_unsigned(y));
    }

    template <class T> static inline T bit_shift_left(T x, int y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) << y);
    }

    template <class T> static inline T bit_shift_right(T x, int y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) >> y);
    }

    // math {{{2
    template <class T> _GLIBCXX_SIMD_INTRINSIC static T __abs(T x) { return T(std::abs(x)); }
    template <class T> _GLIBCXX_SIMD_INTRINSIC static T __sqrt(T x) { return std::sqrt(x); }
    template <class T> _GLIBCXX_SIMD_INTRINSIC static T __trunc(T x) { return std::trunc(x); }
    template <class T> _GLIBCXX_SIMD_INTRINSIC static T __floor(T x) { return std::floor(x); }
    template <class T> _GLIBCXX_SIMD_INTRINSIC static T __ceil(T x) { return std::ceil(x); }

    template <class T> _GLIBCXX_SIMD_INTRINSIC static simd_tuple<int, abi> __fpclassify(T x)
    {
        return {std::fpclassify(x)};
    }
    template <class T> _GLIBCXX_SIMD_INTRINSIC static bool __isfinite(T x) { return std::isfinite(x); }
    template <class T> _GLIBCXX_SIMD_INTRINSIC static bool __isinf(T x) { return std::isinf(x); }
    template <class T> _GLIBCXX_SIMD_INTRINSIC static bool __isnan(T x) { return std::isnan(x); }
    template <class T> _GLIBCXX_SIMD_INTRINSIC static bool __isnormal(T x) { return std::isnormal(x); }
    template <class T> _GLIBCXX_SIMD_INTRINSIC static bool __signbit(T x) { return std::signbit(x); }
    template <class T> _GLIBCXX_SIMD_INTRINSIC static bool __isunordered(T x, T y) { return std::isunordered(x, y); }

    // increment & decrement{{{2
    template <class T> static inline void increment(T &x) { ++x; }
    template <class T> static inline void decrement(T &x) { --x; }

    // compares {{{2
    template <class T> static bool equal_to(T x, T y) { return x == y; }
    template <class T> static bool not_equal_to(T x, T y) { return x != y; }
    template <class T> static bool less(T x, T y) { return x < y; }
    template <class T> static bool greater(T x, T y) { return x > y; }
    template <class T> static bool less_equal(T x, T y) { return x <= y; }
    template <class T> static bool greater_equal(T x, T y) { return x >= y; }

    // smart_reference access {{{2
    template <class T, class U> static void set(T &v, int i, U &&x) noexcept
    {
        _GLIBCXX_SIMD_ASSERT(i == 0);
        unused(i);
        v = std::forward<U>(x);
    }

    // masked_assign {{{2
    template <typename T> _GLIBCXX_SIMD_INTRINSIC static void masked_assign(bool k, T &lhs, T rhs)
    {
        if (k) {
            lhs = rhs;
        }
    }

    // masked_cassign {{{2
    template <template <typename> class Op, typename T>
    _GLIBCXX_SIMD_INTRINSIC static void masked_cassign(const bool k, T &lhs, const T rhs)
    {
        if (k) {
            lhs = Op<T>{}(lhs, rhs);
        }
    }

    // masked_unary {{{2
    template <template <typename> class Op, typename T>
    _GLIBCXX_SIMD_INTRINSIC static T masked_unary(const bool k, const T v)
    {
        return static_cast<T>(k ? Op<T>{}(v) : v);
    }

    // }}}2
};

// }}}
// scalar_mask_impl {{{
struct scalar_mask_impl {
    // member types {{{2
    template <class T> using simd_mask = std::experimental::simd_mask<T, simd_abi::scalar>;
    template <class T> using type_tag = T *;

    // from_bitset {{{2
    template <class T>
    _GLIBCXX_SIMD_INTRINSIC static bool from_bitset(std::bitset<1> bs, type_tag<T>) noexcept
    {
        return bs[0];
    }

    // masked load {{{2
    template <class F>
    _GLIBCXX_SIMD_INTRINSIC static bool masked_load(bool merge, bool mask, const bool *mem,
                                         F) noexcept
    {
        if (mask) {
            merge = mem[0];
        }
        return merge;
    }

    // store {{{2
    template <class F> _GLIBCXX_SIMD_INTRINSIC static void store(bool v, bool *mem, F) noexcept
    {
        mem[0] = v;
    }

    // masked store {{{2
    template <class F>
    _GLIBCXX_SIMD_INTRINSIC static void masked_store(const bool v, bool *mem, F,
                                          const bool k) noexcept
    {
        if (k) {
            mem[0] = v;
        }
    }

    // logical and bitwise operators {{{2
    static constexpr bool logical_and(bool x, bool y) { return x && y; }
    static constexpr bool logical_or(bool x, bool y) { return x || y; }
    static constexpr bool bit_and(bool x, bool y) { return x && y; }
    static constexpr bool bit_or(bool x, bool y) { return x || y; }
    static constexpr bool bit_xor(bool x, bool y) { return x != y; }

    // smart_reference access {{{2
    static void set(bool &k, int i, bool x) noexcept
    {
        _GLIBCXX_SIMD_ASSERT(i == 0);
        detail::unused(i);
        k = x;
    }

    // masked_assign {{{2
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(bool k, bool &lhs, bool rhs)
    {
        if (k) {
            lhs = rhs;
        }
    }

    // }}}2
};

// }}}

// ISA & type detection {{{1
template <class T, size_t N> constexpr bool is_sse_ps()
{
    return have_sse && std::is_same_v<T, float> && N == 4;
}
template <class T, size_t N> constexpr bool is_sse_pd()
{
    return have_sse2 && std::is_same_v<T, double> && N == 2;
}
template <class T, size_t N> constexpr bool is_avx_ps()
{
    return have_avx && std::is_same_v<T, float> && N == 8;
}
template <class T, size_t N> constexpr bool is_avx_pd()
{
    return have_avx && std::is_same_v<T, double> && N == 4;
}
template <class T, size_t N> constexpr bool is_avx512_ps()
{
    return have_avx512f && std::is_same_v<T, float> && N == 16;
}
template <class T, size_t N> constexpr bool is_avx512_pd()
{
    return have_avx512f && std::is_same_v<T, double> && N == 8;
}

template <class T, size_t N> constexpr bool is_neon_ps()
{
    return have_neon && std::is_same_v<T, float> && N == 4;
}
template <class T, size_t N> constexpr bool is_neon_pd()
{
    return have_neon && std::is_same_v<T, double> && N == 2;
}

// generic_simd_impl {{{1
template <class Abi> struct generic_simd_impl : simd_math_fallback<Abi> {
    // member types {{{2
    template <class T> using type_tag = T *;
    template <class T>
    using simd_member_type = typename Abi::template traits<T>::simd_member_type;
    template <class T>
    using mask_member_type = typename Abi::template traits<T>::mask_member_type;
    template <class T> static constexpr size_t full_size = simd_member_type<T>::width;

    // make_simd(Storage) {{{2
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static simd<T, Abi> make_simd(Storage<T, N> x)
    {
        return {detail::private_init, x};
    }

    // broadcast {{{2
    template <class T>
    _GLIBCXX_SIMD_INTRINSIC static constexpr simd_member_type<T> broadcast(T x) noexcept
    {
        return simd_member_type<T>::broadcast(x);
    }

    // generator {{{2
    template <class F, class T>
    _GLIBCXX_SIMD_INTRINSIC static simd_member_type<T> generator(F &&gen, type_tag<T>)
    {
        return detail::generate_storage<T, full_size<T>>(std::forward<F>(gen));
    }

    // load {{{2
    template <class T, class U, class F>
    _GLIBCXX_SIMD_INTRINSIC static simd_member_type<T> load(const U *mem, F,
                                                 type_tag<T>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        constexpr size_t N = simd_member_type<T>::width;
        constexpr size_t max_load_size =
            (sizeof(U) >= 4 && have_avx512f) || have_avx512bw
                ? 64
                : (std::is_floating_point_v<U> && have_avx) || have_avx2 ? 32 : 16;
        if constexpr (sizeof(U) > 8) {
            return detail::generate_storage<T, N>(
                [&](auto i) { return static_cast<T>(mem[i]); });
        } else if constexpr (std::is_same_v<U, T>) {
            return detail::builtin_load<U, N>(mem, F());
        } else if constexpr (sizeof(U) * N < 16) {
            return x86::convert<simd_member_type<T>>(
                detail::builtin_load16<U, sizeof(U) * N>(mem, F()));
        } else if constexpr (sizeof(U) * N <= max_load_size) {
            return x86::convert<simd_member_type<T>>(detail::builtin_load<U, N>(mem, F()));
        } else if constexpr (sizeof(U) * N == 2 * max_load_size) {
            return x86::convert<simd_member_type<T>>(
                detail::builtin_load<U, N / 2>(mem, F()),
                detail::builtin_load<U, N / 2>(mem + N / 2, F()));
        } else if constexpr (sizeof(U) * N == 4 * max_load_size) {
            return x86::convert<simd_member_type<T>>(
                detail::builtin_load<U, N / 4>(mem, F()),
                detail::builtin_load<U, N / 4>(mem + 1 * N / 4, F()),
                detail::builtin_load<U, N / 4>(mem + 2 * N / 4, F()),
                detail::builtin_load<U, N / 4>(mem + 3 * N / 4, F()));
        } else if constexpr (sizeof(U) * N == 8 * max_load_size) {
            return x86::convert<simd_member_type<T>>(
                detail::builtin_load<U, N / 8>(mem, F()),
                detail::builtin_load<U, N / 8>(mem + 1 * N / 8, F()),
                detail::builtin_load<U, N / 8>(mem + 2 * N / 8, F()),
                detail::builtin_load<U, N / 8>(mem + 3 * N / 8, F()),
                detail::builtin_load<U, N / 8>(mem + 4 * N / 8, F()),
                detail::builtin_load<U, N / 8>(mem + 5 * N / 8, F()),
                detail::builtin_load<U, N / 8>(mem + 6 * N / 8, F()),
                detail::builtin_load<U, N / 8>(mem + 7 * N / 8, F()));
        } else {
            assert_unreachable<T>();
        }
    }

    // masked load {{{2
    template <class T, size_t N, class U, class F>
    static inline detail::Storage<T, N> masked_load(detail::Storage<T, N> merge, mask_member_type<T> k,
                                   const U *mem, F) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        if constexpr (std::is_same_v<T, U> ||  // no conversion
                      (sizeof(T) == sizeof(U) &&
                       std::is_integral_v<T> ==
                           std::is_integral_v<U>)  // conversion via bit reinterpretation
        ) {
            constexpr bool have_avx512bw_vl_or_zmm =
                have_avx512bw_vl || (have_avx512bw && sizeof(merge) == 64);
            if constexpr (have_avx512bw_vl_or_zmm && sizeof(T) == 1) {
                if constexpr (sizeof(merge) == 16) {
                    merge = _mm_mask_loadu_epi8(merge, _mm_movemask_epi8(k), mem);
                } else if constexpr (sizeof(merge) == 32) {
                    merge = _mm256_mask_loadu_epi8(merge, _mm256_movemask_epi8(k), mem);
                } else if constexpr (sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_epi8(merge, k, mem);
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (have_avx512bw_vl_or_zmm && sizeof(T) == 2) {
                if constexpr (sizeof(merge) == 16) {
                    merge = _mm_mask_loadu_epi16(merge, movemask_epi16(k), mem);
                } else if constexpr (sizeof(merge) == 32) {
                    merge = _mm256_mask_loadu_epi16(merge, movemask_epi16(k), mem);
                } else if constexpr (sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_epi16(merge, k, mem);
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (have_avx2 && sizeof(T) == 4 && std::is_integral_v<U>) {
                if constexpr (sizeof(merge) == 16) {
                    merge =
                        (~k.d & merge.d) | builtin_cast<T>(_mm_maskload_epi32(
                                               reinterpret_cast<const int *>(mem), k));
                } else if constexpr (sizeof(merge) == 32) {
                    merge =
                        (~k.d & merge.d) | builtin_cast<T>(_mm256_maskload_epi32(
                                               reinterpret_cast<const int *>(mem), k));
                } else if constexpr (have_avx512f && sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_epi32(merge, k, mem);
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (have_avx && sizeof(T) == 4) {
                if constexpr (sizeof(merge) == 16) {
                    merge = or_(andnot_(k.d, merge.d),
                                builtin_cast<T>(
                                    _mm_maskload_ps(reinterpret_cast<const float *>(mem),
                                                    builtin_cast<llong>(k))));
                } else if constexpr (sizeof(merge) == 32) {
                    merge = or_(andnot_(k.d, merge.d),
                                _mm256_maskload_ps(reinterpret_cast<const float *>(mem),
                                                   builtin_cast<llong>(k)));
                } else if constexpr (have_avx512f && sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_ps(merge, k, mem);
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (have_avx2 && sizeof(T) == 8 && std::is_integral_v<U>) {
                if constexpr (sizeof(merge) == 16) {
                    merge =
                        (~k.d & merge.d) | builtin_cast<T>(_mm_maskload_epi64(
                                               reinterpret_cast<const llong *>(mem), k));
                } else if constexpr (sizeof(merge) == 32) {
                    merge =
                        (~k.d & merge.d) | builtin_cast<T>(_mm256_maskload_epi64(
                                               reinterpret_cast<const llong *>(mem), k));
                } else if constexpr (have_avx512f && sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_epi64(merge, k, mem);
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (have_avx && sizeof(T) == 8) {
                if constexpr (sizeof(merge) == 16) {
                    merge = or_(andnot_(k.d, merge.d),
                                builtin_cast<T>(_mm_maskload_pd(
                                    reinterpret_cast<const double *>(mem), builtin_cast<llong>(k))));
                } else if constexpr (sizeof(merge) == 32) {
                    merge = or_(andnot_(k.d, merge.d),
                                _mm256_maskload_pd(reinterpret_cast<const double *>(mem),
                                                   builtin_cast<llong>(k)));
                } else if constexpr (have_avx512f && sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_pd(merge, k, mem);
                } else {
                    assert_unreachable<T>();
                }
            } else {
                detail::bit_iteration(to_bitset(k.d).to_ullong(), [&](auto i) {
                    merge.set(i, static_cast<T>(mem[i]));
                });
            }
        } else if constexpr (sizeof(U) <= 8 &&  // no long double
                             !detail::converts_via_decomposition_v<
                                 U, T, sizeof(merge)>  // conversion via decomposition is
                                                       // better handled via the
                                                       // bit_iteration fallback below
        ) {
            // TODO: copy pattern from masked_store, which doesn't resort to fixed_size
            using A = simd_abi::deduce_t<
                U, std::max(N, 16 / sizeof(U))  // N or more, so that at least a 16 Byte
                                                // vector is used instead of a fixed_size
                                                // filled with scalars
                >;
            using ATraits = detail::traits<U, A>;
            using AImpl = typename ATraits::simd_impl_type;
            typename ATraits::simd_member_type uncvted{};
            typename ATraits::mask_member_type kk;
            if constexpr (detail::is_fixed_size_abi_v<A>) {
                kk = detail::to_bitset(k.d);
            } else {
                kk = convert_mask<typename ATraits::mask_member_type>(k);
            }
            uncvted = AImpl::masked_load(uncvted, kk, mem, F());
            detail::simd_converter<U, A, T, Abi> converter;
            masked_assign(k, merge, converter(uncvted));
        } else {
            detail::bit_iteration(to_bitset(k.d).to_ullong(),
                                  [&](auto i) { merge.set(i, static_cast<T>(mem[i])); });
        }
        return merge;
    }

    // store {{{2
    template <class T, class U, class F>
    _GLIBCXX_SIMD_INTRINSIC static void store(simd_member_type<T> v, U *mem, F,
                                   type_tag<T>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        // TODO: converting int -> "smaller int" can be optimized with AVX512
        constexpr size_t N = simd_member_type<T>::width;
        constexpr size_t max_store_size =
            (sizeof(U) >= 4 && have_avx512f) || have_avx512bw
                ? 64
                : (std::is_floating_point_v<U> && have_avx) || have_avx2 ? 32 : 16;
        if constexpr (sizeof(U) > 8) {
            detail::execute_n_times<N>([&](auto i) { mem[i] = v[i]; });
        } else if constexpr (std::is_same_v<U, T>) {
            detail::builtin_store(v.d, mem, F());
        } else if constexpr (sizeof(U) * N < 16) {
            detail::builtin_store<sizeof(U) * N>(x86::convert<builtin_type16_t<U>>(v),
                                                 mem, F());
        } else if constexpr (sizeof(U) * N <= max_store_size) {
            detail::builtin_store(x86::convert<builtin_type_t<U, N>>(v), mem, F());
        } else {
            constexpr size_t VSize = max_store_size / sizeof(U);
            constexpr size_t stores = N / VSize;
            using V = builtin_type_t<U, VSize>;
            const std::array<V, stores> converted = x86::convert_all<V>(v);
            detail::execute_n_times<stores>([&](auto i) {
                detail::builtin_store(converted[i], mem + i * VSize, F());
            });
        }
    }

    // masked store {{{2
    template <class T, size_t N, class U, class F>
    static inline void masked_store(const Storage<T, N> v, U *mem, F,
                                    const mask_member_type<T> k) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        constexpr size_t max_store_size =
            (sizeof(U) >= 4 && have_avx512f) || have_avx512bw
                ? 64
                : (std::is_floating_point_v<U> && have_avx) || have_avx2 ? 32 : 16;
        if constexpr (std::is_same_v<T, U> ||
                      (std::is_integral_v<T> && std::is_integral_v<U> &&
                       sizeof(T) == sizeof(U))) {
            // bitwise or no conversion, reinterpret:
            const auto kk = [&]() {
                if constexpr (detail::is_bitmask_v<decltype(k)>) {
                    return mask_member_type<U>(k.d);
                } else {
                    return detail::storage_bitcast<U>(k);
                }
            }();
            x86::maskstore(storage_bitcast<U>(v), mem, F(), kk);
        } else if constexpr (std::is_integral_v<T> && std::is_integral_v<U> &&
                             sizeof(T) > sizeof(U) && have_avx512f &&
                             (sizeof(T) >= 4 || have_avx512bw) &&
                             (sizeof(v) == 64 || have_avx512vl)) {  // truncating store
            const auto kk = [&]() {
                if constexpr (detail::is_bitmask_v<decltype(k)>) {
                    return k;
                } else {
                    return convert_mask<Storage<bool, N>>(k);
                }
            }();
            if constexpr (sizeof(T) == 8 && sizeof(U) == 4) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi32(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi32(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi64_storeu_epi32(mem, kk, v);
                }
            } else if constexpr (sizeof(T) == 8 && sizeof(U) == 2) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi16(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi16(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi64_storeu_epi16(mem, kk, v);
                }
            } else if constexpr (sizeof(T) == 8 && sizeof(U) == 1) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi64_storeu_epi8(mem, kk, v);
                }
            } else if constexpr (sizeof(T) == 4 && sizeof(U) == 2) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi32_storeu_epi16(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi32_storeu_epi16(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi32_storeu_epi16(mem, kk, v);
                }
            } else if constexpr (sizeof(T) == 4 && sizeof(U) == 1) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi32_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi32_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi32_storeu_epi8(mem, kk, v);
                }
            } else if constexpr (sizeof(T) == 2 && sizeof(U) == 1) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi16_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi16_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi16_storeu_epi8(mem, kk, v);
                }
            } else {
                assert_unreachable<T>();
            }
        } else if constexpr (sizeof(U) <= 8 &&  // no long double
                             !detail::converts_via_decomposition_v<
                                 T, U, max_store_size>  // conversion via decomposition is
                                                        // better handled via the
                                                        // bit_iteration fallback below
        ) {
            using VV = Storage<U, std::clamp(N, 16 / sizeof(U), max_store_size / sizeof(U))>;
            using V = typename VV::register_type;
            constexpr bool prefer_bitmask =
                (have_avx512f && sizeof(U) >= 4) || have_avx512bw;
            using M = Storage<std::conditional_t<prefer_bitmask, bool, U>, VV::width>;
            constexpr size_t VN = builtin_traits<V>::width;

            if constexpr (VN >= N) {
                x86::maskstore(VV(convert<V>(v)), mem,
                               // careful, if V has more elements than the input v (N),
                               // vector_aligned is incorrect:
                               std::conditional_t<(builtin_traits<V>::width > N),
                                                  overaligned_tag<sizeof(U) * N>, F>(),
                               convert_mask<M>(k));
            } else if constexpr (VN * 2 == N) {
                const std::array<V, 2> converted = x86::convert_all<V>(v);
                x86::maskstore(VV(converted[0]), mem, F(), convert_mask<M>(detail::extract_part<0, 2>(k)));
                x86::maskstore(VV(converted[1]), mem + VV::width, F(), convert_mask<M>(detail::extract_part<1, 2>(k)));
            } else if constexpr (VN * 4 == N) {
                const std::array<V, 4> converted = x86::convert_all<V>(v);
                x86::maskstore(VV(converted[0]), mem, F(), convert_mask<M>(detail::extract_part<0, 4>(k)));
                x86::maskstore(VV(converted[1]), mem + 1 * VV::width, F(), convert_mask<M>(detail::extract_part<1, 4>(k)));
                x86::maskstore(VV(converted[2]), mem + 2 * VV::width, F(), convert_mask<M>(detail::extract_part<2, 4>(k)));
                x86::maskstore(VV(converted[3]), mem + 3 * VV::width, F(), convert_mask<M>(detail::extract_part<3, 4>(k)));
            } else if constexpr (VN * 8 == N) {
                const std::array<V, 8> converted = x86::convert_all<V>(v);
                x86::maskstore(VV(converted[0]), mem, F(), convert_mask<M>(detail::extract_part<0, 8>(k)));
                x86::maskstore(VV(converted[1]), mem + 1 * VV::width, F(), convert_mask<M>(detail::extract_part<1, 8>(k)));
                x86::maskstore(VV(converted[2]), mem + 2 * VV::width, F(), convert_mask<M>(detail::extract_part<2, 8>(k)));
                x86::maskstore(VV(converted[3]), mem + 3 * VV::width, F(), convert_mask<M>(detail::extract_part<3, 8>(k)));
                x86::maskstore(VV(converted[4]), mem + 4 * VV::width, F(), convert_mask<M>(detail::extract_part<4, 8>(k)));
                x86::maskstore(VV(converted[5]), mem + 5 * VV::width, F(), convert_mask<M>(detail::extract_part<5, 8>(k)));
                x86::maskstore(VV(converted[6]), mem + 6 * VV::width, F(), convert_mask<M>(detail::extract_part<6, 8>(k)));
                x86::maskstore(VV(converted[7]), mem + 7 * VV::width, F(), convert_mask<M>(detail::extract_part<7, 8>(k)));
            } else {
                assert_unreachable<T>();
            }
        } else {
            detail::bit_iteration(to_bitset(k.d).to_ullong(),
                                  [&](auto i) { mem[i] = static_cast<U>(v[i]); });
        }
    }

    // complement {{{2
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> complement(Storage<T, N> x) noexcept
    {
        return detail::x86::complement(x);
    }

    // unary minus {{{2
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> unary_minus(Storage<T, N> x) noexcept
    {
        return detail::x86::unary_minus(x);
    }

    // arithmetic operators {{{2
#define _GLIBCXX_SIMD_ARITHMETIC_OP_(name_)                                                         \
    template <class T, size_t N>                                                         \
    _GLIBCXX_SIMD_INTRINSIC static Storage<T, N> name_(Storage<T, N> x, Storage<T, N> y)            \
    {                                                                                    \
        return detail::x86::name_(x, y);                                                 \
    }                                                                                    \
    _GLIBCXX_SIMD_NOTHING_EXPECTING_SEMICOLON
    _GLIBCXX_SIMD_ARITHMETIC_OP_(plus);
    _GLIBCXX_SIMD_ARITHMETIC_OP_(minus);
    _GLIBCXX_SIMD_ARITHMETIC_OP_(multiplies);
#undef _GLIBCXX_SIMD_ARITHMETIC_OP_
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> divides(Storage<T, N> x, Storage<T, N> y)
    {
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX4
        return detail::divides(x.d, y.d);
#else
        return x.d / y.d;
#endif
    }
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> modulus(Storage<T, N> x, Storage<T, N> y)
    {
        static_assert(std::is_integral<T>::value, "modulus is only supported for integral types");
        return x.d % y.d;
    }
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> bit_and(Storage<T, N> x, Storage<T, N> y)
    {
        return builtin_cast<T>(builtin_cast<llong>(x.d) & builtin_cast<llong>(y.d));
    }
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> bit_or(Storage<T, N> x, Storage<T, N> y)
    {
        return builtin_cast<T>(builtin_cast<llong>(x.d) | builtin_cast<llong>(y.d));
    }
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> bit_xor(Storage<T, N> x, Storage<T, N> y)
    {
        return builtin_cast<T>(builtin_cast<llong>(x.d) ^ builtin_cast<llong>(y.d));
    }
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static Storage<T, N> bit_shift_left(Storage<T, N> x, Storage<T, N> y)
    {
        return x.d << y.d;
    }
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static Storage<T, N> bit_shift_right(Storage<T, N> x, Storage<T, N> y)
    {
        return x.d >> y.d;
    }

    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> bit_shift_left(Storage<T, N> x, int y)
    {
        return x.d << y;
    }
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> bit_shift_right(Storage<T, N> x,
                                                                         int y)
    {
        return x.d >> y;
    }

    // compares {{{2
    // equal_to {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr mask_member_type<T> equal_to(Storage<T, N> x,
                                                               Storage<T, N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_EQ_OQ);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ);
                } else { assert_unreachable<T>(); }
            } else {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmpeq_epi64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmpeq_epi32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmpeq_epi16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmpeq_epi8_mask(x, y);
                } else { assert_unreachable<T>(); }
            }
        } else {
            return to_storage(x.d == y.d);
        }
    }

    // not_equal_to {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr mask_member_type<T> not_equal_to(Storage<T, N> x,
                                                                   Storage<T, N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_NEQ_UQ);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_NEQ_UQ);
                } else { assert_unreachable<T>(); }
            } else {
                       if constexpr (sizeof(T) == 8) { return ~_mm512_cmpeq_epi64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return ~_mm512_cmpeq_epi32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return ~_mm512_cmpeq_epi16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return ~_mm512_cmpeq_epi8_mask(x, y);
                } else { assert_unreachable<T>(); }
            }
        } else {
            return to_storage(x.d != y.d);
        }
    }

    // less {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr mask_member_type<T> less(Storage<T, N> x,
                                                           Storage<T, N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_LT_OS);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_LT_OS);
                } else { assert_unreachable<T>(); }
            } else if constexpr (std::is_signed_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmplt_epi64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmplt_epi32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmplt_epi16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmplt_epi8_mask(x, y);
                } else { assert_unreachable<T>(); }
            } else {
                static_assert(std::is_unsigned_v<T>);
                       if constexpr (sizeof(T) == 8) { return _mm512_cmplt_epu64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmplt_epu32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmplt_epu16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmplt_epu8_mask(x, y);
                } else { assert_unreachable<T>(); }
            }
        } else {
            return to_storage(x.d < y.d);
        }
    }

    // less_equal {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr mask_member_type<T> less_equal(Storage<T, N> x,
                                                                 Storage<T, N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_LE_OS);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_LE_OS);
                } else { assert_unreachable<T>(); }
            } else if constexpr (std::is_signed_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmple_epi64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmple_epi32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmple_epi16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmple_epi8_mask(x, y);
                } else { assert_unreachable<T>(); }
            } else {
                static_assert(std::is_unsigned_v<T>);
                       if constexpr (sizeof(T) == 8) { return _mm512_cmple_epu64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmple_epu32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmple_epu16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmple_epu8_mask(x, y);
                } else { assert_unreachable<T>(); }
            }
        } else {
            return to_storage(x.d <= y.d);
        }
    }

    // negation {{{2
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr mask_member_type<T> negate(Storage<T, N> x) noexcept
    {
        if constexpr (detail::is_abi<Abi, simd_abi::__avx512_abi>()) {
            return equal_to(x, simd_member_type<T>());
        } else {
            return detail::to_storage(!x.d);
        }
    }

    // min, max, clamp {{{2
    template <class T, size_t N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> min(Storage<T, N> a,
                                                                   Storage<T, N> b)
    {
        return a.d < b.d ? a.d : b.d;
    }
    template <class T, size_t N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> max(Storage<T, N> a,
                                                                   Storage<T, N> b)
    {
        return a.d > b.d ? a.d : b.d;
    }

    template <class T, size_t N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr std::pair<Storage<T, N>, Storage<T, N>>
    minmax(Storage<T, N> a, Storage<T, N> b)
    {
        return {a.d < b.d ? a.d : b.d, a.d < b.d ? b.d : a.d};
    }

    // reductions {{{2
    template <class T, class BinaryOperation>
    _GLIBCXX_SIMD_INTRINSIC static T reduce(simd<T, Abi> x, BinaryOperation &&binary_op)
    {
        constexpr size_t N = simd_size_v<T, Abi>;
        if constexpr (sizeof(x) > 16) {
            using A = simd_abi::deduce_t<T, N / 2>;
            using V = std::experimental::simd<T, A>;
            return traits<T, A>::simd_impl_type::reduce(
                binary_op(V(detail::private_init, extract<0, 2>(data(x).d)),
                          V(detail::private_init, extract<1, 2>(data(x).d))),
                std::forward<BinaryOperation>(binary_op));
        } else {
            const auto &xx = x.d;
            if constexpr (N == 16) {
                x = binary_op(make_simd<T, N>(_mm_unpacklo_epi8(xx, xx)),
                              make_simd<T, N>(_mm_unpackhi_epi8(xx, xx)));
            }
            if constexpr (N >= 8) {
                x = binary_op(make_simd<T, N>(_mm_unpacklo_epi16(xx, xx)),
                              make_simd<T, N>(_mm_unpackhi_epi16(xx, xx)));
            }
            if constexpr (N >= 4) {
                using U = std::conditional_t<std::is_floating_point_v<T>, float, int>;
                const auto y = builtin_cast<U>(xx.d);
                x = binary_op(x, make_simd<T, N>(to_storage(
                                     builtin_type_t<U, 4>{y[3], y[2], y[1], y[0]})));
            }
            const auto y = builtin_cast<llong>(xx.d);
            return binary_op(
                x, make_simd<T, N>(to_storage(builtin_type_t<llong, 2>{y[1], y[1]})))[0];
        }
    }

    // math {{{2
    // sqrt {{{3
    template <class T, size_t N> _GLIBCXX_SIMD_INTRINSIC static Storage<T, N> __sqrt(Storage<T, N> x)
    {
               if constexpr (is_sse_ps   <T, N>()) { return _mm_sqrt_ps(x);
        } else if constexpr (is_sse_pd   <T, N>()) { return _mm_sqrt_pd(x);
        } else if constexpr (is_avx_ps   <T, N>()) { return _mm256_sqrt_ps(x);
        } else if constexpr (is_avx_pd   <T, N>()) { return _mm256_sqrt_pd(x);
        } else if constexpr (is_avx512_ps<T, N>()) { return _mm512_sqrt_ps(x);
        } else if constexpr (is_avx512_pd<T, N>()) { return _mm512_sqrt_pd(x);
        } else { assert_unreachable<T>(); }
    }

    // abs {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static Storage<T, N> __abs(Storage<T, N> x) noexcept
    {
        return detail::x86::abs(x);
    }

    // trunc {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static Storage<T, N> __trunc(Storage<T, N> x)
    {
        if constexpr (is_avx512_ps<T, N>()) {
            return _mm512_roundscale_round_ps(x, 0x03, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx512_pd<T, N>()) {
            return _mm512_roundscale_round_pd(x, 0x03, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx_ps<T, N>()) {
            return _mm256_round_ps(x, 0x3);
        } else if constexpr (is_avx_pd<T, N>()) {
            return _mm256_round_pd(x, 0x3);
        } else if constexpr (have_sse4_1 && is_sse_ps<T, N>()) {
            return _mm_round_ps(x, 0x3);
        } else if constexpr (have_sse4_1 && is_sse_pd<T, N>()) {
            return _mm_round_pd(x, 0x3);
        } else if constexpr (is_sse_ps<T, N>()) {
            auto truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
            const auto no_fractional_values = builtin_cast<float>(
                builtin_cast<int>(builtin_cast<uint>(x.d) & 0x7f800000u) <
                0x4b000000);  // the exponent is so large that no mantissa bits signify
                              // fractional values (0x3f8 + 23*8 = 0x4b0)
            return x86::blend(no_fractional_values, x, truncated);
        } else if constexpr (is_sse_pd<T, N>()) {
            const auto abs_x = __abs(x).d;
            const auto min_no_fractional_bits = builtin_cast<double>(
                builtin_broadcast<2>(0x4330'0000'0000'0000ull));  // 0x3ff + 52 = 0x433
            builtin_type16_t<double> truncated =
                (abs_x + min_no_fractional_bits) - min_no_fractional_bits;
            // due to rounding, the result can be too large. In this case `truncated >
            // abs(x)` holds, so subtract 1 to truncated if `abs(x) < truncated`
            truncated -=
                and_(builtin_cast<double>(abs_x < truncated), builtin_broadcast<2>(1.));
            // finally, fix the sign bit:
            return or_(
                and_(builtin_cast<double>(builtin_broadcast<2>(0x8000'0000'0000'0000ull)),
                     x),
                truncated);
        } else {
            assert_unreachable<T>();
        }
    }

    // floor {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static Storage<T, N> __floor(Storage<T, N> x)
    {
        if constexpr (is_avx512_ps<T, N>()) {
            return _mm512_roundscale_round_ps(x, 0x01, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx512_pd<T, N>()) {
            return _mm512_roundscale_round_pd(x, 0x01, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx_ps<T, N>()) {
            return _mm256_round_ps(x, 0x1);
        } else if constexpr (is_avx_pd<T, N>()) {
            return _mm256_round_pd(x, 0x1);
        } else if constexpr (have_sse4_1 && is_sse_ps<T, N>()) {
            return _mm_floor_ps(x);
        } else if constexpr (have_sse4_1 && is_sse_pd<T, N>()) {
            return _mm_floor_pd(x);
        } else {
            const auto y = __trunc(x).d;
            const auto negative_input = builtin_cast<T>(x.d < builtin_broadcast<N, T>(0));
            const auto mask = andnot_(builtin_cast<T>(y == x.d), negative_input);
            return or_(andnot_(mask, y), and_(mask, y - builtin_broadcast<N, T>(1)));
        }
    }

    // ceil {{{3
    template <class T, size_t N> _GLIBCXX_SIMD_INTRINSIC static Storage<T, N> __ceil(Storage<T, N> x)
    {
        if constexpr (is_avx512_ps<T, N>()) {
            return _mm512_roundscale_round_ps(x, 0x02, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx512_pd<T, N>()) {
            return _mm512_roundscale_round_pd(x, 0x02, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx_ps<T, N>()) {
            return _mm256_round_ps(x, 0x2);
        } else if constexpr (is_avx_pd<T, N>()) {
            return _mm256_round_pd(x, 0x2);
        } else if constexpr (have_sse4_1 && is_sse_ps<T, N>()) {
            return _mm_ceil_ps(x);
        } else if constexpr (have_sse4_1 && is_sse_pd<T, N>()) {
            return _mm_ceil_pd(x);
        } else {
            const auto y = __trunc(x).d;
            const auto negative_input = builtin_cast<T>(x.d < builtin_broadcast<N, T>(0));
            const auto inv_mask = or_(builtin_cast<T>(y == x.d), negative_input);
            return or_(and_(inv_mask, y),
                       andnot_(inv_mask, y + builtin_broadcast<N, T>(1)));
        }
    }

    // isnan {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static mask_member_type<T> __isnan(Storage<T, N> x)
    {
             if constexpr (is_sse_ps   <T, N>()) { return _mm_cmpunord_ps(x, x); }
        else if constexpr (is_avx_ps   <T, N>()) { return _mm256_cmp_ps(x, x, _CMP_UNORD_Q); }
        else if constexpr (is_avx512_ps<T, N>()) { return _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q); }
        else if constexpr (is_sse_pd   <T, N>()) { return _mm_cmpunord_pd(x, x); }
        else if constexpr (is_avx_pd   <T, N>()) { return _mm256_cmp_pd(x, x, _CMP_UNORD_Q); }
        else if constexpr (is_avx512_pd<T, N>()) { return _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q); }
        else { assert_unreachable<T>(); }
    }

    // isfinite {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static mask_member_type<T> __isfinite(Storage<T, N> x)
    {
        return x86::cmpord(x, x.d * T());
    }

    // isunordered {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static mask_member_type<T> __isunordered(Storage<T, N> x,
                                                          Storage<T, N> y)
    {
        return x86::cmpunord(x, y);
    }

    // signbit {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static mask_member_type<T> __signbit(Storage<T, N> x)
    {
        using I = int_for_sizeof_t<T>;
        if constexpr (have_avx512dq && is_avx512_ps<T, N>()) {
            return _mm512_movepi32_mask(builtin_cast<llong>(x));
        } else if constexpr (have_avx512dq && is_avx512_pd<T, N>()) {
            return _mm512_movepi64_mask(builtin_cast<llong>(x));
        } else if constexpr (sizeof(x) == 64) {
            const auto signmask = builtin_broadcast<N>(std::numeric_limits<I>::min());
            return equal_to(Storage<I, N>(builtin_cast<I>(x.d) & signmask),
                            Storage<I, N>(signmask));
        } else {
            const auto xx = builtin_cast<I>(x.d);
            constexpr I signmask = std::numeric_limits<I>::min();
            if constexpr ((sizeof(T) == 4 && (have_avx2 || sizeof(x) == 16)) ||
                          have_avx512vl) {
                (void)signmask;
                return builtin_cast<T>(xx >> std::numeric_limits<I>::digits);
            } else if constexpr ((have_avx2 || (have_ssse3 && sizeof(x) == 16))) {
                return builtin_cast<T>((xx & signmask) == signmask);
            } else {  // SSE2/3 or AVX (w/o AVX2)
                constexpr auto one = builtin_broadcast<N, T>(1);
                return builtin_cast<T>(
                    builtin_cast<T>((xx & signmask) | builtin_cast<I>(one))  // -1 or 1
                    != one);
            }
        }
    }

    // isnonzerovalue (isnormal | is subnormal == !isinf & !isnan & !is zero) {{{3
    template <class T> _GLIBCXX_SIMD_INTRINSIC static auto isnonzerovalue(T x)
    {
        using U = typename builtin_traits<T>::value_type;
        return x86::cmpord(x * std::numeric_limits<U>::infinity(),  // NaN if x == 0
                           x * U()                                  // NaN if x == inf
        );
    }

    template <class T> _GLIBCXX_SIMD_INTRINSIC static auto isnonzerovalue_mask(T x)
    {
        using U = typename builtin_traits<T>::value_type;
        constexpr size_t N = builtin_traits<T>::width;
        const auto a = x * std::numeric_limits<U>::infinity();  // NaN if x == 0
        const auto b = x * U();                                 // NaN if x == inf
        if constexpr (have_avx512vl && is_sse_ps<U, N>()) {
            return _mm_cmp_ps_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (have_avx512f && is_sse_ps<U, N>()) {
            return __mmask8(0xf &
                            _mm512_cmp_ps_mask(auto_cast(a), auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (have_avx512vl && is_sse_pd<U, N>()) {
            return _mm_cmp_pd_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (have_avx512f && is_sse_pd<U, N>()) {
            return __mmask8(0x3 &
                            _mm512_cmp_pd_mask(auto_cast(a), auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (have_avx512vl && is_avx_ps<U, N>()) {
            return _mm256_cmp_ps_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (have_avx512f && is_avx_ps<U, N>()) {
            return __mmask8(_mm512_cmp_ps_mask(auto_cast(a), auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (have_avx512vl && is_avx_pd<U, N>()) {
            return _mm256_cmp_pd_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (have_avx512f && is_avx_pd<U, N>()) {
            return __mmask8(0xf &
                            _mm512_cmp_pd_mask(auto_cast(a), auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (is_avx512_ps<U, N>()) {
            return _mm512_cmp_ps_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (is_avx512_pd<U, N>()) {
            return _mm512_cmp_pd_mask(a, b, _CMP_ORD_Q);
        } else {
            assert_unreachable<T>();
        }
    }

    // isinf {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static mask_member_type<T> __isinf(Storage<T, N> x)
    {
        if constexpr (is_avx512_pd<T, N>()) {
            if constexpr (have_avx512dq) {
                return _mm512_fpclass_pd_mask(x, 0x08) | _mm512_fpclass_pd_mask(x, 0x10);
            } else {
                return _mm512_cmp_epi64_mask(builtin_cast<llong>(x86::abs(x)),
                                             builtin_broadcast<N>(0x7ff0000000000000ll),
                                             _CMP_EQ_OQ);
            }
        } else if constexpr (is_avx512_ps<T, N>()) {
            if constexpr (have_avx512dq) {
                return _mm512_fpclass_ps_mask(x, 0x08) | _mm512_fpclass_ps_mask(x, 0x10);
            } else {
                return _mm512_cmp_epi32_mask(builtin_cast<llong>(x86::abs(x)),
                                             auto_cast(builtin_broadcast<N>(0x7f800000u)),
                                             _CMP_EQ_OQ);
            }
        } else if constexpr (have_avx512dq_vl) {
            if constexpr (is_sse_pd<T, N>()) {
                return builtin_cast<double>(_mm_movm_epi64(_mm_fpclass_pd_mask(x, 0x08) |
                                                           _mm_fpclass_pd_mask(x, 0x10)));
            } else if constexpr (is_avx_pd<T, N>()) {
                return builtin_cast<double>(_mm256_movm_epi64(
                    _mm256_fpclass_pd_mask(x, 0x08) | _mm256_fpclass_pd_mask(x, 0x10)));
            } else if constexpr (is_sse_ps<T, N>()) {
                return builtin_cast<float>(_mm_movm_epi32(_mm_fpclass_ps_mask(x, 0x08) |
                                                          _mm_fpclass_ps_mask(x, 0x10)));
            } else if constexpr (is_avx_ps<T, N>()) {
                return builtin_cast<float>(_mm256_movm_epi32(
                    _mm256_fpclass_ps_mask(x, 0x08) | _mm256_fpclass_ps_mask(x, 0x10)));
            } else {
                assert_unreachable<T>();
            }
        } else {
            // compares to inf using the corresponding integer type
            return builtin_cast<T>(builtin_cast<int_for_sizeof_t<T>>(__abs(x).d) ==
                                   builtin_cast<int_for_sizeof_t<T>>(builtin_broadcast<N>(
                                       std::numeric_limits<T>::infinity())));
            // alternative:
            //return builtin_cast<T>(__abs(x).d > std::numeric_limits<T>::max());
        }
    }
    // isnormal {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static mask_member_type<T> __isnormal(Storage<T, N> x)
    {
        // subnormals -> 0
        // 0 -> 0
        // inf -> inf
        // -inf -> inf
        // nan -> inf
        // normal value -> positive value / not 0
        return isnonzerovalue(
            and_(x.d, builtin_broadcast<N>(std::numeric_limits<T>::infinity())));
    }

    // fpclassify {{{3
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static fixed_size_storage<int, N> __fpclassify(Storage<T, N> x)
    {
        if constexpr (is_avx512_pd<T, N>()) {
            // AVX512 is special because we want to use an __mmask to blend int vectors
            // (coming from double vectors). GCC doesn't allow this combination on the
            // ternary operator. Thus, resort to intrinsics:
            if constexpr (have_avx512vl) {
                auto &&b = [](int y) { return to_intrin(builtin_broadcast<N>(y)); };
                return {_mm256_mask_mov_epi32(
                    _mm256_mask_mov_epi32(
                        _mm256_mask_mov_epi32(b(FP_NORMAL), __isnan(x), b(FP_NAN)),
                        __isinf(x), b(FP_INFINITE)),
                    _mm512_cmp_pd_mask(
                        __abs(x),
                        builtin_broadcast<N>(std::numeric_limits<double>::min()),
                        _CMP_LT_OS),
                    _mm256_mask_mov_epi32(
                        b(FP_SUBNORMAL),
                        _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                        b(FP_ZERO)))};
            } else {
                auto &&b = [](int y) {
                    return _mm512_castsi256_si512(to_intrin(builtin_broadcast<N>(y)));
                };
                return {lo256(_mm512_mask_mov_epi32(
                    _mm512_mask_mov_epi32(
                        _mm512_mask_mov_epi32(b(FP_NORMAL), __isnan(x), b(FP_NAN)),
                        __isinf(x), b(FP_INFINITE)),
                    _mm512_cmp_pd_mask(
                        __abs(x),
                        builtin_broadcast<N>(std::numeric_limits<double>::min()),
                        _CMP_LT_OS),
                    _mm512_mask_mov_epi32(
                        b(FP_SUBNORMAL),
                        _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                        b(FP_ZERO))))};
            }
        } else {
            constexpr auto fp_normal =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_NORMAL));
            constexpr auto fp_nan =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_NAN));
            constexpr auto fp_infinite =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_INFINITE));
            constexpr auto fp_subnormal =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_SUBNORMAL));
            constexpr auto fp_zero =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_ZERO));

            const auto tmp = builtin_cast<llong>(
                __abs(x).d < std::numeric_limits<T>::min()
                    ? (x.d == 0 ? fp_zero : fp_subnormal)
                    : x86::blend(__isinf(x).d, x86::blend(__isnan(x).d, fp_normal, fp_nan),
                                 fp_infinite));
            if constexpr (std::is_same_v<T, float>) {
                if constexpr (fixed_size_storage<int, N>::tuple_size == 1) {
                    return {tmp};
                } else if constexpr (fixed_size_storage<int, N>::tuple_size == 2) {
                    return {extract<0, 2>(tmp), extract<1, 2>(tmp)};
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (is_sse_pd<T, N>()) {
                static_assert(fixed_size_storage<int, N>::tuple_size == 2);
                return {_mm_cvtsi128_si32(tmp),
                        {_mm_cvtsi128_si32(_mm_unpackhi_epi64(tmp, tmp))}};
            } else if constexpr (is_avx_pd<T, N>()) {
                static_assert(fixed_size_storage<int, N>::tuple_size == 1);
                return {_mm_packs_epi32(lo128(tmp), hi128(tmp))};
            } else {
                assert_unreachable<T>();
            }
        }
    }

    // increment & decrement{{{2
    template <class T, size_t N> _GLIBCXX_SIMD_INTRINSIC static void increment(Storage<T, N> &x)
    {
        x = plus(x, Storage<T, N>::broadcast(T(1)));
    }
    template <class T, size_t N> _GLIBCXX_SIMD_INTRINSIC static void decrement(Storage<T, N> &x)
    {
        x = minus(x, Storage<T, N>::broadcast(T(1)));
    }

    // smart_reference access {{{2
    template <class T, size_t N, class U>
    _GLIBCXX_SIMD_INTRINSIC static void set(Storage<T, N> &v, int i, U &&x) noexcept
    {
        v.set(i, std::forward<U>(x));
    }

    // masked_assign{{{2
    template <class T, class K, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(Storage<K, N> k, Storage<T, N> &lhs,
                                           detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(k.d, lhs.d, rhs.d);
    }

    template <class T, class K, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(Storage<K, N> k, Storage<T, N> &lhs,
                                           detail::id<T> rhs)
    {
        if (__builtin_constant_p(rhs) && rhs == 0 && std::is_same<K, T>::value) {
            if constexpr (!detail::is_bitmask(k)) {
                // the andnot_ optimization only makes sense if k.d is a vector register
                lhs.d = andnot_(k.d, lhs.d);
                return;
            } else {
                // for AVX512/__mmask, a _mm512_maskz_mov is best
                lhs = detail::x86::blend(k, lhs, intrinsic_type_t<T, N>());
                return;
            }
        }
        lhs = detail::x86::blend(k.d, lhs.d, builtin_broadcast<N>(rhs));
    }

    // masked_cassign {{{2
    template <template <typename> class Op, class T, class K, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_cassign(const Storage<K, N> k, Storage<T, N> &lhs,
                                            const detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(
            k.d, lhs.d, detail::data(Op<void>{}(make_simd(lhs), make_simd(rhs))).d);
    }

    template <template <typename> class Op, class T, class K, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_cassign(const Storage<K, N> k, Storage<T, N> &lhs,
                                            const detail::id<T> rhs)
    {
        lhs = detail::x86::blend(
            k.d, lhs.d,
            detail::data(
                Op<void>{}(make_simd(lhs), make_simd<T, N>(builtin_broadcast<N>(rhs))))
                .d);
    }

    // masked_unary {{{2
    template <template <typename> class Op, class T, class K, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static Storage<T, N> masked_unary(const Storage<K, N> k,
                                                            const Storage<T, N> v)
    {
        auto vv = make_simd(v);
        Op<decltype(vv)> op;
        return detail::x86::blend(k, v, detail::data(op(vv)));
    }

    //}}}2
};

// generic_mask_impl {{{1
template <class Abi> struct generic_mask_impl {
    // member types {{{2
    template <class T> using type_tag = T *;
    template <class T> using simd_mask = std::experimental::simd_mask<T, Abi>;
    template <class T>
    using simd_member_type = typename Abi::template traits<T>::simd_member_type;
    template <class T>
    using mask_member_type = typename Abi::template traits<T>::mask_member_type;

    // masked load {{{2
    template <class T, size_t N, class F>
    static inline Storage<T, N> masked_load(Storage<T, N> merge, Storage<T, N> mask,
                                            const bool *mem, F) noexcept
    {
        if constexpr (detail::is_abi<Abi, simd_abi::__avx512_abi>()) {
            if constexpr (detail::have_avx512bw_vl) {
                if constexpr (N == 8) {
                    const auto a = _mm_mask_loadu_epi8(__m128i(), mask, mem);
                    return (merge & ~mask) | _mm_test_epi8_mask(a, a);
                } else if constexpr (N == 16) {
                    const auto a = _mm_mask_loadu_epi8(__m128i(), mask, mem);
                    return (merge & ~mask) | _mm_test_epi8_mask(a, a);
                } else if constexpr (N == 32) {
                    const auto a = _mm256_mask_loadu_epi8(__m256i(), mask, mem);
                    return (merge & ~mask) | _mm256_test_epi8_mask(a, a);
                } else if constexpr (N == 64) {
                    const auto a = _mm512_mask_loadu_epi8(__m512i(), mask, mem);
                    return (merge & ~mask) | _mm512_test_epi8_mask(a, a);
                } else {
                    assert_unreachable<T>();
                }
            } else {
                detail::bit_iteration(mask, [&](auto i) { merge.set(i, mem[i]); });
                return merge;
            }
        } else if constexpr (have_avx512bw_vl && N == 32 && sizeof(T) == 1) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(
                _mm256_mask_sub_epi8(builtin_cast<llong>(merge), k, __m256i(),
                                     _mm256_mask_loadu_epi8(__m256i(), k, mem)));
        } else if constexpr (have_avx512bw_vl && N == 16 && sizeof(T) == 1) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm_mask_sub_epi8(builtin_cast<llong>(merge), k, __m128i(),
                                                 _mm_mask_loadu_epi8(__m128i(), k, mem)));
        } else if constexpr (have_avx512bw_vl && N == 16 && sizeof(T) == 2) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm256_mask_sub_epi16(
                builtin_cast<llong>(merge), k, __m256i(),
                _mm256_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 8 && sizeof(T) == 2) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm_mask_sub_epi16(
                builtin_cast<llong>(merge), k, __m128i(),
                _mm_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 8 && sizeof(T) == 4) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm256_mask_sub_epi32(
                builtin_cast<llong>(merge), k, __m256i(),
                _mm256_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 4 && sizeof(T) == 4) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm_mask_sub_epi32(
                builtin_cast<llong>(merge), k, __m128i(),
                _mm_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 4 && sizeof(T) == 8) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm256_mask_sub_epi64(
                builtin_cast<llong>(merge), k, __m256i(),
                _mm256_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 2 && sizeof(T) == 8) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm_mask_sub_epi64(
                builtin_cast<llong>(merge), k, __m128i(),
                _mm_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else {
            // AVX(2) has 32/64 bit maskload, but nothing at 8 bit granularity
            auto tmp = storage_bitcast<detail::int_for_sizeof_t<T>>(merge);
            detail::bit_iteration(to_bitset(mask.d).to_ullong(),
                                  [&](auto i) { tmp.set(i, -mem[i]); });
            merge = storage_bitcast<T>(tmp);
        }
        return merge;
    }

    // store {{{2
    template <class T, size_t N, class F>
    _GLIBCXX_SIMD_INTRINSIC static void store(Storage<T, N> v, bool *mem, F) noexcept
    {
        if constexpr (detail::is_abi<Abi, simd_abi::__sse_abi>()) {
            if constexpr (N == 2 && have_sse2) {
                const auto k = builtin_cast<int>(v.d);
                mem[0] = -k[1];
                mem[1] = -k[3];
            } else if constexpr (N == 4 && have_sse2) {
                const unsigned bool4 =
                    builtin_cast<uint>(_mm_packs_epi16(
                        _mm_packs_epi32(builtin_cast<llong>(v), __m128i()), __m128i()))[0] &
                    0x01010101u;
                std::memcpy(mem, &bool4, 4);
            } else if constexpr (std::is_same_v<T, float> && have_mmx) {
                const __m128 k(v);
                const __m64 kk = _mm_cvtps_pi8(and_(k, _mm_set1_ps(1.f)));
                builtin_store<4>(kk, mem, F());
                _mm_empty();
            } else if constexpr (N == 8 && have_sse2) {
                builtin_store<8>(
                    _mm_packs_epi16(to_intrin(builtin_cast<ushort>(v.d) >> 15),
                                    __m128i()),
                    mem, F());
            } else if constexpr (N == 16 && have_sse2) {
                builtin_store(v.d & 1, mem, F());
            } else {
                assert_unreachable<T>();
            }
        } else if constexpr (detail::is_abi<Abi, simd_abi::__avx_abi>()) {
            if constexpr (N == 4 && have_avx) {
                auto k = builtin_cast<llong>(v);
                int bool4;
                if constexpr (have_avx2) {
                    bool4 = _mm256_movemask_epi8(k);
                } else {
                    bool4 = (_mm_movemask_epi8(lo128(k)) |
                             (_mm_movemask_epi8(hi128(k)) << 16));
                }
                bool4 &= 0x01010101;
                std::memcpy(mem, &bool4, 4);
            } else if constexpr (N == 8 && have_avx) {
                const auto k = builtin_cast<llong>(v);
                const auto k2 = _mm_srli_epi16(_mm_packs_epi16(lo128(k), hi128(k)), 15);
                const auto k3 = _mm_packs_epi16(k2, __m128i());
                builtin_store<8>(k3, mem, F());
            } else if constexpr (N == 16 && have_avx2) {
                const auto x = _mm256_srli_epi16(v, 15);
                const auto bools = _mm_packs_epi16(lo128(x), hi128(x));
                builtin_store<16>(bools, mem, F());
            } else if constexpr (N == 16 && have_avx) {
                const auto bools = 1 & builtin_cast<uchar>(_mm_packs_epi16(
                                           lo128(v.intrin()), hi128(v.intrin())));
                builtin_store<16>(bools, mem, F());
            } else if constexpr (N == 32 && have_avx) {
                builtin_store<32>(1 & v.d, mem, F());
            } else {
                assert_unreachable<T>();
            }
        } else if constexpr (detail::is_abi<Abi, simd_abi::__avx512_abi>()) {
            if constexpr (N == 8) {
                builtin_store<8>(
#if defined _GLIBCXX_SIMD_HAVE_AVX512VL && defined _GLIBCXX_SIMD_HAVE_AVX512BW
                    _mm_maskz_set1_epi8(v, 1),
#elif defined __x86_64__
                    make_storage<ullong>(_pdep_u64(v, 0x0101010101010101ULL), 0ull),
#else
                    make_storage<uint>(_pdep_u32(v, 0x01010101U),
                                       _pdep_u32(v >> 4, 0x01010101U)),
#endif
                    mem, F());
            } else if constexpr (N == 16 && detail::have_avx512bw_vl) {
                builtin_store(_mm_maskz_set1_epi8(v, 1), mem, F());
            } else if constexpr (N == 16 && detail::have_avx512f) {
                _mm512_mask_cvtepi32_storeu_epi8(mem, ~__mmask16(),
                                                 _mm512_maskz_set1_epi32(v, 1));
            } else if constexpr (N == 32 && detail::have_avx512bw_vl) {
                builtin_store(_mm256_maskz_set1_epi8(v, 1), mem, F());
            } else if constexpr (N == 32 && detail::have_avx512bw) {
                builtin_store(lo256(_mm512_maskz_set1_epi8(v, 1)), mem, F());
            } else if constexpr (N == 64 && detail::have_avx512bw) {
                builtin_store(_mm512_maskz_set1_epi8(v, 1), mem, F());
            } else {
                assert_unreachable<T>();
            }
        } else {
            assert_unreachable<T>();
        }
    }

    // masked store {{{2
    template <class T, size_t N, class F>
    static inline void masked_store(const Storage<T, N> v, bool *mem, F,
                                    const Storage<T, N> k) noexcept
    {
        if constexpr (detail::is_abi<Abi, simd_abi::__avx512_abi>()) {
            if constexpr (N == 8 && detail::have_avx512bw_vl) {
                _mm_mask_cvtepi16_storeu_epi8(mem, k, _mm_maskz_set1_epi16(v, 1));
            } else if constexpr (N == 8 && detail::have_avx512vl) {
                _mm256_mask_cvtepi32_storeu_epi8(mem, k, _mm256_maskz_set1_epi32(v, 1));
            } else if constexpr (N == 8) {
                // we rely on k < 0x100:
                _mm512_mask_cvtepi32_storeu_epi8(mem, k, _mm512_maskz_set1_epi32(v, 1));
            } else if constexpr (N == 16 && detail::have_avx512bw_vl) {
                _mm_mask_storeu_epi8(mem, k, _mm_maskz_set1_epi8(v, 1));
            } else if constexpr (N == 16) {
                _mm512_mask_cvtepi32_storeu_epi8(mem, k, _mm512_maskz_set1_epi32(v, 1));
            } else if constexpr (N == 32 && detail::have_avx512bw_vl) {
                _mm256_mask_storeu_epi8(mem, k, _mm256_maskz_set1_epi8(v, 1));
            } else if constexpr (N == 32 && detail::have_avx512bw) {
                _mm256_mask_storeu_epi8(mem, k, lo256(_mm512_maskz_set1_epi8(v, 1)));
            } else if constexpr (N == 64 && detail::have_avx512bw) {
                _mm512_mask_storeu_epi8(mem, k, _mm512_maskz_set1_epi8(v, 1));
            } else {
                assert_unreachable<T>();
            }
        } else {
            detail::bit_iteration(to_bitset(k.d).to_ullong(), [&](auto i) { mem[i] = v[i]; });
        }
    }

    // from_bitset{{{2
    template <size_t N, class T>
    _GLIBCXX_SIMD_INTRINSIC static mask_member_type<T> from_bitset(std::bitset<N> bits, type_tag<T>)
    {
        return convert_mask<typename mask_member_type<T>::register_type>(bits);
    }

    // logical and bitwise operators {{{2
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> logical_and(const Storage<T, N> &x,
                                                            const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d & y.d;
        } else {
            return detail::and_(x.d, y.d);
        }
    }

    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> logical_or(const Storage<T, N> &x,
                                                           const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d | y.d;
        } else {
            return detail::or_(x.d, y.d);
        }
    }

    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> bit_and(const Storage<T, N> &x,
                                                        const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d & y.d;
        } else {
            return detail::and_(x.d, y.d);
        }
    }

    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> bit_or(const Storage<T, N> &x,
                                                       const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d | y.d;
        } else {
            return detail::or_(x.d, y.d);
        }
    }

    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr Storage<T, N> bit_xor(const Storage<T, N> &x,
                                                        const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d ^ y.d;
        } else {
            return detail::xor_(x.d, y.d);
        }
    }

    // smart_reference access {{{2
    template <class T, size_t N> static void set(Storage<T, N> &k, int i, bool x) noexcept
    {
        if constexpr (std::is_same_v<T, bool>) {
            k.set(i, x);
        } else {
            using int_t = builtin_type_t<int_for_sizeof_t<T>, N>;
            auto tmp = reinterpret_cast<int_t>(k.d);
            tmp[i] = -x;
            k.d = auto_cast(tmp);
        }
    }
    // masked_assign{{{2
    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(Storage<T, N> k, Storage<T, N> &lhs,
                                           detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(k.d, lhs.d, rhs.d);
    }

    template <class T, size_t N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(Storage<T, N> k, Storage<T, N> &lhs, bool rhs)
    {
        if (__builtin_constant_p(rhs)) {
            if (rhs == false) {
                lhs = andnot_(k.d, lhs.d);
            } else {
                lhs = or_(k.d, lhs.d);
            }
            return;
        }
        lhs = detail::x86::blend(k, lhs, detail::data(simd_mask<T>(rhs)));
    }

    //}}}2
};

//}}}1

struct sse_mask_impl : generic_mask_impl<simd_abi::__sse> {};
struct sse_simd_impl : generic_simd_impl<simd_abi::__sse> {};

struct avx_mask_impl : generic_mask_impl<simd_abi::__avx> {};
struct avx_simd_impl : generic_simd_impl<simd_abi::__avx> {};

struct avx512_simd_impl : generic_simd_impl<simd_abi::__avx512> {};
struct avx512_mask_impl : generic_mask_impl<simd_abi::__avx512> {};

struct neon_mask_impl : generic_mask_impl<simd_abi::__neon> {};
struct neon_simd_impl : generic_simd_impl<simd_abi::__neon> {};

/**
 * The fixed_size ABI gives the following guarantees:
 *  - simd objects are passed via the stack
 *  - memory layout of `simd<T, N>` is equivalent to `std::array<T, N>`
 *  - alignment of `simd<T, N>` is `N * sizeof(T)` if N is a power-of-2 value,
 *    otherwise `next_power_of_2(N * sizeof(T))` (Note: if the alignment were to
 *    exceed the system/compiler maximum, it is bounded to that maximum)
 *  - simd_mask objects are passed like std::bitset<N>
 *  - memory layout of `simd_mask<T, N>` is equivalent to `std::bitset<N>`
 *  - alignment of `simd_mask<T, N>` is equal to the alignment of `std::bitset<N>`
 */
// autocvt_to_simd {{{
template <class T, bool = std::is_arithmetic_v<std::decay_t<T>>>
struct autocvt_to_simd {
    T d;
    using TT = std::decay_t<T>;
    operator TT() { return d; }
    operator TT &()
    {
        static_assert(std::is_lvalue_reference<T>::value, "");
        static_assert(!std::is_const<T>::value, "");
        return d;
    }
    operator TT *()
    {
        static_assert(std::is_lvalue_reference<T>::value, "");
        static_assert(!std::is_const<T>::value, "");
        return &d;
    }

    constexpr inline autocvt_to_simd(T dd) : d(dd) {}

    template <class Abi> operator simd<typename TT::value_type, Abi>()
    {
        return {detail::private_init, d};
    }

    template <class Abi> operator simd<typename TT::value_type, Abi> &()
    {
        return *reinterpret_cast<simd<typename TT::value_type, Abi> *>(&d);
    }

    template <class Abi> operator simd<typename TT::value_type, Abi> *()
    {
        return reinterpret_cast<simd<typename TT::value_type, Abi> *>(&d);
    }
};
template <class T> autocvt_to_simd(T &&)->autocvt_to_simd<T>;

template <class T> struct autocvt_to_simd<T, true> {
    using TT = std::decay_t<T>;
    T d;
    fixed_size_simd<TT, 1> fd;

    constexpr inline autocvt_to_simd(T dd) : d(dd), fd(d) {}
    ~autocvt_to_simd()
    {
        d = detail::data(fd).first;
    }

    operator fixed_size_simd<TT, 1>()
    {
        return fd;
    }
    operator fixed_size_simd<TT, 1> &()
    {
        static_assert(std::is_lvalue_reference<T>::value, "");
        static_assert(!std::is_const<T>::value, "");
        return fd;
    }
    operator fixed_size_simd<TT, 1> *()
    {
        static_assert(std::is_lvalue_reference<T>::value, "");
        static_assert(!std::is_const<T>::value, "");
        return &fd;
    }
};

// }}}
// fixed_size_storage<T, N>{{{1
template <class T, int N, class Tuple,
          class Next = simd<T, all_native_abis::best_abi<T, N>>,
          int Remain = N - int(Next::size())>
struct fixed_size_storage_builder;

template <class T, int N>
struct fixed_size_storage_builder_wrapper
    : public fixed_size_storage_builder<T, N, simd_tuple<T>> {
};

template <class T, int N, class... As, class Next>
struct fixed_size_storage_builder<T, N, simd_tuple<T, As...>, Next, 0> {
    using type = simd_tuple<T, As..., typename Next::abi_type>;
};

template <class T, int N, class... As, class Next, int Remain>
struct fixed_size_storage_builder<T, N, simd_tuple<T, As...>, Next, Remain> {
    using type = typename fixed_size_storage_builder<
        T, Remain, simd_tuple<T, As..., typename Next::abi_type>>::type;
};

// n_abis_in_tuple {{{1
template <class T> struct seq_op;
template <size_t I0, size_t... Is> struct seq_op<std::index_sequence<I0, Is...>> {
    using first_plus_one = std::index_sequence<I0 + 1, Is...>;
    using notfirst_plus_one = std::index_sequence<I0, (Is + 1)...>;
    template <size_t First, size_t Add>
    using prepend = std::index_sequence<First, I0 + Add, (Is + Add)...>;
};

template <class T> struct n_abis_in_tuple;
template <class T> struct n_abis_in_tuple<simd_tuple<T>> {
    using counts = std::index_sequence<0>;
    using begins = std::index_sequence<0>;
};
template <class T, class A> struct n_abis_in_tuple<simd_tuple<T, A>> {
    using counts = std::index_sequence<1>;
    using begins = std::index_sequence<0>;
};
template <class T, class A0, class... As>
struct n_abis_in_tuple<simd_tuple<T, A0, A0, As...>> {
    using counts = typename seq_op<
        typename n_abis_in_tuple<simd_tuple<T, A0, As...>>::counts>::first_plus_one;
    using begins = typename seq_op<typename n_abis_in_tuple<
        simd_tuple<T, A0, As...>>::begins>::notfirst_plus_one;
};
template <class T, class A0, class A1, class... As>
struct n_abis_in_tuple<simd_tuple<T, A0, A1, As...>> {
    using counts = typename seq_op<typename n_abis_in_tuple<
        simd_tuple<T, A1, As...>>::counts>::template prepend<1, 0>;
    using begins = typename seq_op<typename n_abis_in_tuple<
        simd_tuple<T, A1, As...>>::begins>::template prepend<0, 1>;
};

namespace tests
{
static_assert(
    std::is_same<n_abis_in_tuple<simd_tuple<int, simd_abi::__sse, simd_abi::__sse,
                                                simd_abi::scalar, simd_abi::scalar,
                                                simd_abi::scalar>>::counts,
                 std::index_sequence<2, 3>>::value,
    "");
static_assert(
    std::is_same<n_abis_in_tuple<simd_tuple<int, simd_abi::__sse, simd_abi::__sse,
                                                simd_abi::scalar, simd_abi::scalar,
                                                simd_abi::scalar>>::begins,
                 std::index_sequence<0, 2>>::value,
    "");
}  // namespace tests

// tree_reduction {{{1
template <size_t Count, size_t Begin> struct tree_reduction {
    static_assert(Count > 0,
                  "tree_reduction requires at least one simd object to work with");
    template <class T, class... As, class BinaryOperation>
    auto operator()(const simd_tuple<T, As...> &tup,
                    const BinaryOperation &binary_op) const noexcept
    {
        constexpr size_t left = next_power_of_2(Count) / 2;
        constexpr size_t right = Count - left;
        return binary_op(tree_reduction<left, Begin>()(tup, binary_op),
                         tree_reduction<right, Begin + left>()(tup, binary_op));
    }
};
template <size_t Begin> struct tree_reduction<1, Begin> {
    template <class T, class... As, class BinaryOperation>
    auto operator()(const simd_tuple<T, As...> &tup, const BinaryOperation &) const
        noexcept
    {
        return detail::get_simd<Begin>(tup);
    }
};
template <size_t Begin> struct tree_reduction<2, Begin> {
    template <class T, class... As, class BinaryOperation>
    auto operator()(const simd_tuple<T, As...> &tup,
                    const BinaryOperation &binary_op) const noexcept
    {
        return binary_op(detail::get_simd<Begin>(tup),
                         detail::get_simd<Begin + 1>(tup));
    }
};

// vec_to_scalar_reduction {{{1
// This helper function implements the second step in a generic fixed_size reduction.
// -  Input: a tuple of native simd (or scalar) objects of decreasing size.
// - Output: a scalar (the reduction).
// - Approach:
//   1. reduce the first two tuple elements
//      a) If the number of elements differs by a factor of 2, split the first object into
//         two objects of the second type and reduce all three to one object of second
//         type.
//      b) If the number of elements differs by a factor of 4, split the first object into
//         two equally sized objects, reduce, and split to two objects of the second type.
//         Finally, reduce all three remaining objects to one object of second type.
//      c) Otherwise use std::experimental::reduce to reduce both inputs to a scalar, and binary_op to
//         reduce to a single scalar.
//
//      (This optimizes all native cases on x86, e.g. <AVX512, SSE, Scalar>.)
//
//   2. Concate the result of (1) with the remaining tuple elements to recurse into
//      vec_to_scalar_reduction.
//
//   3. If vec_to_scalar_reduction is called with a one-element tuple, call std::experimental::reduce to
//      reduce to a scalar and return.
template <class T, class A0, class A1, class BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC simd<T, A1> vec_to_scalar_reduction_first_pair(
    const simd<T, A0> left, const simd<T, A1> right, const BinaryOperation &binary_op,
    size_constant<2>) noexcept
{
    const std::array<simd<T, A1>, 2> splitted = split<simd<T, A1>>(left);
    return binary_op(binary_op(splitted[0], right), splitted[1]);
}

template <class T, class A0, class A1, class BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC simd<T, A1> vec_to_scalar_reduction_first_pair(
    const simd<T, A0> left, const simd<T, A1> right, const BinaryOperation &binary_op,
    size_constant<4>) noexcept
{
    constexpr auto N0 = simd_size_v<T, A0> / 2;
    const auto left2 = split<simd<T, simd_abi::deduce_t<T, N0>>>(left);
    const std::array<simd<T, A1>, 2> splitted =
        split<simd<T, A1>>(binary_op(left2[0], left2[1]));
    return binary_op(binary_op(splitted[0], right), splitted[1]);
}

template <class T, class A0, class A1, class BinaryOperation, size_t Factor>
_GLIBCXX_SIMD_INTRINSIC simd<T, simd_abi::scalar> vec_to_scalar_reduction_first_pair(
    const simd<T, A0> left, const simd<T, A1> right, const BinaryOperation &binary_op,
    size_constant<Factor>) noexcept
{
    return binary_op(std::experimental::reduce(left, binary_op), std::experimental::reduce(right, binary_op));
}

template <class T, class A0, class BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC T vec_to_scalar_reduction(const simd_tuple<T, A0> &tup,
                                       const BinaryOperation &binary_op) noexcept
{
    return std::experimental::reduce(simd<T, A0>(detail::private_init, tup.first), binary_op);
}

template <class T, class A0, class A1, class... As, class BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC T vec_to_scalar_reduction(const simd_tuple<T, A0, A1, As...> &tup,
                                       const BinaryOperation &binary_op) noexcept
{
    return vec_to_scalar_reduction(
        detail::tuple_concat(
            detail::make_tuple(
                vec_to_scalar_reduction_first_pair<T, A0, A1, BinaryOperation>(
                    {private_init, tup.first}, {private_init, tup.second.first},
                    binary_op, size_constant<simd_size_v<T, A0> / simd_size_v<T, A1>>())),
            tup.second.second),
        binary_op);
}

// partial_bitset_to_member_type {{{1
template <class V, size_t N>
_GLIBCXX_SIMD_INTRINSIC auto partial_bitset_to_member_type(std::bitset<N> shifted_bits)
{
    static_assert(V::size() <= N, "");
    using M = typename V::mask_type;
    using T = typename V::value_type;
    constexpr T *type_tag = nullptr;
    return detail::get_impl_t<M>::from_bitset(
        std::bitset<V::size()>(shifted_bits.to_ullong()), type_tag);
}

// fixed_size_simd_impl {{{1
template <int N> struct fixed_size_simd_impl {
    // member types {{{2
    using mask_member_type = std::bitset<N>;
    template <class T> using simd_member_type = fixed_size_storage<T, N>;
    template <class T>
    static constexpr std::size_t tuple_size = simd_member_type<T>::tuple_size;
    template <class T>
    static constexpr std::make_index_sequence<simd_member_type<T>::tuple_size> index_seq = {};
    template <class T> using simd = std::experimental::simd<T, simd_abi::fixed_size<N>>;
    template <class T> using simd_mask = std::experimental::simd_mask<T, simd_abi::fixed_size<N>>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    template <class T> static constexpr inline simd_member_type<T> broadcast(T x) noexcept
    {
        return simd_member_type<T>::generate(
            [&](auto meta) { return meta.broadcast(x); });
    }

    // generator {{{2
    template <class F, class T>
    _GLIBCXX_SIMD_INTRINSIC static simd_member_type<T> generator(F &&gen, type_tag<T>)
    {
        return simd_member_type<T>::generate([&gen](auto meta) {
            return meta.generator(
                [&](auto i_) {
                    return gen(size_constant<meta.offset + decltype(i_)::value>());
                },
                type_tag<T>());
        });
    }

    // load {{{2
    template <class T, class U, class F>
    static inline simd_member_type<T> load(const U *mem, F f,
                                              type_tag<T>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        return simd_member_type<T>::generate(
            [&](auto meta) { return meta.load(&mem[meta.offset], f, type_tag<T>()); });
    }

    // masked load {{{2
    template <class T, class... As, class U, class F>
    static inline simd_tuple<T, As...> masked_load(simd_tuple<T, As...> merge,
                                                   const mask_member_type bits,
                                                   const U *mem,
                                                   F f) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        detail::for_each(merge, [&](auto meta, auto &native) {
            native = meta.masked_load(native, meta.make_mask(bits), &mem[meta.offset], f);
        });
        return merge;
    }

    // store {{{2
    template <class T, class U, class F>
    static inline void store(const simd_member_type<T> v, U *mem, F f,
                             type_tag<T>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        detail::for_each(v, [&](auto meta, auto native) {
            meta.store(native, &mem[meta.offset], f, type_tag<T>());
        });
    }

    // masked store {{{2
    template <class T, class... As, class U, class F>
    static inline void masked_store(const simd_tuple<T, As...> v, U *mem, F f,
                                    const mask_member_type bits) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        detail::for_each(v, [&](auto meta, auto native) {
            meta.masked_store(native, &mem[meta.offset], f, meta.make_mask(bits));
        });
    }

    // negation {{{2
    template <class T, class... As>
    static inline mask_member_type negate(simd_tuple<T, As...> x) noexcept
    {
        mask_member_type bits = 0;
        for_each(x, [&bits](auto meta, auto native) {
            bits |= meta.mask_to_shifted_ullong(meta.negate(native));
        });
        return bits;
    }

    // reductions {{{2
private:
    template <class T, class... As, class BinaryOperation, size_t... Counts,
              size_t... Begins>
    static inline T reduce(const simd_tuple<T, As...> &tup,
                           const BinaryOperation &binary_op,
                           std::index_sequence<Counts...>, std::index_sequence<Begins...>)
    {
        // 1. reduce all tuple elements with equal ABI to a single element in the output
        // tuple
        const auto reduced_vec = detail::make_tuple(detail::tree_reduction<Counts, Begins>()(tup, binary_op)...);
        // 2. split and reduce until a scalar results
        return detail::vec_to_scalar_reduction(reduced_vec, binary_op);
    }

public:
    template <class T, class BinaryOperation>
    static inline T reduce(const simd<T> &x, const BinaryOperation &binary_op)
    {
        using ranges = n_abis_in_tuple<simd_member_type<T>>;
        return fixed_size_simd_impl::reduce(x.d, binary_op, typename ranges::counts(),
                                               typename ranges::begins());
    }

    // min, max, clamp {{{2
    template <class T, class... As>
    static inline simd_tuple<T, As...> min(const simd_tuple<T, As...> a,
                                              const simd_tuple<T, As...> b)
    {
        return simd_tuple_apply(
            [](auto impl, auto aa, auto bb) { return impl.min(aa, bb); }, a, b);
    }

    template <class T, class... As>
    static inline simd_tuple<T, As...> max(const simd_tuple<T, As...> a,
                                              const simd_tuple<T, As...> b)
    {
        return simd_tuple_apply(
            [](auto impl, auto aa, auto bb) { return impl.max(aa, bb); }, a, b);
    }

    // complement {{{2
    template <class T, class... As>
    static inline simd_tuple<T, As...> complement(simd_tuple<T, As...> x) noexcept
    {
        return simd_tuple_apply([](auto impl, auto xx) { return impl.complement(xx); },
                                x);
    }

    // unary minus {{{2
    template <class T, class... As>
    static inline simd_tuple<T, As...> unary_minus(simd_tuple<T, As...> x) noexcept
    {
        return simd_tuple_apply([](auto impl, auto xx) { return impl.unary_minus(xx); },
                                x);
    }

    // arithmetic operators {{{2

#define _GLIBCXX_SIMD_FIXED_OP(name_, op_)                                                          \
    template <class T, class... As>                                                      \
    static inline simd_tuple<T, As...> name_(simd_tuple<T, As...> x,                     \
                                             simd_tuple<T, As...> y)                     \
    {                                                                                    \
        return simd_tuple_apply(                                                         \
            [](auto impl, auto xx, auto yy) { return impl.name_(xx, yy); }, x, y);       \
    }                                                                                    \
    _GLIBCXX_SIMD_NOTHING_EXPECTING_SEMICOLON

    _GLIBCXX_SIMD_FIXED_OP(plus, +);
    _GLIBCXX_SIMD_FIXED_OP(minus, -);
    _GLIBCXX_SIMD_FIXED_OP(multiplies, *);
    _GLIBCXX_SIMD_FIXED_OP(divides, /);
    _GLIBCXX_SIMD_FIXED_OP(modulus, %);
    _GLIBCXX_SIMD_FIXED_OP(bit_and, &);
    _GLIBCXX_SIMD_FIXED_OP(bit_or, |);
    _GLIBCXX_SIMD_FIXED_OP(bit_xor, ^);
    _GLIBCXX_SIMD_FIXED_OP(bit_shift_left, <<);
    _GLIBCXX_SIMD_FIXED_OP(bit_shift_right, >>);
#undef _GLIBCXX_SIMD_FIXED_OP

    template <class T, class... As>
    static inline simd_tuple<T, As...> bit_shift_left(simd_tuple<T, As...> x, int y)
    {
        return simd_tuple_apply(
            [y](auto impl, auto xx) { return impl.bit_shift_left(xx, y); }, x);
    }

    template <class T, class... As>
    static inline simd_tuple<T, As...> bit_shift_right(simd_tuple<T, As...> x,
                                                          int y)
    {
        return simd_tuple_apply(
            [y](auto impl, auto xx) { return impl.bit_shift_right(xx, y); }, x);
    }

    // math {{{2
#define _GLIBCXX_SIMD_APPLY_ON_TUPLE_(name_)                                                        \
    template <class T, class... As>                                                      \
    static inline simd_tuple<T, As...> __##name_(simd_tuple<T, As...> x) noexcept        \
    {                                                                                    \
        return simd_tuple_apply(                                                         \
            [](auto impl, auto xx) {                                                     \
                using V = typename decltype(impl)::simd_type;                            \
                return data(name_(V(private_init, xx)));                                 \
            },                                                                           \
            x);                                                                          \
    }                                                                                    \
    _GLIBCXX_SIMD_NOTHING_EXPECTING_SEMICOLON
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(sqrt);
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(abs);
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(trunc);
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(floor);
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(ceil);
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(sin);
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(cos);
#undef _GLIBCXX_SIMD_APPLY_ON_TUPLE_

    template <class T, class... As>
    static inline simd_tuple<T, As...> __frexp(const simd_tuple<T, As...> &x,
                                             fixed_size_storage<int, N> &exp) noexcept
    {
        return simd_tuple_apply(
            [](auto impl, const auto &a, auto &b) {
                return data(
                    impl.__frexp(typename decltype(impl)::simd_type(private_init, a),
                                 autocvt_to_simd(b)));
            },
            x, exp);
    }

    template <class T, class... As>
    static inline fixed_size_storage<int, N> __fpclassify(simd_tuple<T, As...> x) noexcept
    {
        return detail::optimize_tuple(x.template apply_r<int>(
            [](auto impl, auto xx) { return impl.__fpclassify(xx); }));
    }

#define _GLIBCXX_SIMD_TEST_ON_TUPLE_(name_)                                                         \
    template <class T, class... As>                                                      \
    static inline mask_member_type __##name_(simd_tuple<T, As...> x) noexcept            \
    {                                                                                    \
        return test([](auto impl, auto xx) { return impl.__##name_(xx); }, x);           \
    }
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isinf)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isfinite)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isnan)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isnormal)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(signbit)
#undef _GLIBCXX_SIMD_TEST_ON_TUPLE_

    // increment & decrement{{{2
    template <class... Ts> static inline void increment(simd_tuple<Ts...> &x)
    {
        for_each(x, [](auto meta, auto &native) { meta.increment(native); });
    }

    template <class... Ts> static inline void decrement(simd_tuple<Ts...> &x)
    {
        for_each(x, [](auto meta, auto &native) { meta.decrement(native); });
    }

    // compares {{{2
#define _GLIBCXX_SIMD_CMP_OPERATIONS(cmp_)                                                          \
    template <class T, class... As>                                                      \
    static inline mask_member_type cmp_(const simd_tuple<T, As...> &x,                   \
                                        const simd_tuple<T, As...> &y)                   \
    {                                                                                    \
        mask_member_type bits = 0;                                                       \
        detail::for_each(x, y, [&bits](auto meta, auto native_x, auto native_y) {        \
            bits |= meta.mask_to_shifted_ullong(meta.cmp_(native_x, native_y));          \
        });                                                                              \
        return bits;                                                                     \
    }
    _GLIBCXX_SIMD_CMP_OPERATIONS(equal_to)
    _GLIBCXX_SIMD_CMP_OPERATIONS(not_equal_to)
    _GLIBCXX_SIMD_CMP_OPERATIONS(less)
    _GLIBCXX_SIMD_CMP_OPERATIONS(greater)
    _GLIBCXX_SIMD_CMP_OPERATIONS(less_equal)
    _GLIBCXX_SIMD_CMP_OPERATIONS(greater_equal)
    _GLIBCXX_SIMD_CMP_OPERATIONS(isunordered)
#undef _GLIBCXX_SIMD_CMP_OPERATIONS

    // smart_reference access {{{2
    template <class T, class... As, class U>
    _GLIBCXX_SIMD_INTRINSIC static void set(simd_tuple<T, As...> &v, int i, U &&x) noexcept
    {
        v.set(i, std::forward<U>(x));
    }

    // masked_assign {{{2
    template <typename T, class... As>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(
        const mask_member_type bits, detail::simd_tuple<T, As...> &lhs,
        const detail::id<detail::simd_tuple<T, As...>> rhs)
    {
        detail::for_each(lhs, rhs, [&](auto meta, auto &native_lhs, auto native_rhs) {
            meta.masked_assign(meta.make_mask(bits), native_lhs, native_rhs);
        });
    }

    // Optimization for the case where the RHS is a scalar. No need to broadcast the
    // scalar to a simd first.
    template <typename T, class... As>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(const mask_member_type bits,
                                           detail::simd_tuple<T, As...> &lhs,
                                           const detail::id<T> rhs)
    {
        detail::for_each(lhs, [&](auto meta, auto &native_lhs) {
            meta.masked_assign(meta.make_mask(bits), native_lhs, rhs);
        });
    }

    // masked_cassign {{{2
    template <template <typename> class Op, typename T, class... As>
    static inline void masked_cassign(const mask_member_type bits,
                                      detail::simd_tuple<T, As...> &lhs,
                                      const detail::simd_tuple<T, As...> rhs)
    {
        detail::for_each(lhs, rhs, [&](auto meta, auto &native_lhs, auto native_rhs) {
            meta.template masked_cassign<Op>(meta.make_mask(bits), native_lhs,
                                             native_rhs);
        });
    }

    // Optimization for the case where the RHS is a scalar. No need to broadcast the
    // scalar to a simd
    // first.
    template <template <typename> class Op, typename T, class... As>
    static inline void masked_cassign(const mask_member_type bits,
                                      detail::simd_tuple<T, As...> &lhs, const T rhs)
    {
        detail::for_each(lhs, [&](auto meta, auto &native_lhs) {
            meta.template masked_cassign<Op>(meta.make_mask(bits), native_lhs, rhs);
        });
    }

    // masked_unary {{{2
    template <template <typename> class Op, class T, class... As>
    static inline detail::simd_tuple<T, As...> masked_unary(
        const mask_member_type bits,
        const detail::simd_tuple<T, As...> v)  // TODO: const-ref v?
    {
        return v.apply_wrapped([&bits](auto meta, auto native) {
            return meta.template masked_unary<Op>(meta.make_mask(bits), native);
        });
    }

    // }}}2
};

// fixed_size_mask_impl {{{1
template <int N> struct fixed_size_mask_impl {
    static_assert(sizeof(ullong) * CHAR_BIT >= N,
                  "The fixed_size implementation relies on one "
                  "ullong being able to store all boolean "
                  "elements.");  // required in load & store

    // member types {{{2
    static constexpr std::make_index_sequence<N> index_seq = {};
    using mask_member_type = std::bitset<N>;
    template <class T> using simd_mask = std::experimental::simd_mask<T, simd_abi::fixed_size<N>>;
    template <class T> using type_tag = T *;

    // from_bitset {{{2
    template <class T>
    _GLIBCXX_SIMD_INTRINSIC static mask_member_type from_bitset(const mask_member_type &bs,
                                                     type_tag<T>) noexcept
    {
        return bs;
    }

    // load {{{2
    template <class F> static inline mask_member_type load(const bool *mem, F f) noexcept
    {
        // TODO: uchar is not necessarily the best type to use here. For smaller N ushort,
        // uint, ullong, float, and double can be more efficient.
        ullong r = 0;
        using Vs = fixed_size_storage<uchar, N>;
        detail::for_each(Vs{}, [&](auto meta, auto) {
            r |= meta.mask_to_shifted_ullong(
                meta.simd_mask.load(&mem[meta.offset], f, size_constant<meta.size()>()));
        });
        return r;
    }

    // masked load {{{2
    template <class F>
    static inline mask_member_type masked_load(mask_member_type merge,
                                               mask_member_type mask, const bool *mem,
                                               F) noexcept
    {
        detail::bit_iteration(mask.to_ullong(), [&](auto i) { merge[i] = mem[i]; });
        return merge;
    }

    // store {{{2
    template <class F>
    static inline void store(mask_member_type bs, bool *mem, F f) noexcept
    {
#ifdef _GLIBCXX_SIMD_HAVE_AVX512BW
        const __m512i bool64 = _mm512_movm_epi8(bs.to_ullong()) & 0x0101010101010101ULL;
        builtin_store<N>(bool64, mem, f);
#elif defined _GLIBCXX_SIMD_HAVE_BMI2
#ifdef __x86_64__
        unused(f);
        execute_n_times<N / 8>([&](auto i) {
            constexpr size_t offset = i * 8;
            const ullong bool8 =
                _pdep_u64(bs.to_ullong() >> offset, 0x0101010101010101ULL);
            std::memcpy(&mem[offset], &bool8, 8);
        });
        if (N % 8 > 0) {
            constexpr size_t offset = (N / 8) * 8;
            const ullong bool8 =
                _pdep_u64(bs.to_ullong() >> offset, 0x0101010101010101ULL);
            std::memcpy(&mem[offset], &bool8, N % 8);
        }
#else   // __x86_64__
        unused(f);
        execute_n_times<N / 4>([&](auto i) {
            constexpr size_t offset = i * 4;
            const ullong bool4 =
                _pdep_u32(bs.to_ullong() >> offset, 0x01010101U);
            std::memcpy(&mem[offset], &bool4, 4);
        });
        if (N % 4 > 0) {
            constexpr size_t offset = (N / 4) * 4;
            const ullong bool4 =
                _pdep_u32(bs.to_ullong() >> offset, 0x01010101U);
            std::memcpy(&mem[offset], &bool4, N % 4);
        }
#endif  // __x86_64__
#elif defined _GLIBCXX_SIMD_HAVE_SSE2   // !AVX512BW && !BMI2
        using V = simd<uchar, simd_abi::__sse>;
        ullong bits = bs.to_ullong();
        execute_n_times<(N + 15) / 16>([&](auto i) {
            constexpr size_t offset = i * 16;
            constexpr size_t remaining = N - offset;
            if constexpr (remaining == 1) {
                mem[offset] = static_cast<bool>(bits >> offset);
            } else if constexpr (remaining <= 4) {
                const uint bool4 = ((bits >> offset) * 0x00204081U) & 0x01010101U;
                std::memcpy(&mem[offset], &bool4, remaining);
            } else if constexpr (remaining <= 7) {
                const ullong bool8 =
                    ((bits >> offset) * 0x40810204081ULL) & 0x0101010101010101ULL;
                std::memcpy(&mem[offset], &bool8, remaining);
            } else if constexpr (have_sse2) {
                auto tmp = _mm_cvtsi32_si128(bits >> offset);
                tmp = _mm_unpacklo_epi8(tmp, tmp);
                tmp = _mm_unpacklo_epi16(tmp, tmp);
                tmp = _mm_unpacklo_epi32(tmp, tmp);
                V tmp2(tmp);
                tmp2 &= V([](auto j) {
                    return static_cast<uchar>(1 << (j % CHAR_BIT));
                });  // mask bit index
                const __m128i bool16 =
                    _mm_add_epi8(detail::data(tmp2 == 0),
                                 _mm_set1_epi8(1));  // 0xff -> 0x00 | 0x00 -> 0x01
                if constexpr (remaining >= 16) {
                    builtin_store<16>(bool16, &mem[offset], f);
                } else if constexpr (remaining & 3) {
                    constexpr int to_shift = 16 - int(remaining);
                    _mm_maskmoveu_si128(bool16,
                                        _mm_srli_si128(allbits<__m128i>, to_shift),
                                        reinterpret_cast<char *>(&mem[offset]));
                } else  // at this point: 8 < remaining < 16
                    if constexpr (remaining >= 8) {
                    builtin_store<8>(bool16, &mem[offset], f);
                    if constexpr (remaining == 12) {
                        builtin_store<4>(_mm_unpackhi_epi64(bool16, bool16),
                                         &mem[offset + 8], f);
                    }
                }
            } else {
                assert_unreachable<F>();
            }
        });
#else
        // TODO: uchar is not necessarily the best type to use here. For smaller N ushort,
        // uint, ullong, float, and double can be more efficient.
        using Vs = fixed_size_storage<uchar, N>;
        detail::for_each(Vs{}, [&](auto meta, auto) {
            meta.store(meta.make_mask(bs), &mem[meta.offset], f);
        });
//#else
        //execute_n_times<N>([&](auto i) { mem[i] = bs[i]; });
#endif  // _GLIBCXX_SIMD_HAVE_BMI2
    }

    // masked store {{{2
    template <class F>
    static inline void masked_store(const mask_member_type v, bool *mem, F,
                                    const mask_member_type k) noexcept
    {
        detail::bit_iteration(k, [&](auto i) { mem[i] = v[i]; });
    }

    // logical and bitwise operators {{{2
    _GLIBCXX_SIMD_INTRINSIC static mask_member_type logical_and(const mask_member_type &x,
                                                     const mask_member_type &y) noexcept
    {
        return x & y;
    }

    _GLIBCXX_SIMD_INTRINSIC static mask_member_type logical_or(const mask_member_type &x,
                                                    const mask_member_type &y) noexcept
    {
        return x | y;
    }

    _GLIBCXX_SIMD_INTRINSIC static mask_member_type bit_and(const mask_member_type &x,
                                                 const mask_member_type &y) noexcept
    {
        return x & y;
    }

    _GLIBCXX_SIMD_INTRINSIC static mask_member_type bit_or(const mask_member_type &x,
                                                const mask_member_type &y) noexcept
    {
        return x | y;
    }

    _GLIBCXX_SIMD_INTRINSIC static mask_member_type bit_xor(const mask_member_type &x,
                                                 const mask_member_type &y) noexcept
    {
        return x ^ y;
    }

    // smart_reference access {{{2
    _GLIBCXX_SIMD_INTRINSIC static void set(mask_member_type &k, int i, bool x) noexcept
    {
        k.set(i, x);
    }

    // masked_assign {{{2
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(const mask_member_type k,
                                           mask_member_type &lhs,
                                           const mask_member_type rhs)
    {
        lhs = (lhs & ~k) | (rhs & k);
    }

    // Optimization for the case where the RHS is a scalar.
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(const mask_member_type k,
                                           mask_member_type &lhs, const bool rhs)
    {
        if (rhs) {
            lhs |= k;
        } else {
            lhs &= ~k;
        }
    }

    // }}}2
};
// }}}1

// simd_converter scalar -> scalar {{{
template <class T> struct simd_converter<T, simd_abi::scalar, T, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC T operator()(T a) { return a; }
};
template <class From, class To>
struct simd_converter<From, simd_abi::scalar, To, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC To operator()(From a)
    {
        return static_cast<To>(a);
    }
};

// }}}
// simd_converter __sse -> scalar {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::__sse, To, simd_abi::scalar> {
    using Arg = sse_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC std::array<To, Arg::width> all(Arg a)
    {
        return impl(std::make_index_sequence<Arg::width>(), a);
    }

    template <size_t... Indexes>
    _GLIBCXX_SIMD_INTRINSIC std::array<To, Arg::width> impl(std::index_sequence<Indexes...>, Arg a)
    {
        return {static_cast<To>(a[Indexes])...};
    }
};

// }}}1
// simd_converter scalar -> __sse {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::scalar, To, simd_abi::__sse> {
    using R = sse_simd_member_type<To>;
    template <class... More> _GLIBCXX_SIMD_INTRINSIC constexpr R operator()(From a, More... b)
    {
        static_assert(sizeof...(More) + 1 == R::width);
        static_assert(std::conjunction_v<std::is_same<From, More>...>);
        return builtin_type16_t<To>{static_cast<To>(a), static_cast<To>(b)...};
    }
};

// }}}1
// simd_converter __sse -> __sse {{{1
template <class T> struct simd_converter<T, simd_abi::__sse, T, simd_abi::__sse> {
    using Arg = sse_simd_member_type<T>;
    _GLIBCXX_SIMD_INTRINSIC const Arg &operator()(const Arg &x) { return x; }
};

template <class From, class To>
struct simd_converter<From, simd_abi::__sse, To, simd_abi::__sse> {
    using Arg = sse_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a)
    {
        return x86::convert_all<builtin_type16_t<To>>(a);
    }

    _GLIBCXX_SIMD_INTRINSIC sse_simd_member_type<To> operator()(Arg a)
    {
        return x86::convert<builtin_type16_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return x86::convert<sse_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<sse_simd_member_type<To>>(a, b, c, d);
    }
    _GLIBCXX_SIMD_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d, Arg e,
                                                     Arg f, Arg g, Arg h)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<sse_simd_member_type<To>>(a, b, c, d, e, f, g, h);
    }
};

// }}}1
// simd_converter __avx -> scalar {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::__avx, To, simd_abi::scalar> {
    using Arg = avx_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC std::array<To, Arg::width> all(Arg a)
    {
        return impl(std::make_index_sequence<Arg::width>(), a);
    }

    template <size_t... Indexes>
    _GLIBCXX_SIMD_INTRINSIC std::array<To, Arg::width> impl(std::index_sequence<Indexes...>, Arg a)
    {
        return {static_cast<To>(a[Indexes])...};
    }
};

// }}}1
// simd_converter scalar -> __avx {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::scalar, To, simd_abi::__avx> {
    using R = avx_simd_member_type<To>;
    template <class... More> _GLIBCXX_SIMD_INTRINSIC constexpr R operator()(From a, More... b)
    {
        static_assert(sizeof...(More) + 1 == R::width);
        static_assert(std::conjunction_v<std::is_same<From, More>...>);
        return builtin_type32_t<To>{static_cast<To>(a), static_cast<To>(b)...};
    }
};

// }}}1
// simd_converter __sse -> __avx {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::__sse, To, simd_abi::__avx> {
    using Arg = sse_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a) { return x86::convert_all<builtin_type32_t<To>>(a); }

    _GLIBCXX_SIMD_INTRINSIC avx_simd_member_type<To> operator()(Arg a)
    {
        return x86::convert<builtin_type32_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 1 * sizeof(To), "");
        return x86::convert<avx_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return x86::convert<avx_simd_member_type<To>>(a, b, c, d);
    }
    _GLIBCXX_SIMD_INTRINSIC avx_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<avx_simd_member_type<To>>(x0, x1, x2, x3, x4, x5, x6, x7);
    }
    _GLIBCXX_SIMD_INTRINSIC avx_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7,
                                                     Arg x8, Arg x9, Arg x10, Arg x11,
                                                     Arg x12, Arg x13, Arg x14, Arg x15)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<avx_simd_member_type<To>>(x0, x1, x2, x3, x4, x5, x6, x7, x8,
                                                      x9, x10, x11, x12, x13, x14, x15);
    }
};

// }}}1
// simd_converter __avx -> __sse {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::__avx, To, simd_abi::__sse> {
    using Arg = avx_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a) { return x86::convert_all<builtin_type16_t<To>>(a); }

    _GLIBCXX_SIMD_INTRINSIC sse_simd_member_type<To> operator()(Arg a)
    {
        return x86::convert<builtin_type16_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<sse_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<sse_simd_member_type<To>>(a, b, c, d);
    }
};

// }}}1
// simd_converter __avx -> __avx {{{1
template <class T> struct simd_converter<T, simd_abi::__avx, T, simd_abi::__avx> {
    using Arg = avx_simd_member_type<T>;
    _GLIBCXX_SIMD_INTRINSIC const Arg &operator()(const Arg &x) { return x; }
};

template <class From, class To>
struct simd_converter<From, simd_abi::__avx, To, simd_abi::__avx> {
    using Arg = avx_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a) { return x86::convert_all<builtin_type32_t<To>>(a); }

    _GLIBCXX_SIMD_INTRINSIC avx_simd_member_type<To> operator()(Arg a)
    {
        return x86::convert<builtin_type32_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return x86::convert<avx_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<avx_simd_member_type<To>>(a, b, c, d);
    }
    _GLIBCXX_SIMD_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d, Arg e,
                                                     Arg f, Arg g, Arg h)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<avx_simd_member_type<To>>(a, b, c, d, e, f, g, h);
    }
};

// }}}1
// simd_converter __avx512 -> scalar {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::__avx512, To, simd_abi::scalar> {
    using Arg = avx512_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC std::array<To, Arg::width> all(Arg a)
    {
        return impl(std::make_index_sequence<Arg::width>(), a);
    }

    template <size_t... Indexes>
    _GLIBCXX_SIMD_INTRINSIC std::array<To, Arg::width> impl(std::index_sequence<Indexes...>, Arg a)
    {
        return {static_cast<To>(a[Indexes])...};
    }
};

// }}}1
// simd_converter scalar -> __avx512 {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::scalar, To, simd_abi::__avx512> {
    using R = avx512_simd_member_type<To>;

    _GLIBCXX_SIMD_INTRINSIC R operator()(From a)
    {
        R r{};
        r.set(0, static_cast<To>(a));
        return r;
    }
    _GLIBCXX_SIMD_INTRINSIC R operator()(From a, From b)
    {
        R r{};
        r.set(0, static_cast<To>(a));
        r.set(1, static_cast<To>(b));
        return r;
    }
    _GLIBCXX_SIMD_INTRINSIC R operator()(From a, From b, From c, From d)
    {
        R r{};
        r.set(0, static_cast<To>(a));
        r.set(1, static_cast<To>(b));
        r.set(2, static_cast<To>(c));
        r.set(3, static_cast<To>(d));
        return r;
    }
    _GLIBCXX_SIMD_INTRINSIC R operator()(From a, From b, From c, From d, From e, From f, From g,
                              From h)
    {
        R r{};
        r.set(0, static_cast<To>(a));
        r.set(1, static_cast<To>(b));
        r.set(2, static_cast<To>(c));
        r.set(3, static_cast<To>(d));
        r.set(4, static_cast<To>(e));
        r.set(5, static_cast<To>(f));
        r.set(6, static_cast<To>(g));
        r.set(7, static_cast<To>(h));
        return r;
    }
    _GLIBCXX_SIMD_INTRINSIC R operator()(From x0, From x1, From x2, From x3, From x4, From x5,
                              From x6, From x7, From x8, From x9, From x10, From x11,
                              From x12, From x13, From x14, From x15)
    {
        R r{};
        r.set(0, static_cast<To>(x0));
        r.set(1, static_cast<To>(x1));
        r.set(2, static_cast<To>(x2));
        r.set(3, static_cast<To>(x3));
        r.set(4, static_cast<To>(x4));
        r.set(5, static_cast<To>(x5));
        r.set(6, static_cast<To>(x6));
        r.set(7, static_cast<To>(x7));
        r.set(8, static_cast<To>(x8));
        r.set(9, static_cast<To>(x9));
        r.set(10, static_cast<To>(x10));
        r.set(11, static_cast<To>(x11));
        r.set(12, static_cast<To>(x12));
        r.set(13, static_cast<To>(x13));
        r.set(14, static_cast<To>(x14));
        r.set(15, static_cast<To>(x15));
        return r;
    }
    _GLIBCXX_SIMD_INTRINSIC R operator()(From x0, From x1, From x2, From x3, From x4, From x5,
                              From x6, From x7, From x8, From x9, From x10, From x11,
                              From x12, From x13, From x14, From x15, From x16, From x17,
                              From x18, From x19, From x20, From x21, From x22, From x23,
                              From x24, From x25, From x26, From x27, From x28, From x29,
                              From x30, From x31)
    {
        R r{};
        r.set(0, static_cast<To>(x0));
        r.set(1, static_cast<To>(x1));
        r.set(2, static_cast<To>(x2));
        r.set(3, static_cast<To>(x3));
        r.set(4, static_cast<To>(x4));
        r.set(5, static_cast<To>(x5));
        r.set(6, static_cast<To>(x6));
        r.set(7, static_cast<To>(x7));
        r.set(8, static_cast<To>(x8));
        r.set(9, static_cast<To>(x9));
        r.set(10, static_cast<To>(x10));
        r.set(11, static_cast<To>(x11));
        r.set(12, static_cast<To>(x12));
        r.set(13, static_cast<To>(x13));
        r.set(14, static_cast<To>(x14));
        r.set(15, static_cast<To>(x15));
        r.set(16, static_cast<To>(x16));
        r.set(17, static_cast<To>(x17));
        r.set(18, static_cast<To>(x18));
        r.set(19, static_cast<To>(x19));
        r.set(20, static_cast<To>(x20));
        r.set(21, static_cast<To>(x21));
        r.set(22, static_cast<To>(x22));
        r.set(23, static_cast<To>(x23));
        r.set(24, static_cast<To>(x24));
        r.set(25, static_cast<To>(x25));
        r.set(26, static_cast<To>(x26));
        r.set(27, static_cast<To>(x27));
        r.set(28, static_cast<To>(x28));
        r.set(29, static_cast<To>(x29));
        r.set(30, static_cast<To>(x30));
        r.set(31, static_cast<To>(x31));
        return r;
    }
    _GLIBCXX_SIMD_INTRINSIC R operator()(From x0, From x1, From x2, From x3, From x4, From x5,
                              From x6, From x7, From x8, From x9, From x10, From x11,
                              From x12, From x13, From x14, From x15, From x16, From x17,
                              From x18, From x19, From x20, From x21, From x22, From x23,
                              From x24, From x25, From x26, From x27, From x28, From x29,
                              From x30, From x31, From x32, From x33, From x34, From x35,
                              From x36, From x37, From x38, From x39, From x40, From x41,
                              From x42, From x43, From x44, From x45, From x46, From x47,
                              From x48, From x49, From x50, From x51, From x52, From x53,
                              From x54, From x55, From x56, From x57, From x58, From x59,
                              From x60, From x61, From x62, From x63)
    {
        return R(static_cast<To>(x0), static_cast<To>(x1), static_cast<To>(x2),
                 static_cast<To>(x3), static_cast<To>(x4), static_cast<To>(x5),
                 static_cast<To>(x6), static_cast<To>(x7), static_cast<To>(x8),
                 static_cast<To>(x9), static_cast<To>(x10), static_cast<To>(x11),
                 static_cast<To>(x12), static_cast<To>(x13), static_cast<To>(x14),
                 static_cast<To>(x15), static_cast<To>(x16), static_cast<To>(x17),
                 static_cast<To>(x18), static_cast<To>(x19), static_cast<To>(x20),
                 static_cast<To>(x21), static_cast<To>(x22), static_cast<To>(x23),
                 static_cast<To>(x24), static_cast<To>(x25), static_cast<To>(x26),
                 static_cast<To>(x27), static_cast<To>(x28), static_cast<To>(x29),
                 static_cast<To>(x30), static_cast<To>(x31), static_cast<To>(x32),
                 static_cast<To>(x33), static_cast<To>(x34), static_cast<To>(x35),
                 static_cast<To>(x36), static_cast<To>(x37), static_cast<To>(x38),
                 static_cast<To>(x39), static_cast<To>(x40), static_cast<To>(x41),
                 static_cast<To>(x42), static_cast<To>(x43), static_cast<To>(x44),
                 static_cast<To>(x45), static_cast<To>(x46), static_cast<To>(x47),
                 static_cast<To>(x48), static_cast<To>(x49), static_cast<To>(x50),
                 static_cast<To>(x51), static_cast<To>(x52), static_cast<To>(x53),
                 static_cast<To>(x54), static_cast<To>(x55), static_cast<To>(x56),
                 static_cast<To>(x57), static_cast<To>(x58), static_cast<To>(x59),
                 static_cast<To>(x60), static_cast<To>(x61), static_cast<To>(x62),
                 static_cast<To>(x63));
    }
};

// }}}1
// simd_converter __sse -> __avx512 {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::__sse, To, simd_abi::__avx512> {
    using Arg = sse_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a)
    {
        return x86::convert_all<builtin_type64_t<To>>(a);
    }

    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg a)
    {
        return x86::convert<builtin_type64_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(2 * sizeof(From) >= sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 1 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b, c, d);
    }
    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(x0, x1, x2, x3, x4, x5, x6,
                                                           x7);
    }
    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7,
                                                     Arg x8, Arg x9, Arg x10, Arg x11,
                                                     Arg x12, Arg x13, Arg x14, Arg x15)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
    }
    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(
        Arg x0, Arg x1, Arg x2, Arg x3, Arg x4, Arg x5, Arg x6, Arg x7, Arg x8, Arg x9,
        Arg x10, Arg x11, Arg x12, Arg x13, Arg x14, Arg x15, Arg x16, Arg x17, Arg x18,
        Arg x19, Arg x20, Arg x21, Arg x22, Arg x23, Arg x24, Arg x25, Arg x26, Arg x27,
        Arg x28, Arg x29, Arg x30, Arg x31)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16,
            x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31);
    }
};

// }}}1
// simd_converter __avx512 -> __sse {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::__avx512, To, simd_abi::__sse> {
    using Arg = avx512_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a)
    {
        return x86::convert_all<builtin_type16_t<To>>(a);
    }

    _GLIBCXX_SIMD_INTRINSIC sse_simd_member_type<To> operator()(Arg a)
    {
        return x86::convert<builtin_type16_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC sse_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<sse_simd_member_type<To>>(a, b);
    }
};

// }}}1
// simd_converter __avx -> __avx512 {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::__avx, To, simd_abi::__avx512> {
    using Arg = avx_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a)
    {
        return x86::convert_all<builtin_type64_t<To>>(a);
    }

    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg a)
    {
        return x86::convert<builtin_type64_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 1 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b, c, d);
    }
    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(x0, x1, x2, x3, x4, x5, x6,
                                                           x7);
    }
    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7,
                                                     Arg x8, Arg x9, Arg x10, Arg x11,
                                                     Arg x12, Arg x13, Arg x14, Arg x15)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
    }
};

// }}}1
// simd_converter __avx512 -> __avx {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::__avx512, To, simd_abi::__avx> {
    using Arg = avx512_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a)
    {
        return x86::convert_all<builtin_type32_t<To>>(a);
    }

    _GLIBCXX_SIMD_INTRINSIC avx_simd_member_type<To> operator()(Arg a)
    {
        return x86::convert<builtin_type32_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<avx_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<avx_simd_member_type<To>>(a, b, c, d);
    }
};

// }}}1
// simd_converter __avx512 -> __avx512 {{{1
template <class T> struct simd_converter<T, simd_abi::__avx512, T, simd_abi::__avx512> {
    using Arg = avx512_simd_member_type<T>;
    _GLIBCXX_SIMD_INTRINSIC const Arg &operator()(const Arg &x) { return x; }
};

template <class From, class To>
struct simd_converter<From, simd_abi::__avx512, To, simd_abi::__avx512> {
    using Arg = avx512_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a)
    {
        return x86::convert_all<builtin_type64_t<To>>(a);
    }

    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg a)
    {
        return x86::convert<builtin_type64_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b, c, d);
    }
    _GLIBCXX_SIMD_INTRINSIC avx512_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d, Arg e,
                                                     Arg f, Arg g, Arg h)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return x86::convert<avx512_simd_member_type<To>>(a, b, c, d, e, f, g, h);
    }
};

// }}}1
// simd_converter scalar -> fixed_size<1> {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::scalar, To, simd_abi::fixed_size<1>> {
    simd_tuple<To, simd_abi::scalar> operator()(From x) { return {static_cast<To>(x)}; }
};

// simd_converter fixed_size<1> -> scalar {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::fixed_size<1>, To, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC To operator()(simd_tuple<From, simd_abi::scalar> x)
    {
        return {static_cast<To>(x.first)};
    }
};

// simd_converter fixed_size<N> -> fixed_size<N> {{{1
template <class T, int N>
struct simd_converter<T, simd_abi::fixed_size<N>, T, simd_abi::fixed_size<N>> {
    using arg = fixed_size_storage<T, N>;
    _GLIBCXX_SIMD_INTRINSIC const arg &operator()(const arg &x) { return x; }
};

template <size_t ChunkSize, class T> struct determine_required_input_chunks;

template <class T, class... Abis>
struct determine_required_input_chunks<0, simd_tuple<T, Abis...>>
    : public std::integral_constant<size_t, 0> {
};

template <size_t ChunkSize, class T, class Abi0, class... Abis>
struct determine_required_input_chunks<ChunkSize, simd_tuple<T, Abi0, Abis...>>
    : public std::integral_constant<
          size_t, determine_required_input_chunks<ChunkSize - simd_size_v<T, Abi0>,
                                                  simd_tuple<T, Abis...>>::value> {
};

template <class From, class To> struct fixed_size_converter {
    struct OneToMultipleChunks {
    };
    template <int N> struct MultipleToOneChunk {
    };
    struct EqualChunks {
    };
    template <class FromAbi, class ToAbi, size_t ToSize = simd_size_v<To, ToAbi>,
              size_t FromSize = simd_size_v<From, FromAbi>>
    using ChunkRelation = std::conditional_t<
        (ToSize < FromSize), OneToMultipleChunks,
        std::conditional_t<(ToSize == FromSize), EqualChunks,
                           MultipleToOneChunk<int(ToSize / FromSize)>>>;

    template <class... Abis>
    using return_type = fixed_size_storage<To, simd_tuple<From, Abis...>::size()>;


protected:
    // OneToMultipleChunks {{{2
    template <class A0>
    _GLIBCXX_SIMD_INTRINSIC return_type<A0> impl(OneToMultipleChunks, const simd_tuple<From, A0> &x)
    {
        using R = return_type<A0>;
        simd_converter<From, A0, To, typename R::first_abi> native_cvt;
        auto &&multiple_return_chunks = native_cvt.all(x.first);
        return detail::to_tuple<To, typename R::first_abi>(multiple_return_chunks);
    }

    template <class... Abis>
    _GLIBCXX_SIMD_INTRINSIC return_type<Abis...> impl(OneToMultipleChunks,
                                           const simd_tuple<From, Abis...> &x)
    {
        using R = return_type<Abis...>;
        using arg = simd_tuple<From, Abis...>;
        constexpr size_t first_chunk = simd_size_v<From, typename arg::first_abi>;
        simd_converter<From, typename arg::first_abi, To, typename R::first_abi>
            native_cvt;
        auto &&multiple_return_chunks = native_cvt.all(x.first);
        constexpr size_t n_output_chunks =
            first_chunk / simd_size_v<To, typename R::first_abi>;
        return detail::tuple_concat(
            detail::to_tuple<To, typename R::first_abi>(multiple_return_chunks),
            impl(ChunkRelation<typename arg::second_type::first_abi,
                               typename tuple_element<n_output_chunks, R>::type::abi_type>(),
                 x.second));
    }

    // MultipleToOneChunk {{{2
    template <int N, class A0, class... Abis>
    _GLIBCXX_SIMD_INTRINSIC return_type<A0, Abis...> impl(MultipleToOneChunk<N>,
                                               const simd_tuple<From, A0, Abis...> &x)
    {
        return impl_mto(std::integral_constant<bool, sizeof...(Abis) + 1 == N>(),
                        std::make_index_sequence<N>(), x);
    }

    template <size_t... Indexes, class A0, class... Abis>
    _GLIBCXX_SIMD_INTRINSIC return_type<A0, Abis...> impl_mto(std::true_type,
                                                   std::index_sequence<Indexes...>,
                                                   const simd_tuple<From, A0, Abis...> &x)
    {
        using R = return_type<A0, Abis...>;
        simd_converter<From, A0, To, typename R::first_abi> native_cvt;
        return {native_cvt(detail::get<Indexes>(x)...)};
    }

    template <size_t... Indexes, class A0, class... Abis>
    _GLIBCXX_SIMD_INTRINSIC return_type<A0, Abis...> impl_mto(std::false_type,
                                                   std::index_sequence<Indexes...>,
                                                   const simd_tuple<From, A0, Abis...> &x)
    {
        using R = return_type<A0, Abis...>;
        simd_converter<From, A0, To, typename R::first_abi> native_cvt;
        return {
            native_cvt(detail::get<Indexes>(x)...),
            impl(
                ChunkRelation<
                    typename tuple_element<sizeof...(Indexes),
                                           simd_tuple<From, A0, Abis...>>::type::abi_type,
                    typename R::second_type::first_abi>(),
                tuple_pop_front(size_constant<sizeof...(Indexes)>(), x))};
    }

    // EqualChunks {{{2
    template <class A0>
    _GLIBCXX_SIMD_INTRINSIC return_type<A0> impl(EqualChunks, const simd_tuple<From, A0> &x)
    {
        simd_converter<From, A0, To, typename return_type<A0>::first_abi> native_cvt;
        return {native_cvt(x.first)};
    }

    template <class A0, class A1, class... Abis>
    _GLIBCXX_SIMD_INTRINSIC return_type<A0, A1, Abis...> impl(
        EqualChunks, const simd_tuple<From, A0, A1, Abis...> &x)
    {
        using R = return_type<A0, A1, Abis...>;
        using Rem = typename R::second_type;
        simd_converter<From, A0, To, typename R::first_abi> native_cvt;
        return {native_cvt(x.first),
                impl(ChunkRelation<A1, typename Rem::first_abi>(), x.second)};
    }

    //}}}2
};

template <class From, class To, int N>
struct simd_converter<From, simd_abi::fixed_size<N>, To, simd_abi::fixed_size<N>>
    : public fixed_size_converter<From, To> {
    using base = fixed_size_converter<From, To>;
    using return_type = fixed_size_storage<To, N>;
    using arg = fixed_size_storage<From, N>;

    _GLIBCXX_SIMD_INTRINSIC return_type operator()(const arg &x)
    {
        using CR = typename base::template ChunkRelation<typename arg::first_abi,
                                                         typename return_type::first_abi>;
        return base::impl(CR(), x);
    }
};

// simd_converter "native" -> fixed_size<N> {{{1
// i.e. 1 register to ? registers
template <class From, class A, class To, int N>
struct simd_converter<From, A, To, simd_abi::fixed_size<N>> {
    using traits = detail::traits<From, A>;
    using arg = typename traits::simd_member_type;
    using return_type = fixed_size_storage<To, N>;
    static_assert(N == simd_size_v<From, A>,
                  "simd_converter to fixed_size only works for equal element counts");

    _GLIBCXX_SIMD_INTRINSIC return_type operator()(arg x)
    {
        return impl(std::make_index_sequence<return_type::tuple_size>(), x);
    }

private:
    return_type impl(std::index_sequence<0>, arg x)
    {
        simd_converter<From, A, To, typename return_type::first_abi> native_cvt;
        return {native_cvt(x)};
    }
    template <size_t... Indexes> return_type impl(std::index_sequence<Indexes...>, arg x)
    {
        simd_converter<From, A, To, typename return_type::first_abi> native_cvt;
        const auto &tmp = native_cvt.all(x);
        return {tmp[Indexes]...};
    }
};

// simd_converter fixed_size<N> -> "native" {{{1
// i.e. ? register to 1 registers
template <class From, int N, class To, class A>
struct simd_converter<From, simd_abi::fixed_size<N>, To, A> {
    using traits = detail::traits<To, A>;
    using return_type = typename traits::simd_member_type;
    using arg = fixed_size_storage<From, N>;
    static_assert(N == simd_size_v<To, A>,
                  "simd_converter to fixed_size only works for equal element counts");

    _GLIBCXX_SIMD_INTRINSIC return_type operator()(arg x)
    {
        return impl(std::make_index_sequence<arg::tuple_size>(), x);
    }

private:
    template <size_t... Indexes> return_type impl(std::index_sequence<Indexes...>, arg x)
    {
        simd_converter<From, typename arg::first_abi, To, A> native_cvt;
        return native_cvt(detail::get<Indexes>(x)...);
    }
};

// }}}1
// generic_simd_impl::masked_cassign specializations {{{1
#ifdef _GLIBCXX_SIMD_HAVE_AVX512_ABI
#define _GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(TYPE_, TYPE_SUFFIX_, OP_, OP_NAME_)             \
    template <>                                                                          \
    template <>                                                                          \
    _GLIBCXX_SIMD_INTRINSIC void generic_simd_impl<simd_abi::__avx512>::masked_cassign<             \
        OP_, TYPE_, bool, 64 / sizeof(TYPE_)>(                                           \
        const Storage<bool, 64 / sizeof(TYPE_)> k,                                       \
        Storage<TYPE_, 64 / sizeof(TYPE_)> &lhs,                                         \
        const detail::id<Storage<TYPE_, 64 / sizeof(TYPE_)>> rhs)                        \
    {                                                                                    \
        lhs = _mm512_mask_##OP_NAME_##_##TYPE_SUFFIX_(lhs, k, lhs, rhs);                 \
    }                                                                                    \
    _GLIBCXX_SIMD_NOTHING_EXPECTING_SEMICOLON

    _GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(double, pd, std::plus, add);
    _GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(float, ps, std::plus, add);
    _GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail::llong, epi64, std::plus, add);
    _GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail::ullong, epi64, std::plus, add);
    _GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(long, epi64, std::plus, add);
    _GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail::ulong, epi64, std::plus, add);
    _GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(int, epi32, std::plus, add);
    _GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail::uint, epi32, std::plus, add);
#ifdef _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(         short, epi16, std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail::ushort, epi16, std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail:: schar, epi8 , std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail:: uchar, epi8 , std::plus, add);
#endif  // _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI

_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(        double,  pd  , std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(         float,  ps  , std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail:: llong, epi64, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail::ullong, epi64, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(          long, epi64, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail:: ulong, epi64, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(           int, epi32, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail::  uint, epi32, std::minus, sub);
#ifdef _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(         short, epi16, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail::ushort, epi16, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail:: schar, epi8 , std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(detail:: uchar, epi8 , std::minus, sub);
#endif  // _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
#undef _GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION
#endif  // _GLIBCXX_SIMD_HAVE_AVX512_ABI

// }}}1
}  // namespace detail
_GLIBCXX_SIMD_END_NAMESPACE
#endif  // __cplusplus >= 201703L
#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_ABIS_H_
