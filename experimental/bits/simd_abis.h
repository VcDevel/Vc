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
// __subscript_read/_write {{{1
template <class _T> _T __subscript_read(_Vectorizable<_T> __x, size_t) noexcept
{
    return __x;
}
template <class _T>
void __subscript_write(_Vectorizable<_T> &__x, size_t, __id<_T> __y) noexcept
{
    __x = __y;
}

template <class _T>
typename _T::value_type __subscript_read(const _T &__x, size_t i) noexcept
{
    return __x[i];
}
template <class _T>
void __subscript_write(_T &__x, size_t i, typename _T::value_type __y) noexcept
{
    return __x.set(i, __y);
}

// __simd_tuple_element {{{1
template <size_t I, class _T> struct __simd_tuple_element;
template <class _T, class _A0, class... As>
struct __simd_tuple_element<0, __simd_tuple<_T, _A0, As...>> {
    using type = std::experimental::simd<_T, _A0>;
};
template <size_t I, class _T, class _A0, class... As>
struct __simd_tuple_element<I, __simd_tuple<_T, _A0, As...>> {
    using type = typename __simd_tuple_element<I - 1, __simd_tuple<_T, As...>>::type;
};
template <size_t I, class _T>
using __simd_tuple_element_t = typename __simd_tuple_element<I, _T>::type;

// __simd_tuple_concat {{{1
template <class _T, class... _A1s>
_GLIBCXX_SIMD_INTRINSIC constexpr __simd_tuple<_T, _A1s...> __simd_tuple_concat(
    const __simd_tuple<_T>, const __simd_tuple<_T, _A1s...> __right)
{
    return __right;
}

template <class _T, class _A00, class... _A0s, class _A10, class... _A1s>
_GLIBCXX_SIMD_INTRINSIC constexpr __simd_tuple<_T, _A00, _A0s..., _A10, _A1s...>
__simd_tuple_concat(const __simd_tuple<_T, _A00, _A0s...> __left,
                    const __simd_tuple<_T, _A10, _A1s...> __right)
{
    return {__left.first, __simd_tuple_concat(__left.second, __right)};
}

template <class _T, class _A00, class... _A0s>
_GLIBCXX_SIMD_INTRINSIC constexpr __simd_tuple<_T, _A00, _A0s...> __simd_tuple_concat(
    const __simd_tuple<_T, _A00, _A0s...> __left, const __simd_tuple<_T>)
{
    return __left;
}

template <class _T, class _A10, class... _A1s>
_GLIBCXX_SIMD_INTRINSIC constexpr __simd_tuple<_T, simd_abi::scalar, _A10, _A1s...>
__simd_tuple_concat(const _T __left, const __simd_tuple<_T, _A10, _A1s...> __right)
{
    return {__left, __right};
}

// __simd_tuple_pop_front {{{1
template <class _T>
_GLIBCXX_SIMD_INTRINSIC constexpr const _T &__simd_tuple_pop_front(__size_constant<0>,
                                                                   const _T &x)
{
    return x;
}
template <class _T>
_GLIBCXX_SIMD_INTRINSIC constexpr _T &__simd_tuple_pop_front(__size_constant<0>, _T &x)
{
    return x;
}
template <size_t K, class _T>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto &__simd_tuple_pop_front(__size_constant<K>,
                                                                     const _T &x)
{
    return __simd_tuple_pop_front(__size_constant<K - 1>(), x.second);
}
template <size_t K, class _T>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__simd_tuple_pop_front(__size_constant<K>, _T &x)
{
    return __simd_tuple_pop_front(__size_constant<K - 1>(), x.second);
}

// __get_simd_at<_N> {{{1
struct __as_simd {};
struct __as_simd_tuple {};
template <class _T, class _A0, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr simd<_T, _A0> __simd_tuple_get_impl(
    __as_simd, const __simd_tuple<_T, _A0, _Abis...> &t, __size_constant<0>)
{
    return {__private_init, t.first};
}
template <class _T, class _A0, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto &__simd_tuple_get_impl(
    __as_simd_tuple, const __simd_tuple<_T, _A0, _Abis...> &t, __size_constant<0>)
{
    return t.first;
}
template <class _T, class _A0, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__simd_tuple_get_impl(
    __as_simd_tuple, __simd_tuple<_T, _A0, _Abis...> &t, __size_constant<0>)
{
    return t.first;
}

template <class R, size_t _N, class _T, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __simd_tuple_get_impl(
    R, const __simd_tuple<_T, _Abis...> &t, __size_constant<_N>)
{
    return __simd_tuple_get_impl(R(), t.second, __size_constant<_N - 1>());
}
template <size_t _N, class _T, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__simd_tuple_get_impl(
    __as_simd_tuple, __simd_tuple<_T, _Abis...> &t, __size_constant<_N>)
{
    return __simd_tuple_get_impl(__as_simd_tuple(), t.second, __size_constant<_N - 1>());
}

template <size_t _N, class _T, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __get_simd_at(const __simd_tuple<_T, _Abis...> &t)
{
    return __simd_tuple_get_impl(__as_simd(), t, __size_constant<_N>());
}

// }}}
// __get_tuple_at<_N> {{{
template <size_t _N, class _T, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __get_tuple_at(const __simd_tuple<_T, _Abis...> &t)
{
    return __simd_tuple_get_impl(__as_simd_tuple(), t, __size_constant<_N>());
}

template <size_t _N, class _T, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__get_tuple_at(__simd_tuple<_T, _Abis...> &t)
{
    return __simd_tuple_get_impl(__as_simd_tuple(), t, __size_constant<_N>());
}

// __how_many_to_extract {{{1
template <size_t LeftN, class RightT> constexpr size_t __tuple_elements_for () {
    if constexpr (LeftN == 0) {
        return 0;
    } else {
        return 1 + __tuple_elements_for<LeftN - RightT::__first_size_v,
                                        typename RightT::__second_type>();
    }
}
template <size_t LeftN, class RightT, bool = (RightT::__first_size_v < LeftN)>
struct __how_many_to_extract;
template <size_t LeftN, class RightT> struct __how_many_to_extract<LeftN, RightT, true> {
    static constexpr std::make_index_sequence<__tuple_elements_for<LeftN, RightT>()> tag()
    {
        return {};
    }
};
template <class _T, size_t _Offset, size_t Length, bool Done, class IndexSeq>
struct chunked {
};
template <size_t LeftN, class RightT> struct __how_many_to_extract<LeftN, RightT, false> {
    static_assert(LeftN != RightT::__first_size_v, "");
    static constexpr chunked<typename RightT::__first_type, 0, LeftN, false,
                             std::make_index_sequence<LeftN>>
    tag()
    {
        return {};
    }
};

// __tuple_element_meta {{{1
template <class _T, class _Abi, size_t _Offset>
struct __tuple_element_meta : public _Abi::__simd_impl_type {
    using value_type = _T;
    using abi_type = _Abi;
    using __traits = __simd_traits<_T, _Abi>;
    using maskimpl = typename __traits::__mask_impl_type;
    using __member_type = typename __traits::__simd_member_type;
    using __mask_member_type = typename __traits::__mask_member_type;
    using simd_type = std::experimental::simd<_T, _Abi>;
    static constexpr size_t offset = _Offset;
    static constexpr size_t size() { return simd_size<_T, _Abi>::value; }
    static constexpr maskimpl simd_mask = {};

    template <size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type make_mask(std::bitset<_N> bits)
    {
        constexpr _T *__type_tag = nullptr;
        return maskimpl::__from_bitset(std::bitset<size()>((bits >> _Offset).to_ullong()),
                                     __type_tag);
    }

    _GLIBCXX_SIMD_INTRINSIC static __ullong mask_to_shifted_ullong(__mask_member_type k)
    {
        return __vector_to_bitset(k).to_ullong() << _Offset;
    }
};

template <size_t _Offset, class _T, class _Abi, class... As>
__tuple_element_meta<_T, _Abi, _Offset> make_meta(const __simd_tuple<_T, _Abi, As...> &)
{
    return {};
}

// __simd_tuple specializations {{{1
// empty {{{2
template <class _T> struct __simd_tuple<_T> {
    using value_type = _T;
    static constexpr size_t tuple_size = 0;
    static constexpr size_t size() { return 0; }
};

// 1 member {{{2
template <class _T, class Abi0> struct __simd_tuple<_T, Abi0> {
    using value_type = _T;
    using __first_type = typename __simd_traits<_T, Abi0>::__simd_member_type;
    using __second_type = __simd_tuple<_T>;
    using __first_abi = Abi0;
    static constexpr size_t tuple_size = 1;
    static constexpr size_t size() { return simd_size_v<_T, Abi0>; }
    static constexpr size_t __first_size_v = simd_size_v<_T, Abi0>;
    alignas(sizeof(__first_type)) __first_type first;
    static constexpr __second_type second = {};

    template <size_t _Offset = 0, class F>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple generate(F &&gen, __size_constant<_Offset> = {})
    {
        return {gen(__tuple_element_meta<_T, Abi0, _Offset>())};
    }

    template <size_t _Offset = 0, class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC __simd_tuple apply_wrapped(F &&fun, const More &... more) const
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return {fun(make_meta<_Offset>(*this), first, more.first...)};
    }

    template <class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC friend __simd_tuple __simd_tuple_apply(F &&fun, const __simd_tuple &x,
                                                    More &&... more)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return __simd_tuple::apply_impl(
            __bool_constant<conjunction<__is_equal<
                size_t, __first_size_v, std::decay_t<More>::__first_size_v>...>::value>(),
            std::forward<F>(fun), x, std::forward<More>(more)...);
    }

private:
    template <class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl(true_type,  // __first_size_v is equal for all arguments
               F &&fun, const __simd_tuple &x, More &&... more)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("more.first = ", more.first..., "more = ", more...);
        return {fun(__tuple_element_meta<_T, Abi0, 0>(), x.first, more.first...)};
    }

    template <class F, class More>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple apply_impl(false_type,  // at least one argument in
                                                           // More has different
                                                           // __first_size_v, x has only one
                                                           // member, so More has 2 or
                                                           // more
                                              F &&fun, const __simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("y = ", y);
        return apply_impl(std::make_index_sequence<std::decay_t<More>::tuple_size>(),
                          std::forward<F>(fun), x, std::forward<More>(y));
    }

    template <class F, class More, size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple apply_impl(std::index_sequence<_Indexes...>, F &&fun,
                                              const __simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("y = ", y);
        auto tmp = std::experimental::concat(__get_simd_at<_Indexes>(y)...);
        const auto first = fun(__tuple_element_meta<_T, Abi0, 0>(), x.first, tmp);
        if constexpr (std::is_lvalue_reference<More>::value &&
                      !std::is_const<More>::value) {
            // if y is non-const lvalue ref, assume write back is necessary
            const auto tup =
                std::experimental::split<__simd_tuple_element_t<_Indexes, std::decay_t<More>>::size()...>(tmp);
            auto &&ignore = {
                (__get_tuple_at<_Indexes>(y) = __data(std::get<_Indexes>(tup)), 0)...};
            __unused(ignore);
        }
        return {first};
    }

public:
    // apply_impl2 can only be called from a 2-element __simd_tuple
    template <class Tuple, size_t _Offset, class F2>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple __extract(
        __size_constant<_Offset>, __size_constant<std::decay_t<Tuple>::__first_size_v - _Offset>,
        Tuple &&tup, F2 &&fun2)
    {
        static_assert(_Offset > 0, "");
        auto splitted =
            split<_Offset, std::decay_t<Tuple>::__first_size_v - _Offset>(__get_simd_at<0>(tup));
        __simd_tuple r = fun2(__data(std::get<1>(splitted)));
        // if tup is non-const lvalue ref, write __get_tuple_at<0>(splitted) back
        tup.first = __data(concat(std::get<0>(splitted), std::get<1>(splitted)));
        return r;
    }

    template <class F, class More, class _U, size_t Length, size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl2(chunked<_U, std::decay_t<More>::__first_size_v, Length, true,
                        std::index_sequence<_Indexes...>>,
                F &&fun, const __simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return __simd_tuple_apply(std::forward<F>(fun), x, y.second);
    }

    template <class F, class More, class _U, size_t _Offset, size_t Length,
              size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl2(chunked<_U, _Offset, Length, false, std::index_sequence<_Indexes...>>,
                F &&fun, const __simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        static_assert(_Offset < std::decay_t<More>::__first_size_v, "");
        static_assert(_Offset > 0, "");
        return __extract(__size_constant<_Offset>(), __size_constant<Length>(), y,
                       [&](auto &&yy) -> __simd_tuple {
                           return {fun(__tuple_element_meta<_T, Abi0, 0>(), x.first, yy)};
                       });
    }

    template <class R = _T, class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC __fixed_size_storage<R, size()> apply_r(F &&fun,
                                                       const More &... more) const
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return {fun(__tuple_element_meta<_T, Abi0, 0>(), first, more.first...)};
    }

    template <class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC friend std::bitset<size()> test(F &&fun, const __simd_tuple &x,
                                                 const More &... more)
    {
        return __vector_to_bitset(
            fun(__tuple_element_meta<_T, Abi0, 0>(), x.first, more.first...));
    }

    _T operator[](size_t i) const noexcept { return __subscript_read(first, i); }
    void set(size_t i, _T val) noexcept { __subscript_write(first, i, val); }
};

// 2 or more {{{2
template <class _T, class Abi0, class... _Abis> struct __simd_tuple<_T, Abi0, _Abis...> {
    using value_type = _T;
    using __first_type = typename __simd_traits<_T, Abi0>::__simd_member_type;
    using __first_abi = Abi0;
    using __second_type = __simd_tuple<_T, _Abis...>;
    static constexpr size_t tuple_size = sizeof...(_Abis) + 1;
    static constexpr size_t size() { return simd_size_v<_T, Abi0> + __second_type::size(); }
    static constexpr size_t __first_size_v = simd_size_v<_T, Abi0>;
    static constexpr size_t alignment =
        std::clamp(__next_power_of_2(sizeof(_T) * size()), size_t(16), size_t(256));
    alignas(alignment) __first_type first;
    __second_type second;

    template <size_t _Offset = 0, class F>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple generate(F &&gen, __size_constant<_Offset> = {})
    {
        return {gen(__tuple_element_meta<_T, Abi0, _Offset>()),
                __second_type::generate(
                    std::forward<F>(gen),
                    __size_constant<_Offset + simd_size_v<_T, Abi0>>())};
    }

    template <size_t _Offset = 0, class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC __simd_tuple apply_wrapped(F &&fun, const More &... more) const
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return {fun(make_meta<_Offset>(*this), first, more.first...),
                second.template apply_wrapped<_Offset + simd_size_v<_T, Abi0>>(
                    std::forward<F>(fun), more.second...)};
    }

    template <class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC friend __simd_tuple __simd_tuple_apply(F &&fun, const __simd_tuple &x,
                                                    More &&... more)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("more = ", more...);
        return __simd_tuple::apply_impl(
            __bool_constant<conjunction<__is_equal<size_t, __first_size_v,
                                       std::decay_t<More>::__first_size_v>...>::value>(),
            std::forward<F>(fun), x, std::forward<More>(more)...);
    }

private:
    template <class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl(true_type,  // __first_size_v is equal for all arguments
               F &&fun, const __simd_tuple &x, More &&... more)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return {fun(__tuple_element_meta<_T, Abi0, 0>(), x.first, more.first...),
                __simd_tuple_apply(std::forward<F>(fun), x.second, more.second...)};
    }

    template <class F, class More>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl(false_type,  // at least one argument in More has different __first_size_v
               F &&fun, const __simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("y = ", y);
        return apply_impl2(__how_many_to_extract<__first_size_v, std::decay_t<More>>::tag(),
                           std::forward<F>(fun), x, y);
    }

    template <class F, class More, size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple apply_impl2(std::index_sequence<_Indexes...>, F &&fun,
                                               const __simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("y = ", y);
        auto tmp = std::experimental::concat(__get_simd_at<_Indexes>(y)...);
        const auto first = fun(__tuple_element_meta<_T, Abi0, 0>(), x.first, tmp);
        if constexpr (std::is_lvalue_reference<More>::value &&
                      !std::is_const<More>::value) {
            // if y is non-const lvalue ref, assume write back is necessary
            const auto tup =
                std::experimental::split<__simd_tuple_element_t<_Indexes, std::decay_t<More>>::size()...>(tmp);
            [](std::initializer_list<int>) {
            }({(__get_tuple_at<_Indexes>(y) = __data(std::get<_Indexes>(tup)), 0)...});
        }
        return {first, __simd_tuple_apply(
                           std::forward<F>(fun), x.second,
                           __simd_tuple_pop_front(__size_constant<sizeof...(_Indexes)>(), y))};
    }

public:
    template <class F, class More, class _U, size_t Length, size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl2(chunked<_U, std::decay_t<More>::__first_size_v, Length, true,
                        std::index_sequence<_Indexes...>>,
                F &&fun, const __simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return __simd_tuple_apply(std::forward<F>(fun), x, y.second);
    }

    template <class Tuple, size_t Length, class F2>
    _GLIBCXX_SIMD_INTRINSIC static auto __extract(__size_constant<0>, __size_constant<Length>, Tuple &&tup,
                                     F2 &&fun2)
    {
        auto splitted =
            split<Length, std::decay_t<Tuple>::__first_size_v - Length>(__get_simd_at<0>(tup));
        auto r = fun2(__data(std::get<0>(splitted)));
        // if tup is non-const lvalue ref, write __get_tuple_at<0>(splitted) back
        tup.first = __data(concat(std::get<0>(splitted), std::get<1>(splitted)));
        return r;
    }

    template <class Tuple, size_t _Offset, class F2>
    _GLIBCXX_SIMD_INTRINSIC static auto __extract(
        __size_constant<_Offset>, __size_constant<std::decay_t<Tuple>::__first_size_v - _Offset>,
        Tuple &&tup, F2 &&fun2)
    {
        auto splitted =
            split<_Offset, std::decay_t<Tuple>::__first_size_v - _Offset>(__get_simd_at<0>(tup));
        auto r = fun2(__data(std::get<1>(splitted)));
        // if tup is non-const lvalue ref, write __get_tuple_at<0>(splitted) back
        tup.first = __data(concat(std::get<0>(splitted), std::get<1>(splitted)));
        return r;
    }

    template <
        class Tuple, size_t _Offset, size_t Length, class F2,
        class = enable_if_t<(_Offset + Length < std::decay_t<Tuple>::__first_size_v)>>
    _GLIBCXX_SIMD_INTRINSIC static auto __extract(__size_constant<_Offset>, __size_constant<Length>,
                                     Tuple &&tup, F2 &&fun2)
    {
        static_assert(_Offset + Length < std::decay_t<Tuple>::__first_size_v, "");
        auto splitted =
            split<_Offset, Length, std::decay_t<Tuple>::__first_size_v - _Offset - Length>(
                __get_simd_at<0>(tup));
        auto r = fun2(__data(std::get<1>(splitted)));
        // if tup is non-const lvalue ref, write __get_tuple_at<0>(splitted) back
        tup.first = __data(
            concat(std::get<0>(splitted), std::get<1>(splitted), std::get<2>(splitted)));
        return r;
    }

    template <class F, class More, class _U, size_t _Offset, size_t Length,
              size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl2(chunked<_U, _Offset, Length, false, std::index_sequence<_Indexes...>>,
                F &&fun, const __simd_tuple &x, More &&y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        static_assert(_Offset < std::decay_t<More>::__first_size_v, "");
        return {__extract(__size_constant<_Offset>(), __size_constant<Length>(), y,
                        [&](auto &&yy) {
                            return fun(__tuple_element_meta<_T, Abi0, 0>(), x.first, yy);
                        }),
                __second_type::apply_impl2(
                    chunked<_U, _Offset + Length, Length,
                            _Offset + Length == std::decay_t<More>::__first_size_v,
                            std::index_sequence<_Indexes...>>(),
                    std::forward<F>(fun), x.second, y)};
    }

    template <class R = _T, class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC auto apply_r(F &&fun, const More &... more) const
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return __simd_tuple_concat<R>(
            fun(__tuple_element_meta<_T, Abi0, 0>(), first, more.first...),
            second.template apply_r<R>(std::forward<F>(fun), more.second...));
    }

    template <class F, class... More>
    _GLIBCXX_SIMD_INTRINSIC friend std::bitset<size()> test(F &&fun, const __simd_tuple &x,
                                                 const More &... more)
    {
        return __vector_to_bitset(
                   fun(__tuple_element_meta<_T, Abi0, 0>(), x.first, more.first...))
                   .to_ullong() |
               (test(fun, x.second, more.second...).to_ullong() << simd_size_v<_T, Abi0>);
    }

    template <class _U, _U I>
    _GLIBCXX_SIMD_INTRINSIC constexpr _T operator[](std::integral_constant<_U, I>) const noexcept
    {
        if constexpr (I < simd_size_v<_T, Abi0>) {
            return __subscript_read(first, I);
        } else {
            return second[std::integral_constant<_U, I - simd_size_v<_T, Abi0>>()];
        }
    }

    _T operator[](size_t i) const noexcept
    {
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        return reinterpret_cast<const __may_alias<_T> *>(this)[i];
#else
        return i < simd_size_v<_T, Abi0> ? __subscript_read(first, i)
                                        : second[i - simd_size_v<_T, Abi0>];
#endif
    }
    void set(size_t i, _T val) noexcept
    {
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        reinterpret_cast<__may_alias<_T> *>(this)[i] = val;
#else
        if (i < simd_size_v<_T, Abi0>) {
            __subscript_write(first, i, val);
        } else {
            second.set(i - simd_size_v<_T, Abi0>, val);
        }
#endif
    }
};

// __make_simd_tuple {{{1
template <class _T, class _A0>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_T, _A0> __make_simd_tuple(
    std::experimental::simd<_T, _A0> x0)
{
    return {__data(x0)};
}
template <class _T, class _A0, class... As>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_T, _A0, As...> __make_simd_tuple(
    const std::experimental::simd<_T, _A0> &x0,
    const std::experimental::simd<_T, As> &... xs)
{
    return {__data(x0), __make_simd_tuple(xs...)};
}

template <class _T, class _A0>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_T, _A0> __make_simd_tuple(
    const typename __simd_traits<_T, _A0>::__simd_member_type &arg0)
{
    return {arg0};
}

template <class _T, class _A0, class _A1, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_T, _A0, _A1, _Abis...> __make_simd_tuple(
    const typename __simd_traits<_T, _A0>::__simd_member_type &arg0,
    const typename __simd_traits<_T, _A1>::__simd_member_type &arg1,
    const typename __simd_traits<_T, _Abis>::__simd_member_type &... args)
{
    return {arg0, __make_simd_tuple<_T, _A1, _Abis...>(arg1, args...)};
}

// __to_simd_tuple {{{1
template <size_t, class _T> using __to_tuple_helper = _T;
template <class _T, class _A0, size_t... _Indexes>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_T, __to_tuple_helper<_Indexes, _A0>...>
__to_simd_tuple_impl(std::index_sequence<_Indexes...>,
                     const std::array<__vector_type_t<_T, simd_size_v<_T, _A0>>,
                                      sizeof...(_Indexes)> &args)
{
    return __make_simd_tuple<_T, __to_tuple_helper<_Indexes, _A0>...>(args[_Indexes]...);
}

template <class _T, class _A0, size_t _N>
_GLIBCXX_SIMD_INTRINSIC auto __to_simd_tuple(
    const std::array<__vector_type_t<_T, simd_size_v<_T, _A0>>, _N> &args)
{
    return __to_simd_tuple_impl<_T, _A0>(std::make_index_sequence<_N>(), args);
}

// __optimize_simd_tuple {{{1
template <class _T> _GLIBCXX_SIMD_INTRINSIC __simd_tuple<_T> __optimize_simd_tuple(const __simd_tuple<_T>)
{
    return {};
}

template <class _T, class _A>
_GLIBCXX_SIMD_INTRINSIC const __simd_tuple<_T, _A> &__optimize_simd_tuple(const __simd_tuple<_T, _A> &x)
{
    return x;
}

template <class _T, class _A0, class _A1, class... _Abis,
          class R = __fixed_size_storage<_T, __simd_tuple<_T, _A0, _A1, _Abis...>::size()>>
_GLIBCXX_SIMD_INTRINSIC R __optimize_simd_tuple(const __simd_tuple<_T, _A0, _A1, _Abis...> &x)
{
    using Tup = __simd_tuple<_T, _A0, _A1, _Abis...>;
    if constexpr (R::__first_size_v == simd_size_v<_T, _A0>) {
        return __simd_tuple_concat(__simd_tuple<_T, typename R::__first_abi>{x.first},
                            __optimize_simd_tuple(x.second));
    } else if constexpr (R::__first_size_v == simd_size_v<_T, _A0> + simd_size_v<_T, _A1>) {
        return __simd_tuple_concat(__simd_tuple<_T, typename R::__first_abi>{__data(
                                std::experimental::concat(__get_simd_at<0>(x), __get_simd_at<1>(x)))},
                            __optimize_simd_tuple(x.second.second));
    } else if constexpr (sizeof...(_Abis) >= 2) {
        if constexpr (R::__first_size_v == __simd_tuple_element_t<0, Tup>::size() +
                                             __simd_tuple_element_t<1, Tup>::size() +
                                             __simd_tuple_element_t<2, Tup>::size() +
                                             __simd_tuple_element_t<3, Tup>::size()) {
            return __simd_tuple_concat(
                __simd_tuple<_T, typename R::__first_abi>{__data(concat(
                    __get_simd_at<0>(x), __get_simd_at<1>(x), __get_simd_at<2>(x), __get_simd_at<3>(x)))},
                __optimize_simd_tuple(x.second.second.second.second));
        }
    } else {
        return x;
    }
}

// __for_each(const __simd_tuple &, Fun) {{{1
template <size_t _Offset = 0, class _T, class _A0, class F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(const __simd_tuple<_T, _A0> &t_, F &&fun_)
{
    std::forward<F>(fun_)(make_meta<_Offset>(t_), t_.first);
}
template <size_t _Offset = 0, class _T, class _A0, class _A1, class... As, class F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(const __simd_tuple<_T, _A0, _A1, As...> &t_, F &&fun_)
{
    fun_(make_meta<_Offset>(t_), t_.first);
    __for_each<_Offset + simd_size<_T, _A0>::value>(t_.second, std::forward<F>(fun_));
}

// __for_each(__simd_tuple &, Fun) {{{1
template <size_t _Offset = 0, class _T, class _A0, class F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(__simd_tuple<_T, _A0> &t_, F &&fun_)
{
    std::forward<F>(fun_)(make_meta<_Offset>(t_), t_.first);
}
template <size_t _Offset = 0, class _T, class _A0, class _A1, class... As, class F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(__simd_tuple<_T, _A0, _A1, As...> &t_, F &&fun_)
{
    fun_(make_meta<_Offset>(t_), t_.first);
    __for_each<_Offset + simd_size<_T, _A0>::value>(t_.second, std::forward<F>(fun_));
}

// __for_each(__simd_tuple &, const __simd_tuple &, Fun) {{{1
template <size_t _Offset = 0, class _T, class _A0, class F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(__simd_tuple<_T, _A0> &a_, const __simd_tuple<_T, _A0> &b_,
                           F &&fun_)
{
    std::forward<F>(fun_)(make_meta<_Offset>(a_), a_.first, b_.first);
}
template <size_t _Offset = 0, class _T, class _A0, class _A1, class... As, class F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(__simd_tuple<_T, _A0, _A1, As...> & a_,
                           const __simd_tuple<_T, _A0, _A1, As...> &b_, F &&fun_)
{
    fun_(make_meta<_Offset>(a_), a_.first, b_.first);
    __for_each<_Offset + simd_size<_T, _A0>::value>(a_.second, b_.second,
                                                  std::forward<F>(fun_));
}

// __for_each(const __simd_tuple &, const __simd_tuple &, Fun) {{{1
template <size_t _Offset = 0, class _T, class _A0, class F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(const __simd_tuple<_T, _A0> &a_, const __simd_tuple<_T, _A0> &b_,
                           F &&fun_)
{
    std::forward<F>(fun_)(make_meta<_Offset>(a_), a_.first, b_.first);
}
template <size_t _Offset = 0, class _T, class _A0, class _A1, class... As, class F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(const __simd_tuple<_T, _A0, _A1, As...> &a_,
                           const __simd_tuple<_T, _A0, _A1, As...> &b_, F &&fun_)
{
    fun_(make_meta<_Offset>(a_), a_.first, b_.first);
    __for_each<_Offset + simd_size<_T, _A0>::value>(a_.second, b_.second,
                                                  std::forward<F>(fun_));
}

// }}}1
#if _GLIBCXX_SIMD_HAVE_SSE || _GLIBCXX_SIMD_HAVE_MMX
namespace __x86
{
// missing _mmXXX_mask_cvtepi16_storeu_epi8 intrinsics {{{
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
// __fixup_avx_xzyw{{{
template <class _T, class Traits = __vector_traits<_T>> _GLIBCXX_SIMD_INTRINSIC _T __fixup_avx_xzyw(_T a)
{
    static_assert(sizeof(_T) == 32);
    using _V = std::conditional_t<std::is_floating_point_v<typename Traits::value_type>,
                                 __m256d, __m256i>;
    const _V x = reinterpret_cast<_V>(a);
    return reinterpret_cast<_T>(_V{x[0], x[2], x[1], x[3]});
}

// }}}
// __shift_right{{{
template <int n> _GLIBCXX_SIMD_INTRINSIC __m128  __shift_right(__m128  v);
template <> _GLIBCXX_SIMD_INTRINSIC __m128  __shift_right< 0>(__m128  v) { return v; }
template <> _GLIBCXX_SIMD_INTRINSIC __m128  __shift_right<16>(__m128   ) { return _mm_setzero_ps(); }

#if _GLIBCXX_SIMD_HAVE_SSE2
template <int n> _GLIBCXX_SIMD_INTRINSIC __m128  __shift_right(__m128  v) { return _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), n)); }
template <int n> _GLIBCXX_SIMD_INTRINSIC __m128d __shift_right(__m128d v) { return _mm_castsi128_pd(_mm_srli_si128(_mm_castpd_si128(v), n)); }
template <int n> _GLIBCXX_SIMD_INTRINSIC __m128i __shift_right(__m128i v) { return _mm_srli_si128(v, n); }

template <> _GLIBCXX_SIMD_INTRINSIC __m128  __shift_right< 8>(__m128  v) { return _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(v), _mm_setzero_pd())); }
template <> _GLIBCXX_SIMD_INTRINSIC __m128d __shift_right< 0>(__m128d v) { return v; }
template <> _GLIBCXX_SIMD_INTRINSIC __m128d __shift_right< 8>(__m128d v) { return _mm_unpackhi_pd(v, _mm_setzero_pd()); }
template <> _GLIBCXX_SIMD_INTRINSIC __m128d __shift_right<16>(__m128d  ) { return _mm_setzero_pd(); }
template <> _GLIBCXX_SIMD_INTRINSIC __m128i __shift_right< 0>(__m128i v) { return v; }
template <> _GLIBCXX_SIMD_INTRINSIC __m128i __shift_right<16>(__m128i  ) { return _mm_setzero_si128(); }
#endif  // _GLIBCXX_SIMD_HAVE_SSE2

#if _GLIBCXX_SIMD_HAVE_AVX2
template <int n> _GLIBCXX_SIMD_INTRINSIC __m256 __shift_right(__m256 v)
{
    __m256i vi = _mm256_castps_si256(v);
    return _mm256_castsi256_ps(
        n < 16 ? _mm256_srli_si256(vi, n)
               : _mm256_srli_si256(_mm256_permute2x128_si256(vi, vi, 0x81), n));
}
template <> _GLIBCXX_SIMD_INTRINSIC __m256 __shift_right<0>(__m256 v) { return v; }
template <> _GLIBCXX_SIMD_INTRINSIC __m256 __shift_right<16>(__m256 v) { return __intrin_cast<__m256>(__lo128(v)); }
template <int n> _GLIBCXX_SIMD_INTRINSIC __m256d __shift_right(__m256d v)
{
    __m256i vi = _mm256_castpd_si256(v);
    return _mm256_castsi256_pd(
        n < 16 ? _mm256_srli_si256(vi, n)
               : _mm256_srli_si256(_mm256_permute2x128_si256(vi, vi, 0x81), n));
}
template <> _GLIBCXX_SIMD_INTRINSIC __m256d __shift_right<0>(__m256d v) { return v; }
template <> _GLIBCXX_SIMD_INTRINSIC __m256d __shift_right<16>(__m256d v) { return __intrin_cast<__m256d>(__lo128(v)); }
template <int n> _GLIBCXX_SIMD_INTRINSIC __m256i __shift_right(__m256i v)
{
    return n < 16 ? _mm256_srli_si256(v, n)
                  : _mm256_srli_si256(_mm256_permute2x128_si256(v, v, 0x81), n);
}
template <> _GLIBCXX_SIMD_INTRINSIC __m256i __shift_right<0>(__m256i v) { return v; }
template <> _GLIBCXX_SIMD_INTRINSIC __m256i __shift_right<16>(__m256i v) { return _mm256_permute2x128_si256(v, v, 0x81); }
#endif

// }}}
// __cmpord{{{
_GLIBCXX_SIMD_INTRINSIC __vector_type16_t<float> __cmpord(__vector_type16_t<float> x,
                                            __vector_type16_t<float> y)
{
    return _mm_cmpord_ps(x, y);
}
_GLIBCXX_SIMD_INTRINSIC __vector_type16_t<double> __cmpord(__vector_type16_t<double> x,
                                             __vector_type16_t<double> y)
{
    return _mm_cmpord_pd(x, y);
}

#if _GLIBCXX_SIMD_HAVE_AVX
_GLIBCXX_SIMD_INTRINSIC __vector_type32_t<float> __cmpord(__vector_type32_t<float> x,
                                            __vector_type32_t<float> y)
{
    return _mm256_cmp_ps(x, y, _CMP_ORD_Q);
}
_GLIBCXX_SIMD_INTRINSIC __vector_type32_t<double> __cmpord(__vector_type32_t<double> x,
                                             __vector_type32_t<double> y)
{
    return _mm256_cmp_pd(x, y, _CMP_ORD_Q);
}
#endif  // _GLIBCXX_SIMD_HAVE_AVX

#if _GLIBCXX_SIMD_HAVE_AVX512F
_GLIBCXX_SIMD_INTRINSIC __mmask16 __cmpord(__vector_type64_t<float> x, __vector_type64_t<float> y)
{
    return _mm512_cmp_ps_mask(x, y, _CMP_ORD_Q);
}
_GLIBCXX_SIMD_INTRINSIC __mmask8 __cmpord(__vector_type64_t<double> x, __vector_type64_t<double> y)
{
    return _mm512_cmp_pd_mask(x, y, _CMP_ORD_Q);
}
#endif  // _GLIBCXX_SIMD_HAVE_AVX512F

// }}}
// __cmpunord{{{
_GLIBCXX_SIMD_INTRINSIC __vector_type16_t<float> __cmpunord(__vector_type16_t<float> x,
                                              __vector_type16_t<float> y)
{
    return _mm_cmpunord_ps(x, y);
}
_GLIBCXX_SIMD_INTRINSIC __vector_type16_t<double> __cmpunord(__vector_type16_t<double> x,
                                               __vector_type16_t<double> y)
{
    return _mm_cmpunord_pd(x, y);
}

#if _GLIBCXX_SIMD_HAVE_AVX
_GLIBCXX_SIMD_INTRINSIC __vector_type32_t<float> __cmpunord(__vector_type32_t<float> x,
                                              __vector_type32_t<float> y)
{
    return _mm256_cmp_ps(x, y, _CMP_UNORD_Q);
}
_GLIBCXX_SIMD_INTRINSIC __vector_type32_t<double> __cmpunord(__vector_type32_t<double> x,
                                               __vector_type32_t<double> y)
{
    return _mm256_cmp_pd(x, y, _CMP_UNORD_Q);
}
#endif  // _GLIBCXX_SIMD_HAVE_AVX

#if _GLIBCXX_SIMD_HAVE_AVX512F
_GLIBCXX_SIMD_INTRINSIC __mmask16 __cmpunord(__vector_type64_t<float> x, __vector_type64_t<float> y)
{
    return _mm512_cmp_ps_mask(x, y, _CMP_UNORD_Q);
}
_GLIBCXX_SIMD_INTRINSIC __mmask8 __cmpunord(__vector_type64_t<double> x, __vector_type64_t<double> y)
{
    return _mm512_cmp_pd_mask(x, y, _CMP_UNORD_Q);
}
#endif  // _GLIBCXX_SIMD_HAVE_AVX512F

// }}}
// non-converting maskstore (SSE-AVX512BWVL) {{{
template <class _T, class F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage64_t<_T> v, _T *mem, F,
                            __storage<bool, __storage64_t<_T>::width> k)
{
    static_assert(sizeof(v) == 64 && __have_avx512f);
    if constexpr (__have_avx512bw && sizeof(_T) == 1) {
        _mm512_mask_storeu_epi8(mem, k, v);
    } else if constexpr (__have_avx512bw && sizeof(_T) == 2) {
        _mm512_mask_storeu_epi16(mem, k, v);
    } else if constexpr (sizeof(_T) == 4) {
        if constexpr (__is_aligned_v<F, 64> && std::is_integral_v<_T>) {
            _mm512_mask_store_epi32(mem, k, v);
        } else if constexpr (__is_aligned_v<F, 64> && std::is_floating_point_v<_T>) {
            _mm512_mask_store_ps(mem, k, v);
        } else if constexpr (std::is_integral_v<_T>) {
            _mm512_mask_storeu_epi32(mem, k, v);
        } else {
            _mm512_mask_storeu_ps(mem, k, v);
        }
    } else if constexpr (sizeof(_T) == 8) {
        if constexpr (__is_aligned_v<F, 64> && std::is_integral_v<_T>) {
            _mm512_mask_store_epi64(mem, k, v);
        } else if constexpr (__is_aligned_v<F, 64> && std::is_floating_point_v<_T>) {
            _mm512_mask_store_pd(mem, k, v);
        } else if constexpr (std::is_integral_v<_T>) {
            _mm512_mask_storeu_epi64(mem, k, v);
        } else {
            _mm512_mask_storeu_pd(mem, k, v);
        }
    } else {
        constexpr int _N = 16 / sizeof(_T);
        using M = __vector_type_t<_T, _N>;
        _mm_maskmoveu_si128(__auto_cast(__extract<0, 4>(v.d)),
                            __auto_cast(__convert_mask<M>(k.d)),
                            reinterpret_cast<char *>(mem));
        _mm_maskmoveu_si128(__auto_cast(__extract<1, 4>(v.d)),
                            __auto_cast(__convert_mask<M>(k.d >> 1 * _N)),
                            reinterpret_cast<char *>(mem) + 1 * 16);
        _mm_maskmoveu_si128(__auto_cast(__extract<2, 4>(v.d)),
                            __auto_cast(__convert_mask<M>(k.d >> 2 * _N)),
                            reinterpret_cast<char *>(mem) + 2 * 16);
        _mm_maskmoveu_si128(__auto_cast(__extract<3, 4>(v.d)),
                            __auto_cast(__convert_mask<M>(k.d >> 3 * _N)),
                            reinterpret_cast<char *>(mem) + 3 * 16);
    }
}

template <class _T, class F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage32_t<_T> v, _T *mem, F, __storage32_t<_T> k)
{
    static_assert(sizeof(v) == 32 && __have_avx);
    if constexpr (__have_avx512bw_vl && sizeof(_T) == 1) {
        _mm256_mask_storeu_epi8(mem, _mm256_movepi8_mask(k), v);
    } else if constexpr (__have_avx512bw_vl && sizeof(_T) == 2) {
        _mm256_mask_storeu_epi16(mem, _mm256_movepi16_mask(k), v);
    } else if constexpr (__have_avx2 && sizeof(_T) == 4 && std::is_integral_v<_T>) {
        _mm256_maskstore_epi32(reinterpret_cast<int *>(mem), k, v);
    } else if constexpr (sizeof(_T) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float *>(mem), __vector_cast<__llong>(k),
                            __vector_cast<float>(v));
    } else if constexpr (__have_avx2 && sizeof(_T) == 8 && std::is_integral_v<_T>) {
        _mm256_maskstore_epi64(reinterpret_cast<__llong *>(mem), k, v);
    } else if constexpr (sizeof(_T) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double *>(mem), __vector_cast<__llong>(k),
                            __vector_cast<double>(v));
    } else {
        _mm_maskmoveu_si128(__vector_cast<__llong>(__lo128(v)), __vector_cast<__llong>(__lo128(k)),
                            reinterpret_cast<char *>(mem));
        _mm_maskmoveu_si128(__vector_cast<__llong>(__hi128(v)), __vector_cast<__llong>(__hi128(k)),
                            reinterpret_cast<char *>(mem) + 16);
    }
}

template <class _T, class F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage32_t<_T> v, _T *mem, F,
                            __storage<bool, __storage32_t<_T>::width> k)
{
    static_assert(sizeof(v) == 32 && __have_avx512f);
    if constexpr (__have_avx512bw_vl && sizeof(_T) == 1) {
        _mm256_mask_storeu_epi8(mem, k, v);
    } else if constexpr (__have_avx512bw_vl && sizeof(_T) == 2) {
        _mm256_mask_storeu_epi16(mem, k, v);
    } else if constexpr (__have_avx512vl && sizeof(_T) == 4) {
        if constexpr (__is_aligned_v<F, 32> && std::is_integral_v<_T>) {
            _mm256_mask_store_epi32(mem, k, v);
        } else if constexpr (__is_aligned_v<F, 32> && std::is_floating_point_v<_T>) {
            _mm256_mask_store_ps(mem, k, v);
        } else if constexpr (std::is_integral_v<_T>) {
            _mm256_mask_storeu_epi32(mem, k, v);
        } else {
            _mm256_mask_storeu_ps(mem, k, v);
        }
    } else if constexpr (__have_avx512vl && sizeof(_T) == 8) {
        if constexpr (__is_aligned_v<F, 32> && std::is_integral_v<_T>) {
            _mm256_mask_store_epi64(mem, k, v);
        } else if constexpr (__is_aligned_v<F, 32> && std::is_floating_point_v<_T>) {
            _mm256_mask_store_pd(mem, k, v);
        } else if constexpr (std::is_integral_v<_T>) {
            _mm256_mask_storeu_epi64(mem, k, v);
        } else {
            _mm256_mask_storeu_pd(mem, k, v);
        }
    } else if constexpr (__have_avx512f && (sizeof(_T) >= 4 || __have_avx512bw)) {
        // use a 512-bit maskstore, using zero-extension of the bitmask
        __maskstore(
            __storage64_t<_T>(
                __intrin_cast<__intrinsic_type64_t<_T>>(v.d)),
            mem,
            // careful, vector_aligned has a stricter meaning in the 512-bit maskstore:
            std::conditional_t<std::is_same_v<F, vector_aligned_tag>, overaligned_tag<32>,
                               F>(),
            __storage<bool, 64 / sizeof(_T)>(k.d));
    } else {
        __maskstore(
            v, mem, F(),
            __storage32_t<_T>(__convert_mask<__vector_type_t<_T, 32 / sizeof(_T)>>(k)));
    }
}

template <class _T, class F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage16_t<_T> v, _T *mem, F, __storage16_t<_T> k)
{
    if constexpr (__have_avx512bw_vl && sizeof(_T) == 1) {
        _mm_mask_storeu_epi8(mem, _mm_movepi8_mask(k), v);
    } else if constexpr (__have_avx512bw_vl && sizeof(_T) == 2) {
        _mm_mask_storeu_epi16(mem, _mm_movepi16_mask(k), v);
    } else if constexpr (__have_avx2 && sizeof(_T) == 4 && std::is_integral_v<_T>) {
        _mm_maskstore_epi32(reinterpret_cast<int *>(mem), k, v);
    } else if constexpr (__have_avx && sizeof(_T) == 4) {
        _mm_maskstore_ps(reinterpret_cast<float *>(mem), __vector_cast<__llong>(k),
                         __vector_cast<float>(v));
    } else if constexpr (__have_avx2 && sizeof(_T) == 8 && std::is_integral_v<_T>) {
        _mm_maskstore_epi64(reinterpret_cast<__llong *>(mem), k, v);
    } else if constexpr (__have_avx && sizeof(_T) == 8) {
        _mm_maskstore_pd(reinterpret_cast<double *>(mem), __vector_cast<__llong>(k),
                         __vector_cast<double>(v));
    } else if constexpr (__have_sse2) {
        _mm_maskmoveu_si128(__vector_cast<__llong>(v), __vector_cast<__llong>(k),
                            reinterpret_cast<char *>(mem));
    } else {
        __execute_n_times<__storage16_t<_T>::width>([&](auto i) {
            if (k[i]) {
                mem[i] = v[i];
            }
        });
    }
}

template <class _T, class F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage16_t<_T> v, _T *mem, F,
                            __storage<bool, __storage16_t<_T>::width> k)
{
    static_assert(sizeof(v) == 16 && __have_avx512f);
    if constexpr (__have_avx512bw_vl && sizeof(_T) == 1) {
        _mm_mask_storeu_epi8(mem, k, v);
    } else if constexpr (__have_avx512bw_vl && sizeof(_T) == 2) {
        _mm_mask_storeu_epi16(mem, k, v);
    } else if constexpr (__have_avx512vl && sizeof(_T) == 4) {
        if constexpr (__is_aligned_v<F, 16> && std::is_integral_v<_T>) {
            _mm_mask_store_epi32(mem, k, v);
        } else if constexpr (__is_aligned_v<F, 16> && std::is_floating_point_v<_T>) {
            _mm_mask_store_ps(mem, k, v);
        } else if constexpr (std::is_integral_v<_T>) {
            _mm_mask_storeu_epi32(mem, k, v);
        } else {
            _mm_mask_storeu_ps(mem, k, v);
        }
    } else if constexpr (__have_avx512vl && sizeof(_T) == 8) {
        if constexpr (__is_aligned_v<F, 16> && std::is_integral_v<_T>) {
            _mm_mask_store_epi64(mem, k, v);
        } else if constexpr (__is_aligned_v<F, 16> && std::is_floating_point_v<_T>) {
            _mm_mask_store_pd(mem, k, v);
        } else if constexpr (std::is_integral_v<_T>) {
            _mm_mask_storeu_epi64(mem, k, v);
        } else {
            _mm_mask_storeu_pd(mem, k, v);
        }
    } else if constexpr (__have_avx512f && (sizeof(_T) >= 4 || __have_avx512bw)) {
        // use a 512-bit maskstore, using zero-extension of the bitmask
        __maskstore(
            __storage64_t<_T>(
                __intrin_cast<__intrinsic_type64_t<_T>>(v.d)),
            mem,
            // careful, vector_aligned has a stricter meaning in the 512-bit maskstore:
            std::conditional_t<std::is_same_v<F, vector_aligned_tag>, overaligned_tag<16>,
                               F>(),
            __storage<bool, 64 / sizeof(_T)>(k.d));
    } else {
        __maskstore(
            v, mem, F(),
            __storage16_t<_T>(__convert_mask<__vector_type_t<_T, 16 / sizeof(_T)>>(k)));
    }
}

// }}}
}  // namespace __x86
using namespace __x86;
#endif  // SSE || MMX
// __extract_part(__storage<_T, _N>) {{{
template <size_t _Index, size_t _Total, class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST
    __storage<_T, std::max(16 / sizeof(_T), _N / _Total)>
    __extract_part(__storage<_T, _N> x)
{
    constexpr size_t _NewN = _N / _Total;
    static_assert(_Total > _Index, "_Total must be greater than _Index");
    static_assert(_NewN * _Total == _N, "_N must be divisible by _Total");
    if constexpr (_Index == 0 && _Total == 1) {
        return x;
    } else if constexpr (sizeof(_T) * _NewN >= 16) {
        return __extract<_Index, _Total>(x.d);
    } else {
        static_assert(__have_sse && _N == _N);
        constexpr int split = sizeof(x) / 16;
        constexpr int shift = (sizeof(x) / _Total * _Index) % 16;
        return __x86::__shift_right<shift>(
            __extract_part<_Index * split / _Total, split>(x));
    }
}

// }}}
// __extract_part(__storage<bool, _N>) {{{
template <size_t _Index, size_t _Total, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<bool, _N / _Total> __extract_part(
    __storage<bool, _N> x)
{
    static_assert(__have_avx512f && _N == _N);
    static_assert(_Total >= 2 && _Index < _Total && _Index >= 0);
    return x.d >> (_Index * _N / _Total);
}

// }}}
// __extract_part(__simd_tuple) {{{
template <int _Index, int Parts, class _T, class _A0, class... As>
_GLIBCXX_SIMD_INTRINSIC auto __extract_part(const __simd_tuple<_T, _A0, As...> &x)
{
    // worst cases:
    // (a) 4, 4, 4 => 3, 3, 3, 3 (Parts = 4)
    // (b) 2, 2, 2 => 3, 3       (Parts = 2)
    // (c) 4, 2 => 2, 2, 2       (Parts = 3)
    using Tuple = __simd_tuple<_T, _A0, As...>;
    static_assert(_Index < Parts && _Index >= 0 && Parts >= 1);
    constexpr size_t _N = Tuple::size();
    static_assert(_N >= Parts && _N % Parts == 0);
    constexpr size_t values_per_part = _N / Parts;
    if constexpr (Parts == 1) {
        if constexpr (Tuple::tuple_size == 1) {
            return x.first;
        } else {
            return x;
        }
    } else if constexpr (simd_size_v<_T, _A0> % values_per_part != 0) {
        // nasty case: The requested partition does not match the partition of the
        // __simd_tuple. Fall back to construction via scalar copies.
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        const __may_alias<_T> *const element_ptr =
            reinterpret_cast<const __may_alias<_T> *>(&x) + _Index * values_per_part;
        return __data(simd<_T, simd_abi::deduce_t<_T, values_per_part>>(
            [&](auto i) { return element_ptr[i]; }));
#else
        constexpr size_t offset = _Index * values_per_part;
        __unused(offset);  // not really
        return __data(simd<_T, simd_abi::deduce_t<_T, values_per_part>>([&](auto i) {
            constexpr __size_constant<i + offset> k;
            return x[k];
        }));
#endif
    } else if constexpr (values_per_part * _Index >= simd_size_v<_T, _A0>) {  // recurse
        constexpr int parts_in_first = simd_size_v<_T, _A0> / values_per_part;
        return __extract_part<_Index - parts_in_first, Parts - parts_in_first>(x.second);
    } else {  // at this point we know that all of the return values are in x.first
        static_assert(values_per_part * (1 + _Index) <= simd_size_v<_T, _A0>);
        if constexpr (simd_size_v<_T, _A0> == values_per_part) {
            return x.first;
        } else {
            return __extract_part<_Index, simd_size_v<_T, _A0> / values_per_part>(
                x.first);
        }
    }
}
// }}}
// __to_storage specializations for bitset and __mmask<_N> {{{
#if _GLIBCXX_SIMD_HAVE_AVX512_ABI
template <size_t _N> class __to_storage<std::bitset<_N>>
{
    std::bitset<_N> d;

public:
    // can convert to larger storage for _Abi::is_partial == true
    template <class _U, size_t M> constexpr operator __storage<_U, M>() const
    {
        static_assert(M >= _N);
        return __convert_mask<__storage<_U, M>>(d);
    }
};

#define _GLIBCXX_SIMD_TO_STORAGE(_Type)                                                  \
    template <> class __to_storage<_Type>                                                \
    {                                                                                    \
        _Type d;                                                                         \
                                                                                         \
    public:                                                                              \
        template <class _U, size_t _N> constexpr operator __storage<_U, _N>() const      \
        {                                                                                \
            static_assert(_N >= sizeof(_Type) * CHAR_BIT);                               \
            return reinterpret_cast<__vector_type_t<_U, _N>>(                            \
                __convert_mask<__storage<_U, _N>>(d));                                   \
        }                                                                                \
                                                                                         \
        template <size_t _N> constexpr operator __storage<bool, _N>() const              \
        {                                                                                \
            static_assert(                                                               \
                std::is_same_v<_Type, typename __bool_storage_member_type<_N>::type>);   \
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
#if _GLIBCXX_SIMD_HAVE_SSE

#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85048
#include "simd_x86_conversions.h"
#endif  // _GLIBCXX_SIMD_WORKAROUND_PR85048

namespace __x86
{
// __converts_via_decomposition{{{
template <class _From, class _To, size_t _ToSize> struct __converts_via_decomposition {
private:
    static constexpr bool __i_to_i = is_integral_v<_From> && is_integral_v<_To>;
    static constexpr bool __f_to_i = is_floating_point_v<_From> && is_integral_v<_To>;
    static constexpr bool __f_to_f = is_floating_point_v<_From> && is_floating_point_v<_To>;
    static constexpr bool __i_to_f = is_integral_v<_From> && is_floating_point_v<_To>;

    template <size_t _A, size_t B>
    static constexpr bool __sizes = sizeof(_From) == _A && sizeof(_To) == B;

public:
    static constexpr bool value =
        (__i_to_i && __sizes<8, 2> && !__have_ssse3 && _ToSize == 16) ||
        (__i_to_i && __sizes<8, 1> && !__have_avx512f && _ToSize == 16) ||
        (__f_to_i && __sizes<4, 8> && !__have_avx512dq) ||
        (__f_to_i && __sizes<8, 8> && !__have_avx512dq) ||
        (__f_to_i && __sizes<8, 4> && !__have_sse4_1 && _ToSize == 16) ||
        (__i_to_f && __sizes<8, 4> && !__have_avx512dq && _ToSize == 16) ||
        (__i_to_f && __sizes<8, 8> && !__have_avx512dq && _ToSize < 64);
};

template <class _From, class _To, size_t _ToSize>
inline constexpr bool __converts_via_decomposition_v =
    __converts_via_decomposition<_From, _To, _ToSize>::value;

// }}}
// convert function{{{
template <class To, class From, class... _More>
_GLIBCXX_SIMD_INTRINSIC auto convert(From v0, _More... vs)
{
    static_assert((true && ... && is_same_v<From, _More>));
    if constexpr (__is_vectorizable_v<From>) {
        if constexpr (__is_vector_type_v<To>) {
            using _T = typename __vector_traits<To>::value_type;
            return __make_builtin(v0, vs...);
        } else {
            using _T = typename To::value_type;
            return __make_storage<_T>(v0, vs...);
        }
    } else if constexpr (!__is_vector_type_v<From>) {
        return convert<To>(v0.d, vs.d...);
    } else if constexpr (!__is_vector_type_v<To>) {
        return To(convert<typename To::register_type>(v0, vs...));
    } else if constexpr (__is_vectorizable_v<To>) {
        return convert<__vector_type_t<To, (__vector_traits<From>::width *
                                            (1 + sizeof...(_More)))>>(v0, vs...)
            .d;
    } else {
        static_assert(sizeof...(_More) == 0 ||
                          __vector_traits<To>::width >=
                              (1 + sizeof...(_More)) * __vector_traits<From>::width,
                      "convert(...) requires the input to fit into the output");
        return __vector_convert<To>(v0, vs...);
    }
}

// }}}
// convert_all function{{{
template <typename To, typename From> _GLIBCXX_SIMD_INTRINSIC auto convert_all(From v)
{
    static_assert(__is_vector_type_v<To>);
    if constexpr (__is_vector_type_v<From>) {
        using _Trait = __vector_traits<From>;
        using S = __storage<typename _Trait::value_type, _Trait::width>;
        return convert_all<To>(S(v));
    } else if constexpr (From::width > __vector_traits<To>::width) {
        constexpr size_t _N = From::width / __vector_traits<To>::width;
        return __generate_from_n_evaluations<_N, std::array<To, _N>>([&](auto i) {
            auto part = __extract_part<decltype(i)::value, _N>(v);
            return convert<To>(part);
        });
    } else {
        return convert<To>(v);
    }
}

// }}}
// plus{{{
template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, _N> plus(__storage<_T, _N> a, __storage<_T, _N> b)
{
    return a.d + b.d;
}

// minus{{{
template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, _N> minus(__storage<_T, _N> a, __storage<_T, _N> b)
{
    return a.d - b.d;
}

// multiplies{{{
template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, _N> multiplies(__storage<_T, _N> a, __storage<_T, _N> b)
{
    if constexpr (sizeof(_T) == 1) {
        return __vector_cast<_T>(
            ((__vector_cast<short>(a) * __vector_cast<short>(b)) &
             __vector_cast<short>(~__vector_type_t<ushort, _N / 2>() >> 8)) |
            (((__vector_cast<short>(a) >> 8) * (__vector_cast<short>(b) >> 8)) << 8));
    }
    return a.d * b.d;
}

// complement{{{
template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, _N> complement(__storage<_T, _N> v)
{
    return ~v.d;
}

//}}}1
// unary_minus{{{
// GCC doesn't use the psign instructions, but pxor & psub seem to be just as good a
// choice as pcmpeqd & psign. So meh.
template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, _N> unary_minus(__storage<_T, _N> v)
{
    return -v.d;
}

// }}}
// abs{{{
template <class _T, size_t _N>
_GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, _N> abs(__storage<_T, _N> v)
{
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85572
    if constexpr (!__have_avx512vl && std::is_integral_v<_T> && sizeof(_T) == 8 && _N <= 4) {
        // positive value:
        //   negative == 0
        //   a unchanged after xor
        //   a - 0 -> a
        // negative value:
        //   negative == ~0 == -1
        //   a xor ~0    -> -a - 1
        //   -a - 1 - -1 -> -a
        if constexpr(__have_sse4_2) {
            const auto negative = reinterpret_cast<__vector_type_t<_T, _N>>(v.d < 0);
            return (v.d ^ negative) - negative;
        } else {
            // arithmetic right shift doesn't exist for 64-bit integers, use the following
            // instead:
            // >>63: negative ->  1, positive ->  0
            //  -  : negative -> -1, positive ->  0
            const auto negative = -reinterpret_cast<__vector_type_t<_T, _N>>(
                reinterpret_cast<__vector_type_t<__ullong, _N>>(v.d) >> 63);
            return (v.d ^ negative) - negative;
        }
    } else
#endif
        if constexpr (std::is_floating_point_v<_T>) {
        // this workaround is only required because __builtin_abs is not a constant
        // expression
        using I = std::make_unsigned_t<__int_for_sizeof_t<_T>>;
        return __vector_cast<_T>(__vector_cast<I>(v.d) & __vector_broadcast<_N, I>(~I() >> 1));
    } else {
        return v.d < 0 ? -v.d : v.d;
    }
}

//}}}
}  // namespace __x86
#endif  // _GLIBCXX_SIMD_HAVE_SSE

// interleave (lo/hi/128) {{{
template <class _A, class B, class _T = std::common_type_t<_A, B>,
          class _Trait = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T __interleave_lo(_A _a, B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const _T a(_a);
    const _T b(_b);
    if constexpr (sizeof(_T) == 16 && needs_intrinsics) {
        if constexpr (_Trait::width == 2) {
            if constexpr (std::is_integral_v<typename _Trait::value_type>) {
                return reinterpret_cast<_T>(
                    _mm_unpacklo_epi64(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
            } else {
                return reinterpret_cast<_T>(
                    _mm_unpacklo_pd(__vector_cast<double>(a), __vector_cast<double>(b)));
            }
        } else if constexpr (_Trait::width == 4) {
            if constexpr (std::is_integral_v<typename _Trait::value_type>) {
                return reinterpret_cast<_T>(
                    _mm_unpacklo_epi32(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
            } else {
                return reinterpret_cast<_T>(
                    _mm_unpacklo_ps(__vector_cast<float>(a), __vector_cast<float>(b)));
            }
        } else if constexpr (_Trait::width == 8) {
            return reinterpret_cast<_T>(
                _mm_unpacklo_epi16(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
        } else if constexpr (_Trait::width == 16) {
            return reinterpret_cast<_T>(
                _mm_unpacklo_epi8(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
        }
    } else if constexpr (_Trait::width == 2) {
        return _T{a[0], b[0]};
    } else if constexpr (_Trait::width == 4) {
        return _T{a[0], b[0], a[1], b[1]};
    } else if constexpr (_Trait::width == 8) {
        return _T{a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]};
    } else if constexpr (_Trait::width == 16) {
        return _T{a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3],
                 a[4], b[4], a[5], b[5], a[6], b[6], a[7], b[7]};
    } else if constexpr (_Trait::width == 32) {
        return _T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],
                 a[4],  b[4],  a[5],  b[5],  a[6],  b[6],  a[7],  b[7],
                 a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11],
                 a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15]};
    } else if constexpr (_Trait::width == 64) {
        return _T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],  a[4],  b[4],
                 a[5],  b[5],  a[6],  b[6],  a[7],  b[7],  a[8],  b[8],  a[9],  b[9],
                 a[10], b[10], a[11], b[11], a[12], b[12], a[13], b[13], a[14], b[14],
                 a[15], b[15], a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19],
                 a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23], a[24], b[24],
                 a[25], b[25], a[26], b[26], a[27], b[27], a[28], b[28], a[29], b[29],
                 a[30], b[30], a[31], b[31]};
    } else {
        __assert_unreachable<_T>();
    }
}

template <class _A, class B, class _T = std::common_type_t<_A, B>,
          class _Trait = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T __interleave_hi(_A _a, B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const _T a(_a);
    const _T b(_b);
    if constexpr (sizeof(_T) == 16 && needs_intrinsics) {
        if constexpr (_Trait::width == 2) {
            if constexpr (std::is_integral_v<typename _Trait::value_type>) {
                return reinterpret_cast<_T>(
                    _mm_unpackhi_epi64(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
            } else {
                return reinterpret_cast<_T>(
                    _mm_unpackhi_pd(__vector_cast<double>(a), __vector_cast<double>(b)));
            }
        } else if constexpr (_Trait::width == 4) {
            if constexpr (std::is_integral_v<typename _Trait::value_type>) {
                return reinterpret_cast<_T>(
                    _mm_unpackhi_epi32(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
            } else {
                return reinterpret_cast<_T>(
                    _mm_unpackhi_ps(__vector_cast<float>(a), __vector_cast<float>(b)));
            }
        } else if constexpr (_Trait::width == 8) {
            return reinterpret_cast<_T>(
                _mm_unpackhi_epi16(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
        } else if constexpr (_Trait::width == 16) {
            return reinterpret_cast<_T>(
                _mm_unpackhi_epi8(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
        }
    } else if constexpr (_Trait::width == 2) {
        return _T{a[1], b[1]};
    } else if constexpr (_Trait::width == 4) {
        return _T{a[2], b[2], a[3], b[3]};
    } else if constexpr (_Trait::width == 8) {
        return _T{a[4], b[4], a[5], b[5], a[6], b[6], a[7], b[7]};
    } else if constexpr (_Trait::width == 16) {
        return _T{a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11],
                 a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15]};
    } else if constexpr (_Trait::width == 32) {
        return _T{a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19],
                 a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23],
                 a[24], b[24], a[25], b[25], a[26], b[26], a[27], b[27],
                 a[28], b[28], a[29], b[29], a[30], b[30], a[31], b[31]};
    } else if constexpr (_Trait::width == 64) {
        return _T{a[32], b[32], a[33], b[33], a[34], b[34], a[35], b[35],
                 a[36], b[36], a[37], b[37], a[38], b[38], a[39], b[39],
                 a[40], b[40], a[41], b[41], a[42], b[42], a[43], b[43],
                 a[44], b[44], a[45], b[45], a[46], b[46], a[47], b[47],
                 a[48], b[48], a[49], b[49], a[50], b[50], a[51], b[51],
                 a[52], b[52], a[53], b[53], a[54], b[54], a[55], b[55],
                 a[56], b[56], a[57], b[57], a[58], b[58], a[59], b[59],
                 a[60], b[60], a[61], b[61], a[62], b[62], a[63], b[63]};
    } else {
        __assert_unreachable<_T>();
    }
}

template <class _A, class B, class _T = std::common_type_t<_A, B>,
          class _Trait = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T interleave128_lo(_A _a, B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const _T a(_a);
    const _T b(_b);
    if constexpr (sizeof(_T) == 16) {
        return __interleave_lo(a, b);
    } else if constexpr (sizeof(_T) == 32 && needs_intrinsics) {
        if constexpr (_Trait::width == 4) {
            return reinterpret_cast<_T>(
                _mm256_unpacklo_pd(__vector_cast<double>(a), __vector_cast<double>(b)));
        } else if constexpr (_Trait::width == 8) {
            return reinterpret_cast<_T>(
                _mm256_unpacklo_ps(__vector_cast<float>(a), __vector_cast<float>(b)));
        } else if constexpr (_Trait::width == 16) {
            return reinterpret_cast<_T>(
                _mm256_unpacklo_epi16(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
        } else if constexpr (_Trait::width == 32) {
            return reinterpret_cast<_T>(
                _mm256_unpacklo_epi8(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
        }
    } else if constexpr (sizeof(_T) == 32) {
        if constexpr (_Trait::width == 4) {
            return _T{a[0], b[0], a[2], b[2]};
        } else if constexpr (_Trait::width == 8) {
            return _T{a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5]};
        } else if constexpr (_Trait::width == 16) {
            return _T{a[0], b[0], a[1], b[1], a[2],  b[2],  a[3],  b[3],
                     a[8], b[8], a[9], b[9], a[10], b[10], a[11], b[11]};
        } else if constexpr (_Trait::width == 32) {
            return _T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],
                     a[4],  b[4],  a[5],  b[5],  a[6],  b[6],  a[7],  b[7],
                     a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19],
                     a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23]};
        } else if constexpr (_Trait::width == 64) {
            return _T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],  a[4],  b[4],
                     a[5],  b[5],  a[6],  b[6],  a[7],  b[7],  a[8],  b[8],  a[9],  b[9],
                     a[10], b[10], a[11], b[11], a[12], b[12], a[13], b[13], a[14], b[14],
                     a[15], b[15], a[32], b[32], a[33], b[33], a[34], b[34], a[35], b[35],
                     a[36], b[36], a[37], b[37], a[38], b[38], a[39], b[39], a[40], b[40],
                     a[41], b[41], a[42], b[42], a[43], b[43], a[44], b[44], a[45], b[45],
                     a[46], b[46], a[47], b[47]};
        } else {
            __assert_unreachable<_T>();
        }
    } else if constexpr (sizeof(_T) == 64 && needs_intrinsics) {
        if constexpr (_Trait::width == 8) {
            return reinterpret_cast<_T>(
                _mm512_unpacklo_pd(__vector_cast<double>(a), __vector_cast<double>(b)));
        } else if constexpr (_Trait::width == 16) {
            return reinterpret_cast<_T>(
                _mm512_unpacklo_ps(__vector_cast<float>(a), __vector_cast<float>(b)));
        } else if constexpr (_Trait::width == 32) {
            return reinterpret_cast<_T>(
                _mm512_unpacklo_epi16(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
        } else if constexpr (_Trait::width == 64) {
            return reinterpret_cast<_T>(
                _mm512_unpacklo_epi8(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
        }
    } else if constexpr (sizeof(_T) == 64) {
        if constexpr (_Trait::width == 8) {
            return _T{a[0], b[0], a[2], b[2], a[4], b[4], a[6], b[6]};
        } else if constexpr (_Trait::width == 16) {
            return _T{a[0], b[0], a[1], b[1], a[4],  b[4],  a[5],  b[5],
                     a[8], b[8], a[9], b[9], a[12], b[12], a[13], b[13]};
        } else if constexpr (_Trait::width == 32) {
            return _T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],
                     a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11],
                     a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19],
                     a[24], b[24], a[25], b[25], a[26], b[26], a[27], b[27]};
        } else if constexpr (_Trait::width == 64) {
            return _T{a[0],  b[0],  a[1],  b[1],  a[2],  b[2],  a[3],  b[3],  a[4],  b[4],
                     a[5],  b[5],  a[6],  b[6],  a[7],  b[7],  a[16], b[16], a[17], b[17],
                     a[18], b[18], a[19], b[19], a[20], b[20], a[21], b[21], a[22], b[22],
                     a[23], b[23], a[32], b[32], a[33], b[33], a[34], b[34], a[35], b[35],
                     a[36], b[36], a[37], b[37], a[38], b[38], a[39], b[39], a[48], b[48],
                     a[49], b[49], a[50], b[50], a[51], b[51], a[52], b[52], a[53], b[53],
                     a[54], b[54], a[55], b[55]};
        } else {
            __assert_unreachable<_T>();
        }
    }
}

template <class _A, class B, class _T = std::common_type_t<_A, B>,
          class _Trait = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T interleave128_hi(_A _a, B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const _T a(_a);
    const _T b(_b);
    if constexpr (sizeof(_T) == 16) {
        return __interleave_hi(a, b);
    } else if constexpr (sizeof(_T) == 32 && needs_intrinsics) {
        if constexpr (_Trait::width == 4) {
            return reinterpret_cast<_T>(
                _mm256_unpackhi_pd(__vector_cast<double>(a), __vector_cast<double>(b)));
        } else if constexpr (_Trait::width == 8) {
            return reinterpret_cast<_T>(
                _mm256_unpackhi_ps(__vector_cast<float>(a), __vector_cast<float>(b)));
        } else if constexpr (_Trait::width == 16) {
            return reinterpret_cast<_T>(
                _mm256_unpackhi_epi16(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
        } else if constexpr (_Trait::width == 32) {
            return reinterpret_cast<_T>(
                _mm256_unpackhi_epi8(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
        }
    } else if constexpr (sizeof(_T) == 32) {
        if constexpr (_Trait::width == 4) {
            return _T{a[1], b[1], a[3], b[3]};
        } else if constexpr (_Trait::width == 8) {
            return _T{a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7]};
        } else if constexpr (_Trait::width == 16) {
            return _T{a[4],  b[4],  a[5],  b[5],  a[6],  b[6],  a[7],  b[7],
                     a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15]};
        } else if constexpr (_Trait::width == 32) {
            return _T{a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11],
                     a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15],
                     a[24], b[24], a[25], b[25], a[26], b[26], a[27], b[27],
                     a[28], b[28], a[29], b[29], a[30], b[30], a[31], b[31]};
        } else if constexpr (_Trait::width == 64) {
            return _T{a[16], b[16], a[17], b[17], a[18], b[18], a[19], b[19], a[20], b[20],
                     a[21], b[21], a[22], b[22], a[23], b[23], a[24], b[24], a[25], b[25],
                     a[26], b[26], a[27], b[27], a[28], b[28], a[29], b[29], a[30], b[30],
                     a[31], b[31], a[48], b[48], a[49], b[49], a[50], b[50], a[51], b[51],
                     a[52], b[52], a[53], b[53], a[54], b[54], a[55], b[55], a[56], b[56],
                     a[57], b[57], a[58], b[58], a[59], b[59], a[60], b[60], a[61], b[61],
                     a[62], b[62], a[63], b[63]};
        } else {
            __assert_unreachable<_T>();
        }
    } else if constexpr (sizeof(_T) == 64 && needs_intrinsics) {
        if constexpr (_Trait::width == 8) {
            return reinterpret_cast<_T>(
                _mm512_unpackhi_pd(__vector_cast<double>(a), __vector_cast<double>(b)));
        } else if constexpr (_Trait::width == 16) {
            return reinterpret_cast<_T>(
                _mm512_unpackhi_ps(__vector_cast<float>(a), __vector_cast<float>(b)));
        } else if constexpr (_Trait::width == 32) {
            return reinterpret_cast<_T>(
                _mm512_unpackhi_epi16(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
        } else if constexpr (_Trait::width == 64) {
            return reinterpret_cast<_T>(
                _mm512_unpackhi_epi8(__vector_cast<__llong>(a), __vector_cast<__llong>(b)));
        }
    } else if constexpr (sizeof(_T) == 64) {
        if constexpr (_Trait::width == 8) {
            return _T{a[1], b[1], a[3], b[3], a[5], b[5], a[7], b[7]};
        } else if constexpr (_Trait::width == 16) {
            return _T{a[2],  b[2],  a[3],  b[3],  a[6],  b[6],  a[7],  b[7],
                     a[10], b[10], a[11], b[11], a[14], b[14], a[15], b[15]};
        } else if constexpr (_Trait::width == 32) {
            return _T{a[4],  b[4],  a[5],  b[5],  a[6],  b[6],  a[7],  b[7],
                     a[12], b[12], a[13], b[13], a[14], b[14], a[15], b[15],
                     a[20], b[20], a[21], b[21], a[22], b[22], a[23], b[23],
                     a[28], b[28], a[29], b[29], a[30], b[30], a[31], b[31]};
        } else if constexpr (_Trait::width == 64) {
            return _T{a[8],  b[8],  a[9],  b[9],  a[10], b[10], a[11], b[11], a[12], b[12],
                     a[13], b[13], a[14], b[14], a[15], b[15], a[24], b[24], a[25], b[25],
                     a[26], b[26], a[27], b[27], a[28], b[28], a[29], b[29], a[30], b[30],
                     a[31], b[31], a[40], b[40], a[41], b[41], a[42], b[42], a[43], b[43],
                     a[44], b[44], a[45], b[45], a[46], b[46], a[47], b[47], a[56], b[56],
                     a[57], b[57], a[58], b[58], a[59], b[59], a[60], b[60], a[61], b[61],
                     a[62], b[62], a[63], b[63]};
        } else {
            __assert_unreachable<_T>();
        }
    }
}

template <class _T> struct interleaved_pair {
    _T lo, hi;
};

template <class _A, class B, class _T = std::common_type_t<_A, B>,
          class _Trait = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr interleaved_pair<_T> interleave(_A a, B b)
{
    return {__interleave_lo(a, b), __interleave_hi(a, b)};
}

template <class _A, class B, class _T = std::common_type_t<_A, B>,
          class _Trait = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr interleaved_pair<_T> interleave128(_A a, B b)
{
    return {interleave128_lo(a, b), interleave128_hi(a, b)};
}
// }}}
// __is_bitset {{{
template <class _T> struct __is_bitset : false_type {};
template <size_t _N> struct __is_bitset<std::bitset<_N>> : true_type {};
template <class _T> inline constexpr bool __is_bitset_v = __is_bitset<_T>::value;

// }}}
// __is_storage {{{
template <class _T> struct __is_storage : false_type {};
template <class _T, size_t _N> struct __is_storage<__storage<_T, _N>> : true_type {};
template <class _T> inline constexpr bool __is_storage_v = __is_storage<_T>::value;

// }}}
// __convert_mask{{{
template <class To, class From> inline To __convert_mask(From k) {
    if constexpr (std::is_same_v<To, From>) {  // also covers bool -> bool
        return k;
    } else if constexpr (std::is_unsigned_v<From> && std::is_unsigned_v<To>) {
        // bits -> bits
        return k;  // zero-extends or truncates
    } else if constexpr (__is_bitset_v<From>) {
        // from std::bitset {{{
        static_assert(k.size() <= sizeof(__ullong) * CHAR_BIT);
        using _T = std::conditional_t<
            (k.size() <= sizeof(ushort) * CHAR_BIT),
            std::conditional_t<(k.size() <= CHAR_BIT), __uchar, ushort>,
            std::conditional_t<(k.size() <= sizeof(uint) * CHAR_BIT), uint, __ullong>>;
        return __convert_mask<To>(static_cast<_T>(k.to_ullong()));
        // }}}
    } else if constexpr (__is_bitset_v<To>) {
        // to std::bitset {{{
        static_assert(To().size() <= sizeof(__ullong) * CHAR_BIT);
        using _T = std::conditional_t<
            (To().size() <= sizeof(ushort) * CHAR_BIT),
            std::conditional_t<(To().size() <= CHAR_BIT), __uchar, ushort>,
            std::conditional_t<(To().size() <= sizeof(uint) * CHAR_BIT), uint, __ullong>>;
        return __convert_mask<_T>(k);
        // }}}
    } else if constexpr (__is_storage_v<From>) {
        return __convert_mask<To>(k.d);
    } else if constexpr (__is_storage_v<To>) {
        return __convert_mask<typename To::register_type>(k);
    } else if constexpr (std::is_unsigned_v<From> && __is_vector_type_v<To>) {
        // bits -> vector {{{
        using _Trait = __vector_traits<To>;
        constexpr size_t N_in = sizeof(From) * CHAR_BIT;
        using ToT = typename _Trait::value_type;
        constexpr size_t N_out = _Trait::width;
        constexpr size_t _N = std::min(N_in, N_out);
        constexpr size_t bytes_per_output_element = sizeof(ToT);
        if constexpr (__have_avx512f) {
            if constexpr (bytes_per_output_element == 1 && sizeof(To) == 16) {
                if constexpr (__have_avx512bw_vl) {
                    return __vector_cast<ToT>(_mm_movm_epi8(k));
                } else if constexpr (__have_avx512bw) {
                    return __vector_cast<ToT>(__lo128(_mm512_movm_epi8(k)));
                } else {
                    auto as32bits = _mm512_maskz_mov_epi32(k, ~__m512i());
                    auto as16bits = __fixup_avx_xzyw(
                        _mm256_packs_epi32(__lo256(as32bits), __hi256(as32bits)));
                    return __vector_cast<ToT>(
                        _mm_packs_epi16(__lo128(as16bits), __hi128(as16bits)));
                }
            } else if constexpr (bytes_per_output_element == 1 && sizeof(To) == 32) {
                if constexpr (__have_avx512bw_vl) {
                    return __vector_cast<ToT>(_mm256_movm_epi8(k));
                } else if constexpr (__have_avx512bw) {
                    return __vector_cast<ToT>(__lo256(_mm512_movm_epi8(k)));
                } else {
                    auto as16bits =  // 0 16 1 17 ... 15 31
                        _mm512_srli_epi32(_mm512_maskz_mov_epi32(k, ~__m512i()), 16) |
                        _mm512_slli_epi32(_mm512_maskz_mov_epi32(k >> 16, ~__m512i()),
                                          16);
                    auto _0_16_1_17 = __fixup_avx_xzyw(_mm256_packs_epi16(
                        __lo256(as16bits),
                        __hi256(as16bits))  // 0 16 1 17 2 18 3 19 8 24 9 25 ...
                    );
                    // deinterleave:
                    return __vector_cast<ToT>(__fixup_avx_xzyw(_mm256_shuffle_epi8(
                        _0_16_1_17,  // 0 16 1 17 2 ...
                        _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13,
                                         15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11,
                                         13, 15))));  // 0-7 16-23 8-15 24-31 -> xzyw
                                                      // 0-3  8-11 16-19 24-27
                                                      // 4-7 12-15 20-23 28-31
                }
            } else if constexpr (bytes_per_output_element == 1 && sizeof(To) == 64) {
                return reinterpret_cast<__vector_type_t<__schar, 64>>(_mm512_movm_epi8(k));
            } else if constexpr (bytes_per_output_element == 2 && sizeof(To) == 16) {
                if constexpr (__have_avx512bw_vl) {
                    return __vector_cast<ToT>(_mm_movm_epi16(k));
                } else if constexpr (__have_avx512bw) {
                    return __vector_cast<ToT>(__lo128(_mm512_movm_epi16(k)));
                } else {
                    __m256i as32bits;
                    if constexpr (__have_avx512vl) {
                        as32bits = _mm256_maskz_mov_epi32(k, ~__m256i());
                    } else {
                        as32bits = __lo256(_mm512_maskz_mov_epi32(k, ~__m512i()));
                    }
                    return __vector_cast<ToT>(
                        _mm_packs_epi32(__lo128(as32bits), __hi128(as32bits)));
                }
            } else if constexpr (bytes_per_output_element == 2 && sizeof(To) == 32) {
                if constexpr (__have_avx512bw_vl) {
                    return __vector_cast<ToT>(_mm256_movm_epi16(k));
                } else if constexpr (__have_avx512bw) {
                    return __vector_cast<ToT>(__lo256(_mm512_movm_epi16(k)));
                } else {
                    auto as32bits = _mm512_maskz_mov_epi32(k, ~__m512i());
                    return __vector_cast<ToT>(__fixup_avx_xzyw(
                        _mm256_packs_epi32(__lo256(as32bits), __hi256(as32bits))));
                }
            } else if constexpr (bytes_per_output_element == 2 && sizeof(To) == 64) {
                return __vector_cast<ToT>(_mm512_movm_epi16(k));
            } else if constexpr (bytes_per_output_element == 4 && sizeof(To) == 16) {
                return __vector_cast<ToT>(
                    __have_avx512dq_vl
                        ? _mm_movm_epi32(k)
                        : __have_avx512dq
                              ? __lo128(_mm512_movm_epi32(k))
                              : __have_avx512vl
                                    ? _mm_maskz_mov_epi32(k, ~__m128i())
                                    : __lo128(_mm512_maskz_mov_epi32(k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 4 && sizeof(To) == 32) {
                return __vector_cast<ToT>(
                    __have_avx512dq_vl
                        ? _mm256_movm_epi32(k)
                        : __have_avx512dq
                              ? __lo256(_mm512_movm_epi32(k))
                              : __have_avx512vl
                                    ? _mm256_maskz_mov_epi32(k, ~__m256i())
                                    : __lo256(_mm512_maskz_mov_epi32(k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 4 && sizeof(To) == 64) {
                return __vector_cast<ToT>(__have_avx512dq
                                             ? _mm512_movm_epi32(k)
                                             : _mm512_maskz_mov_epi32(k, ~__m512i()));
            } else if constexpr (bytes_per_output_element == 8 && sizeof(To) == 16) {
                return __vector_cast<ToT>(
                    __have_avx512dq_vl
                        ? _mm_movm_epi64(k)
                        : __have_avx512dq
                              ? __lo128(_mm512_movm_epi64(k))
                              : __have_avx512vl
                                    ? _mm_maskz_mov_epi64(k, ~__m128i())
                                    : __lo128(_mm512_maskz_mov_epi64(k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 8 && sizeof(To) == 32) {
                return __vector_cast<ToT>(
                    __have_avx512dq_vl
                        ? _mm256_movm_epi64(k)
                        : __have_avx512dq
                              ? __lo256(_mm512_movm_epi64(k))
                              : __have_avx512vl
                                    ? _mm256_maskz_mov_epi64(k, ~__m256i())
                                    : __lo256(_mm512_maskz_mov_epi64(k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 8 && sizeof(To) == 64) {
                return __vector_cast<ToT>(__have_avx512dq
                                             ? _mm512_movm_epi64(k)
                                             : _mm512_maskz_mov_epi64(k, ~__m512i()));
            } else {
                __assert_unreachable<To>();
            }
        } else if constexpr (__have_sse) {
            using _U = std::make_unsigned_t<__int_for_sizeof_t<ToT>>;
            using _V = __vector_type_t<_U, _N>;  // simd<_U, _Abi>;
            static_assert(sizeof(_V) <= 32);  // can't be AVX512
            constexpr size_t bits_per_element = sizeof(_U) * CHAR_BIT;
            if constexpr (!__have_avx2 && __have_avx && sizeof(_V) == 32) {
                if constexpr (_N == 8) {
                    return _mm256_cmp_ps(
                        _mm256_and_ps(
                            _mm256_castsi256_ps(_mm256_set1_epi32(k)),
                            _mm256_castsi256_ps(_mm256_setr_epi32(
                                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80))),
                        _mm256_setzero_ps(), _CMP_NEQ_UQ);
                } else if constexpr (_N == 4) {
                    return _mm256_cmp_pd(
                        _mm256_and_pd(
                            _mm256_castsi256_pd(_mm256_set1_epi64x(k)),
                            _mm256_castsi256_pd(
                                _mm256_setr_epi64x(0x01, 0x02, 0x04, 0x08))),
                        _mm256_setzero_pd(), _CMP_NEQ_UQ);
                } else {
                    __assert_unreachable<To>();
                }
            } else if constexpr (bits_per_element >= _N) {
                constexpr auto bitmask = __generate_builtin<__vector_type_t<_U, _N>>(
                    [](auto i) -> _U { return 1ull << i; });
                return __vector_cast<ToT>(
                    (__vector_broadcast<_N, _U>(k) & bitmask) != 0);
            } else if constexpr (sizeof(_V) == 16 && sizeof(ToT) == 1 && __have_ssse3) {
                const auto bitmask = __to_intrin(__make_builtin<__uchar>(
                    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128));
                return __vector_cast<ToT>(
                    __vector_cast<ToT>(
                        _mm_shuffle_epi8(
                            __to_intrin(__vector_type_t<__ullong, 2>{k}),
                            _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                                          1)) &
                        bitmask) != 0);
            } else if constexpr (sizeof(_V) == 32 && sizeof(ToT) == 1 && __have_avx2) {
                const auto bitmask =
                    _mm256_broadcastsi128_si256(__to_intrin(__make_builtin<__uchar>(
                        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128)));
                return __vector_cast<ToT>(
                    __vector_cast<ToT>(_mm256_shuffle_epi8(
                                        _mm256_broadcastsi128_si256(__to_intrin(
                                            __vector_type_t<__ullong, 2>{k})),
                                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                                                         1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                                         2, 2, 3, 3, 3, 3, 3, 3, 3, 3)) &
                                    bitmask) != 0);
                /* TODO:
                } else if constexpr (sizeof(_V) == 32 && sizeof(ToT) == 2 && __have_avx2) {
                    constexpr auto bitmask = _mm256_broadcastsi128_si256(
                        _mm_setr_epi8(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000)); return
                __vector_cast<ToT>( _mm256_shuffle_epi8(
                                   _mm256_broadcastsi128_si256(__m128i{k}),
                                   _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)) & bitmask) != 0;
                */
            } else {
                const _V tmp = __generate_builtin<_V>([&](auto i) {
                                  return static_cast<_U>(
                                      k >> (bits_per_element * (i / bits_per_element)));
                              }) &
                              __generate_builtin<_V>([](auto i) {
                                  return static_cast<_U>(1ull << (i % bits_per_element));
                              });  // mask bit index
                return __vector_cast<ToT>(tmp != _V());
            }
        } else {
            __assert_unreachable<To>();
        } // }}}
    } else if constexpr (__is_vector_type_v<From> && std::is_unsigned_v<To>) {
        // vector -> bits {{{
        using _Trait = __vector_traits<From>;
        using _T = typename _Trait::value_type;
        constexpr size_t FromN = _Trait::width;
        constexpr size_t cvt_id = FromN * 10 + sizeof(_T);
        constexpr bool __have_avx512_int = __have_avx512f && std::is_integral_v<_T>;
        [[maybe_unused]]  // PR85827
        const auto intrin = __to_intrin(k);

             if constexpr (cvt_id == 16'1 && __have_avx512bw_vl) { return    _mm_movepi8_mask(intrin); }
        else if constexpr (cvt_id == 16'1 && __have_avx512bw   ) { return _mm512_movepi8_mask(__zero_extend(intrin)); }
        else if constexpr (cvt_id == 16'1                    ) { return    _mm_movemask_epi8(intrin); }
        else if constexpr (cvt_id == 32'1 && __have_avx512bw_vl) { return _mm256_movepi8_mask(intrin); }
        else if constexpr (cvt_id == 32'1 && __have_avx512bw   ) { return _mm512_movepi8_mask(__zero_extend(intrin)); }
        else if constexpr (cvt_id == 32'1                    ) { return _mm256_movemask_epi8(intrin); }
        else if constexpr (cvt_id == 64'1 && __have_avx512bw   ) { return _mm512_movepi8_mask(intrin); }
        else if constexpr (cvt_id ==  8'2 && __have_avx512bw_vl) { return    _mm_movepi16_mask(intrin); }
        else if constexpr (cvt_id ==  8'2 && __have_avx512bw   ) { return _mm512_movepi16_mask(__zero_extend(intrin)); }
        else if constexpr (cvt_id ==  8'2                    ) { return movemask_epi16(intrin); }
        else if constexpr (cvt_id == 16'2 && __have_avx512bw_vl) { return _mm256_movepi16_mask(intrin); }
        else if constexpr (cvt_id == 16'2 && __have_avx512bw   ) { return _mm512_movepi16_mask(__zero_extend(intrin)); }
        else if constexpr (cvt_id == 16'2                    ) { return movemask_epi16(intrin); }
        else if constexpr (cvt_id == 32'2 && __have_avx512bw   ) { return _mm512_movepi16_mask(intrin); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512dq_vl) { return    _mm_movepi32_mask(__vector_cast<__llong>(k)); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512dq   ) { return _mm512_movepi32_mask(__zero_extend(__vector_cast<__llong>(k))); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512vl   ) { return    _mm_cmp_epi32_mask(__vector_cast<__llong>(k), __m128i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512_int ) { return _mm512_cmp_epi32_mask(__zero_extend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'4                    ) { return    _mm_movemask_ps(k); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512dq_vl) { return _mm256_movepi32_mask(__vector_cast<__llong>(k)); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512dq   ) { return _mm512_movepi32_mask(__zero_extend(__vector_cast<__llong>(k))); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512vl   ) { return _mm256_cmp_epi32_mask(__vector_cast<__llong>(k), __m256i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512_int ) { return _mm512_cmp_epi32_mask(__zero_extend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  8'4                    ) { return _mm256_movemask_ps(k); }
        else if constexpr (cvt_id == 16'4 && __have_avx512dq   ) { return _mm512_movepi32_mask(__vector_cast<__llong>(k)); }
        else if constexpr (cvt_id == 16'4                    ) { return _mm512_cmp_epi32_mask(__vector_cast<__llong>(k), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512dq_vl) { return    _mm_movepi64_mask(__vector_cast<__llong>(k)); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512dq   ) { return _mm512_movepi64_mask(__zero_extend(__vector_cast<__llong>(k))); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512vl   ) { return    _mm_cmp_epi64_mask(__vector_cast<__llong>(k), __m128i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512_int ) { return _mm512_cmp_epi64_mask(__zero_extend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8                    ) { return    _mm_movemask_pd(k); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512dq_vl) { return _mm256_movepi64_mask(__vector_cast<__llong>(k)); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512dq   ) { return _mm512_movepi64_mask(__zero_extend(__vector_cast<__llong>(k))); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512vl   ) { return _mm256_cmp_epi64_mask(__vector_cast<__llong>(k), __m256i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512_int ) { return _mm512_cmp_epi64_mask(__zero_extend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'8                    ) { return _mm256_movemask_pd(k); }
        else if constexpr (cvt_id ==  8'8 && __have_avx512dq   ) { return _mm512_movepi64_mask(__vector_cast<__llong>(k)); }
        else if constexpr (cvt_id ==  8'8                    ) { return _mm512_cmp_epi64_mask(__vector_cast<__llong>(k), __m512i(), _MM_CMPINT_LT); }
        else { __assert_unreachable<To>(); }
        // }}}
    } else if constexpr (__is_vector_type_v<From> && __is_vector_type_v<To>) {
        // vector -> vector {{{
        using ToTrait = __vector_traits<To>;
        using FromTrait = __vector_traits<From>;
        using ToT = typename ToTrait::value_type;
        using _T = typename FromTrait::value_type;
        constexpr size_t FromN = FromTrait::width;
        constexpr size_t ToN = ToTrait::width;
        constexpr int FromBytes = sizeof(_T);
        constexpr int ToBytes = sizeof(ToT);

        if constexpr (FromN == ToN && sizeof(From) == sizeof(To)) {
            // reinterpret the bits
            return reinterpret_cast<To>(k);
        } else if constexpr (sizeof(To) == 16 && sizeof(k) == 16) {
            // SSE -> SSE {{{
            if constexpr (FromBytes == 4 && ToBytes == 8) {
                if constexpr(std::is_integral_v<_T>) {
                    return __vector_cast<ToT>(interleave128_lo(k, k));
                } else {
                    return __vector_cast<ToT>(interleave128_lo(k, k));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 8) {
                const auto y = __vector_cast<int>(interleave128_lo(k, k));
                return __vector_cast<ToT>(interleave128_lo(y, y));
            } else if constexpr (FromBytes == 1 && ToBytes == 8) {
                auto y = __vector_cast<short>(interleave128_lo(k, k));
                auto z = __vector_cast<int>(interleave128_lo(y, y));
                return __vector_cast<ToT>(interleave128_lo(z, z));
            } else if constexpr (FromBytes == 8 && ToBytes == 4) {
                if constexpr (std::is_floating_point_v<_T>) {
                    return __vector_cast<ToT>(_mm_shuffle_ps(__vector_cast<float>(k), __m128(),
                                                     __make_immediate<4>(1, 3, 1, 3)));
                } else {
                    auto y = __vector_cast<__llong>(k);
                    return __vector_cast<ToT>(_mm_packs_epi32(y, __m128i()));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 4) {
                return __vector_cast<ToT>(interleave128_lo(k, k));
            } else if constexpr (FromBytes == 1 && ToBytes == 4) {
                const auto y = __vector_cast<short>(interleave128_lo(k, k));
                return __vector_cast<ToT>(interleave128_lo(y, y));
            } else if constexpr (FromBytes == 8 && ToBytes == 2) {
                if constexpr(__have_ssse3) {
                    return __vector_cast<ToT>(
                        _mm_shuffle_epi8(__vector_cast<__llong>(k),
                                         _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    const auto y = _mm_packs_epi32(__vector_cast<__llong>(k), __m128i());
                    return __vector_cast<ToT>(_mm_packs_epi32(y, __m128i()));
                }
            } else if constexpr (FromBytes == 4 && ToBytes == 2) {
                return __vector_cast<ToT>(
                    _mm_packs_epi32(__vector_cast<__llong>(k), __m128i()));
            } else if constexpr (FromBytes == 1 && ToBytes == 2) {
                return __vector_cast<ToT>(interleave128_lo(k, k));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {
                if constexpr(__have_ssse3) {
                    return __vector_cast<ToT>(
                        _mm_shuffle_epi8(__vector_cast<__llong>(k),
                                         _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    auto y = _mm_packs_epi32(__vector_cast<__llong>(k), __m128i());
                    y = _mm_packs_epi32(y, __m128i());
                    return __vector_cast<ToT>(_mm_packs_epi16(y, __m128i()));
                }
            } else if constexpr (FromBytes == 4 && ToBytes == 1) {
                if constexpr(__have_ssse3) {
                    return __vector_cast<ToT>(
                        _mm_shuffle_epi8(__vector_cast<__llong>(k),
                                         _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    const auto y = _mm_packs_epi32(__vector_cast<__llong>(k), __m128i());
                    return __vector_cast<ToT>(_mm_packs_epi16(y, __m128i()));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 1) {
                return __vector_cast<ToT>(_mm_packs_epi16(__vector_cast<__llong>(k), __m128i()));
            } else {
                static_assert(!std::is_same_v<_T, _T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(To) == 32 && sizeof(k) == 32) {
            // AVX -> AVX {{{
            if constexpr (FromBytes == ToBytes) {  // keep low 1/2
                static_assert(!std::is_same_v<_T, _T>, "should be unreachable");
            } else if constexpr (FromBytes == ToBytes * 2) {
                const auto y = __vector_cast<__llong>(k);
                return __vector_cast<ToT>(
                    _mm256_castsi128_si256(_mm_packs_epi16(__lo128(y), __hi128(y))));
            } else if constexpr (FromBytes == ToBytes * 4) {
                const auto y = __vector_cast<__llong>(k);
                return __vector_cast<ToT>(_mm256_castsi128_si256(
                    _mm_packs_epi16(_mm_packs_epi16(__lo128(y), __hi128(y)), __m128i())));
            } else if constexpr (FromBytes == ToBytes * 8) {
                const auto y = __vector_cast<__llong>(k);
                return __vector_cast<ToT>(_mm256_castsi128_si256(
                    _mm_shuffle_epi8(_mm_packs_epi16(__lo128(y), __hi128(y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1))));
            } else if constexpr (FromBytes * 2 == ToBytes) {
                auto y = __fixup_avx_xzyw(__to_intrin(k));
                if constexpr(std::is_floating_point_v<_T>) {
                    return __vector_cast<ToT>(_mm256_unpacklo_ps(y, y));
                } else {
                    return __vector_cast<ToT>(_mm256_unpacklo_epi8(y, y));
                }
            } else if constexpr (FromBytes * 4 == ToBytes) {
                auto y = _mm_unpacklo_epi8(__lo128(__vector_cast<__llong>(k)),
                                           __lo128(__vector_cast<__llong>(k)));  // drops 3/4 of input
                return __vector_cast<ToT>(
                    __concat(_mm_unpacklo_epi16(y, y), _mm_unpackhi_epi16(y, y)));
            } else if constexpr (FromBytes == 1 && ToBytes == 8) {
                auto y = _mm_unpacklo_epi8(__lo128(__vector_cast<__llong>(k)),
                                           __lo128(__vector_cast<__llong>(k)));  // drops 3/4 of input
                y = _mm_unpacklo_epi16(y, y);  // drops another 1/2 => 7/8 total
                return __vector_cast<ToT>(
                    __concat(_mm_unpacklo_epi32(y, y), _mm_unpackhi_epi32(y, y)));
            } else {
                static_assert(!std::is_same_v<_T, _T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(To) == 32 && sizeof(k) == 16) {
            // SSE -> AVX {{{
            if constexpr (FromBytes == ToBytes) {
                return __vector_cast<ToT>(
                    __intrinsic_type_t<_T, 32 / sizeof(_T)>(__zero_extend(__to_intrin(k))));
            } else if constexpr (FromBytes * 2 == ToBytes) {  // keep all
                return __vector_cast<ToT>(__concat(_mm_unpacklo_epi8(__vector_cast<__llong>(k), __vector_cast<__llong>(k)),
                                         _mm_unpackhi_epi8(__vector_cast<__llong>(k), __vector_cast<__llong>(k))));
            } else if constexpr (FromBytes * 4 == ToBytes) {
                if constexpr (__have_avx2) {
                    return __vector_cast<ToT>(_mm256_shuffle_epi8(
                        __concat(__vector_cast<__llong>(k), __vector_cast<__llong>(k)),
                        _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7,
                                         7)));
                } else {
                    return __vector_cast<ToT>(
                        __concat(_mm_shuffle_epi8(__vector_cast<__llong>(k),
                                                _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2,
                                                              2, 2, 2, 3, 3, 3, 3)),
                               _mm_shuffle_epi8(__vector_cast<__llong>(k),
                                                _mm_setr_epi8(4, 4, 4, 4, 5, 5, 5, 5, 6,
                                                              6, 6, 6, 7, 7, 7, 7))));
                }
            } else if constexpr (FromBytes * 8 == ToBytes) {
                if constexpr (__have_avx2) {
                    return __vector_cast<ToT>(_mm256_shuffle_epi8(
                        __concat(__vector_cast<__llong>(k), __vector_cast<__llong>(k)),
                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                                         3)));
                } else {
                    return __vector_cast<ToT>(
                        __concat(_mm_shuffle_epi8(__vector_cast<__llong>(k),
                                                _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                              1, 1, 1, 1, 1, 1, 1)),
                               _mm_shuffle_epi8(__vector_cast<__llong>(k),
                                                _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3,
                                                              3, 3, 3, 3, 3, 3, 3))));
                }
            } else if constexpr (FromBytes == ToBytes * 2) {
                return __vector_cast<ToT>(
                    __m256i(__zero_extend(_mm_packs_epi16(__vector_cast<__llong>(k), __m128i()))));
            } else if constexpr (FromBytes == 8 && ToBytes == 2) {
                return __vector_cast<ToT>(__m256i(__zero_extend(
                    _mm_shuffle_epi8(__vector_cast<__llong>(k),
                                     _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else if constexpr (FromBytes == 4 && ToBytes == 1) {
                return __vector_cast<ToT>(__m256i(__zero_extend(
                    _mm_shuffle_epi8(__vector_cast<__llong>(k),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {
                return __vector_cast<ToT>(__m256i(__zero_extend(
                    _mm_shuffle_epi8(__vector_cast<__llong>(k),
                                     _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else {
                static_assert(!std::is_same_v<_T, _T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(To) == 16 && sizeof(k) == 32) {
            // AVX -> SSE {{{
            if constexpr (FromBytes == ToBytes) {  // keep low 1/2
                return __vector_cast<ToT>(__lo128(k));
            } else if constexpr (FromBytes == ToBytes * 2) {  // keep all
                auto y = __vector_cast<__llong>(k);
                return __vector_cast<ToT>(_mm_packs_epi16(__lo128(y), __hi128(y)));
            } else if constexpr (FromBytes == ToBytes * 4) {  // add 1/2 undef
                auto y = __vector_cast<__llong>(k);
                return __vector_cast<ToT>(
                    _mm_packs_epi16(_mm_packs_epi16(__lo128(y), __hi128(y)), __m128i()));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {  // add 3/4 undef
                auto y = __vector_cast<__llong>(k);
                return __vector_cast<ToT>(
                    _mm_shuffle_epi8(_mm_packs_epi16(__lo128(y), __hi128(y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)));
            } else if constexpr (FromBytes * 2 == ToBytes) {  // keep low 1/4
                auto y = __lo128(__vector_cast<__llong>(k));
                return __vector_cast<ToT>(_mm_unpacklo_epi8(y, y));
            } else if constexpr (FromBytes * 4 == ToBytes) {  // keep low 1/8
                auto y = __lo128(__vector_cast<__llong>(k));
                y = _mm_unpacklo_epi8(y, y);
                return __vector_cast<ToT>(_mm_unpacklo_epi8(y, y));
            } else if constexpr (FromBytes * 8 == ToBytes) {  // keep low 1/16
                auto y = __lo128(__vector_cast<__llong>(k));
                y = _mm_unpacklo_epi8(y, y);
                y = _mm_unpacklo_epi8(y, y);
                return __vector_cast<ToT>(_mm_unpacklo_epi8(y, y));
            } else {
                static_assert(!std::is_same_v<_T, _T>, "should be unreachable");
            }
            // }}}
        }
        // }}}
    } else {
        __assert_unreachable<To>();
    }
}

// }}}

template <class _Abi> struct __simd_math_fallback {  //{{{
    template <class _T> simd<_T, _Abi> __acos(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::acos(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __asin(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::asin(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __atan(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::atan(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __atan2(const simd<_T, _Abi> &x, const simd<_T, _Abi> &y)
    {
        return simd<_T, _Abi>([&](auto i) { return std::atan2(x[i], y[i]); });
    }

    template <class _T> simd<_T, _Abi> __cos(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::cos(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __sin(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::sin(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __tan(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::tan(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __acosh(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::acosh(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __asinh(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::asinh(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __atanh(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::atanh(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __cosh(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::cosh(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __sinh(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::sinh(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __tanh(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::tanh(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __exp(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::exp(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __exp2(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::exp2(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __expm1(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::expm1(x[i]); });
    }

    template <class _T>
    simd<_T, _Abi> __frexp(const simd<_T, _Abi> &x,
                         fixed_size_simd<int, simd_size_v<_T, _Abi>> &exp)
    {
        return simd<_T, _Abi>([&](auto i) {
            int tmp;
            _T r = std::frexp(x[i], &tmp);
            exp[i] = tmp;
            return r;
        });
    }

    template <class _T>
    simd<_T, _Abi> __ldexp(const simd<_T, _Abi> &x,
                         const fixed_size_simd<int, simd_size_v<_T, _Abi>> &exp)
    {
        return simd<_T, _Abi>([&](auto i) { return std::ldexp(x[i], exp[i]); });
    }

    template <class _T>
    fixed_size_simd<int, simd_size_v<_T, _Abi>> __ilogb(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::ilogb(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __log(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::log(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __log10(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::log10(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __log1p(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::log1p(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __log2(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::log2(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __logb(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::logb(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __modf(const simd<_T, _Abi> &x, simd<_T, _Abi> &y)
    {
        return simd<_T, _Abi>([&](auto i) {
            _T tmp;
            _T r = std::modf(x[i], &tmp);
            y[i] = tmp;
            return r;
        });
    }

    template <class _T>
    simd<_T, _Abi> __scalbn(const simd<_T, _Abi> &x,
                          const fixed_size_simd<int, simd_size_v<_T, _Abi>> &y)
    {
        return simd<_T, _Abi>([&](auto i) { return std::scalbn(x[i], y[i]); });
    }

    template <class _T>
    simd<_T, _Abi> __scalbln(const simd<_T, _Abi> &x,
                           const fixed_size_simd<long, simd_size_v<_T, _Abi>> &y)
    {
        return simd<_T, _Abi>([&](auto i) { return std::scalbln(x[i], y[i]); });
    }

    template <class _T> simd<_T, _Abi> __cbrt(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::cbrt(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __abs(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::abs(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __fabs(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::fabs(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __pow(const simd<_T, _Abi> &x, const simd<_T, _Abi> &y)
    {
        return simd<_T, _Abi>([&](auto i) { return std::pow(x[i], y[i]); });
    }

    template <class _T> simd<_T, _Abi> __sqrt(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::sqrt(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __erf(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::erf(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __erfc(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::erfc(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __lgamma(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::lgamma(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __tgamma(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::tgamma(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __ceil(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::ceil(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __floor(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::floor(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __nearbyint(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::nearbyint(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __rint(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::rint(x[i]); });
    }

    template <class _T>
    fixed_size_simd<long, simd_size_v<_T, _Abi>> __lrint(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::lrint(x[i]); });
    }

    template <class _T>
    fixed_size_simd<long long, simd_size_v<_T, _Abi>> __llrint(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::llrint(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __round(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::round(x[i]); });
    }

    template <class _T>
    fixed_size_simd<long, simd_size_v<_T, _Abi>> __lround(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::lround(x[i]); });
    }

    template <class _T>
    fixed_size_simd<long long, simd_size_v<_T, _Abi>> __llround(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::llround(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __trunc(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::trunc(x[i]); });
    }

    template <class _T> simd<_T, _Abi> __fmod(const simd<_T, _Abi> &x, const simd<_T, _Abi> &y)
    {
        return simd<_T, _Abi>([&](auto i) { return std::fmod(x[i], y[i]); });
    }

    template <class _T>
    simd<_T, _Abi> __remainder(const simd<_T, _Abi> &x, const simd<_T, _Abi> &y)
    {
        return simd<_T, _Abi>([&](auto i) { return std::remainder(x[i], y[i]); });
    }

    template <class _T>
    simd<_T, _Abi> __remquo(const simd<_T, _Abi> &x, const simd<_T, _Abi> &y,
                          fixed_size_simd<int, simd_size_v<_T, _Abi>> &z)
    {
        return simd<_T, _Abi>([&](auto i) {
            int tmp;
            _T r = std::remquo(x[i], y[i], &tmp);
            z[i] = tmp;
            return r;
        });
    }

    template <class _T>
    simd<_T, _Abi> __copysign(const simd<_T, _Abi> &x, const simd<_T, _Abi> &y)
    {
        return simd<_T, _Abi>([&](auto i) { return std::copysign(x[i], y[i]); });
    }

    template <class _T>
    simd<_T, _Abi> __nextafter(const simd<_T, _Abi> &x, const simd<_T, _Abi> &y)
    {
        return simd<_T, _Abi>([&](auto i) { return std::nextafter(x[i], y[i]); });
    }

    template <class _T> simd<_T, _Abi> __fdim(const simd<_T, _Abi> &x, const simd<_T, _Abi> &y)
    {
        return simd<_T, _Abi>([&](auto i) { return std::fdim(x[i], y[i]); });
    }

    template <class _T> simd<_T, _Abi> __fmax(const simd<_T, _Abi> &x, const simd<_T, _Abi> &y)
    {
        return simd<_T, _Abi>([&](auto i) { return std::fmax(x[i], y[i]); });
    }

    template <class _T> simd<_T, _Abi> __fmin(const simd<_T, _Abi> &x, const simd<_T, _Abi> &y)
    {
        return simd<_T, _Abi>([&](auto i) { return std::fmin(x[i], y[i]); });
    }

    template <class _T>
    simd<_T, _Abi> __fma(const simd<_T, _Abi> &x, const simd<_T, _Abi> &y,
                       const simd<_T, _Abi> &z)
    {
        return simd<_T, _Abi>([&](auto i) { return std::fma(x[i], y[i], z[i]); });
    }

    template <class _T>
    fixed_size_simd<int, simd_size_v<_T, _Abi>> __fpclassify(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::fpclassify(x[i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isfinite(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::isfinite(x[i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isinf(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::isinf(x[i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isnan(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::isnan(x[i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isnormal(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::isnormal(x[i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __signbit(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::signbit(x[i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isgreater(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::isgreater(x[i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isgreaterequal(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::isgreaterequal(x[i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isless(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::isless(x[i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __islessequal(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::islessequal(x[i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __islessgreater(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::islessgreater(x[i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isunordered(const simd<_T, _Abi> &x)
    {
        return simd<_T, _Abi>([&](auto i) { return std::isunordered(x[i]); });
    }
};  // }}}
// __scalar_simd_impl {{{
struct __scalar_simd_impl : __simd_math_fallback<simd_abi::scalar> {
    // member types {{{2
    using abi = std::experimental::simd_abi::scalar;
    using __mask_member_type = bool;
    template <class _T> using __simd_member_type = _T;
    template <class _T> using simd = std::experimental::simd<_T, abi>;
    template <class _T> using simd_mask = std::experimental::simd_mask<_T, abi>;
    template <class _T> using __type_tag = _T *;

    // broadcast {{{2
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static constexpr _T __broadcast(_T x) noexcept
    {
        return x;
    }

    // generator {{{2
    template <class F, class _T>
    _GLIBCXX_SIMD_INTRINSIC static _T generator(F &&gen, __type_tag<_T>)
    {
        return gen(__size_constant<0>());
    }

    // load {{{2
    template <class _T, class _U, class F>
    static inline _T load(const _U *mem, F, __type_tag<_T>) noexcept
    {
        return static_cast<_T>(mem[0]);
    }

    // masked load {{{2
    template <class _T, class _U, class F>
    static inline _T masked_load(_T merge, bool k, const _U *mem, F) noexcept
    {
        if (k) {
            merge = static_cast<_T>(mem[0]);
        }
        return merge;
    }

    // store {{{2
    template <class _T, class _U, class F>
    static inline void store(_T v, _U *mem, F, __type_tag<_T>) noexcept
    {
        mem[0] = static_cast<_T>(v);
    }

    // masked store {{{2
    template <class _T, class _U, class F>
    static inline void masked_store(const _T v, _U *mem, F, const bool k) noexcept
    {
        if (k) {
            mem[0] = v;
        }
    }

    // negation {{{2
    template <class _T> static inline bool negate(_T x) noexcept { return !x; }

    // reductions {{{2
    template <class _T, class _BinaryOperation>
    static inline _T reduce(const simd<_T> &x, _BinaryOperation &)
    {
        return x.d;
    }

    // min, max, clamp {{{2
    template <class _T> static inline _T min(const _T a, const _T b)
    {
        return std::min(a, b);
    }

    template <class _T> static inline _T max(const _T a, const _T b)
    {
        return std::max(a, b);
    }

    // complement {{{2
    template <class _T> static inline _T complement(_T x) noexcept
    {
        return static_cast<_T>(~x);
    }

    // unary minus {{{2
    template <class _T> static inline _T unary_minus(_T x) noexcept
    {
        return static_cast<_T>(-x);
    }

    // arithmetic operators {{{2
    template <class _T> static inline _T plus(_T x, _T y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(x) +
                              __promote_preserving_unsigned(y));
    }

    template <class _T> static inline _T minus(_T x, _T y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(x) -
                              __promote_preserving_unsigned(y));
    }

    template <class _T> static inline _T multiplies(_T x, _T y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(x) *
                              __promote_preserving_unsigned(y));
    }

    template <class _T> static inline _T divides(_T x, _T y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(x) /
                              __promote_preserving_unsigned(y));
    }

    template <class _T> static inline _T modulus(_T x, _T y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(x) %
                              __promote_preserving_unsigned(y));
    }

    template <class _T> static inline _T bit_and(_T x, _T y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(x) &
                              __promote_preserving_unsigned(y));
    }
    static inline float bit_and(float x, float y)
    {
        static_assert(sizeof(float) == sizeof(uint), "");
        const uint r = reinterpret_cast<const __may_alias<uint> &>(x) &
                       reinterpret_cast<const __may_alias<uint> &>(y);
        return reinterpret_cast<const __may_alias<float> &>(r);
    }
    static inline double bit_and(double x, double y)
    {
        static_assert(sizeof(double) == sizeof(__ullong), "");
        const __ullong r = reinterpret_cast<const __may_alias<__ullong> &>(x) &
                         reinterpret_cast<const __may_alias<__ullong> &>(y);
        return reinterpret_cast<const __may_alias<double> &>(r);
    }

    template <class _T> static inline _T bit_or(_T x, _T y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(x) |
                              __promote_preserving_unsigned(y));
    }
    static inline float bit_or(float x, float y)
    {
        static_assert(sizeof(float) == sizeof(uint), "");
        const uint r = reinterpret_cast<const __may_alias<uint> &>(x) |
                       reinterpret_cast<const __may_alias<uint> &>(y);
        return reinterpret_cast<const __may_alias<float> &>(r);
    }
    static inline double bit_or(double x, double y)
    {
        static_assert(sizeof(double) == sizeof(__ullong), "");
        const __ullong r = reinterpret_cast<const __may_alias<__ullong> &>(x) |
                         reinterpret_cast<const __may_alias<__ullong> &>(y);
        return reinterpret_cast<const __may_alias<double> &>(r);
    }


    template <class _T> static inline _T bit_xor(_T x, _T y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(x) ^
                              __promote_preserving_unsigned(y));
    }

    template <class _T> static inline _T bit_shift_left(_T x, int y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(x) << y);
    }

    template <class _T> static inline _T bit_shift_right(_T x, int y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(x) >> y);
    }

    // math {{{2
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static _T __abs(_T x) { return _T(std::abs(x)); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static _T __sqrt(_T x) { return std::sqrt(x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static _T __trunc(_T x) { return std::trunc(x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static _T __floor(_T x) { return std::floor(x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static _T __ceil(_T x) { return std::ceil(x); }

    template <class _T> _GLIBCXX_SIMD_INTRINSIC static __simd_tuple<int, abi> __fpclassify(_T x)
    {
        return {std::fpclassify(x)};
    }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static bool __isfinite(_T x) { return std::isfinite(x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static bool __isinf(_T x) { return std::isinf(x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static bool __isnan(_T x) { return std::isnan(x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static bool __isnormal(_T x) { return std::isnormal(x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static bool __signbit(_T x) { return std::signbit(x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static bool __isunordered(_T x, _T y) { return std::isunordered(x, y); }

    // __increment & __decrement{{{2
    template <class _T> static inline void __increment(_T &x) { ++x; }
    template <class _T> static inline void __decrement(_T &x) { --x; }

    // compares {{{2
    template <class _T> static bool equal_to(_T x, _T y) { return x == y; }
    template <class _T> static bool not_equal_to(_T x, _T y) { return x != y; }
    template <class _T> static bool less(_T x, _T y) { return x < y; }
    template <class _T> static bool greater(_T x, _T y) { return x > y; }
    template <class _T> static bool less_equal(_T x, _T y) { return x <= y; }
    template <class _T> static bool greater_equal(_T x, _T y) { return x >= y; }

    // smart_reference access {{{2
    template <class _T, class _U> static void set(_T &v, int i, _U &&x) noexcept
    {
        _GLIBCXX_SIMD_ASSERT(i == 0);
        __unused(i);
        v = std::forward<_U>(x);
    }

    // masked_assign {{{2
    template <typename _T> _GLIBCXX_SIMD_INTRINSIC static void masked_assign(bool k, _T &lhs, _T rhs)
    {
        if (k) {
            lhs = rhs;
        }
    }

    // __masked_cassign {{{2
    template <template <typename> class Op, typename _T>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_cassign(const bool k, _T &lhs, const _T rhs)
    {
        if (k) {
            lhs = Op<_T>{}(lhs, rhs);
        }
    }

    // masked_unary {{{2
    template <template <typename> class Op, typename _T>
    _GLIBCXX_SIMD_INTRINSIC static _T masked_unary(const bool k, const _T v)
    {
        return static_cast<_T>(k ? Op<_T>{}(v) : v);
    }

    // }}}2
};

// }}}
// __scalar_mask_impl {{{
struct __scalar_mask_impl {
    // member types {{{2
    template <class _T> using simd_mask = std::experimental::simd_mask<_T, simd_abi::scalar>;
    template <class _T> using __type_tag = _T *;

    // __from_bitset {{{2
    template <class _T>
    _GLIBCXX_SIMD_INTRINSIC static bool __from_bitset(std::bitset<1> bs, __type_tag<_T>) noexcept
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
        __unused(i);
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
template <class _T, size_t _N> constexpr bool __is_sse_ps()
{
    return __have_sse && std::is_same_v<_T, float> && _N == 4;
}
template <class _T, size_t _N> constexpr bool __is_sse_pd()
{
    return __have_sse2 && std::is_same_v<_T, double> && _N == 2;
}
template <class _T, size_t _N> constexpr bool __is_avx_ps()
{
    return __have_avx && std::is_same_v<_T, float> && _N == 8;
}
template <class _T, size_t _N> constexpr bool __is_avx_pd()
{
    return __have_avx && std::is_same_v<_T, double> && _N == 4;
}
template <class _T, size_t _N> constexpr bool __is_avx512_ps()
{
    return __have_avx512f && std::is_same_v<_T, float> && _N == 16;
}
template <class _T, size_t _N> constexpr bool __is_avx512_pd()
{
    return __have_avx512f && std::is_same_v<_T, double> && _N == 8;
}

template <class _T, size_t _N> constexpr bool __is_neon_ps()
{
    return __have_neon && std::is_same_v<_T, float> && _N == 4;
}
template <class _T, size_t _N> constexpr bool __is_neon_pd()
{
    return __have_neon && std::is_same_v<_T, double> && _N == 2;
}

// __generic_simd_impl {{{1
template <class _Abi> struct __generic_simd_impl : __simd_math_fallback<_Abi> {
    // member types {{{2
    template <class _T> using __type_tag = _T *;
    template <class _T>
    using __simd_member_type = typename _Abi::template __traits<_T>::__simd_member_type;
    template <class _T>
    using __mask_member_type = typename _Abi::template __traits<_T>::__mask_member_type;
    template <class _T> static constexpr size_t full_size = __simd_member_type<_T>::width;

    // make_simd(__storage) {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static simd<_T, _Abi> make_simd(__storage<_T, _N> x)
    {
        return {__private_init, x};
    }

    // broadcast {{{2
    template <class _T>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __simd_member_type<_T> __broadcast(_T x) noexcept
    {
        return __vector_broadcast<full_size<_T>>(x);
    }

    // generator {{{2
    template <class F, class _T>
    _GLIBCXX_SIMD_INTRINSIC static __simd_member_type<_T> generator(F &&gen, __type_tag<_T>)
    {
        return __generate_storage<_T, full_size<_T>>(std::forward<F>(gen));
    }

    // load {{{2
    template <class _T, class _U, class F>
    _GLIBCXX_SIMD_INTRINSIC static __simd_member_type<_T> load(const _U *mem, F,
                                                 __type_tag<_T>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        constexpr size_t _N = __simd_member_type<_T>::width;
        constexpr size_t max_load_size =
            (sizeof(_U) >= 4 && __have_avx512f) || __have_avx512bw
                ? 64
                : (std::is_floating_point_v<_U> && __have_avx) || __have_avx2 ? 32 : 16;
        if constexpr (sizeof(_U) > 8) {
            return __generate_storage<_T, _N>(
                [&](auto i) { return static_cast<_T>(mem[i]); });
        } else if constexpr (std::is_same_v<_U, _T>) {
            return __vector_load<_U, _N>(mem, F());
        } else if constexpr (sizeof(_U) * _N < 16) {
            return __x86::convert<__simd_member_type<_T>>(
                __vector_load16<_U, sizeof(_U) * _N>(mem, F()));
        } else if constexpr (sizeof(_U) * _N <= max_load_size) {
            return __x86::convert<__simd_member_type<_T>>(__vector_load<_U, _N>(mem, F()));
        } else if constexpr (sizeof(_U) * _N == 2 * max_load_size) {
            return __x86::convert<__simd_member_type<_T>>(
                __vector_load<_U, _N / 2>(mem, F()),
                __vector_load<_U, _N / 2>(mem + _N / 2, F()));
        } else if constexpr (sizeof(_U) * _N == 4 * max_load_size) {
            return __x86::convert<__simd_member_type<_T>>(
                __vector_load<_U, _N / 4>(mem, F()),
                __vector_load<_U, _N / 4>(mem + 1 * _N / 4, F()),
                __vector_load<_U, _N / 4>(mem + 2 * _N / 4, F()),
                __vector_load<_U, _N / 4>(mem + 3 * _N / 4, F()));
        } else if constexpr (sizeof(_U) * _N == 8 * max_load_size) {
            return __x86::convert<__simd_member_type<_T>>(
                __vector_load<_U, _N / 8>(mem, F()),
                __vector_load<_U, _N / 8>(mem + 1 * _N / 8, F()),
                __vector_load<_U, _N / 8>(mem + 2 * _N / 8, F()),
                __vector_load<_U, _N / 8>(mem + 3 * _N / 8, F()),
                __vector_load<_U, _N / 8>(mem + 4 * _N / 8, F()),
                __vector_load<_U, _N / 8>(mem + 5 * _N / 8, F()),
                __vector_load<_U, _N / 8>(mem + 6 * _N / 8, F()),
                __vector_load<_U, _N / 8>(mem + 7 * _N / 8, F()));
        } else {
            __assert_unreachable<_T>();
        }
    }

    // masked load {{{2
    template <class _T, size_t _N, class _U, class F>
    static inline __storage<_T, _N> masked_load(__storage<_T, _N> merge, __mask_member_type<_T> k,
                                   const _U *mem, F) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        if constexpr (std::is_same_v<_T, _U> ||  // no conversion
                      (sizeof(_T) == sizeof(_U) &&
                       std::is_integral_v<_T> ==
                           std::is_integral_v<_U>)  // conversion via bit reinterpretation
        ) {
            constexpr bool __have_avx512bw_vl_or_zmm =
                __have_avx512bw_vl || (__have_avx512bw && sizeof(merge) == 64);
            if constexpr (__have_avx512bw_vl_or_zmm && sizeof(_T) == 1) {
                if constexpr (sizeof(merge) == 16) {
                    merge = _mm_mask_loadu_epi8(merge, _mm_movemask_epi8(k), mem);
                } else if constexpr (sizeof(merge) == 32) {
                    merge = _mm256_mask_loadu_epi8(merge, _mm256_movemask_epi8(k), mem);
                } else if constexpr (sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_epi8(merge, k, mem);
                } else {
                    __assert_unreachable<_T>();
                }
            } else if constexpr (__have_avx512bw_vl_or_zmm && sizeof(_T) == 2) {
                if constexpr (sizeof(merge) == 16) {
                    merge = _mm_mask_loadu_epi16(merge, movemask_epi16(k), mem);
                } else if constexpr (sizeof(merge) == 32) {
                    merge = _mm256_mask_loadu_epi16(merge, movemask_epi16(k), mem);
                } else if constexpr (sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_epi16(merge, k, mem);
                } else {
                    __assert_unreachable<_T>();
                }
            } else if constexpr (__have_avx2 && sizeof(_T) == 4 && std::is_integral_v<_U>) {
                if constexpr (sizeof(merge) == 16) {
                    merge =
                        (~k.d & merge.d) | __vector_cast<_T>(_mm_maskload_epi32(
                                               reinterpret_cast<const int *>(mem), k));
                } else if constexpr (sizeof(merge) == 32) {
                    merge =
                        (~k.d & merge.d) | __vector_cast<_T>(_mm256_maskload_epi32(
                                               reinterpret_cast<const int *>(mem), k));
                } else if constexpr (__have_avx512f && sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_epi32(merge, k, mem);
                } else {
                    __assert_unreachable<_T>();
                }
            } else if constexpr (__have_avx && sizeof(_T) == 4) {
                if constexpr (sizeof(merge) == 16) {
                    merge = __or(__andnot(k.d, merge.d),
                                __vector_cast<_T>(
                                    _mm_maskload_ps(reinterpret_cast<const float *>(mem),
                                                    __vector_cast<__llong>(k))));
                } else if constexpr (sizeof(merge) == 32) {
                    merge = __or(__andnot(k.d, merge.d),
                                _mm256_maskload_ps(reinterpret_cast<const float *>(mem),
                                                   __vector_cast<__llong>(k)));
                } else if constexpr (__have_avx512f && sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_ps(merge, k, mem);
                } else {
                    __assert_unreachable<_T>();
                }
            } else if constexpr (__have_avx2 && sizeof(_T) == 8 && std::is_integral_v<_U>) {
                if constexpr (sizeof(merge) == 16) {
                    merge =
                        (~k.d & merge.d) | __vector_cast<_T>(_mm_maskload_epi64(
                                               reinterpret_cast<const __llong *>(mem), k));
                } else if constexpr (sizeof(merge) == 32) {
                    merge =
                        (~k.d & merge.d) | __vector_cast<_T>(_mm256_maskload_epi64(
                                               reinterpret_cast<const __llong *>(mem), k));
                } else if constexpr (__have_avx512f && sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_epi64(merge, k, mem);
                } else {
                    __assert_unreachable<_T>();
                }
            } else if constexpr (__have_avx && sizeof(_T) == 8) {
                if constexpr (sizeof(merge) == 16) {
                    merge = __or(__andnot(k.d, merge.d),
                                __vector_cast<_T>(_mm_maskload_pd(
                                    reinterpret_cast<const double *>(mem), __vector_cast<__llong>(k))));
                } else if constexpr (sizeof(merge) == 32) {
                    merge = __or(__andnot(k.d, merge.d),
                                _mm256_maskload_pd(reinterpret_cast<const double *>(mem),
                                                   __vector_cast<__llong>(k)));
                } else if constexpr (__have_avx512f && sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_pd(merge, k, mem);
                } else {
                    __assert_unreachable<_T>();
                }
            } else {
                __bit_iteration(__vector_to_bitset(k.d).to_ullong(), [&](auto i) {
                    merge.set(i, static_cast<_T>(mem[i]));
                });
            }
        } else if constexpr (sizeof(_U) <= 8 &&  // no long double
                             !__converts_via_decomposition_v<
                                 _U, _T, sizeof(merge)>  // conversion via decomposition is
                                                       // better handled via the
                                                       // bit_iteration fallback below
        ) {
            // TODO: copy pattern from masked_store, which doesn't resort to fixed_size
            using _A = simd_abi::deduce_t<
                _U, std::max(_N, 16 / sizeof(_U))  // _N or more, so that at least a 16 Byte
                                                // vector is used instead of a fixed_size
                                                // filled with scalars
                >;
            using ATraits = __simd_traits<_U, _A>;
            using AImpl = typename ATraits::__simd_impl_type;
            typename ATraits::__simd_member_type uncvted{};
            typename ATraits::__mask_member_type kk;
            if constexpr (__is_fixed_size_abi_v<_A>) {
                kk = __vector_to_bitset(k.d);
            } else {
                kk = __convert_mask<typename ATraits::__mask_member_type>(k);
            }
            uncvted = AImpl::masked_load(uncvted, kk, mem, F());
            __simd_converter<_U, _A, _T, _Abi> converter;
            masked_assign(k, merge, converter(uncvted));
        } else {
            __bit_iteration(__vector_to_bitset(k.d).to_ullong(),
                                  [&](auto i) { merge.set(i, static_cast<_T>(mem[i])); });
        }
        return merge;
    }

    // store {{{2
    template <class _T, class _U, class F>
    _GLIBCXX_SIMD_INTRINSIC static void store(__simd_member_type<_T> v, _U *mem, F,
                                   __type_tag<_T>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        // TODO: converting int -> "smaller int" can be optimized with AVX512
        constexpr size_t _N = __simd_member_type<_T>::width;
        constexpr size_t max_store_size =
            (sizeof(_U) >= 4 && __have_avx512f) || __have_avx512bw
                ? 64
                : (std::is_floating_point_v<_U> && __have_avx) || __have_avx2 ? 32 : 16;
        if constexpr (sizeof(_U) > 8) {
            __execute_n_times<_N>([&](auto i) { mem[i] = v[i]; });
        } else if constexpr (std::is_same_v<_U, _T>) {
            __vector_store(v.d, mem, F());
        } else if constexpr (sizeof(_U) * _N < 16) {
            __vector_store<sizeof(_U) * _N>(__x86::convert<__vector_type16_t<_U>>(v),
                                                 mem, F());
        } else if constexpr (sizeof(_U) * _N <= max_store_size) {
            __vector_store(__x86::convert<__vector_type_t<_U, _N>>(v), mem, F());
        } else {
            constexpr size_t VSize = max_store_size / sizeof(_U);
            constexpr size_t stores = _N / VSize;
            using _V = __vector_type_t<_U, VSize>;
            const std::array<_V, stores> converted = __x86::convert_all<_V>(v);
            __execute_n_times<stores>([&](auto i) {
                __vector_store(converted[i], mem + i * VSize, F());
            });
        }
    }

    // masked store {{{2
    template <class _T, size_t _N, class _U, class F>
    static inline void masked_store(const __storage<_T, _N> v, _U *mem, F,
                                    const __mask_member_type<_T> k) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        constexpr size_t max_store_size =
            (sizeof(_U) >= 4 && __have_avx512f) || __have_avx512bw
                ? 64
                : (std::is_floating_point_v<_U> && __have_avx) || __have_avx2 ? 32 : 16;
        if constexpr (std::is_same_v<_T, _U> ||
                      (std::is_integral_v<_T> && std::is_integral_v<_U> &&
                       sizeof(_T) == sizeof(_U))) {
            // bitwise or no conversion, reinterpret:
            const auto kk = [&]() {
                if constexpr (__is_bitmask_v<decltype(k)>) {
                    return __mask_member_type<_U>(k.d);
                } else {
                    return __storage_bitcast<_U>(k);
                }
            }();
            __x86::__maskstore(__storage_bitcast<_U>(v), mem, F(), kk);
        } else if constexpr (std::is_integral_v<_T> && std::is_integral_v<_U> &&
                             sizeof(_T) > sizeof(_U) && __have_avx512f &&
                             (sizeof(_T) >= 4 || __have_avx512bw) &&
                             (sizeof(v) == 64 || __have_avx512vl)) {  // truncating store
            const auto kk = [&]() {
                if constexpr (__is_bitmask_v<decltype(k)>) {
                    return k;
                } else {
                    return __convert_mask<__storage<bool, _N>>(k);
                }
            }();
            if constexpr (sizeof(_T) == 8 && sizeof(_U) == 4) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi32(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi32(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi64_storeu_epi32(mem, kk, v);
                }
            } else if constexpr (sizeof(_T) == 8 && sizeof(_U) == 2) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi16(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi16(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi64_storeu_epi16(mem, kk, v);
                }
            } else if constexpr (sizeof(_T) == 8 && sizeof(_U) == 1) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi64_storeu_epi8(mem, kk, v);
                }
            } else if constexpr (sizeof(_T) == 4 && sizeof(_U) == 2) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi32_storeu_epi16(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi32_storeu_epi16(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi32_storeu_epi16(mem, kk, v);
                }
            } else if constexpr (sizeof(_T) == 4 && sizeof(_U) == 1) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi32_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi32_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi32_storeu_epi8(mem, kk, v);
                }
            } else if constexpr (sizeof(_T) == 2 && sizeof(_U) == 1) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi16_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi16_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi16_storeu_epi8(mem, kk, v);
                }
            } else {
                __assert_unreachable<_T>();
            }
        } else if constexpr (sizeof(_U) <= 8 &&  // no long double
                             !__converts_via_decomposition_v<
                                 _T, _U, max_store_size>  // conversion via decomposition is
                                                        // better handled via the
                                                        // bit_iteration fallback below
        ) {
            using VV = __storage<_U, std::clamp(_N, 16 / sizeof(_U), max_store_size / sizeof(_U))>;
            using _V = typename VV::register_type;
            constexpr bool prefer_bitmask =
                (__have_avx512f && sizeof(_U) >= 4) || __have_avx512bw;
            using M = __storage<std::conditional_t<prefer_bitmask, bool, _U>, VV::width>;
            constexpr size_t VN = __vector_traits<_V>::width;

            if constexpr (VN >= _N) {
                __x86::__maskstore(VV(convert<_V>(v)), mem,
                               // careful, if _V has more elements than the input v (_N),
                               // vector_aligned is incorrect:
                               std::conditional_t<(__vector_traits<_V>::width > _N),
                                                  overaligned_tag<sizeof(_U) * _N>, F>(),
                               __convert_mask<M>(k));
            } else if constexpr (VN * 2 == _N) {
                const std::array<_V, 2> converted = __x86::convert_all<_V>(v);
                __x86::__maskstore(VV(converted[0]), mem, F(), __convert_mask<M>(__extract_part<0, 2>(k)));
                __x86::__maskstore(VV(converted[1]), mem + VV::width, F(), __convert_mask<M>(__extract_part<1, 2>(k)));
            } else if constexpr (VN * 4 == _N) {
                const std::array<_V, 4> converted = __x86::convert_all<_V>(v);
                __x86::__maskstore(VV(converted[0]), mem, F(), __convert_mask<M>(__extract_part<0, 4>(k)));
                __x86::__maskstore(VV(converted[1]), mem + 1 * VV::width, F(), __convert_mask<M>(__extract_part<1, 4>(k)));
                __x86::__maskstore(VV(converted[2]), mem + 2 * VV::width, F(), __convert_mask<M>(__extract_part<2, 4>(k)));
                __x86::__maskstore(VV(converted[3]), mem + 3 * VV::width, F(), __convert_mask<M>(__extract_part<3, 4>(k)));
            } else if constexpr (VN * 8 == _N) {
                const std::array<_V, 8> converted = __x86::convert_all<_V>(v);
                __x86::__maskstore(VV(converted[0]), mem, F(), __convert_mask<M>(__extract_part<0, 8>(k)));
                __x86::__maskstore(VV(converted[1]), mem + 1 * VV::width, F(), __convert_mask<M>(__extract_part<1, 8>(k)));
                __x86::__maskstore(VV(converted[2]), mem + 2 * VV::width, F(), __convert_mask<M>(__extract_part<2, 8>(k)));
                __x86::__maskstore(VV(converted[3]), mem + 3 * VV::width, F(), __convert_mask<M>(__extract_part<3, 8>(k)));
                __x86::__maskstore(VV(converted[4]), mem + 4 * VV::width, F(), __convert_mask<M>(__extract_part<4, 8>(k)));
                __x86::__maskstore(VV(converted[5]), mem + 5 * VV::width, F(), __convert_mask<M>(__extract_part<5, 8>(k)));
                __x86::__maskstore(VV(converted[6]), mem + 6 * VV::width, F(), __convert_mask<M>(__extract_part<6, 8>(k)));
                __x86::__maskstore(VV(converted[7]), mem + 7 * VV::width, F(), __convert_mask<M>(__extract_part<7, 8>(k)));
            } else {
                __assert_unreachable<_T>();
            }
        } else {
            __bit_iteration(__vector_to_bitset(k.d).to_ullong(),
                                  [&](auto i) { mem[i] = static_cast<_U>(v[i]); });
        }
    }

    // complement {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> complement(__storage<_T, _N> x) noexcept
    {
        return __x86::complement(x);
    }

    // unary minus {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> unary_minus(__storage<_T, _N> x) noexcept
    {
        return __x86::unary_minus(x);
    }

    // arithmetic operators {{{2
#define _GLIBCXX_SIMD_ARITHMETIC_OP_(name_)                                                         \
    template <class _T, size_t _N>                                                         \
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> name_(__storage<_T, _N> x, __storage<_T, _N> y)            \
    {                                                                                    \
        return __x86::name_(x, y);                                                 \
    }                                                                                    \
    _GLIBCXX_SIMD_NOTHING_EXPECTING_SEMICOLON
    _GLIBCXX_SIMD_ARITHMETIC_OP_(plus);
    _GLIBCXX_SIMD_ARITHMETIC_OP_(minus);
    _GLIBCXX_SIMD_ARITHMETIC_OP_(multiplies);
#undef _GLIBCXX_SIMD_ARITHMETIC_OP_
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> divides(__storage<_T, _N> x, __storage<_T, _N> y)
    {
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX4
        return __divides(x.d, y.d);
#else
        return x.d / y.d;
#endif
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> modulus(__storage<_T, _N> x, __storage<_T, _N> y)
    {
        static_assert(std::is_integral<_T>::value, "modulus is only supported for integral types");
        return x.d % y.d;
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_and(__storage<_T, _N> x, __storage<_T, _N> y)
    {
        return __vector_cast<_T>(__vector_cast<__llong>(x.d) & __vector_cast<__llong>(y.d));
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_or(__storage<_T, _N> x, __storage<_T, _N> y)
    {
        return __vector_cast<_T>(__vector_cast<__llong>(x.d) | __vector_cast<__llong>(y.d));
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_xor(__storage<_T, _N> x, __storage<_T, _N> y)
    {
        return __vector_cast<_T>(__vector_cast<__llong>(x.d) ^ __vector_cast<__llong>(y.d));
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> bit_shift_left(__storage<_T, _N> x, __storage<_T, _N> y)
    {
        return x.d << y.d;
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> bit_shift_right(__storage<_T, _N> x, __storage<_T, _N> y)
    {
        return x.d >> y.d;
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_shift_left(__storage<_T, _N> x, int y)
    {
        return x.d << y;
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_shift_right(__storage<_T, _N> x,
                                                                         int y)
    {
        return x.d >> y;
    }

    // compares {{{2
    // equal_to {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __mask_member_type<_T> equal_to(__storage<_T, _N> x,
                                                               __storage<_T, _N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<_T>) {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_EQ_OQ);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ);
                } else { __assert_unreachable<_T>(); }
            } else {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmpeq_epi64_mask(x, y);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmpeq_epi32_mask(x, y);
                } else if constexpr (sizeof(_T) == 2) { return _mm512_cmpeq_epi16_mask(x, y);
                } else if constexpr (sizeof(_T) == 1) { return _mm512_cmpeq_epi8_mask(x, y);
                } else { __assert_unreachable<_T>(); }
            }
        } else {
            return __to_storage(x.d == y.d);
        }
    }

    // not_equal_to {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __mask_member_type<_T> not_equal_to(__storage<_T, _N> x,
                                                                   __storage<_T, _N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<_T>) {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_NEQ_UQ);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_NEQ_UQ);
                } else { __assert_unreachable<_T>(); }
            } else {
                       if constexpr (sizeof(_T) == 8) { return ~_mm512_cmpeq_epi64_mask(x, y);
                } else if constexpr (sizeof(_T) == 4) { return ~_mm512_cmpeq_epi32_mask(x, y);
                } else if constexpr (sizeof(_T) == 2) { return ~_mm512_cmpeq_epi16_mask(x, y);
                } else if constexpr (sizeof(_T) == 1) { return ~_mm512_cmpeq_epi8_mask(x, y);
                } else { __assert_unreachable<_T>(); }
            }
        } else {
            return __to_storage(x.d != y.d);
        }
    }

    // less {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __mask_member_type<_T> less(__storage<_T, _N> x,
                                                           __storage<_T, _N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<_T>) {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_LT_OS);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_LT_OS);
                } else { __assert_unreachable<_T>(); }
            } else if constexpr (std::is_signed_v<_T>) {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmplt_epi64_mask(x, y);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmplt_epi32_mask(x, y);
                } else if constexpr (sizeof(_T) == 2) { return _mm512_cmplt_epi16_mask(x, y);
                } else if constexpr (sizeof(_T) == 1) { return _mm512_cmplt_epi8_mask(x, y);
                } else { __assert_unreachable<_T>(); }
            } else {
                static_assert(std::is_unsigned_v<_T>);
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmplt_epu64_mask(x, y);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmplt_epu32_mask(x, y);
                } else if constexpr (sizeof(_T) == 2) { return _mm512_cmplt_epu16_mask(x, y);
                } else if constexpr (sizeof(_T) == 1) { return _mm512_cmplt_epu8_mask(x, y);
                } else { __assert_unreachable<_T>(); }
            }
        } else {
            return __to_storage(x.d < y.d);
        }
    }

    // less_equal {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __mask_member_type<_T> less_equal(__storage<_T, _N> x,
                                                                 __storage<_T, _N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<_T>) {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_LE_OS);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_LE_OS);
                } else { __assert_unreachable<_T>(); }
            } else if constexpr (std::is_signed_v<_T>) {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmple_epi64_mask(x, y);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmple_epi32_mask(x, y);
                } else if constexpr (sizeof(_T) == 2) { return _mm512_cmple_epi16_mask(x, y);
                } else if constexpr (sizeof(_T) == 1) { return _mm512_cmple_epi8_mask(x, y);
                } else { __assert_unreachable<_T>(); }
            } else {
                static_assert(std::is_unsigned_v<_T>);
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmple_epu64_mask(x, y);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmple_epu32_mask(x, y);
                } else if constexpr (sizeof(_T) == 2) { return _mm512_cmple_epu16_mask(x, y);
                } else if constexpr (sizeof(_T) == 1) { return _mm512_cmple_epu8_mask(x, y);
                } else { __assert_unreachable<_T>(); }
            }
        } else {
            return __to_storage(x.d <= y.d);
        }
    }

    // negation {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __mask_member_type<_T> negate(__storage<_T, _N> x) noexcept
    {
        if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
            return equal_to(x, __simd_member_type<_T>());
        } else {
            return __to_storage(!x.d);
        }
    }

    // min, max, clamp {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> min(__storage<_T, _N> a,
                                                                   __storage<_T, _N> b)
    {
        return a.d < b.d ? a.d : b.d;
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> max(__storage<_T, _N> a,
                                                                   __storage<_T, _N> b)
    {
        return a.d > b.d ? a.d : b.d;
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr std::pair<__storage<_T, _N>, __storage<_T, _N>>
    minmax(__storage<_T, _N> a, __storage<_T, _N> b)
    {
        return {a.d < b.d ? a.d : b.d, a.d < b.d ? b.d : a.d};
    }

    // reductions {{{2
    template <class _T, class _BinaryOperation>
    _GLIBCXX_SIMD_INTRINSIC static _T reduce(simd<_T, _Abi> x, _BinaryOperation &&__binary_op)
    {
        constexpr size_t _N = simd_size_v<_T, _Abi>;
        if constexpr (sizeof(x) > 16) {
            using _A = simd_abi::deduce_t<_T, _N / 2>;
            using _V = std::experimental::simd<_T, _A>;
            return __simd_traits<_T, _A>::__simd_impl_type::reduce(
                __binary_op(_V(__private_init, __extract<0, 2>(__data(x).d)),
                          _V(__private_init, __extract<1, 2>(__data(x).d))),
                std::forward<_BinaryOperation>(__binary_op));
        } else {
            const auto &xx = x.d;
            if constexpr (_N == 16) {
                x = __binary_op(make_simd<_T, _N>(_mm_unpacklo_epi8(xx, xx)),
                              make_simd<_T, _N>(_mm_unpackhi_epi8(xx, xx)));
            }
            if constexpr (_N >= 8) {
                x = __binary_op(make_simd<_T, _N>(_mm_unpacklo_epi16(xx, xx)),
                              make_simd<_T, _N>(_mm_unpackhi_epi16(xx, xx)));
            }
            if constexpr (_N >= 4) {
                using _U = std::conditional_t<std::is_floating_point_v<_T>, float, int>;
                const auto y = __vector_cast<_U>(xx.d);
                x = __binary_op(x, make_simd<_T, _N>(__to_storage(
                                     __vector_type_t<_U, 4>{y[3], y[2], y[1], y[0]})));
            }
            const auto y = __vector_cast<__llong>(xx.d);
            return __binary_op(
                x, make_simd<_T, _N>(__to_storage(__vector_type_t<__llong, 2>{y[1], y[1]})))[0];
        }
    }

    // math {{{2
    // sqrt {{{3
    template <class _T, size_t _N> _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> __sqrt(__storage<_T, _N> x)
    {
               if constexpr (__is_sse_ps   <_T, _N>()) { return _mm_sqrt_ps(x);
        } else if constexpr (__is_sse_pd   <_T, _N>()) { return _mm_sqrt_pd(x);
        } else if constexpr (__is_avx_ps   <_T, _N>()) { return _mm256_sqrt_ps(x);
        } else if constexpr (__is_avx_pd   <_T, _N>()) { return _mm256_sqrt_pd(x);
        } else if constexpr (__is_avx512_ps<_T, _N>()) { return _mm512_sqrt_ps(x);
        } else if constexpr (__is_avx512_pd<_T, _N>()) { return _mm512_sqrt_pd(x);
        } else { __assert_unreachable<_T>(); }
    }

    // abs {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> __abs(__storage<_T, _N> x) noexcept
    {
        return __x86::abs(x);
    }

    // trunc {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> __trunc(__storage<_T, _N> x)
    {
        if constexpr (__is_avx512_ps<_T, _N>()) {
            return _mm512_roundscale_round_ps(x, 0x03, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx512_pd<_T, _N>()) {
            return _mm512_roundscale_round_pd(x, 0x03, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx_ps<_T, _N>()) {
            return _mm256_round_ps(x, 0x3);
        } else if constexpr (__is_avx_pd<_T, _N>()) {
            return _mm256_round_pd(x, 0x3);
        } else if constexpr (__have_sse4_1 && __is_sse_ps<_T, _N>()) {
            return _mm_round_ps(x, 0x3);
        } else if constexpr (__have_sse4_1 && __is_sse_pd<_T, _N>()) {
            return _mm_round_pd(x, 0x3);
        } else if constexpr (__is_sse_ps<_T, _N>()) {
            auto truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
            const auto no_fractional_values = __vector_cast<float>(
                __vector_cast<int>(__vector_cast<uint>(x.d) & 0x7f800000u) <
                0x4b000000);  // the exponent is so large that no mantissa bits signify
                              // fractional values (0x3f8 + 23*8 = 0x4b0)
            return __blend(no_fractional_values, x, truncated);
        } else if constexpr (__is_sse_pd<_T, _N>()) {
            const auto abs_x = __abs(x).d;
            const auto min_no_fractional_bits = __vector_cast<double>(
                __vector_broadcast<2>(0x4330'0000'0000'0000ull));  // 0x3ff + 52 = 0x433
            __vector_type16_t<double> truncated =
                (abs_x + min_no_fractional_bits) - min_no_fractional_bits;
            // due to rounding, the result can be too large. In this case `truncated >
            // abs(x)` holds, so subtract 1 to truncated if `abs(x) < truncated`
            truncated -=
                __and(__vector_cast<double>(abs_x < truncated), __vector_broadcast<2>(1.));
            // finally, fix the sign bit:
            return __or(
                __and(__vector_cast<double>(__vector_broadcast<2>(0x8000'0000'0000'0000ull)),
                     x),
                truncated);
        } else {
            __assert_unreachable<_T>();
        }
    }

    // floor {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> __floor(__storage<_T, _N> x)
    {
        if constexpr (__is_avx512_ps<_T, _N>()) {
            return _mm512_roundscale_round_ps(x, 0x01, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx512_pd<_T, _N>()) {
            return _mm512_roundscale_round_pd(x, 0x01, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx_ps<_T, _N>()) {
            return _mm256_round_ps(x, 0x1);
        } else if constexpr (__is_avx_pd<_T, _N>()) {
            return _mm256_round_pd(x, 0x1);
        } else if constexpr (__have_sse4_1 && __is_sse_ps<_T, _N>()) {
            return _mm_floor_ps(x);
        } else if constexpr (__have_sse4_1 && __is_sse_pd<_T, _N>()) {
            return _mm_floor_pd(x);
        } else {
            const auto y = __trunc(x).d;
            const auto negative_input = __vector_cast<_T>(x.d < __vector_broadcast<_N, _T>(0));
            const auto mask = __andnot(__vector_cast<_T>(y == x.d), negative_input);
            return __or(__andnot(mask, y), __and(mask, y - __vector_broadcast<_N, _T>(1)));
        }
    }

    // ceil {{{3
    template <class _T, size_t _N> _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> __ceil(__storage<_T, _N> x)
    {
        if constexpr (__is_avx512_ps<_T, _N>()) {
            return _mm512_roundscale_round_ps(x, 0x02, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx512_pd<_T, _N>()) {
            return _mm512_roundscale_round_pd(x, 0x02, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx_ps<_T, _N>()) {
            return _mm256_round_ps(x, 0x2);
        } else if constexpr (__is_avx_pd<_T, _N>()) {
            return _mm256_round_pd(x, 0x2);
        } else if constexpr (__have_sse4_1 && __is_sse_ps<_T, _N>()) {
            return _mm_ceil_ps(x);
        } else if constexpr (__have_sse4_1 && __is_sse_pd<_T, _N>()) {
            return _mm_ceil_pd(x);
        } else {
            const auto y = __trunc(x).d;
            const auto negative_input = __vector_cast<_T>(x.d < __vector_broadcast<_N, _T>(0));
            const auto inv_mask = __or(__vector_cast<_T>(y == x.d), negative_input);
            return __or(__and(inv_mask, y),
                       __andnot(inv_mask, y + __vector_broadcast<_N, _T>(1)));
        }
    }

    // isnan {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type<_T> __isnan(__storage<_T, _N> x)
    {
             if constexpr (__is_sse_ps   <_T, _N>()) { return _mm_cmpunord_ps(x, x); }
        else if constexpr (__is_avx_ps   <_T, _N>()) { return _mm256_cmp_ps(x, x, _CMP_UNORD_Q); }
        else if constexpr (__is_avx512_ps<_T, _N>()) { return _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q); }
        else if constexpr (__is_sse_pd   <_T, _N>()) { return _mm_cmpunord_pd(x, x); }
        else if constexpr (__is_avx_pd   <_T, _N>()) { return _mm256_cmp_pd(x, x, _CMP_UNORD_Q); }
        else if constexpr (__is_avx512_pd<_T, _N>()) { return _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q); }
        else { __assert_unreachable<_T>(); }
    }

    // isfinite {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type<_T> __isfinite(__storage<_T, _N> x)
    {
        return __x86::__cmpord(x, x.d * _T());
    }

    // isunordered {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type<_T> __isunordered(__storage<_T, _N> x,
                                                          __storage<_T, _N> y)
    {
        return __x86::__cmpunord(x, y);
    }

    // signbit {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type<_T> __signbit(__storage<_T, _N> x)
    {
        using I = __int_for_sizeof_t<_T>;
        if constexpr (__have_avx512dq && __is_avx512_ps<_T, _N>()) {
            return _mm512_movepi32_mask(__vector_cast<__llong>(x));
        } else if constexpr (__have_avx512dq && __is_avx512_pd<_T, _N>()) {
            return _mm512_movepi64_mask(__vector_cast<__llong>(x));
        } else if constexpr (sizeof(x) == 64) {
            const auto signmask = __vector_broadcast<_N>(std::numeric_limits<I>::min());
            return equal_to(__storage<I, _N>(__vector_cast<I>(x.d) & signmask),
                            __storage<I, _N>(signmask));
        } else {
            const auto xx = __vector_cast<I>(x.d);
            constexpr I signmask = std::numeric_limits<I>::min();
            if constexpr ((sizeof(_T) == 4 && (__have_avx2 || sizeof(x) == 16)) ||
                          __have_avx512vl) {
                (void)signmask;
                return __vector_cast<_T>(xx >> std::numeric_limits<I>::digits);
            } else if constexpr ((__have_avx2 || (__have_ssse3 && sizeof(x) == 16))) {
                return __vector_cast<_T>((xx & signmask) == signmask);
            } else {  // SSE2/3 or AVX (w/o AVX2)
                constexpr auto one = __vector_broadcast<_N, _T>(1);
                return __vector_cast<_T>(
                    __vector_cast<_T>((xx & signmask) | __vector_cast<I>(one))  // -1 or 1
                    != one);
            }
        }
    }

    // isnonzerovalue (isnormal | is subnormal == !isinf & !isnan & !is zero) {{{3
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static auto isnonzerovalue(_T x)
    {
        using _U = typename __vector_traits<_T>::value_type;
        return __x86::__cmpord(x * std::numeric_limits<_U>::infinity(),  // NaN if x == 0
                           x * _U()                                  // NaN if x == inf
        );
    }

    template <class _T> _GLIBCXX_SIMD_INTRINSIC static auto isnonzerovalue_mask(_T x)
    {
        using _U = typename __vector_traits<_T>::value_type;
        constexpr size_t _N = __vector_traits<_T>::width;
        const auto a = x * std::numeric_limits<_U>::infinity();  // NaN if x == 0
        const auto b = x * _U();                                 // NaN if x == inf
        if constexpr (__have_avx512vl && __is_sse_ps<_U, _N>()) {
            return _mm_cmp_ps_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (__have_avx512f && __is_sse_ps<_U, _N>()) {
            return __mmask8(0xf &
                            _mm512_cmp_ps_mask(__auto_cast(a), __auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (__have_avx512vl && __is_sse_pd<_U, _N>()) {
            return _mm_cmp_pd_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (__have_avx512f && __is_sse_pd<_U, _N>()) {
            return __mmask8(0x3 &
                            _mm512_cmp_pd_mask(__auto_cast(a), __auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (__have_avx512vl && __is_avx_ps<_U, _N>()) {
            return _mm256_cmp_ps_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (__have_avx512f && __is_avx_ps<_U, _N>()) {
            return __mmask8(_mm512_cmp_ps_mask(__auto_cast(a), __auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (__have_avx512vl && __is_avx_pd<_U, _N>()) {
            return _mm256_cmp_pd_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (__have_avx512f && __is_avx_pd<_U, _N>()) {
            return __mmask8(0xf &
                            _mm512_cmp_pd_mask(__auto_cast(a), __auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (__is_avx512_ps<_U, _N>()) {
            return _mm512_cmp_ps_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (__is_avx512_pd<_U, _N>()) {
            return _mm512_cmp_pd_mask(a, b, _CMP_ORD_Q);
        } else {
            __assert_unreachable<_T>();
        }
    }

    // isinf {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type<_T> __isinf(__storage<_T, _N> x)
    {
        if constexpr (__is_avx512_pd<_T, _N>()) {
            if constexpr (__have_avx512dq) {
                return _mm512_fpclass_pd_mask(x, 0x08) | _mm512_fpclass_pd_mask(x, 0x10);
            } else {
                return _mm512_cmp_epi64_mask(__vector_cast<__llong>(__x86::abs(x)),
                                             __vector_broadcast<_N>(0x7ff0000000000000ll),
                                             _CMP_EQ_OQ);
            }
        } else if constexpr (__is_avx512_ps<_T, _N>()) {
            if constexpr (__have_avx512dq) {
                return _mm512_fpclass_ps_mask(x, 0x08) | _mm512_fpclass_ps_mask(x, 0x10);
            } else {
                return _mm512_cmp_epi32_mask(__vector_cast<__llong>(__x86::abs(x)),
                                             __auto_cast(__vector_broadcast<_N>(0x7f800000u)),
                                             _CMP_EQ_OQ);
            }
        } else if constexpr (__have_avx512dq_vl) {
            if constexpr (__is_sse_pd<_T, _N>()) {
                return __vector_cast<double>(_mm_movm_epi64(_mm_fpclass_pd_mask(x, 0x08) |
                                                           _mm_fpclass_pd_mask(x, 0x10)));
            } else if constexpr (__is_avx_pd<_T, _N>()) {
                return __vector_cast<double>(_mm256_movm_epi64(
                    _mm256_fpclass_pd_mask(x, 0x08) | _mm256_fpclass_pd_mask(x, 0x10)));
            } else if constexpr (__is_sse_ps<_T, _N>()) {
                return __vector_cast<float>(_mm_movm_epi32(_mm_fpclass_ps_mask(x, 0x08) |
                                                          _mm_fpclass_ps_mask(x, 0x10)));
            } else if constexpr (__is_avx_ps<_T, _N>()) {
                return __vector_cast<float>(_mm256_movm_epi32(
                    _mm256_fpclass_ps_mask(x, 0x08) | _mm256_fpclass_ps_mask(x, 0x10)));
            } else {
                __assert_unreachable<_T>();
            }
        } else {
            // compares to inf using the corresponding integer type
            return __vector_cast<_T>(__vector_cast<__int_for_sizeof_t<_T>>(__abs(x).d) ==
                                   __vector_cast<__int_for_sizeof_t<_T>>(__vector_broadcast<_N>(
                                       std::numeric_limits<_T>::infinity())));
            // alternative:
            //return __vector_cast<_T>(__abs(x).d > std::numeric_limits<_T>::max());
        }
    }
    // isnormal {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type<_T> __isnormal(__storage<_T, _N> x)
    {
        // subnormals -> 0
        // 0 -> 0
        // inf -> inf
        // -inf -> inf
        // nan -> inf
        // normal value -> positive value / not 0
        return isnonzerovalue(
            __and(x.d, __vector_broadcast<_N>(std::numeric_limits<_T>::infinity())));
    }

    // fpclassify {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __fixed_size_storage<int, _N> __fpclassify(__storage<_T, _N> x)
    {
        if constexpr (__is_avx512_pd<_T, _N>()) {
            // AVX512 is special because we want to use an __mmask to blend int vectors
            // (coming from double vectors). GCC doesn't allow this combination on the
            // ternary operator. Thus, resort to intrinsics:
            if constexpr (__have_avx512vl) {
                auto &&b = [](int y) { return __to_intrin(__vector_broadcast<_N>(y)); };
                return {_mm256_mask_mov_epi32(
                    _mm256_mask_mov_epi32(
                        _mm256_mask_mov_epi32(b(FP_NORMAL), __isnan(x), b(FP_NAN)),
                        __isinf(x), b(FP_INFINITE)),
                    _mm512_cmp_pd_mask(
                        __abs(x),
                        __vector_broadcast<_N>(std::numeric_limits<double>::min()),
                        _CMP_LT_OS),
                    _mm256_mask_mov_epi32(
                        b(FP_SUBNORMAL),
                        _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                        b(FP_ZERO)))};
            } else {
                auto &&b = [](int y) {
                    return _mm512_castsi256_si512(__to_intrin(__vector_broadcast<_N>(y)));
                };
                return {__lo256(_mm512_mask_mov_epi32(
                    _mm512_mask_mov_epi32(
                        _mm512_mask_mov_epi32(b(FP_NORMAL), __isnan(x), b(FP_NAN)),
                        __isinf(x), b(FP_INFINITE)),
                    _mm512_cmp_pd_mask(
                        __abs(x),
                        __vector_broadcast<_N>(std::numeric_limits<double>::min()),
                        _CMP_LT_OS),
                    _mm512_mask_mov_epi32(
                        b(FP_SUBNORMAL),
                        _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                        b(FP_ZERO))))};
            }
        } else {
            constexpr auto fp_normal =
                __vector_cast<_T>(__vector_broadcast<_N, __int_for_sizeof_t<_T>>(FP_NORMAL));
            constexpr auto fp_nan =
                __vector_cast<_T>(__vector_broadcast<_N, __int_for_sizeof_t<_T>>(FP_NAN));
            constexpr auto fp_infinite =
                __vector_cast<_T>(__vector_broadcast<_N, __int_for_sizeof_t<_T>>(FP_INFINITE));
            constexpr auto fp_subnormal =
                __vector_cast<_T>(__vector_broadcast<_N, __int_for_sizeof_t<_T>>(FP_SUBNORMAL));
            constexpr auto fp_zero =
                __vector_cast<_T>(__vector_broadcast<_N, __int_for_sizeof_t<_T>>(FP_ZERO));

            const auto tmp = __vector_cast<__llong>(
                __abs(x).d < std::numeric_limits<_T>::min()
                    ? (x.d == 0 ? fp_zero : fp_subnormal)
                    : __blend(__isinf(x).d, __blend(__isnan(x).d, fp_normal, fp_nan),
                                 fp_infinite));
            if constexpr (std::is_same_v<_T, float>) {
                if constexpr (__fixed_size_storage<int, _N>::tuple_size == 1) {
                    return {tmp};
                } else if constexpr (__fixed_size_storage<int, _N>::tuple_size == 2) {
                    return {__extract<0, 2>(tmp), __extract<1, 2>(tmp)};
                } else {
                    __assert_unreachable<_T>();
                }
            } else if constexpr (__is_sse_pd<_T, _N>()) {
                static_assert(__fixed_size_storage<int, _N>::tuple_size == 2);
                return {_mm_cvtsi128_si32(tmp),
                        {_mm_cvtsi128_si32(_mm_unpackhi_epi64(tmp, tmp))}};
            } else if constexpr (__is_avx_pd<_T, _N>()) {
                static_assert(__fixed_size_storage<int, _N>::tuple_size == 1);
                return {_mm_packs_epi32(__lo128(tmp), __hi128(tmp))};
            } else {
                __assert_unreachable<_T>();
            }
        }
    }

    // __increment & __decrement{{{2
    template <class _T, size_t _N> _GLIBCXX_SIMD_INTRINSIC static void __increment(__storage<_T, _N> &x)
    {
        x = plus<_T, _N>(x, __vector_broadcast<_N, _T>(1));
    }
    template <class _T, size_t _N> _GLIBCXX_SIMD_INTRINSIC static void __decrement(__storage<_T, _N> &x)
    {
        x = minus<_T, _N>(x, __vector_broadcast<_N, _T>(1));
    }

    // smart_reference access {{{2
    template <class _T, size_t _N, class _U>
    _GLIBCXX_SIMD_INTRINSIC static void set(__storage<_T, _N> &v, int i, _U &&x) noexcept
    {
        v.set(i, std::forward<_U>(x));
    }

    // masked_assign{{{2
    template <class _T, class K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(__storage<K, _N> k, __storage<_T, _N> &lhs,
                                           __id<__storage<_T, _N>> rhs)
    {
        lhs = __blend(k.d, lhs.d, rhs.d);
    }

    template <class _T, class K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(__storage<K, _N> k, __storage<_T, _N> &lhs,
                                           __id<_T> rhs)
    {
        if (__builtin_constant_p(rhs) && rhs == 0 && std::is_same<K, _T>::value) {
            if constexpr (!__is_bitmask(k)) {
                // the __andnot optimization only makes sense if k.d is a vector register
                lhs.d = __andnot(k.d, lhs.d);
                return;
            } else {
                // for AVX512/__mmask, a _mm512_maskz_mov is best
                lhs = __blend(k, lhs, __intrinsic_type_t<_T, _N>());
                return;
            }
        }
        lhs = __blend(k.d, lhs.d, __vector_broadcast<_N>(rhs));
    }

    // __masked_cassign {{{2
    template <template <typename> class Op, class _T, class K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_cassign(const __storage<K, _N> k, __storage<_T, _N> &lhs,
                                            const __id<__storage<_T, _N>> rhs)
    {
        lhs = __blend(
            k.d, lhs.d, __data(Op<void>{}(make_simd(lhs), make_simd(rhs))).d);
    }

    template <template <typename> class Op, class _T, class K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_cassign(const __storage<K, _N> k, __storage<_T, _N> &lhs,
                                            const __id<_T> rhs)
    {
        lhs = __blend(
            k.d, lhs.d,
            __data(
                Op<void>{}(make_simd(lhs), make_simd<_T, _N>(__vector_broadcast<_N>(rhs))))
                .d);
    }

    // masked_unary {{{2
    template <template <typename> class Op, class _T, class K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> masked_unary(const __storage<K, _N> k,
                                                            const __storage<_T, _N> v)
    {
        auto vv = make_simd(v);
        Op<decltype(vv)> op;
        return __blend(k, v, __data(op(vv)));
    }

    //}}}2
};

// __generic_mask_impl {{{1
template <class _Abi> struct __generic_mask_impl {
    // member types {{{2
    template <class _T> using __type_tag = _T *;
    template <class _T> using simd_mask = std::experimental::simd_mask<_T, _Abi>;
    template <class _T>
    using __simd_member_type = typename _Abi::template __traits<_T>::__simd_member_type;
    template <class _T>
    using __mask_member_type = typename _Abi::template __traits<_T>::__mask_member_type;

    // masked load {{{2
    template <class _T, size_t _N, class F>
    static inline __storage<_T, _N> masked_load(__storage<_T, _N> merge, __storage<_T, _N> mask,
                                            const bool *mem, F) noexcept
    {
        if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
            if constexpr (__have_avx512bw_vl) {
                if constexpr (_N == 8) {
                    const auto a = _mm_mask_loadu_epi8(__m128i(), mask, mem);
                    return (merge & ~mask) | _mm_test_epi8_mask(a, a);
                } else if constexpr (_N == 16) {
                    const auto a = _mm_mask_loadu_epi8(__m128i(), mask, mem);
                    return (merge & ~mask) | _mm_test_epi8_mask(a, a);
                } else if constexpr (_N == 32) {
                    const auto a = _mm256_mask_loadu_epi8(__m256i(), mask, mem);
                    return (merge & ~mask) | _mm256_test_epi8_mask(a, a);
                } else if constexpr (_N == 64) {
                    const auto a = _mm512_mask_loadu_epi8(__m512i(), mask, mem);
                    return (merge & ~mask) | _mm512_test_epi8_mask(a, a);
                } else {
                    __assert_unreachable<_T>();
                }
            } else {
                __bit_iteration(mask, [&](auto i) { merge.set(i, mem[i]); });
                return merge;
            }
        } else if constexpr (__have_avx512bw_vl && _N == 32 && sizeof(_T) == 1) {
            const auto k = __convert_mask<__storage<bool, _N>>(mask);
            merge = __to_storage(
                _mm256_mask_sub_epi8(__vector_cast<__llong>(merge), k, __m256i(),
                                     _mm256_mask_loadu_epi8(__m256i(), k, mem)));
        } else if constexpr (__have_avx512bw_vl && _N == 16 && sizeof(_T) == 1) {
            const auto k = __convert_mask<__storage<bool, _N>>(mask);
            merge = __to_storage(_mm_mask_sub_epi8(__vector_cast<__llong>(merge), k, __m128i(),
                                                 _mm_mask_loadu_epi8(__m128i(), k, mem)));
        } else if constexpr (__have_avx512bw_vl && _N == 16 && sizeof(_T) == 2) {
            const auto k = __convert_mask<__storage<bool, _N>>(mask);
            merge = __to_storage(_mm256_mask_sub_epi16(
                __vector_cast<__llong>(merge), k, __m256i(),
                _mm256_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 8 && sizeof(_T) == 2) {
            const auto k = __convert_mask<__storage<bool, _N>>(mask);
            merge = __to_storage(_mm_mask_sub_epi16(
                __vector_cast<__llong>(merge), k, __m128i(),
                _mm_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 8 && sizeof(_T) == 4) {
            const auto k = __convert_mask<__storage<bool, _N>>(mask);
            merge = __to_storage(_mm256_mask_sub_epi32(
                __vector_cast<__llong>(merge), k, __m256i(),
                _mm256_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 4 && sizeof(_T) == 4) {
            const auto k = __convert_mask<__storage<bool, _N>>(mask);
            merge = __to_storage(_mm_mask_sub_epi32(
                __vector_cast<__llong>(merge), k, __m128i(),
                _mm_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 4 && sizeof(_T) == 8) {
            const auto k = __convert_mask<__storage<bool, _N>>(mask);
            merge = __to_storage(_mm256_mask_sub_epi64(
                __vector_cast<__llong>(merge), k, __m256i(),
                _mm256_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 2 && sizeof(_T) == 8) {
            const auto k = __convert_mask<__storage<bool, _N>>(mask);
            merge = __to_storage(_mm_mask_sub_epi64(
                __vector_cast<__llong>(merge), k, __m128i(),
                _mm_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else {
            // AVX(2) has 32/64 bit maskload, but nothing at 8 bit granularity
            auto tmp = __storage_bitcast<__int_for_sizeof_t<_T>>(merge);
            __bit_iteration(__vector_to_bitset(mask.d).to_ullong(),
                                  [&](auto i) { tmp.set(i, -mem[i]); });
            merge = __storage_bitcast<_T>(tmp);
        }
        return merge;
    }

    // store {{{2
    template <class _T, size_t _N, class F>
    _GLIBCXX_SIMD_INTRINSIC static void store(__storage<_T, _N> v, bool *mem, F) noexcept
    {
        if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>()) {
            if constexpr (_N == 2 && __have_sse2) {
                const auto k = __vector_cast<int>(v.d);
                mem[0] = -k[1];
                mem[1] = -k[3];
            } else if constexpr (_N == 4 && __have_sse2) {
                const unsigned bool4 =
                    __vector_cast<uint>(_mm_packs_epi16(
                        _mm_packs_epi32(__vector_cast<__llong>(v), __m128i()), __m128i()))[0] &
                    0x01010101u;
                std::memcpy(mem, &bool4, 4);
            } else if constexpr (std::is_same_v<_T, float> && __have_mmx) {
                const __m128 k(v);
                const __m64 kk = _mm_cvtps_pi8(__and(k, _mm_set1_ps(1.f)));
                __vector_store<4>(kk, mem, F());
                _mm_empty();
            } else if constexpr (_N == 8 && __have_sse2) {
                __vector_store<8>(
                    _mm_packs_epi16(__to_intrin(__vector_cast<ushort>(v.d) >> 15),
                                    __m128i()),
                    mem, F());
            } else if constexpr (_N == 16 && __have_sse2) {
                __vector_store(v.d & 1, mem, F());
            } else {
                __assert_unreachable<_T>();
            }
        } else if constexpr (__is_abi<_Abi, simd_abi::__avx_abi>()) {
            if constexpr (_N == 4 && __have_avx) {
                auto k = __vector_cast<__llong>(v);
                int bool4;
                if constexpr (__have_avx2) {
                    bool4 = _mm256_movemask_epi8(k);
                } else {
                    bool4 = (_mm_movemask_epi8(__lo128(k)) |
                             (_mm_movemask_epi8(__hi128(k)) << 16));
                }
                bool4 &= 0x01010101;
                std::memcpy(mem, &bool4, 4);
            } else if constexpr (_N == 8 && __have_avx) {
                const auto k = __vector_cast<__llong>(v);
                const auto k2 = _mm_srli_epi16(_mm_packs_epi16(__lo128(k), __hi128(k)), 15);
                const auto k3 = _mm_packs_epi16(k2, __m128i());
                __vector_store<8>(k3, mem, F());
            } else if constexpr (_N == 16 && __have_avx2) {
                const auto x = _mm256_srli_epi16(v, 15);
                const auto bools = _mm_packs_epi16(__lo128(x), __hi128(x));
                __vector_store<16>(bools, mem, F());
            } else if constexpr (_N == 16 && __have_avx) {
                const auto bools = 1 & __vector_cast<__uchar>(_mm_packs_epi16(
                                           __lo128(v.intrin()), __hi128(v.intrin())));
                __vector_store<16>(bools, mem, F());
            } else if constexpr (_N == 32 && __have_avx) {
                __vector_store<32>(1 & v.d, mem, F());
            } else {
                __assert_unreachable<_T>();
            }
        } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
            if constexpr (_N == 8) {
                __vector_store<8>(
#if _GLIBCXX_SIMD_HAVE_AVX512VL && _GLIBCXX_SIMD_HAVE_AVX512BW
                    _mm_maskz_set1_epi8(v, 1),
#elif defined __x86_64__
                    __make_storage<__ullong>(_pdep_u64(v, 0x0101010101010101ULL), 0ull),
#else
                    __make_storage<uint>(_pdep_u32(v, 0x01010101U),
                                       _pdep_u32(v >> 4, 0x01010101U)),
#endif
                    mem, F());
            } else if constexpr (_N == 16 && __have_avx512bw_vl) {
                __vector_store(_mm_maskz_set1_epi8(v, 1), mem, F());
            } else if constexpr (_N == 16 && __have_avx512f) {
                _mm512_mask_cvtepi32_storeu_epi8(mem, ~__mmask16(),
                                                 _mm512_maskz_set1_epi32(v, 1));
            } else if constexpr (_N == 32 && __have_avx512bw_vl) {
                __vector_store(_mm256_maskz_set1_epi8(v, 1), mem, F());
            } else if constexpr (_N == 32 && __have_avx512bw) {
                __vector_store(__lo256(_mm512_maskz_set1_epi8(v, 1)), mem, F());
            } else if constexpr (_N == 64 && __have_avx512bw) {
                __vector_store(_mm512_maskz_set1_epi8(v, 1), mem, F());
            } else {
                __assert_unreachable<_T>();
            }
        } else {
            __assert_unreachable<_T>();
        }
    }

    // masked store {{{2
    template <class _T, size_t _N, class F>
    static inline void masked_store(const __storage<_T, _N> v, bool *mem, F,
                                    const __storage<_T, _N> k) noexcept
    {
        if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
            if constexpr (_N == 8 && __have_avx512bw_vl) {
                _mm_mask_cvtepi16_storeu_epi8(mem, k, _mm_maskz_set1_epi16(v, 1));
            } else if constexpr (_N == 8 && __have_avx512vl) {
                _mm256_mask_cvtepi32_storeu_epi8(mem, k, _mm256_maskz_set1_epi32(v, 1));
            } else if constexpr (_N == 8) {
                // we rely on k < 0x100:
                _mm512_mask_cvtepi32_storeu_epi8(mem, k, _mm512_maskz_set1_epi32(v, 1));
            } else if constexpr (_N == 16 && __have_avx512bw_vl) {
                _mm_mask_storeu_epi8(mem, k, _mm_maskz_set1_epi8(v, 1));
            } else if constexpr (_N == 16) {
                _mm512_mask_cvtepi32_storeu_epi8(mem, k, _mm512_maskz_set1_epi32(v, 1));
            } else if constexpr (_N == 32 && __have_avx512bw_vl) {
                _mm256_mask_storeu_epi8(mem, k, _mm256_maskz_set1_epi8(v, 1));
            } else if constexpr (_N == 32 && __have_avx512bw) {
                _mm256_mask_storeu_epi8(mem, k, __lo256(_mm512_maskz_set1_epi8(v, 1)));
            } else if constexpr (_N == 64 && __have_avx512bw) {
                _mm512_mask_storeu_epi8(mem, k, _mm512_maskz_set1_epi8(v, 1));
            } else {
                __assert_unreachable<_T>();
            }
        } else {
            __bit_iteration(__vector_to_bitset(k.d).to_ullong(), [&](auto i) { mem[i] = v[i]; });
        }
    }

    // __from_bitset{{{2
    template <size_t _N, class _T>
    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type<_T> __from_bitset(std::bitset<_N> bits, __type_tag<_T>)
    {
        return __convert_mask<typename __mask_member_type<_T>::register_type>(bits);
    }

    // logical and bitwise operators {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> logical_and(const __storage<_T, _N> &x,
                                                            const __storage<_T, _N> &y)
    {
        if constexpr (std::is_same_v<_T, bool>) {
            return x.d & y.d;
        } else {
            return __and(x.d, y.d);
        }
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> logical_or(const __storage<_T, _N> &x,
                                                           const __storage<_T, _N> &y)
    {
        if constexpr (std::is_same_v<_T, bool>) {
            return x.d | y.d;
        } else {
            return __or(x.d, y.d);
        }
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_and(const __storage<_T, _N> &x,
                                                        const __storage<_T, _N> &y)
    {
        if constexpr (std::is_same_v<_T, bool>) {
            return x.d & y.d;
        } else {
            return __and(x.d, y.d);
        }
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_or(const __storage<_T, _N> &x,
                                                       const __storage<_T, _N> &y)
    {
        if constexpr (std::is_same_v<_T, bool>) {
            return x.d | y.d;
        } else {
            return __or(x.d, y.d);
        }
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_xor(const __storage<_T, _N> &x,
                                                        const __storage<_T, _N> &y)
    {
        if constexpr (std::is_same_v<_T, bool>) {
            return x.d ^ y.d;
        } else {
            return __xor(x.d, y.d);
        }
    }

    // smart_reference access {{{2
    template <class _T, size_t _N> static void set(__storage<_T, _N> &k, int i, bool x) noexcept
    {
        if constexpr (std::is_same_v<_T, bool>) {
            k.set(i, x);
        } else {
            using int_t = __vector_type_t<__int_for_sizeof_t<_T>, _N>;
            auto tmp = reinterpret_cast<int_t>(k.d);
            tmp[i] = -x;
            k.d = __auto_cast(tmp);
        }
    }
    // masked_assign{{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(__storage<_T, _N> k, __storage<_T, _N> &lhs,
                                           __id<__storage<_T, _N>> rhs)
    {
        lhs = __blend(k.d, lhs.d, rhs.d);
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(__storage<_T, _N> k, __storage<_T, _N> &lhs, bool rhs)
    {
        if (__builtin_constant_p(rhs)) {
            if (rhs == false) {
                lhs = __andnot(k.d, lhs.d);
            } else {
                lhs = __or(k.d, lhs.d);
            }
            return;
        }
        lhs = __blend(k, lhs, __data(simd_mask<_T>(rhs)));
    }

    //}}}2
};

//}}}1

struct __sse_mask_impl : __generic_mask_impl<simd_abi::__sse> {};
struct __sse_simd_impl : __generic_simd_impl<simd_abi::__sse> {};

struct __avx_mask_impl : __generic_mask_impl<simd_abi::__avx> {};
struct __avx_simd_impl : __generic_simd_impl<simd_abi::__avx> {};

struct __avx512_simd_impl : __generic_simd_impl<simd_abi::__avx512> {};
struct __avx512_mask_impl : __generic_mask_impl<simd_abi::__avx512> {};

struct __neon_mask_impl : __generic_mask_impl<simd_abi::__neon> {};
struct __neon_simd_impl : __generic_simd_impl<simd_abi::__neon> {};

/**
 * The fixed_size ABI gives the following guarantees:
 *  - simd objects are passed via the stack
 *  - memory layout of `simd<_T, _N>` is equivalent to `std::array<_T, _N>`
 *  - alignment of `simd<_T, _N>` is `_N * sizeof(_T)` if _N is a power-of-2 value,
 *    otherwise `__next_power_of_2(_N * sizeof(_T))` (Note: if the alignment were to
 *    exceed the system/compiler maximum, it is bounded to that maximum)
 *  - simd_mask objects are passed like std::bitset<_N>
 *  - memory layout of `simd_mask<_T, _N>` is equivalent to `std::bitset<_N>`
 *  - alignment of `simd_mask<_T, _N>` is equal to the alignment of `std::bitset<_N>`
 */
// __autocvt_to_simd {{{
template <class _T, bool = std::is_arithmetic_v<std::decay_t<_T>>>
struct __autocvt_to_simd {
    _T d;
    using TT = std::decay_t<_T>;
    operator TT() { return d; }
    operator TT &()
    {
        static_assert(std::is_lvalue_reference<_T>::value, "");
        static_assert(!std::is_const<_T>::value, "");
        return d;
    }
    operator TT *()
    {
        static_assert(std::is_lvalue_reference<_T>::value, "");
        static_assert(!std::is_const<_T>::value, "");
        return &d;
    }

    constexpr inline __autocvt_to_simd(_T dd) : d(dd) {}

    template <class _Abi> operator simd<typename TT::value_type, _Abi>()
    {
        return {__private_init, d};
    }

    template <class _Abi> operator simd<typename TT::value_type, _Abi> &()
    {
        return *reinterpret_cast<simd<typename TT::value_type, _Abi> *>(&d);
    }

    template <class _Abi> operator simd<typename TT::value_type, _Abi> *()
    {
        return reinterpret_cast<simd<typename TT::value_type, _Abi> *>(&d);
    }
};
template <class _T> __autocvt_to_simd(_T &&)->__autocvt_to_simd<_T>;

template <class _T> struct __autocvt_to_simd<_T, true> {
    using TT = std::decay_t<_T>;
    _T d;
    fixed_size_simd<TT, 1> fd;

    constexpr inline __autocvt_to_simd(_T dd) : d(dd), fd(d) {}
    ~__autocvt_to_simd()
    {
        d = __data(fd).first;
    }

    operator fixed_size_simd<TT, 1>()
    {
        return fd;
    }
    operator fixed_size_simd<TT, 1> &()
    {
        static_assert(std::is_lvalue_reference<_T>::value, "");
        static_assert(!std::is_const<_T>::value, "");
        return fd;
    }
    operator fixed_size_simd<TT, 1> *()
    {
        static_assert(std::is_lvalue_reference<_T>::value, "");
        static_assert(!std::is_const<_T>::value, "");
        return &fd;
    }
};

// }}}
// __fixed_size_storage<_T, _N>{{{1
template <class _T, int _N, class Tuple,
          class Next = simd<_T, __all_native_abis::best_abi<_T, _N>>,
          int Remain = _N - int(Next::size())>
struct __fixed_size_storage_builder;

template <class _T, int _N>
struct __fixed_size_storage_builder_wrapper
    : public __fixed_size_storage_builder<_T, _N, __simd_tuple<_T>> {
};

template <class _T, int _N, class... As, class Next>
struct __fixed_size_storage_builder<_T, _N, __simd_tuple<_T, As...>, Next, 0> {
    using type = __simd_tuple<_T, As..., typename Next::abi_type>;
};

template <class _T, int _N, class... As, class Next, int Remain>
struct __fixed_size_storage_builder<_T, _N, __simd_tuple<_T, As...>, Next, Remain> {
    using type = typename __fixed_size_storage_builder<
        _T, Remain, __simd_tuple<_T, As..., typename Next::abi_type>>::type;
};

// __n_abis_in_tuple {{{1
template <class _T> struct __seq_op;
template <size_t I0, size_t... Is> struct __seq_op<std::index_sequence<I0, Is...>> {
    using __first_plus_one = std::index_sequence<I0 + 1, Is...>;
    using __notfirst_plus_one = std::index_sequence<I0, (Is + 1)...>;
    template <size_t First, size_t Add>
    using __prepend = std::index_sequence<First, I0 + Add, (Is + Add)...>;
};

template <class _T> struct __n_abis_in_tuple;
template <class _T> struct __n_abis_in_tuple<__simd_tuple<_T>> {
    using __counts = std::index_sequence<0>;
    using __begins = std::index_sequence<0>;
};
template <class _T, class _A> struct __n_abis_in_tuple<__simd_tuple<_T, _A>> {
    using __counts = std::index_sequence<1>;
    using __begins = std::index_sequence<0>;
};
template <class _T, class _A0, class... As>
struct __n_abis_in_tuple<__simd_tuple<_T, _A0, _A0, As...>> {
    using __counts = typename __seq_op<typename __n_abis_in_tuple<
        __simd_tuple<_T, _A0, As...>>::__counts>::__first_plus_one;
    using __begins = typename __seq_op<typename __n_abis_in_tuple<
        __simd_tuple<_T, _A0, As...>>::__begins>::__notfirst_plus_one;
};
template <class _T, class _A0, class _A1, class... As>
struct __n_abis_in_tuple<__simd_tuple<_T, _A0, _A1, As...>> {
    using __counts = typename __seq_op<typename __n_abis_in_tuple<
        __simd_tuple<_T, _A1, As...>>::__counts>::template __prepend<1, 0>;
    using __begins = typename __seq_op<typename __n_abis_in_tuple<
        __simd_tuple<_T, _A1, As...>>::__begins>::template __prepend<0, 1>;
};

// __tree_reduction {{{1
template <size_t Count, size_t Begin> struct __tree_reduction {
    static_assert(Count > 0,
                  "__tree_reduction requires at least one simd object to work with");
    template <class _T, class... As, class _BinaryOperation>
    auto operator()(const __simd_tuple<_T, As...> &tup,
                    const _BinaryOperation &__binary_op) const noexcept
    {
        constexpr size_t left = __next_power_of_2(Count) / 2;
        constexpr size_t right = Count - left;
        return __binary_op(__tree_reduction<left, Begin>()(tup, __binary_op),
                         __tree_reduction<right, Begin + left>()(tup, __binary_op));
    }
};
template <size_t Begin> struct __tree_reduction<1, Begin> {
    template <class _T, class... As, class _BinaryOperation>
    auto operator()(const __simd_tuple<_T, As...> &tup, const _BinaryOperation &) const
        noexcept
    {
        return __get_simd_at<Begin>(tup);
    }
};
template <size_t Begin> struct __tree_reduction<2, Begin> {
    template <class _T, class... As, class _BinaryOperation>
    auto operator()(const __simd_tuple<_T, As...> &tup,
                    const _BinaryOperation &__binary_op) const noexcept
    {
        return __binary_op(__get_simd_at<Begin>(tup),
                         __get_simd_at<Begin + 1>(tup));
    }
};

// __vec_to_scalar_reduction {{{1
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
//      __vec_to_scalar_reduction.
//
//   3. If __vec_to_scalar_reduction is called with a one-element tuple, call std::experimental::reduce to
//      reduce to a scalar and return.
template <class _T, class _A0, class _A1, class _BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC simd<_T, _A1> __vec_to_scalar_reduction_first_pair(
    const simd<_T, _A0> left, const simd<_T, _A1> right, const _BinaryOperation &__binary_op,
    __size_constant<2>) noexcept
{
    const std::array<simd<_T, _A1>, 2> splitted = split<simd<_T, _A1>>(left);
    return __binary_op(__binary_op(splitted[0], right), splitted[1]);
}

template <class _T, class _A0, class _A1, class _BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC simd<_T, _A1> __vec_to_scalar_reduction_first_pair(
    const simd<_T, _A0> left, const simd<_T, _A1> right, const _BinaryOperation &__binary_op,
    __size_constant<4>) noexcept
{
    constexpr auto N0 = simd_size_v<_T, _A0> / 2;
    const auto left2 = split<simd<_T, simd_abi::deduce_t<_T, N0>>>(left);
    const std::array<simd<_T, _A1>, 2> splitted =
        split<simd<_T, _A1>>(__binary_op(left2[0], left2[1]));
    return __binary_op(__binary_op(splitted[0], right), splitted[1]);
}

template <class _T, class _A0, class _A1, class _BinaryOperation, size_t Factor>
_GLIBCXX_SIMD_INTRINSIC simd<_T, simd_abi::scalar> __vec_to_scalar_reduction_first_pair(
    const simd<_T, _A0> left, const simd<_T, _A1> right, const _BinaryOperation &__binary_op,
    __size_constant<Factor>) noexcept
{
    return __binary_op(std::experimental::reduce(left, __binary_op), std::experimental::reduce(right, __binary_op));
}

template <class _T, class _A0, class _BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC _T __vec_to_scalar_reduction(const __simd_tuple<_T, _A0> &tup,
                                       const _BinaryOperation &__binary_op) noexcept
{
    return std::experimental::reduce(simd<_T, _A0>(__private_init, tup.first), __binary_op);
}

template <class _T, class _A0, class _A1, class... As, class _BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC _T __vec_to_scalar_reduction(const __simd_tuple<_T, _A0, _A1, As...> &tup,
                                       const _BinaryOperation &__binary_op) noexcept
{
    return __vec_to_scalar_reduction(
        __simd_tuple_concat(
            __make_simd_tuple(
                __vec_to_scalar_reduction_first_pair<_T, _A0, _A1, _BinaryOperation>(
                    {__private_init, tup.first}, {__private_init, tup.second.first},
                    __binary_op,
                    __size_constant<simd_size_v<_T, _A0> / simd_size_v<_T, _A1>>())),
            tup.second.second),
        __binary_op);
}

// __partial_bitset_to_member_type {{{1
template <class _V, size_t _N>
_GLIBCXX_SIMD_INTRINSIC auto __partial_bitset_to_member_type(std::bitset<_N> shifted_bits)
{
    static_assert(_V::size() <= _N, "");
    using M = typename _V::mask_type;
    using _T = typename _V::value_type;
    constexpr _T *__type_tag = nullptr;
    return __get_impl_t<M>::__from_bitset(
        std::bitset<_V::size()>(shifted_bits.to_ullong()), __type_tag);
}

// __fixed_size_simd_impl {{{1
template <int _N> struct __fixed_size_simd_impl {
    // member types {{{2
    using __mask_member_type = std::bitset<_N>;
    template <class _T> using __simd_member_type = __fixed_size_storage<_T, _N>;
    template <class _T>
    static constexpr std::size_t tuple_size = __simd_member_type<_T>::tuple_size;
    template <class _T>
    static constexpr std::make_index_sequence<__simd_member_type<_T>::tuple_size> index_seq = {};
    template <class _T> using simd = std::experimental::simd<_T, simd_abi::fixed_size<_N>>;
    template <class _T> using simd_mask = std::experimental::simd_mask<_T, simd_abi::fixed_size<_N>>;
    template <class _T> using __type_tag = _T *;

    // broadcast {{{2
    template <class _T> static constexpr inline __simd_member_type<_T> __broadcast(_T x) noexcept
    {
        return __simd_member_type<_T>::generate(
            [&](auto meta) { return meta.__broadcast(x); });
    }

    // generator {{{2
    template <class F, class _T>
    _GLIBCXX_SIMD_INTRINSIC static __simd_member_type<_T> generator(F &&gen, __type_tag<_T>)
    {
        return __simd_member_type<_T>::generate([&gen](auto meta) {
            return meta.generator(
                [&](auto i_) {
                    return gen(__size_constant<meta.offset + decltype(i_)::value>());
                },
                __type_tag<_T>());
        });
    }

    // load {{{2
    template <class _T, class _U, class F>
    static inline __simd_member_type<_T> load(const _U *mem, F f,
                                              __type_tag<_T>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        return __simd_member_type<_T>::generate(
            [&](auto meta) { return meta.load(&mem[meta.offset], f, __type_tag<_T>()); });
    }

    // masked load {{{2
    template <class _T, class... As, class _U, class F>
    static inline __simd_tuple<_T, As...> masked_load(__simd_tuple<_T, As...> merge,
                                                   const __mask_member_type bits,
                                                   const _U *mem,
                                                   F f) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        __for_each(merge, [&](auto meta, auto &native) {
            native = meta.masked_load(native, meta.make_mask(bits), &mem[meta.offset], f);
        });
        return merge;
    }

    // store {{{2
    template <class _T, class _U, class F>
    static inline void store(const __simd_member_type<_T> v, _U *mem, F f,
                             __type_tag<_T>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        __for_each(v, [&](auto meta, auto native) {
            meta.store(native, &mem[meta.offset], f, __type_tag<_T>());
        });
    }

    // masked store {{{2
    template <class _T, class... As, class _U, class F>
    static inline void masked_store(const __simd_tuple<_T, As...> v, _U *mem, F f,
                                    const __mask_member_type bits) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        __for_each(v, [&](auto meta, auto native) {
            meta.masked_store(native, &mem[meta.offset], f, meta.make_mask(bits));
        });
    }

    // negation {{{2
    template <class _T, class... As>
    static inline __mask_member_type negate(__simd_tuple<_T, As...> x) noexcept
    {
        __mask_member_type bits = 0;
        __for_each(x, [&bits](auto meta, auto native) {
            bits |= meta.mask_to_shifted_ullong(meta.negate(native));
        });
        return bits;
    }

    // reductions {{{2
private:
    template <class _T, class... As, class _BinaryOperation, size_t... Counts,
              size_t... Begins>
    static inline _T reduce(const __simd_tuple<_T, As...> &tup,
                           const _BinaryOperation &__binary_op,
                           std::index_sequence<Counts...>, std::index_sequence<Begins...>)
    {
        // 1. reduce all tuple elements with equal ABI to a single element in the output
        // tuple
        const auto reduced_vec =
            __make_simd_tuple(__tree_reduction<Counts, Begins>()(tup, __binary_op)...);
        // 2. split and reduce until a scalar results
        return __vec_to_scalar_reduction(reduced_vec, __binary_op);
    }

public:
    template <class _T, class _BinaryOperation>
    static inline _T reduce(const simd<_T> &x, const _BinaryOperation &__binary_op)
    {
        using ranges = __n_abis_in_tuple<__simd_member_type<_T>>;
        return __fixed_size_simd_impl::reduce(x.d, __binary_op, typename ranges::__counts(),
                                               typename ranges::__begins());
    }

    // min, max, clamp {{{2
    template <class _T, class... As>
    static inline __simd_tuple<_T, As...> min(const __simd_tuple<_T, As...> a,
                                              const __simd_tuple<_T, As...> b)
    {
        return __simd_tuple_apply(
            [](auto __impl, auto aa, auto bb) { return __impl.min(aa, bb); }, a, b);
    }

    template <class _T, class... As>
    static inline __simd_tuple<_T, As...> max(const __simd_tuple<_T, As...> a,
                                              const __simd_tuple<_T, As...> b)
    {
        return __simd_tuple_apply(
            [](auto __impl, auto aa, auto bb) { return __impl.max(aa, bb); }, a, b);
    }

    // complement {{{2
    template <class _T, class... As>
    static inline __simd_tuple<_T, As...> complement(__simd_tuple<_T, As...> x) noexcept
    {
        return __simd_tuple_apply([](auto __impl, auto xx) { return __impl.complement(xx); },
                                x);
    }

    // unary minus {{{2
    template <class _T, class... As>
    static inline __simd_tuple<_T, As...> unary_minus(__simd_tuple<_T, As...> x) noexcept
    {
        return __simd_tuple_apply([](auto __impl, auto xx) { return __impl.unary_minus(xx); },
                                x);
    }

    // arithmetic operators {{{2

#define _GLIBCXX_SIMD_FIXED_OP(name_, op_)                                                          \
    template <class _T, class... As>                                                      \
    static inline __simd_tuple<_T, As...> name_(__simd_tuple<_T, As...> x,                     \
                                             __simd_tuple<_T, As...> y)                     \
    {                                                                                    \
        return __simd_tuple_apply(                                                         \
            [](auto __impl, auto xx, auto yy) { return __impl.name_(xx, yy); }, x, y);       \
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

    template <class _T, class... As>
    static inline __simd_tuple<_T, As...> bit_shift_left(__simd_tuple<_T, As...> x, int y)
    {
        return __simd_tuple_apply(
            [y](auto __impl, auto xx) { return __impl.bit_shift_left(xx, y); }, x);
    }

    template <class _T, class... As>
    static inline __simd_tuple<_T, As...> bit_shift_right(__simd_tuple<_T, As...> x,
                                                          int y)
    {
        return __simd_tuple_apply(
            [y](auto __impl, auto xx) { return __impl.bit_shift_right(xx, y); }, x);
    }

    // math {{{2
#define _GLIBCXX_SIMD_APPLY_ON_TUPLE_(name_)                                             \
    template <class _T, class... As>                                                     \
    static inline __simd_tuple<_T, As...> __##name_(__simd_tuple<_T, As...> x) noexcept  \
    {                                                                                    \
        return __simd_tuple_apply(                                                       \
            [](auto __impl, auto xx) {                                                   \
                using _V = typename decltype(__impl)::simd_type;                         \
                return __data(name_(_V(__private_init, xx)));                            \
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

    template <class _T, class... As>
    static inline __simd_tuple<_T, As...> __frexp(const __simd_tuple<_T, As...> &x,
                                             __fixed_size_storage<int, _N> &exp) noexcept
    {
        return __simd_tuple_apply(
            [](auto __impl, const auto &a, auto &b) {
                return __data(
                    __impl.__frexp(typename decltype(__impl)::simd_type(__private_init, a),
                                 __autocvt_to_simd(b)));
            },
            x, exp);
    }

    template <class _T, class... As>
    static inline __fixed_size_storage<int, _N> __fpclassify(__simd_tuple<_T, As...> x) noexcept
    {
        return __optimize_simd_tuple(x.template apply_r<int>(
            [](auto __impl, auto xx) { return __impl.__fpclassify(xx); }));
    }

#define _GLIBCXX_SIMD_TEST_ON_TUPLE_(name_)                                              \
    template <class _T, class... As>                                                     \
    static inline __mask_member_type __##name_(__simd_tuple<_T, As...> x) noexcept         \
    {                                                                                    \
        return test([](auto __impl, auto xx) { return __impl.__##name_(xx); }, x);           \
    }
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isinf)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isfinite)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isnan)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isnormal)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(signbit)
#undef _GLIBCXX_SIMD_TEST_ON_TUPLE_

    // __increment & __decrement{{{2
    template <class... _Ts> static inline void __increment(__simd_tuple<_Ts...> &x)
    {
        __for_each(x, [](auto meta, auto &native) { meta.__increment(native); });
    }

    template <class... _Ts> static inline void __decrement(__simd_tuple<_Ts...> &x)
    {
        __for_each(x, [](auto meta, auto &native) { meta.__decrement(native); });
    }

    // compares {{{2
#define _GLIBCXX_SIMD_CMP_OPERATIONS(cmp_)                                                          \
    template <class _T, class... As>                                                      \
    static inline __mask_member_type cmp_(const __simd_tuple<_T, As...> &x,                   \
                                        const __simd_tuple<_T, As...> &y)                   \
    {                                                                                    \
        __mask_member_type bits = 0;                                                       \
        __for_each(x, y, [&bits](auto meta, auto native_x, auto native_y) {        \
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
    template <class _T, class... As, class _U>
    _GLIBCXX_SIMD_INTRINSIC static void set(__simd_tuple<_T, As...> &v, int i, _U &&x) noexcept
    {
        v.set(i, std::forward<_U>(x));
    }

    // masked_assign {{{2
    template <typename _T, class... As>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(
        const __mask_member_type bits, __simd_tuple<_T, As...> &lhs,
        const __id<__simd_tuple<_T, As...>> rhs)
    {
        __for_each(lhs, rhs, [&](auto meta, auto &native_lhs, auto native_rhs) {
            meta.masked_assign(meta.make_mask(bits), native_lhs, native_rhs);
        });
    }

    // Optimization for the case where the RHS is a scalar. No need to broadcast the
    // scalar to a simd first.
    template <typename _T, class... As>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(const __mask_member_type bits,
                                           __simd_tuple<_T, As...> &lhs,
                                           const __id<_T> rhs)
    {
        __for_each(lhs, [&](auto meta, auto &native_lhs) {
            meta.masked_assign(meta.make_mask(bits), native_lhs, rhs);
        });
    }

    // __masked_cassign {{{2
    template <template <typename> class Op, typename _T, class... As>
    static inline void __masked_cassign(const __mask_member_type bits,
                                      __simd_tuple<_T, As...> &lhs,
                                      const __simd_tuple<_T, As...> rhs)
    {
        __for_each(lhs, rhs, [&](auto meta, auto &native_lhs, auto native_rhs) {
            meta.template __masked_cassign<Op>(meta.make_mask(bits), native_lhs,
                                             native_rhs);
        });
    }

    // Optimization for the case where the RHS is a scalar. No need to broadcast the
    // scalar to a simd
    // first.
    template <template <typename> class Op, typename _T, class... As>
    static inline void __masked_cassign(const __mask_member_type bits,
                                      __simd_tuple<_T, As...> &lhs, const _T rhs)
    {
        __for_each(lhs, [&](auto meta, auto &native_lhs) {
            meta.template __masked_cassign<Op>(meta.make_mask(bits), native_lhs, rhs);
        });
    }

    // masked_unary {{{2
    template <template <typename> class Op, class _T, class... As>
    static inline __simd_tuple<_T, As...> masked_unary(
        const __mask_member_type bits,
        const __simd_tuple<_T, As...> v)  // TODO: const-ref v?
    {
        return v.apply_wrapped([&bits](auto meta, auto native) {
            return meta.template masked_unary<Op>(meta.make_mask(bits), native);
        });
    }

    // }}}2
};

// __fixed_size_mask_impl {{{1
template <int _N> struct __fixed_size_mask_impl {
    static_assert(sizeof(__ullong) * CHAR_BIT >= _N,
                  "The fixed_size implementation relies on one "
                  "__ullong being able to store all boolean "
                  "elements.");  // required in load & store

    // member types {{{2
    static constexpr std::make_index_sequence<_N> index_seq = {};
    using __mask_member_type = std::bitset<_N>;
    template <class _T> using simd_mask = std::experimental::simd_mask<_T, simd_abi::fixed_size<_N>>;
    template <class _T> using __type_tag = _T *;

    // __from_bitset {{{2
    template <class _T>
    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type __from_bitset(const __mask_member_type &bs,
                                                     __type_tag<_T>) noexcept
    {
        return bs;
    }

    // load {{{2
    template <class F> static inline __mask_member_type load(const bool *mem, F f) noexcept
    {
        // TODO: __uchar is not necessarily the best type to use here. For smaller _N ushort,
        // uint, __ullong, float, and double can be more efficient.
        __ullong r = 0;
        using Vs = __fixed_size_storage<__uchar, _N>;
        __for_each(Vs{}, [&](auto meta, auto) {
            r |= meta.mask_to_shifted_ullong(
                meta.simd_mask.load(&mem[meta.offset], f, __size_constant<meta.size()>()));
        });
        return r;
    }

    // masked load {{{2
    template <class F>
    static inline __mask_member_type masked_load(__mask_member_type merge,
                                               __mask_member_type mask, const bool *mem,
                                               F) noexcept
    {
        __bit_iteration(mask.to_ullong(), [&](auto i) { merge[i] = mem[i]; });
        return merge;
    }

    // store {{{2
    template <class F>
    static inline void store(__mask_member_type bs, bool *mem, F f) noexcept
    {
#if _GLIBCXX_SIMD_HAVE_AVX512BW
        const __m512i bool64 = _mm512_movm_epi8(bs.to_ullong()) & 0x0101010101010101ULL;
        __vector_store<_N>(bool64, mem, f);
#elif _GLIBCXX_SIMD_HAVE_BMI2
#ifdef __x86_64__
        __unused(f);
        __execute_n_times<_N / 8>([&](auto i) {
            constexpr size_t offset = i * 8;
            const __ullong bool8 =
                _pdep_u64(bs.to_ullong() >> offset, 0x0101010101010101ULL);
            std::memcpy(&mem[offset], &bool8, 8);
        });
        if (_N % 8 > 0) {
            constexpr size_t offset = (_N / 8) * 8;
            const __ullong bool8 =
                _pdep_u64(bs.to_ullong() >> offset, 0x0101010101010101ULL);
            std::memcpy(&mem[offset], &bool8, _N % 8);
        }
#else   // __x86_64__
        __unused(f);
        __execute_n_times<_N / 4>([&](auto i) {
            constexpr size_t offset = i * 4;
            const __ullong bool4 =
                _pdep_u32(bs.to_ullong() >> offset, 0x01010101U);
            std::memcpy(&mem[offset], &bool4, 4);
        });
        if (_N % 4 > 0) {
            constexpr size_t offset = (_N / 4) * 4;
            const __ullong bool4 =
                _pdep_u32(bs.to_ullong() >> offset, 0x01010101U);
            std::memcpy(&mem[offset], &bool4, _N % 4);
        }
#endif  // __x86_64__
#elif  _GLIBCXX_SIMD_HAVE_SSE2   // !AVX512BW && !BMI2
        using _V = simd<__uchar, simd_abi::__sse>;
        __ullong bits = bs.to_ullong();
        __execute_n_times<(_N + 15) / 16>([&](auto i) {
            constexpr size_t offset = i * 16;
            constexpr size_t remaining = _N - offset;
            if constexpr (remaining == 1) {
                mem[offset] = static_cast<bool>(bits >> offset);
            } else if constexpr (remaining <= 4) {
                const uint bool4 = ((bits >> offset) * 0x00204081U) & 0x01010101U;
                std::memcpy(&mem[offset], &bool4, remaining);
            } else if constexpr (remaining <= 7) {
                const __ullong bool8 =
                    ((bits >> offset) * 0x40810204081ULL) & 0x0101010101010101ULL;
                std::memcpy(&mem[offset], &bool8, remaining);
            } else if constexpr (__have_sse2) {
                auto tmp = _mm_cvtsi32_si128(bits >> offset);
                tmp = _mm_unpacklo_epi8(tmp, tmp);
                tmp = _mm_unpacklo_epi16(tmp, tmp);
                tmp = _mm_unpacklo_epi32(tmp, tmp);
                _V tmp2(tmp);
                tmp2 &= _V([](auto j) {
                    return static_cast<__uchar>(1 << (j % CHAR_BIT));
                });  // mask bit index
                const __m128i bool16 =
                    _mm_add_epi8(__data(tmp2 == 0),
                                 _mm_set1_epi8(1));  // 0xff -> 0x00 | 0x00 -> 0x01
                if constexpr (remaining >= 16) {
                    __vector_store<16>(bool16, &mem[offset], f);
                } else if constexpr (remaining & 3) {
                    constexpr int to_shift = 16 - int(remaining);
                    _mm_maskmoveu_si128(bool16,
                                        _mm_srli_si128(__allbits<__m128i>, to_shift),
                                        reinterpret_cast<char *>(&mem[offset]));
                } else  // at this point: 8 < remaining < 16
                    if constexpr (remaining >= 8) {
                    __vector_store<8>(bool16, &mem[offset], f);
                    if constexpr (remaining == 12) {
                        __vector_store<4>(_mm_unpackhi_epi64(bool16, bool16),
                                         &mem[offset + 8], f);
                    }
                }
            } else {
                __assert_unreachable<F>();
            }
        });
#else
        // TODO: __uchar is not necessarily the best type to use here. For smaller _N ushort,
        // uint, __ullong, float, and double can be more efficient.
        using Vs = __fixed_size_storage<__uchar, _N>;
        __for_each(Vs{}, [&](auto meta, auto) {
            meta.store(meta.make_mask(bs), &mem[meta.offset], f);
        });
//#else
        //__execute_n_times<_N>([&](auto i) { mem[i] = bs[i]; });
#endif  // _GLIBCXX_SIMD_HAVE_BMI2
    }

    // masked store {{{2
    template <class F>
    static inline void masked_store(const __mask_member_type v, bool *mem, F,
                                    const __mask_member_type k) noexcept
    {
        __bit_iteration(k, [&](auto i) { mem[i] = v[i]; });
    }

    // logical and bitwise operators {{{2
    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type logical_and(const __mask_member_type &x,
                                                     const __mask_member_type &y) noexcept
    {
        return x & y;
    }

    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type logical_or(const __mask_member_type &x,
                                                    const __mask_member_type &y) noexcept
    {
        return x | y;
    }

    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type bit_and(const __mask_member_type &x,
                                                 const __mask_member_type &y) noexcept
    {
        return x & y;
    }

    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type bit_or(const __mask_member_type &x,
                                                const __mask_member_type &y) noexcept
    {
        return x | y;
    }

    _GLIBCXX_SIMD_INTRINSIC static __mask_member_type bit_xor(const __mask_member_type &x,
                                                 const __mask_member_type &y) noexcept
    {
        return x ^ y;
    }

    // smart_reference access {{{2
    _GLIBCXX_SIMD_INTRINSIC static void set(__mask_member_type &k, int i, bool x) noexcept
    {
        k.set(i, x);
    }

    // masked_assign {{{2
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(const __mask_member_type k,
                                           __mask_member_type &lhs,
                                           const __mask_member_type rhs)
    {
        lhs = (lhs & ~k) | (rhs & k);
    }

    // Optimization for the case where the RHS is a scalar.
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(const __mask_member_type k,
                                           __mask_member_type &lhs, const bool rhs)
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

// __simd_converter scalar -> scalar {{{
template <class _T> struct __simd_converter<_T, simd_abi::scalar, _T, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC _T operator()(_T a) { return a; }
};
template <class From, class To>
struct __simd_converter<From, simd_abi::scalar, To, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC To operator()(From a)
    {
        return static_cast<To>(a);
    }
};

// }}}
// __simd_converter __sse -> scalar {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::__sse, To, simd_abi::scalar> {
    using Arg = __sse_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC std::array<To, Arg::width> all(Arg a)
    {
        return __impl(std::make_index_sequence<Arg::width>(), a);
    }

    template <size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC std::array<To, Arg::width> __impl(std::index_sequence<_Indexes...>, Arg a)
    {
        return {static_cast<To>(a[_Indexes])...};
    }
};

// }}}1
// __simd_converter scalar -> __sse {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::scalar, To, simd_abi::__sse> {
    using R = __sse_simd_member_type<To>;
    template <class... More> _GLIBCXX_SIMD_INTRINSIC constexpr R operator()(From a, More... b)
    {
        static_assert(sizeof...(More) + 1 == R::width);
        static_assert(std::conjunction_v<std::is_same<From, More>...>);
        return __vector_type16_t<To>{static_cast<To>(a), static_cast<To>(b)...};
    }
};

// }}}1
// __simd_converter __sse -> __sse {{{1
template <class _T> struct __simd_converter<_T, simd_abi::__sse, _T, simd_abi::__sse> {
    using Arg = __sse_simd_member_type<_T>;
    _GLIBCXX_SIMD_INTRINSIC const Arg &operator()(const Arg &x) { return x; }
};

template <class From, class To>
struct __simd_converter<From, simd_abi::__sse, To, simd_abi::__sse> {
    using Arg = __sse_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a)
    {
        return __x86::convert_all<__vector_type16_t<To>>(a);
    }

    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<To> operator()(Arg a)
    {
        return __x86::convert<__vector_type16_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return __x86::convert<__sse_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return __x86::convert<__sse_simd_member_type<To>>(a, b, c, d);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d, Arg e,
                                                     Arg f, Arg g, Arg h)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return __x86::convert<__sse_simd_member_type<To>>(a, b, c, d, e, f, g, h);
    }
};

// }}}1
// __simd_converter __avx -> scalar {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::__avx, To, simd_abi::scalar> {
    using Arg = __avx_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC std::array<To, Arg::width> all(Arg a)
    {
        return __impl(std::make_index_sequence<Arg::width>(), a);
    }

    template <size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC std::array<To, Arg::width> __impl(std::index_sequence<_Indexes...>, Arg a)
    {
        return {static_cast<To>(a[_Indexes])...};
    }
};

// }}}1
// __simd_converter scalar -> __avx {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::scalar, To, simd_abi::__avx> {
    using R = __avx_simd_member_type<To>;
    template <class... More> _GLIBCXX_SIMD_INTRINSIC constexpr R operator()(From a, More... b)
    {
        static_assert(sizeof...(More) + 1 == R::width);
        static_assert(std::conjunction_v<std::is_same<From, More>...>);
        return __vector_type32_t<To>{static_cast<To>(a), static_cast<To>(b)...};
    }
};

// }}}1
// __simd_converter __sse -> __avx {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::__sse, To, simd_abi::__avx> {
    using Arg = __sse_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a) { return __x86::convert_all<__vector_type32_t<To>>(a); }

    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<To> operator()(Arg a)
    {
        return __x86::convert<__vector_type32_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 1 * sizeof(To), "");
        return __x86::convert<__avx_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return __x86::convert<__avx_simd_member_type<To>>(a, b, c, d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return __x86::convert<__avx_simd_member_type<To>>(x0, x1, x2, x3, x4, x5, x6, x7);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7,
                                                     Arg x8, Arg x9, Arg x10, Arg x11,
                                                     Arg x12, Arg x13, Arg x14, Arg x15)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return __x86::convert<__avx_simd_member_type<To>>(x0, x1, x2, x3, x4, x5, x6, x7, x8,
                                                      x9, x10, x11, x12, x13, x14, x15);
    }
};

// }}}1
// __simd_converter __avx -> __sse {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::__avx, To, simd_abi::__sse> {
    using Arg = __avx_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a) { return __x86::convert_all<__vector_type16_t<To>>(a); }

    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<To> operator()(Arg a)
    {
        return __x86::convert<__vector_type16_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return __x86::convert<__sse_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return __x86::convert<__sse_simd_member_type<To>>(a, b, c, d);
    }
};

// }}}1
// __simd_converter __avx -> __avx {{{1
template <class _T> struct __simd_converter<_T, simd_abi::__avx, _T, simd_abi::__avx> {
    using Arg = __avx_simd_member_type<_T>;
    _GLIBCXX_SIMD_INTRINSIC const Arg &operator()(const Arg &x) { return x; }
};

template <class From, class To>
struct __simd_converter<From, simd_abi::__avx, To, simd_abi::__avx> {
    using Arg = __avx_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a) { return __x86::convert_all<__vector_type32_t<To>>(a); }

    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<To> operator()(Arg a)
    {
        return __x86::convert<__vector_type32_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return __x86::convert<__avx_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return __x86::convert<__avx_simd_member_type<To>>(a, b, c, d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d, Arg e,
                                                     Arg f, Arg g, Arg h)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return __x86::convert<__avx_simd_member_type<To>>(a, b, c, d, e, f, g, h);
    }
};

// }}}1
// __simd_converter __avx512 -> scalar {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::__avx512, To, simd_abi::scalar> {
    using Arg = __avx512_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC std::array<To, Arg::width> all(Arg a)
    {
        return __impl(std::make_index_sequence<Arg::width>(), a);
    }

    template <size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC std::array<To, Arg::width> __impl(std::index_sequence<_Indexes...>, Arg a)
    {
        return {static_cast<To>(a[_Indexes])...};
    }
};

// }}}1
// __simd_converter scalar -> __avx512 {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::scalar, To, simd_abi::__avx512> {
    using R = __avx512_simd_member_type<To>;

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
// __simd_converter __sse -> __avx512 {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::__sse, To, simd_abi::__avx512> {
    using Arg = __sse_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a)
    {
        return __x86::convert_all<__vector_type64_t<To>>(a);
    }

    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg a)
    {
        return __x86::convert<__vector_type64_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(2 * sizeof(From) >= sizeof(To), "");
        return __x86::convert<__avx512_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 1 * sizeof(To), "");
        return __x86::convert<__avx512_simd_member_type<To>>(a, b, c, d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return __x86::convert<__avx512_simd_member_type<To>>(x0, x1, x2, x3, x4, x5, x6,
                                                           x7);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7,
                                                     Arg x8, Arg x9, Arg x10, Arg x11,
                                                     Arg x12, Arg x13, Arg x14, Arg x15)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return __x86::convert<__avx512_simd_member_type<To>>(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(
        Arg x0, Arg x1, Arg x2, Arg x3, Arg x4, Arg x5, Arg x6, Arg x7, Arg x8, Arg x9,
        Arg x10, Arg x11, Arg x12, Arg x13, Arg x14, Arg x15, Arg x16, Arg x17, Arg x18,
        Arg x19, Arg x20, Arg x21, Arg x22, Arg x23, Arg x24, Arg x25, Arg x26, Arg x27,
        Arg x28, Arg x29, Arg x30, Arg x31)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return __x86::convert<__avx512_simd_member_type<To>>(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16,
            x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31);
    }
};

// }}}1
// __simd_converter __avx512 -> __sse {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::__avx512, To, simd_abi::__sse> {
    using Arg = __avx512_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a)
    {
        return __x86::convert_all<__vector_type16_t<To>>(a);
    }

    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<To> operator()(Arg a)
    {
        return __x86::convert<__vector_type16_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return __x86::convert<__sse_simd_member_type<To>>(a, b);
    }
};

// }}}1
// __simd_converter __avx -> __avx512 {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::__avx, To, simd_abi::__avx512> {
    using Arg = __avx_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a)
    {
        return __x86::convert_all<__vector_type64_t<To>>(a);
    }

    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg a)
    {
        return __x86::convert<__vector_type64_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 1 * sizeof(To), "");
        return __x86::convert<__avx512_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return __x86::convert<__avx512_simd_member_type<To>>(a, b, c, d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return __x86::convert<__avx512_simd_member_type<To>>(x0, x1, x2, x3, x4, x5, x6,
                                                           x7);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg x0, Arg x1, Arg x2, Arg x3,
                                                     Arg x4, Arg x5, Arg x6, Arg x7,
                                                     Arg x8, Arg x9, Arg x10, Arg x11,
                                                     Arg x12, Arg x13, Arg x14, Arg x15)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return __x86::convert<__avx512_simd_member_type<To>>(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
    }
};

// }}}1
// __simd_converter __avx512 -> __avx {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::__avx512, To, simd_abi::__avx> {
    using Arg = __avx512_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a)
    {
        return __x86::convert_all<__vector_type32_t<To>>(a);
    }

    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<To> operator()(Arg a)
    {
        return __x86::convert<__vector_type32_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return __x86::convert<__avx_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return __x86::convert<__avx_simd_member_type<To>>(a, b, c, d);
    }
};

// }}}1
// __simd_converter __avx512 -> __avx512 {{{1
template <class _T> struct __simd_converter<_T, simd_abi::__avx512, _T, simd_abi::__avx512> {
    using Arg = __avx512_simd_member_type<_T>;
    _GLIBCXX_SIMD_INTRINSIC const Arg &operator()(const Arg &x) { return x; }
};

template <class From, class To>
struct __simd_converter<From, simd_abi::__avx512, To, simd_abi::__avx512> {
    using Arg = __avx512_simd_member_type<From>;

    _GLIBCXX_SIMD_INTRINSIC auto all(Arg a)
    {
        return __x86::convert_all<__vector_type64_t<To>>(a);
    }

    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg a)
    {
        return __x86::convert<__vector_type64_t<To>>(a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg a, Arg b)
    {
        static_assert(sizeof(From) >= 2 * sizeof(To), "");
        return __x86::convert<__avx512_simd_member_type<To>>(a, b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d)
    {
        static_assert(sizeof(From) >= 4 * sizeof(To), "");
        return __x86::convert<__avx512_simd_member_type<To>>(a, b, c, d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<To> operator()(Arg a, Arg b, Arg c, Arg d, Arg e,
                                                     Arg f, Arg g, Arg h)
    {
        static_assert(sizeof(From) >= 8 * sizeof(To), "");
        return __x86::convert<__avx512_simd_member_type<To>>(a, b, c, d, e, f, g, h);
    }
};

// }}}1
// __simd_converter scalar -> fixed_size<1> {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::scalar, To, simd_abi::fixed_size<1>> {
    __simd_tuple<To, simd_abi::scalar> operator()(From x) { return {static_cast<To>(x)}; }
};

// __simd_converter fixed_size<1> -> scalar {{{1
template <class From, class To>
struct __simd_converter<From, simd_abi::fixed_size<1>, To, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC To operator()(__simd_tuple<From, simd_abi::scalar> x)
    {
        return {static_cast<To>(x.first)};
    }
};

// __simd_converter fixed_size<_N> -> fixed_size<_N> {{{1
template <class _T, int _N>
struct __simd_converter<_T, simd_abi::fixed_size<_N>, _T, simd_abi::fixed_size<_N>> {
    using arg = __fixed_size_storage<_T, _N>;
    _GLIBCXX_SIMD_INTRINSIC const arg &operator()(const arg &x) { return x; }
};

template <size_t ChunkSize, class _T> struct determine_required_input_chunks;

template <class _T, class... _Abis>
struct determine_required_input_chunks<0, __simd_tuple<_T, _Abis...>>
    : public std::integral_constant<size_t, 0> {
};

template <size_t ChunkSize, class _T, class Abi0, class... _Abis>
struct determine_required_input_chunks<ChunkSize, __simd_tuple<_T, Abi0, _Abis...>>
    : public std::integral_constant<
          size_t, determine_required_input_chunks<ChunkSize - simd_size_v<_T, Abi0>,
                                                  __simd_tuple<_T, _Abis...>>::value> {
};

template <class From, class To> struct fixed_size_converter {
    struct OneToMultipleChunks {
    };
    template <int _N> struct MultipleToOneChunk {
    };
    struct EqualChunks {
    };
    template <class FromAbi, class ToAbi, size_t _ToSize = simd_size_v<To, ToAbi>,
              size_t FromSize = simd_size_v<From, FromAbi>>
    using ChunkRelation = std::conditional_t<
        (_ToSize < FromSize), OneToMultipleChunks,
        std::conditional_t<(_ToSize == FromSize), EqualChunks,
                           MultipleToOneChunk<int(_ToSize / FromSize)>>>;

    template <class... _Abis>
    using __return_type = __fixed_size_storage<To, __simd_tuple<From, _Abis...>::size()>;


protected:
    // OneToMultipleChunks {{{2
    template <class _A0>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0> __impl(OneToMultipleChunks, const __simd_tuple<From, _A0> &x)
    {
        using R = __return_type<_A0>;
        __simd_converter<From, _A0, To, typename R::__first_abi> native_cvt;
        auto &&multiple_return_chunks = native_cvt.all(x.first);
        return __to_simd_tuple<To, typename R::__first_abi>(multiple_return_chunks);
    }

    template <class... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_Abis...> __impl(OneToMultipleChunks,
                                           const __simd_tuple<From, _Abis...> &x)
    {
        using R = __return_type<_Abis...>;
        using arg = __simd_tuple<From, _Abis...>;
        constexpr size_t first_chunk = simd_size_v<From, typename arg::__first_abi>;
        __simd_converter<From, typename arg::__first_abi, To, typename R::__first_abi>
            native_cvt;
        auto &&multiple_return_chunks = native_cvt.all(x.first);
        constexpr size_t n_output_chunks =
            first_chunk / simd_size_v<To, typename R::__first_abi>;
        return __simd_tuple_concat(
            __to_simd_tuple<To, typename R::__first_abi>(multiple_return_chunks),
            __impl(ChunkRelation<typename arg::__second_type::__first_abi,
                               typename __simd_tuple_element<n_output_chunks, R>::type::abi_type>(),
                 x.second));
    }

    // MultipleToOneChunk {{{2
    template <int _N, class _A0, class... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0, _Abis...> __impl(MultipleToOneChunk<_N>,
                                               const __simd_tuple<From, _A0, _Abis...> &x)
    {
        return impl_mto(std::integral_constant<bool, sizeof...(_Abis) + 1 == _N>(),
                        std::make_index_sequence<_N>(), x);
    }

    template <size_t... _Indexes, class _A0, class... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0, _Abis...> impl_mto(true_type,
                                                   std::index_sequence<_Indexes...>,
                                                   const __simd_tuple<From, _A0, _Abis...> &x)
    {
        using R = __return_type<_A0, _Abis...>;
        __simd_converter<From, _A0, To, typename R::__first_abi> native_cvt;
        return {native_cvt(__get_tuple_at<_Indexes>(x)...)};
    }

    template <size_t... _Indexes, class _A0, class... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0, _Abis...> impl_mto(false_type,
                                                   std::index_sequence<_Indexes...>,
                                                   const __simd_tuple<From, _A0, _Abis...> &x)
    {
        using R = __return_type<_A0, _Abis...>;
        __simd_converter<From, _A0, To, typename R::__first_abi> native_cvt;
        return {
            native_cvt(__get_tuple_at<_Indexes>(x)...),
            __impl(
                ChunkRelation<
                    typename __simd_tuple_element<sizeof...(_Indexes),
                                           __simd_tuple<From, _A0, _Abis...>>::type::abi_type,
                    typename R::__second_type::__first_abi>(),
                __simd_tuple_pop_front(__size_constant<sizeof...(_Indexes)>(), x))};
    }

    // EqualChunks {{{2
    template <class _A0>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0> __impl(EqualChunks, const __simd_tuple<From, _A0> &x)
    {
        __simd_converter<From, _A0, To, typename __return_type<_A0>::__first_abi> native_cvt;
        return {native_cvt(x.first)};
    }

    template <class _A0, class _A1, class... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0, _A1, _Abis...> __impl(
        EqualChunks, const __simd_tuple<From, _A0, _A1, _Abis...> &x)
    {
        using R = __return_type<_A0, _A1, _Abis...>;
        using Rem = typename R::__second_type;
        __simd_converter<From, _A0, To, typename R::__first_abi> native_cvt;
        return {native_cvt(x.first),
                __impl(ChunkRelation<_A1, typename Rem::__first_abi>(), x.second)};
    }

    //}}}2
};

template <class From, class To, int _N>
struct __simd_converter<From, simd_abi::fixed_size<_N>, To, simd_abi::fixed_size<_N>>
    : public fixed_size_converter<From, To> {
    using base = fixed_size_converter<From, To>;
    using __return_type = __fixed_size_storage<To, _N>;
    using arg = __fixed_size_storage<From, _N>;

    _GLIBCXX_SIMD_INTRINSIC __return_type operator()(const arg &x)
    {
        using CR = typename base::template ChunkRelation<typename arg::__first_abi,
                                                         typename __return_type::__first_abi>;
        return base::__impl(CR(), x);
    }
};

// __simd_converter "native" -> fixed_size<_N> {{{1
// i.e. 1 register to ? registers
template <class From, class _A, class To, int _N>
struct __simd_converter<From, _A, To, simd_abi::fixed_size<_N>> {
    using __traits = __simd_traits<From, _A>;
    using arg = typename __traits::__simd_member_type;
    using __return_type = __fixed_size_storage<To, _N>;
    static_assert(_N == simd_size_v<From, _A>,
                  "__simd_converter to fixed_size only works for equal element counts");

    _GLIBCXX_SIMD_INTRINSIC __return_type operator()(arg x)
    {
        return __impl(std::make_index_sequence<__return_type::tuple_size>(), x);
    }

private:
    __return_type __impl(std::index_sequence<0>, arg x)
    {
        __simd_converter<From, _A, To, typename __return_type::__first_abi> native_cvt;
        return {native_cvt(x)};
    }
    template <size_t... _Indexes> __return_type __impl(std::index_sequence<_Indexes...>, arg x)
    {
        __simd_converter<From, _A, To, typename __return_type::__first_abi> native_cvt;
        const auto &tmp = native_cvt.all(x);
        return {tmp[_Indexes]...};
    }
};

// __simd_converter fixed_size<_N> -> "native" {{{1
// i.e. ? register to 1 registers
template <class From, int _N, class To, class _A>
struct __simd_converter<From, simd_abi::fixed_size<_N>, To, _A> {
    using __traits = __simd_traits<To, _A>;
    using __return_type = typename __traits::__simd_member_type;
    using arg = __fixed_size_storage<From, _N>;
    static_assert(_N == simd_size_v<To, _A>,
                  "__simd_converter to fixed_size only works for equal element counts");

    _GLIBCXX_SIMD_INTRINSIC __return_type operator()(arg x)
    {
        return __impl(std::make_index_sequence<arg::tuple_size>(), x);
    }

private:
    template <size_t... _Indexes> __return_type __impl(std::index_sequence<_Indexes...>, arg x)
    {
        __simd_converter<From, typename arg::__first_abi, To, _A> native_cvt;
        return native_cvt(__get_tuple_at<_Indexes>(x)...);
    }
};

// }}}1
// __generic_simd_impl::__masked_cassign specializations {{{1
#if _GLIBCXX_SIMD_HAVE_AVX512_ABI
#define _GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(_TYPE, _TYPE_SUFFIX, _OP, _OP_NAME)  \
    template <>                                                                          \
    template <>                                                                          \
    _GLIBCXX_SIMD_INTRINSIC void                                                         \
    __generic_simd_impl<simd_abi::__avx512>::__masked_cassign<_OP, _TYPE, bool,          \
                                                              64 / sizeof(_TYPE)>(       \
        const __storage<bool, 64 / sizeof(_TYPE)> k,                                     \
        __storage<_TYPE, 64 / sizeof(_TYPE)> &lhs,                                       \
        const __id<__storage<_TYPE, 64 / sizeof(_TYPE)>> rhs)                            \
    {                                                                                    \
        lhs = _mm512_mask_##_OP_NAME##_##_TYPE_SUFFIX(lhs, k, lhs, rhs);                 \
    }                                                                                    \
    _GLIBCXX_SIMD_NOTHING_EXPECTING_SEMICOLON

_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(double, pd, std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(float, ps, std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(__llong, epi64, std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(__ullong, epi64, std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(long, epi64, std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(__ulong, epi64, std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(int, epi32, std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(__uint, epi32, std::plus, add);
#if _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(   short, epi16, std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(__ushort, epi16, std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION( __schar, epi8 , std::plus, add);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION( __uchar, epi8 , std::plus, add);
#endif  // _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI

_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(  double,  pd  , std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(   float,  ps  , std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION( __llong, epi64, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(__ullong, epi64, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(    long, epi64, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION( __ulong, epi64, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(     int, epi32, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(  __uint, epi32, std::minus, sub);
#if _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(   short, epi16, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION(__ushort, epi16, std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION( __schar, epi8 , std::minus, sub);
_GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION( __uchar, epi8 , std::minus, sub);
#endif  // _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
#undef _GLIBCXX_SIMD_MASKED_CASSIGN_SPECIALIZATION
#endif  // _GLIBCXX_SIMD_HAVE_AVX512_ABI

// }}}1
_GLIBCXX_SIMD_END_NAMESPACE
#endif  // __cplusplus >= 201703L
#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_ABIS_H_
// vim: foldmethod=marker
