#ifndef _GLIBCXX_EXPERIMENTAL_SIMD_ABIS_H_
#define _GLIBCXX_EXPERIMENTAL_SIMD_ABIS_H_

//#pragma GCC system_header

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
typename _T::value_type __subscript_read(const _T &__x, size_t __i) noexcept
{
    return __x[__i];
}
template <class _T>
void __subscript_write(_T &__x, size_t __i, typename _T::value_type __y) noexcept
{
    return __x.set(__i, __y);
}

// __simd_tuple_element {{{1
template <size_t _I, class _T> struct __simd_tuple_element;
template <class _T, class _A0, class... As>
struct __simd_tuple_element<0, __simd_tuple<_T, _A0, As...>> {
    using type = std::experimental::simd<_T, _A0>;
};
template <size_t _I, class _T, class _A0, class... As>
struct __simd_tuple_element<_I, __simd_tuple<_T, _A0, As...>> {
    using type = typename __simd_tuple_element<_I - 1, __simd_tuple<_T, As...>>::type;
};
template <size_t _I, class _T>
using __simd_tuple_element_t = typename __simd_tuple_element<_I, _T>::type;

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
                                                                   const _T &__x)
{
    return __x;
}
template <class _T>
_GLIBCXX_SIMD_INTRINSIC constexpr _T &__simd_tuple_pop_front(__size_constant<0>, _T &__x)
{
    return __x;
}
template <size_t _K, class _T>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto &__simd_tuple_pop_front(__size_constant<_K>,
                                                                     const _T &__x)
{
    return __simd_tuple_pop_front(__size_constant<_K - 1>(), __x.second);
}
template <size_t _K, class _T>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__simd_tuple_pop_front(__size_constant<_K>, _T &__x)
{
    return __simd_tuple_pop_front(__size_constant<_K - 1>(), __x.second);
}

// __get_simd_at<_N> {{{1
struct __as_simd {};
struct __as_simd_tuple {};
template <class _T, class _A0, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr simd<_T, _A0> __simd_tuple_get_impl(
    __as_simd, const __simd_tuple<_T, _A0, _Abis...> &__t, __size_constant<0>)
{
    return {__private_init, __t.first};
}
template <class _T, class _A0, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto &__simd_tuple_get_impl(
    __as_simd_tuple, const __simd_tuple<_T, _A0, _Abis...> &__t, __size_constant<0>)
{
    return __t.first;
}
template <class _T, class _A0, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__simd_tuple_get_impl(
    __as_simd_tuple, __simd_tuple<_T, _A0, _Abis...> &__t, __size_constant<0>)
{
    return __t.first;
}

template <class _R, size_t _N, class _T, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __simd_tuple_get_impl(
    _R, const __simd_tuple<_T, _Abis...> &__t, __size_constant<_N>)
{
    return __simd_tuple_get_impl(_R(), __t.second, __size_constant<_N - 1>());
}
template <size_t _N, class _T, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__simd_tuple_get_impl(
    __as_simd_tuple, __simd_tuple<_T, _Abis...> &__t, __size_constant<_N>)
{
    return __simd_tuple_get_impl(__as_simd_tuple(), __t.second, __size_constant<_N - 1>());
}

template <size_t _N, class _T, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __get_simd_at(const __simd_tuple<_T, _Abis...> &__t)
{
    return __simd_tuple_get_impl(__as_simd(), __t, __size_constant<_N>());
}

// }}}
// __get_tuple_at<_N> {{{
template <size_t _N, class _T, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __get_tuple_at(const __simd_tuple<_T, _Abis...> &__t)
{
    return __simd_tuple_get_impl(__as_simd_tuple(), __t, __size_constant<_N>());
}

template <size_t _N, class _T, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__get_tuple_at(__simd_tuple<_T, _Abis...> &__t)
{
    return __simd_tuple_get_impl(__as_simd_tuple(), __t, __size_constant<_N>());
}

// __how_many_to_extract {{{1
template <size_t LeftN, class _RightT> constexpr size_t __tuple_elements_for () {
    if constexpr (LeftN == 0) {
        return 0;
    } else {
        return 1 + __tuple_elements_for<LeftN - _RightT::_S_first_size,
                                        typename _RightT::_Second_type>();
    }
}
template <size_t LeftN, class _RightT, bool = (_RightT::_S_first_size < LeftN)>
struct __how_many_to_extract;
template <size_t LeftN, class _RightT> struct __how_many_to_extract<LeftN, _RightT, true> {
    static constexpr std::make_index_sequence<__tuple_elements_for<LeftN, _RightT>()> tag()
    {
        return {};
    }
};
template <class _T, size_t _Offset, size_t _Length, bool _Done, class _IndexSeq>
struct chunked {
};
template <size_t LeftN, class _RightT> struct __how_many_to_extract<LeftN, _RightT, false> {
    static_assert(LeftN != _RightT::_S_first_size, "");
    static constexpr chunked<typename _RightT::_First_type, 0, LeftN, false,
                             std::make_index_sequence<LeftN>>
    tag()
    {
        return {};
    }
};

// __tuple_element_meta {{{1
template <class _T, class _Abi, size_t _Offset>
struct __tuple_element_meta : public _Abi::_Simd_impl_type {
    using value_type = _T;
    using abi_type = _Abi;
    using __traits = __simd_traits<_T, _Abi>;
    using maskimpl = typename __traits::_Mask_impl_type;
    using __member_type = typename __traits::_Simd_member_type;
    using _Mask_member_type = typename __traits::_Mask_member_type;
    using simd_type = std::experimental::simd<_T, _Abi>;
    static constexpr size_t offset = _Offset;
    static constexpr size_t size() { return simd_size<_T, _Abi>::value; }
    static constexpr maskimpl simd_mask = {};

    template <size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type make_mask(std::bitset<_N> __bits)
    {
        constexpr _T *__type_tag = nullptr;
        return maskimpl::__from_bitset(std::bitset<size()>((__bits >> _Offset).to_ullong()),
                                     __type_tag);
    }

    _GLIBCXX_SIMD_INTRINSIC static __ullong mask_to_shifted_ullong(_Mask_member_type __k)
    {
        return __vector_to_bitset(__k).to_ullong() << _Offset;
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
template <class _T, class _Abi0> struct __simd_tuple<_T, _Abi0> {
    using value_type = _T;
    using _First_type = typename __simd_traits<_T, _Abi0>::_Simd_member_type;
    using _Second_type = __simd_tuple<_T>;
    using _First_abi = _Abi0;
    static constexpr size_t tuple_size = 1;
    static constexpr size_t size() { return simd_size_v<_T, _Abi0>; }
    static constexpr size_t _S_first_size = simd_size_v<_T, _Abi0>;
    alignas(sizeof(_First_type)) _First_type first;
    static constexpr _Second_type second = {};

    template <size_t _Offset = 0, class _F>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple generate(_F &&gen, __size_constant<_Offset> = {})
    {
        return {gen(__tuple_element_meta<_T, _Abi0, _Offset>())};
    }

    template <size_t _Offset = 0, class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC __simd_tuple apply_wrapped(_F &&fun, const _More &... more) const
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return {fun(make_meta<_Offset>(*this), first, more.first...)};
    }

    template <class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC friend __simd_tuple __simd_tuple_apply(_F &&fun, const __simd_tuple &__x,
                                                    _More &&... more)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return __simd_tuple::apply_impl(
            __bool_constant<conjunction<__is_equal<
                size_t, _S_first_size, std::decay_t<_More>::_S_first_size>...>::value>(),
            std::forward<_F>(fun), __x, std::forward<_More>(more)...);
    }

private:
    template <class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl(true_type,  // _S_first_size is equal for all arguments
               _F &&fun, const __simd_tuple &__x, _More &&... more)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("more.first = ", more.first..., "more = ", more...);
        return {fun(__tuple_element_meta<_T, _Abi0, 0>(), __x.first, more.first...)};
    }

    template <class _F, class _More>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple apply_impl(false_type,  // at least one argument in
                                                           // _More has different
                                                           // _S_first_size, __x has only one
                                                           // member, so _More has 2 or
                                                           // more
                                              _F &&fun, const __simd_tuple &__x, _More &&__y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("__y = ", __y);
        return apply_impl(std::make_index_sequence<std::decay_t<_More>::tuple_size>(),
                          std::forward<_F>(fun), __x, std::forward<_More>(__y));
    }

    template <class _F, class _More, size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple apply_impl(std::index_sequence<_Indexes...>, _F &&fun,
                                              const __simd_tuple &__x, _More &&__y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("__y = ", __y);
        auto tmp = std::experimental::concat(__get_simd_at<_Indexes>(__y)...);
        const auto first = fun(__tuple_element_meta<_T, _Abi0, 0>(), __x.first, tmp);
        if constexpr (std::is_lvalue_reference<_More>::value &&
                      !std::is_const<_More>::value) {
            // if __y is non-const lvalue ref, assume write back is necessary
            const auto tup =
                std::experimental::split<__simd_tuple_element_t<_Indexes, std::decay_t<_More>>::size()...>(tmp);
            auto &&ignore = {
                (__get_tuple_at<_Indexes>(__y) = __data(std::get<_Indexes>(tup)), 0)...};
            __unused(ignore);
        }
        return {first};
    }

public:
    // apply_impl2 can only be called from a 2-element __simd_tuple
    template <class Tuple, size_t _Offset, class F2>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple __extract(
        __size_constant<_Offset>, __size_constant<std::decay_t<Tuple>::_S_first_size - _Offset>,
        Tuple &&tup, F2 &&fun2)
    {
        static_assert(_Offset > 0, "");
        auto splitted =
            split<_Offset, std::decay_t<Tuple>::_S_first_size - _Offset>(__get_simd_at<0>(tup));
        __simd_tuple __r = fun2(__data(std::get<1>(splitted)));
        // if tup is non-const lvalue ref, write __get_tuple_at<0>(splitted) back
        tup.first = __data(concat(std::get<0>(splitted), std::get<1>(splitted)));
        return __r;
    }

    template <class _F, class _More, class _U, size_t _Length, size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl2(chunked<_U, std::decay_t<_More>::_S_first_size, _Length, true,
                        std::index_sequence<_Indexes...>>,
                _F &&fun, const __simd_tuple &__x, _More &&__y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return __simd_tuple_apply(std::forward<_F>(fun), __x, __y.second);
    }

    template <class _F, class _More, class _U, size_t _Offset, size_t _Length,
              size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl2(chunked<_U, _Offset, _Length, false, std::index_sequence<_Indexes...>>,
                _F &&fun, const __simd_tuple &__x, _More &&__y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        static_assert(_Offset < std::decay_t<_More>::_S_first_size, "");
        static_assert(_Offset > 0, "");
        return __extract(__size_constant<_Offset>(), __size_constant<_Length>(), __y,
                       [&](auto &&yy) -> __simd_tuple {
                           return {fun(__tuple_element_meta<_T, _Abi0, 0>(), __x.first, yy)};
                       });
    }

    template <class _R = _T, class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC __fixed_size_storage<_R, size()> apply_r(_F &&fun,
                                                       const _More &... more) const
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return {fun(__tuple_element_meta<_T, _Abi0, 0>(), first, more.first...)};
    }

    template <class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC friend std::bitset<size()> test(_F &&fun, const __simd_tuple &__x,
                                                 const _More &... more)
    {
        return __vector_to_bitset(
            fun(__tuple_element_meta<_T, _Abi0, 0>(), __x.first, more.first...));
    }

    _T operator[](size_t __i) const noexcept { return __subscript_read(first, __i); }
    void set(size_t __i, _T val) noexcept { __subscript_write(first, __i, val); }
};

// 2 or more {{{2
template <class _T, class _Abi0, class... _Abis> struct __simd_tuple<_T, _Abi0, _Abis...> {
    using value_type = _T;
    using _First_type = typename __simd_traits<_T, _Abi0>::_Simd_member_type;
    using _First_abi = _Abi0;
    using _Second_type = __simd_tuple<_T, _Abis...>;
    static constexpr size_t tuple_size = sizeof...(_Abis) + 1;
    static constexpr size_t size() { return simd_size_v<_T, _Abi0> + _Second_type::size(); }
    static constexpr size_t _S_first_size = simd_size_v<_T, _Abi0>;
    static constexpr size_t alignment =
        std::clamp(__next_power_of_2(sizeof(_T) * size()), size_t(16), size_t(256));
    alignas(alignment) _First_type first;
    _Second_type second;

    template <size_t _Offset = 0, class _F>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple generate(_F &&gen, __size_constant<_Offset> = {})
    {
        return {gen(__tuple_element_meta<_T, _Abi0, _Offset>()),
                _Second_type::generate(
                    std::forward<_F>(gen),
                    __size_constant<_Offset + simd_size_v<_T, _Abi0>>())};
    }

    template <size_t _Offset = 0, class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC __simd_tuple apply_wrapped(_F &&fun, const _More &... more) const
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return {fun(make_meta<_Offset>(*this), first, more.first...),
                second.template apply_wrapped<_Offset + simd_size_v<_T, _Abi0>>(
                    std::forward<_F>(fun), more.second...)};
    }

    template <class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC friend __simd_tuple __simd_tuple_apply(_F &&fun, const __simd_tuple &__x,
                                                    _More &&... more)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("more = ", more...);
        return __simd_tuple::apply_impl(
            __bool_constant<conjunction<__is_equal<size_t, _S_first_size,
                                       std::decay_t<_More>::_S_first_size>...>::value>(),
            std::forward<_F>(fun), __x, std::forward<_More>(more)...);
    }

private:
    template <class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl(true_type,  // _S_first_size is equal for all arguments
               _F &&fun, const __simd_tuple &__x, _More &&... more)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return {fun(__tuple_element_meta<_T, _Abi0, 0>(), __x.first, more.first...),
                __simd_tuple_apply(std::forward<_F>(fun), __x.second, more.second...)};
    }

    template <class _F, class _More>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl(false_type,  // at least one argument in _More has different _S_first_size
               _F &&fun, const __simd_tuple &__x, _More &&__y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("__y = ", __y);
        return apply_impl2(__how_many_to_extract<_S_first_size, std::decay_t<_More>>::tag(),
                           std::forward<_F>(fun), __x, __y);
    }

    template <class _F, class _More, size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple apply_impl2(std::index_sequence<_Indexes...>, _F &&fun,
                                               const __simd_tuple &__x, _More &&__y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("__y = ", __y);
        auto tmp = std::experimental::concat(__get_simd_at<_Indexes>(__y)...);
        const auto first = fun(__tuple_element_meta<_T, _Abi0, 0>(), __x.first, tmp);
        if constexpr (std::is_lvalue_reference<_More>::value &&
                      !std::is_const<_More>::value) {
            // if __y is non-const lvalue ref, assume write back is necessary
            const auto tup =
                std::experimental::split<__simd_tuple_element_t<_Indexes, std::decay_t<_More>>::size()...>(tmp);
            [](std::initializer_list<int>) {
            }({(__get_tuple_at<_Indexes>(__y) = __data(std::get<_Indexes>(tup)), 0)...});
        }
        return {first, __simd_tuple_apply(
                           std::forward<_F>(fun), __x.second,
                           __simd_tuple_pop_front(__size_constant<sizeof...(_Indexes)>(), __y))};
    }

public:
    template <class _F, class _More, class _U, size_t _Length, size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl2(chunked<_U, std::decay_t<_More>::_S_first_size, _Length, true,
                        std::index_sequence<_Indexes...>>,
                _F &&fun, const __simd_tuple &__x, _More &&__y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return __simd_tuple_apply(std::forward<_F>(fun), __x, __y.second);
    }

    template <class Tuple, size_t _Length, class F2>
    _GLIBCXX_SIMD_INTRINSIC static auto __extract(__size_constant<0>, __size_constant<_Length>, Tuple &&tup,
                                     F2 &&fun2)
    {
        auto splitted =
            split<_Length, std::decay_t<Tuple>::_S_first_size - _Length>(__get_simd_at<0>(tup));
        auto __r = fun2(__data(std::get<0>(splitted)));
        // if tup is non-const lvalue ref, write __get_tuple_at<0>(splitted) back
        tup.first = __data(concat(std::get<0>(splitted), std::get<1>(splitted)));
        return __r;
    }

    template <class Tuple, size_t _Offset, class F2>
    _GLIBCXX_SIMD_INTRINSIC static auto __extract(
        __size_constant<_Offset>, __size_constant<std::decay_t<Tuple>::_S_first_size - _Offset>,
        Tuple &&tup, F2 &&fun2)
    {
        auto splitted =
            split<_Offset, std::decay_t<Tuple>::_S_first_size - _Offset>(__get_simd_at<0>(tup));
        auto __r = fun2(__data(std::get<1>(splitted)));
        // if tup is non-const lvalue ref, write __get_tuple_at<0>(splitted) back
        tup.first = __data(concat(std::get<0>(splitted), std::get<1>(splitted)));
        return __r;
    }

    template <
        class Tuple, size_t _Offset, size_t _Length, class F2,
        class = enable_if_t<(_Offset + _Length < std::decay_t<Tuple>::_S_first_size)>>
    _GLIBCXX_SIMD_INTRINSIC static auto __extract(__size_constant<_Offset>, __size_constant<_Length>,
                                     Tuple &&tup, F2 &&fun2)
    {
        static_assert(_Offset + _Length < std::decay_t<Tuple>::_S_first_size, "");
        auto splitted =
            split<_Offset, _Length, std::decay_t<Tuple>::_S_first_size - _Offset - _Length>(
                __get_simd_at<0>(tup));
        auto __r = fun2(__data(std::get<1>(splitted)));
        // if tup is non-const lvalue ref, write __get_tuple_at<0>(splitted) back
        tup.first = __data(
            concat(std::get<0>(splitted), std::get<1>(splitted), std::get<2>(splitted)));
        return __r;
    }

    template <class _F, class _More, class _U, size_t _Offset, size_t _Length,
              size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple
    apply_impl2(chunked<_U, _Offset, _Length, false, std::index_sequence<_Indexes...>>,
                _F &&fun, const __simd_tuple &__x, _More &&__y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        static_assert(_Offset < std::decay_t<_More>::_S_first_size, "");
        return {__extract(__size_constant<_Offset>(), __size_constant<_Length>(), __y,
                        [&](auto &&yy) {
                            return fun(__tuple_element_meta<_T, _Abi0, 0>(), __x.first, yy);
                        }),
                _Second_type::apply_impl2(
                    chunked<_U, _Offset + _Length, _Length,
                            _Offset + _Length == std::decay_t<_More>::_S_first_size,
                            std::index_sequence<_Indexes...>>(),
                    std::forward<_F>(fun), __x.second, __y)};
    }

    template <class _R = _T, class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC auto apply_r(_F &&fun, const _More &... more) const
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return __simd_tuple_concat<_R>(
            fun(__tuple_element_meta<_T, _Abi0, 0>(), first, more.first...),
            second.template apply_r<_R>(std::forward<_F>(fun), more.second...));
    }

    template <class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC friend std::bitset<size()> test(_F &&fun, const __simd_tuple &__x,
                                                 const _More &... more)
    {
        return __vector_to_bitset(
                   fun(__tuple_element_meta<_T, _Abi0, 0>(), __x.first, more.first...))
                   .to_ullong() |
               (test(fun, __x.second, more.second...).to_ullong() << simd_size_v<_T, _Abi0>);
    }

    template <class _U, _U _I>
    _GLIBCXX_SIMD_INTRINSIC constexpr _T operator[](std::integral_constant<_U, _I>) const noexcept
    {
        if constexpr (_I < simd_size_v<_T, _Abi0>) {
            return __subscript_read(first, _I);
        } else {
            return second[std::integral_constant<_U, _I - simd_size_v<_T, _Abi0>>()];
        }
    }

    _T operator[](size_t __i) const noexcept
    {
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        return reinterpret_cast<const __may_alias<_T> *>(this)[__i];
#else
        return __i < simd_size_v<_T, _Abi0> ? __subscript_read(first, __i)
                                        : second[__i - simd_size_v<_T, _Abi0>];
#endif
    }
    void set(size_t __i, _T val) noexcept
    {
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        reinterpret_cast<__may_alias<_T> *>(this)[__i] = val;
#else
        if (__i < simd_size_v<_T, _Abi0>) {
            __subscript_write(first, __i, val);
        } else {
            second.set(__i - simd_size_v<_T, _Abi0>, val);
        }
#endif
    }
};

// __make_simd_tuple {{{1
template <class _T, class _A0>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_T, _A0> __make_simd_tuple(
    std::experimental::simd<_T, _A0> __x0)
{
    return {__data(__x0)};
}
template <class _T, class _A0, class... As>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_T, _A0, As...> __make_simd_tuple(
    const std::experimental::simd<_T, _A0> &__x0,
    const std::experimental::simd<_T, As> &... __xs)
{
    return {__data(__x0), __make_simd_tuple(__xs...)};
}

template <class _T, class _A0>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_T, _A0> __make_simd_tuple(
    const typename __simd_traits<_T, _A0>::_Simd_member_type &arg0)
{
    return {arg0};
}

template <class _T, class _A0, class _A1, class... _Abis>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_T, _A0, _A1, _Abis...> __make_simd_tuple(
    const typename __simd_traits<_T, _A0>::_Simd_member_type &arg0,
    const typename __simd_traits<_T, _A1>::_Simd_member_type &arg1,
    const typename __simd_traits<_T, _Abis>::_Simd_member_type &... args)
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
_GLIBCXX_SIMD_INTRINSIC const __simd_tuple<_T, _A> &__optimize_simd_tuple(const __simd_tuple<_T, _A> &__x)
{
    return __x;
}

template <class _T, class _A0, class _A1, class... _Abis,
          class _R = __fixed_size_storage<_T, __simd_tuple<_T, _A0, _A1, _Abis...>::size()>>
_GLIBCXX_SIMD_INTRINSIC _R __optimize_simd_tuple(const __simd_tuple<_T, _A0, _A1, _Abis...> &__x)
{
    using Tup = __simd_tuple<_T, _A0, _A1, _Abis...>;
    if constexpr (_R::_S_first_size == simd_size_v<_T, _A0>) {
        return __simd_tuple_concat(__simd_tuple<_T, typename _R::_First_abi>{__x.first},
                            __optimize_simd_tuple(__x.second));
    } else if constexpr (_R::_S_first_size == simd_size_v<_T, _A0> + simd_size_v<_T, _A1>) {
        return __simd_tuple_concat(__simd_tuple<_T, typename _R::_First_abi>{__data(
                                std::experimental::concat(__get_simd_at<0>(__x), __get_simd_at<1>(__x)))},
                            __optimize_simd_tuple(__x.second.second));
    } else if constexpr (sizeof...(_Abis) >= 2) {
        if constexpr (_R::_S_first_size == __simd_tuple_element_t<0, Tup>::size() +
                                             __simd_tuple_element_t<1, Tup>::size() +
                                             __simd_tuple_element_t<2, Tup>::size() +
                                             __simd_tuple_element_t<3, Tup>::size()) {
            return __simd_tuple_concat(
                __simd_tuple<_T, typename _R::_First_abi>{__data(concat(
                    __get_simd_at<0>(__x), __get_simd_at<1>(__x), __get_simd_at<2>(__x), __get_simd_at<3>(__x)))},
                __optimize_simd_tuple(__x.second.second.second.second));
        }
    } else {
        return __x;
    }
}

// __for_each(const __simd_tuple &, Fun) {{{1
template <size_t _Offset = 0, class _T, class _A0, class _F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(const __simd_tuple<_T, _A0> &t_, _F &&fun_)
{
    std::forward<_F>(fun_)(make_meta<_Offset>(t_), t_.first);
}
template <size_t _Offset = 0, class _T, class _A0, class _A1, class... As, class _F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(const __simd_tuple<_T, _A0, _A1, As...> &t_, _F &&fun_)
{
    fun_(make_meta<_Offset>(t_), t_.first);
    __for_each<_Offset + simd_size<_T, _A0>::value>(t_.second, std::forward<_F>(fun_));
}

// __for_each(__simd_tuple &, Fun) {{{1
template <size_t _Offset = 0, class _T, class _A0, class _F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(__simd_tuple<_T, _A0> &t_, _F &&fun_)
{
    std::forward<_F>(fun_)(make_meta<_Offset>(t_), t_.first);
}
template <size_t _Offset = 0, class _T, class _A0, class _A1, class... As, class _F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(__simd_tuple<_T, _A0, _A1, As...> &t_, _F &&fun_)
{
    fun_(make_meta<_Offset>(t_), t_.first);
    __for_each<_Offset + simd_size<_T, _A0>::value>(t_.second, std::forward<_F>(fun_));
}

// __for_each(__simd_tuple &, const __simd_tuple &, Fun) {{{1
template <size_t _Offset = 0, class _T, class _A0, class _F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(__simd_tuple<_T, _A0> &a_, const __simd_tuple<_T, _A0> &b_,
                           _F &&fun_)
{
    std::forward<_F>(fun_)(make_meta<_Offset>(a_), a_.first, b_.first);
}
template <size_t _Offset = 0, class _T, class _A0, class _A1, class... As, class _F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(__simd_tuple<_T, _A0, _A1, As...> & a_,
                           const __simd_tuple<_T, _A0, _A1, As...> &b_, _F &&fun_)
{
    fun_(make_meta<_Offset>(a_), a_.first, b_.first);
    __for_each<_Offset + simd_size<_T, _A0>::value>(a_.second, b_.second,
                                                  std::forward<_F>(fun_));
}

// __for_each(const __simd_tuple &, const __simd_tuple &, Fun) {{{1
template <size_t _Offset = 0, class _T, class _A0, class _F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(const __simd_tuple<_T, _A0> &a_, const __simd_tuple<_T, _A0> &b_,
                           _F &&fun_)
{
    std::forward<_F>(fun_)(make_meta<_Offset>(a_), a_.first, b_.first);
}
template <size_t _Offset = 0, class _T, class _A0, class _A1, class... As, class _F>
_GLIBCXX_SIMD_INTRINSIC void __for_each(const __simd_tuple<_T, _A0, _A1, As...> &a_,
                           const __simd_tuple<_T, _A0, _A1, As...> &b_, _F &&fun_)
{
    fun_(make_meta<_Offset>(a_), a_.first, b_.first);
    __for_each<_Offset + simd_size<_T, _A0>::value>(a_.second, b_.second,
                                                  std::forward<_F>(fun_));
}

// }}}1
// missing _mmXXX_mask_cvtepi16_storeu_epi8 intrinsics {{{
#if defined __GNUC__ && !defined __clang__ &&  __GNUC__ < 8
_GLIBCXX_SIMD_INTRINSIC void _mm_mask_cvtepi16_storeu_epi8(void *p, __mmask8 __k, __m128i __x)
{
    asm("vpmovwb %0,(%2)%{%1%}" :: "__x"(__x), "k"(__k), "g"(p) : "k0");
}
_GLIBCXX_SIMD_INTRINSIC void _mm256_mask_cvtepi16_storeu_epi8(void *p, __mmask16 __k, __m256i __x)
{
    asm("vpmovwb %0,(%2)%{%1%}" :: "__x"(__x), "k"(__k), "g"(p) : "k0");
}
_GLIBCXX_SIMD_INTRINSIC void _mm512_mask_cvtepi16_storeu_epi8(void *p, __mmask32 __k, __m512i __x)
{
    asm("vpmovwb %0,(%2)%{%1%}" :: "__x"(__x), "k"(__k), "g"(p) : "k0");
}
#endif

// }}}
// __shift16_right{{{
// if (__shift % 2ⁿ == 0) => the low n Bytes are correct
template <unsigned __shift, class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC _T __shift16_right(_T __v)
{
    static_assert(__shift <= sizeof(_T));
    static_assert(sizeof(_T) == 16);
    if constexpr (__shift == 0) {
        return __v;
    } else if constexpr(__shift == sizeof(_T)) {
        return _T();
    } else if constexpr (__have_sse && __shift == 8 && _TVT::template is<float, 4>) {
        return _mm_movehl_ps(__v, __v);
    } else if constexpr (__have_sse2 && __shift == 8 && _TVT::template is<double, 2>) {
        return _mm_unpackhi_pd(__v, __v);
    } else if constexpr (__have_sse2 && sizeof(_T) == 16) {
        return __intrin_bitcast<_T>(
            _mm_srli_si128(__intrin_bitcast<__m128i>(__v), __shift));
/*
    } else if constexpr (__shift == 16 && sizeof(_T) == 32) {
        if constexpr (__have_avx && _TVT::template is<double, 4>) {
            return _mm256_permute2f128_pd(__v, __v, 0x81);
        } else if constexpr (__have_avx && _TVT::template is<float, 8>) {
            return _mm256_permute2f128_ps(__v, __v, 0x81);
        } else if constexpr (__have_avx) {
            return _mm256_permute2f128_si256(__v, __v, 0x81);
        } else {
            return __auto_bitcast(__hi128(__v));
        }
    } else if constexpr (__have_avx2 && sizeof(_T) == 32) {
        const auto __vi = __intrin_bitcast<__m256i>(__v);
        return __intrin_bitcast<_T>(_mm256_srli_si256(
            __shift < 16 ? __vi : _mm256_permute2x128_si256(__vi, __vi, 0x81),
            __shift % 16));
    } else if constexpr (sizeof(_T) == 32) {
        __shift % 16
        return __intrin_bitcast<_T>(
        __extract<_shift/16, 2>(__v)
        );
    } else if constexpr (__have512f && sizeof(_T) == 64) {
        if constexpr (__shift % 8 == 0) {
            return __mm512_alignr_epi64(__m512i(), __intrin_bitcast<__m512i>(__v),
                                        __shift / 8);
        } else if constexpr (__shift % 4 == 0) {
            return __mm512_alignr_epi32(__m512i(), __intrin_bitcast<__m512i>(__v),
                                        __shift / 4);
        } else {
            const auto __shifted = __mm512_alignr_epi8(
                __m512i(), __intrin_bitcast<__m512i>(__v), __shift % 16);
            return __intrin_bitcast<_T>(
                __shift < 16
                    ? __shifted
                    : _mm512_shuffle_i32x4(__shifted, __shifted, 0xe4 + (__shift / 16)));
        }
    } else if constexpr (__shift == 32 && sizeof(_T) == 64) {
        return __auto_bitcast(__hi256(__v));
    } else if constexpr (__shift % 16 == 0 && sizeof(_T) == 64) {
        return __auto_bitcast(__extract<__shift / 16, 4>(__v));
*/
    } else {
        constexpr int __chunksize =
            __shift % 8 == 0 ? 8 : __shift % 4 == 0 ? 4 : __shift % 2 == 0 ? 2 : 1;
        auto __w = __vector_bitcast<__int_with_sizeof_t<__chunksize>>(__v);
        return __intrin_bitcast<_T>(decltype(__w){__v[__shift / __chunksize], 0});
    }
}

// }}}
// __cmpord{{{
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC auto __cmpord(_T __x, _T __y)
{
    static_assert(is_floating_point_v<typename _TVT::value_type>);
    if constexpr (__have_sse && _TVT::template is<float, 4>) {
        return __intrin_bitcast<_T>(_mm_cmpord_ps(__x, __y));
    } else if constexpr (__have_sse2 && _TVT::template is<double, 2>) {
        return __intrin_bitcast<_T>(_mm_cmpord_pd(__x, __y));
    } else if constexpr (__have_avx && _TVT::template is<float, 8>) {
        return __intrin_bitcast<_T>(_mm256_cmp_ps(__x, __y, _CMP_ORD_Q));
    } else if constexpr (__have_avx && _TVT::template is<double, 4>) {
        return __intrin_bitcast<_T>(_mm256_cmp_pd(__x, __y, _CMP_ORD_Q));
    } else if constexpr (__have_avx512f && _TVT::template is<float, 16>) {
        return _mm512_cmp_ps_mask(__x, __y, _CMP_ORD_Q);
    } else if constexpr (__have_avx512f && _TVT::template is<double, 8>) {
        return _mm512_cmp_pd_mask(__x, __y, _CMP_ORD_Q);
    } else {
        _T __r;
        __execute_n_times<_TVT::_S_width>(
            [&](auto __i) { __r[__i] = (!isnan(__x[__i]) && !isnan(__y[__i])) ? -1 : 0; });
        return __r;
    }
}

// }}}
// __cmpunord{{{
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC auto __cmpunord(_T __x, _T __y)
{
    static_assert(is_floating_point_v<typename _TVT::value_type>);
    if constexpr (__have_sse && _TVT::template is<float, 4>) {
        return __intrin_bitcast<_T>(_mm_cmpunord_ps(__x, __y));
    } else if constexpr (__have_sse2 && _TVT::template is<double, 2>) {
        return __intrin_bitcast<_T>(_mm_cmpunord_pd(__x, __y));
    } else if constexpr (__have_avx && _TVT::template is<float, 8>) {
        return __intrin_bitcast<_T>(_mm256_cmp_ps(__x, __y, _CMP_UNORD_Q));
    } else if constexpr (__have_avx && _TVT::template is<double, 4>) {
        return __intrin_bitcast<_T>(_mm256_cmp_pd(__x, __y, _CMP_UNORD_Q));
    } else if constexpr (__have_avx512f && _TVT::template is<float, 16>) {
        return _mm512_cmp_ps_mask(__x, __y, _CMP_UNORD_Q);
    } else if constexpr (__have_avx512f && _TVT::template is<double, 8>) {
        return _mm512_cmp_pd_mask(__x, __y, _CMP_UNORD_Q);
    } else {
        _T __r;
        __execute_n_times<_TVT::_S_width>(
            [&](auto __i) { __r[__i] = isunordered(__x[__i], __y[__i]) ? -1 : 0; });
        return __r;
    }
}

// }}}
// __maskstore (non-converting; with optimizations for SSE2-AVX512BWVL) {{{
template <class _T, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage64_t<_T> __v, _T* __mem, _F,
                                         __storage<bool, __storage64_t<_T>::_S_width> __k)
{
    [[maybe_unused]] const auto __vi = __to_intrin(__v);
    static_assert(sizeof(__v) == 64 && __have_avx512f);
    if constexpr (__have_avx512bw && sizeof(_T) == 1) {
        _mm512_mask_storeu_epi8(__mem, __k, __vi);
    } else if constexpr (__have_avx512bw && sizeof(_T) == 2) {
        _mm512_mask_storeu_epi16(__mem, __k, __vi);
    } else if constexpr (__have_avx512f && sizeof(_T) == 4) {
        if constexpr (__is_aligned_v<_F, 64> && std::is_integral_v<_T>) {
            _mm512_mask_store_epi32(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 64> && std::is_floating_point_v<_T>) {
            _mm512_mask_store_ps(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_T>) {
            _mm512_mask_storeu_epi32(__mem, __k, __vi);
        } else {
            _mm512_mask_storeu_ps(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512f && sizeof(_T) == 8) {
        if constexpr (__is_aligned_v<_F, 64> && std::is_integral_v<_T>) {
            _mm512_mask_store_epi64(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 64> && std::is_floating_point_v<_T>) {
            _mm512_mask_store_pd(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_T>) {
            _mm512_mask_storeu_epi64(__mem, __k, __vi);
        } else {
            _mm512_mask_storeu_pd(__mem, __k, __vi);
        }
    } else if constexpr (__have_sse2) {
        constexpr int _N = 16 / sizeof(_T);
        using _M          = __vector_type_t<_T, _N>;
        _mm_maskmoveu_si128(__auto_bitcast(__extract<0, 4>(__v._M_data)),
                            __auto_bitcast(__convert_mask<_M>(__k._M_data)),
                            reinterpret_cast<char*>(__mem));
        _mm_maskmoveu_si128(__auto_bitcast(__extract<1, 4>(__v._M_data)),
                            __auto_bitcast(__convert_mask<_M>(__k._M_data >> 1 * _N)),
                            reinterpret_cast<char*>(__mem) + 1 * 16);
        _mm_maskmoveu_si128(__auto_bitcast(__extract<2, 4>(__v._M_data)),
                            __auto_bitcast(__convert_mask<_M>(__k._M_data >> 2 * _N)),
                            reinterpret_cast<char*>(__mem) + 2 * 16);
        _mm_maskmoveu_si128(__auto_bitcast(__extract<3, 4>(__v._M_data)),
                            __auto_bitcast(__convert_mask<_M>(__k._M_data >> 3 * _N)),
                            reinterpret_cast<char*>(__mem) + 3 * 16);
    } else {
        __assert_unreachable<_T>();
    }
}

template <class _T, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage32_t<_T> __v, _T* __mem, _F,
                                         __storage32_t<_T> __k)
{
    [[maybe_unused]] const auto __vi = __vector_bitcast<__llong>(__v);
    [[maybe_unused]] const auto __ki = __vector_bitcast<__llong>(__k);
    if constexpr (__have_avx512bw_vl && sizeof(_T) == 1) {
        _mm256_mask_storeu_epi8(__mem, _mm256_movepi8_mask(__ki), __vi);
    } else if constexpr (__have_avx512bw_vl && sizeof(_T) == 2) {
        _mm256_mask_storeu_epi16(__mem, _mm256_movepi16_mask(__ki), __vi);
    } else if constexpr (__have_avx2 && sizeof(_T) == 4 && std::is_integral_v<_T>) {
        _mm256_maskstore_epi32(reinterpret_cast<int*>(__mem), __ki, __vi);
    } else if constexpr (sizeof(_T) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__mem), __ki,
                            __vector_bitcast<float>(__v));
    } else if constexpr (__have_avx2 && sizeof(_T) == 8 && std::is_integral_v<_T>) {
        _mm256_maskstore_epi64(reinterpret_cast<__llong*>(__mem), __ki, __vi);
    } else if constexpr (__have_avx && sizeof(_T) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__mem), __ki,
                            __vector_bitcast<double>(__v));
    } else if constexpr (__have_sse2) {
        _mm_maskmoveu_si128(__lo128(__vi), __lo128(__ki), reinterpret_cast<char*>(__mem));
        _mm_maskmoveu_si128(__hi128(__vi), __hi128(__ki),
                            reinterpret_cast<char*>(__mem) + 16);
    } else {
        __assert_unreachable<_T>();
    }
}

template <class _T, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage32_t<_T> __v, _T* __mem, _F,
                                         __storage<bool, __storage32_t<_T>::_S_width> __k)
{
    [[maybe_unused]] const auto __vi = __to_intrin(__v);
    if constexpr (__have_avx512bw_vl && sizeof(_T) == 1) {
        _mm256_mask_storeu_epi8(__mem, __k, __vi);
    } else if constexpr (__have_avx512bw_vl && sizeof(_T) == 2) {
        _mm256_mask_storeu_epi16(__mem, __k, __vi);
    } else if constexpr (__have_avx512vl && sizeof(_T) == 4) {
        if constexpr (__is_aligned_v<_F, 32> && std::is_integral_v<_T>) {
            _mm256_mask_store_epi32(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 32> && std::is_floating_point_v<_T>) {
            _mm256_mask_store_ps(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_T>) {
            _mm256_mask_storeu_epi32(__mem, __k, __vi);
        } else {
            _mm256_mask_storeu_ps(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512vl && sizeof(_T) == 8) {
        if constexpr (__is_aligned_v<_F, 32> && std::is_integral_v<_T>) {
            _mm256_mask_store_epi64(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 32> && std::is_floating_point_v<_T>) {
            _mm256_mask_store_pd(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_T>) {
            _mm256_mask_storeu_epi64(__mem, __k, __vi);
        } else {
            _mm256_mask_storeu_pd(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512f && (sizeof(_T) >= 4 || __have_avx512bw)) {
        // use a 512-bit maskstore, using zero-extension of the bitmask
        __maskstore(
            __storage64_t<_T>(__intrin_bitcast<__vector_type64_t<_T>>(__v._M_data)),
            __mem,
            // careful, vector_aligned has a stricter meaning in the 512-bit maskstore:
            std::conditional_t<std::is_same_v<_F, vector_aligned_tag>,
                               overaligned_tag<32>, _F>(),
            __storage<bool, 64 / sizeof(_T)>(__k._M_data));
    } else {
        __maskstore(
            __v, __mem, _F(),
            __storage32_t<_T>(__convert_mask<__vector_type_t<_T, 32 / sizeof(_T)>>(__k)));
    }
}

template <class _T, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage16_t<_T> __v, _T* __mem, _F,
                                         __storage16_t<_T> __k)
{
    [[maybe_unused]] const auto __vi = __vector_bitcast<__llong>(__v);
    [[maybe_unused]] const auto __ki = __vector_bitcast<__llong>(__k);
    if constexpr (__have_avx512bw_vl && sizeof(_T) == 1) {
        _mm_mask_storeu_epi8(__mem, _mm_movepi8_mask(__ki), __vi);
    } else if constexpr (__have_avx512bw_vl && sizeof(_T) == 2) {
        _mm_mask_storeu_epi16(__mem, _mm_movepi16_mask(__ki), __vi);
    } else if constexpr (__have_avx2 && sizeof(_T) == 4 && std::is_integral_v<_T>) {
        _mm_maskstore_epi32(reinterpret_cast<int*>(__mem), __ki, __vi);
    } else if constexpr (__have_avx && sizeof(_T) == 4) {
        _mm_maskstore_ps(reinterpret_cast<float*>(__mem), __ki,
                         __vector_bitcast<float>(__v));
    } else if constexpr (__have_avx2 && sizeof(_T) == 8 && std::is_integral_v<_T>) {
        _mm_maskstore_epi64(reinterpret_cast<__llong*>(__mem), __ki, __vi);
    } else if constexpr (__have_avx && sizeof(_T) == 8) {
        _mm_maskstore_pd(reinterpret_cast<double*>(__mem), __ki,
                         __vector_bitcast<double>(__v));
    } else if constexpr (__have_sse2) {
        _mm_maskmoveu_si128(__vi, __ki, reinterpret_cast<char*>(__mem));
    } else {
        __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(),
                        [&](auto __i) { __mem[__i] = __v[__i]; });
    }
}

template <class _T, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage16_t<_T> __v, _T* __mem, _F,
                                         __storage<bool, __storage16_t<_T>::_S_width> __k)
{
    [[maybe_unused]] const auto __vi = __to_intrin(__v);
    if constexpr (__have_avx512bw_vl && sizeof(_T) == 1) {
        _mm_mask_storeu_epi8(__mem, __k, __vi);
    } else if constexpr (__have_avx512bw_vl && sizeof(_T) == 2) {
        _mm_mask_storeu_epi16(__mem, __k, __vi);
    } else if constexpr (__have_avx512vl && sizeof(_T) == 4) {
        if constexpr (__is_aligned_v<_F, 16> && std::is_integral_v<_T>) {
            _mm_mask_store_epi32(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 16> && std::is_floating_point_v<_T>) {
            _mm_mask_store_ps(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_T>) {
            _mm_mask_storeu_epi32(__mem, __k, __vi);
        } else {
            _mm_mask_storeu_ps(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512vl && sizeof(_T) == 8) {
        if constexpr (__is_aligned_v<_F, 16> && std::is_integral_v<_T>) {
            _mm_mask_store_epi64(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 16> && std::is_floating_point_v<_T>) {
            _mm_mask_store_pd(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_T>) {
            _mm_mask_storeu_epi64(__mem, __k, __vi);
        } else {
            _mm_mask_storeu_pd(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512f && (sizeof(_T) >= 4 || __have_avx512bw)) {
        // use a 512-bit maskstore, using zero-extension of the bitmask
        __maskstore(
            __storage64_t<_T>(__intrin_bitcast<__intrinsic_type64_t<_T>>(__v._M_data)), __mem,
            // careful, vector_aligned has a stricter meaning in the 512-bit maskstore:
            std::conditional_t<std::is_same_v<_F, vector_aligned_tag>,
                               overaligned_tag<16>, _F>(),
            __storage<bool, 64 / sizeof(_T)>(__k._M_data));
    } else {
        __maskstore(
            __v, __mem, _F(),
            __storage16_t<_T>(__convert_mask<__vector_type_t<_T, 16 / sizeof(_T)>>(__k)));
    }
}

// }}}
// __xzyw{{{
// shuffles the complete vector, swapping the inner two quarters. Often useful for AVX for
// fixing up a shuffle result.
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC _T __xzyw(_T __a)
{
    if constexpr (sizeof(_T) == 16) {
        static_assert(sizeof(float) == 4 && sizeof(int) == 4);
        const auto __x = __vector_bitcast<
            conditional_t<is_floating_point_v<typename _TVT::value_type>, float, int>>(
            __a);
        return reinterpret_cast<_T>(decltype(__x){__x[0], __x[2], __x[1], __x[3]});
    } else if constexpr (sizeof(_T) == 32) {
        static_assert(sizeof(double) == 8 && sizeof(__llong) == 8);
        const auto __x =
            __vector_bitcast<conditional_t<is_floating_point_v<typename _TVT::value_type>,
                                           double, __llong>>(__a);
        return reinterpret_cast<_T>(decltype(__x){__x[0], __x[2], __x[1], __x[3]});
    } else if constexpr (sizeof(_T) == 64) {
        static_assert(sizeof(double) == 8 && sizeof(__llong) == 8);
        const auto __x =
            __vector_bitcast<conditional_t<is_floating_point_v<typename _TVT::value_type>,
                                           double, __llong>>(__a);
        return reinterpret_cast<_T>(decltype(__x){__x[0], __x[1], __x[4], __x[5], __x[2],
                                                  __x[3], __x[6], __x[7]});
    } else {
        __assert_unreachable<_T>();
    }
}

// }}}
// __extract_part(__storage<_T, _N>) {{{
template <size_t _Index, size_t _Total, class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST
    __vector_type_t<_T, std::max(16 / sizeof(_T), _N / _Total)>
    __extract_part(__storage<_T, _N> __x)
{
    constexpr size_t _NewN = _N / _Total;
    static_assert(_Total > _Index, "_Total must be greater than _Index");
    static_assert(_NewN * _Total == _N, "_N must be divisible by _Total");
    if constexpr (_Index == 0 && _Total == 1) {
        return __x._M_data;
    } else if constexpr (sizeof(_T) * _NewN >= 16) {
        return __extract<_Index, _Total>(__x._M_data);
    } else {
        static_assert(__have_sse && _N == _N);
        constexpr int split = sizeof(__x) / 16;
        constexpr int shift = (sizeof(__x) / _Total * _Index) % 16;
        return __shift16_right<shift>(
            __extract_part<_Index * split / _Total, split>(__x));
    }
}

// }}}
// __extract_part(__storage<bool, _N>) {{{
template <size_t _Index, size_t _Total, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __bool_storage_member_type_t<_N / _Total>
__extract_part(__storage<bool, _N> __x)
{
    static_assert(__have_avx512f && _N == _N);
    static_assert(_Total >= 2 && _Index < _Total && _Index >= 0);
    return __x._M_data >> (_Index * _N / _Total);
}

// }}}
// __extract_part(__simd_tuple) {{{
template <int _Index, int Parts, class _T, class _A0, class... As>
_GLIBCXX_SIMD_INTRINSIC auto  // __vector_type_t or __simd_tuple
__extract_part(const __simd_tuple<_T, _A0, As...> &__x)
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
            return __x.first._M_data;
        } else {
            return __x;
        }
    } else if constexpr (simd_size_v<_T, _A0> % values_per_part != 0) {
        // nasty case: The requested partition does not match the partition of the
        // __simd_tuple. Fall back to construction via scalar copies.
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        const __may_alias<_T> *const element_ptr =
            reinterpret_cast<const __may_alias<_T> *>(&__x) + _Index * values_per_part;
        return __data(simd<_T, simd_abi::deduce_t<_T, values_per_part>>(
                          [&](auto __i) { return element_ptr[__i]; }))
            ._M_data;
#else
        constexpr size_t offset = _Index * values_per_part;
        __unused(offset);  // not really
        return __data(simd<_T, simd_abi::deduce_t<_T, values_per_part>>([&](auto __i) {
                   constexpr __size_constant<__i + offset> __k;
                   return __x[__k];
               }))
            ._M_data;
#endif
    } else if constexpr (values_per_part * _Index >= simd_size_v<_T, _A0>) {  // recurse
        constexpr int parts_in_first = simd_size_v<_T, _A0> / values_per_part;
        return __extract_part<_Index - parts_in_first, Parts - parts_in_first>(__x.second);
    } else {  // at this point we know that all of the return values are in __x.first
        static_assert(values_per_part * (1 + _Index) <= simd_size_v<_T, _A0>);
        if constexpr (simd_size_v<_T, _A0> == values_per_part) {
            return __x.first._M_data;
        } else {
            return __extract_part<_Index, simd_size_v<_T, _A0> / values_per_part>(
                __x.first);
        }
    }
}
// }}}
// _To_storage specializations for bitset and __mmask<_N> {{{
#if _GLIBCXX_SIMD_HAVE_AVX512_ABI
template <size_t _N> class _To_storage<std::bitset<_N>>
{
    std::bitset<_N> _M_data;

public:
    // can convert to larger storage for _Abi::is_partial == true
    template <class _U, size_t _M> constexpr operator __storage<_U, _M>() const
    {
        static_assert(_M >= _N);
        return __convert_mask<__storage<_U, _M>>(_M_data);
    }
};

#define _GLIBCXX_SIMD_TO_STORAGE(_Type)                                                  \
    template <> class _To_storage<_Type>                                                \
    {                                                                                    \
        _Type _M_data;                                                                         \
                                                                                         \
    public:                                                                              \
        template <class _U, size_t _N> constexpr operator __storage<_U, _N>() const      \
        {                                                                                \
            static_assert(_N >= sizeof(_Type) * CHAR_BIT);                               \
            return reinterpret_cast<__vector_type_t<_U, _N>>(                            \
                __convert_mask<__storage<_U, _N>>(_M_data));                                   \
        }                                                                                \
                                                                                         \
        template <size_t _N> constexpr operator __storage<bool, _N>() const              \
        {                                                                                \
            static_assert(                                                               \
                std::is_same_v<_Type, typename __bool_storage_member_type<_N>::type>);   \
            return _M_data;                                                                    \
        }                                                                                \
    }
_GLIBCXX_SIMD_TO_STORAGE(__mmask8);
_GLIBCXX_SIMD_TO_STORAGE(__mmask16);
_GLIBCXX_SIMD_TO_STORAGE(__mmask32);
_GLIBCXX_SIMD_TO_STORAGE(__mmask64);
#undef _GLIBCXX_SIMD_TO_STORAGE
#endif  // _GLIBCXX_SIMD_HAVE_AVX512_ABI

// }}}

#if _GLIBCXX_SIMD_HAVE_SSE && defined _GLIBCXX_SIMD_WORKAROUND_PR85048
#include "simd_x86_conversions.h"
#endif  // SSE && _GLIBCXX_SIMD_WORKAROUND_PR85048

// __convert function{{{
template <class _To, class _From, class... _More>
_GLIBCXX_SIMD_INTRINSIC auto __convert(_From __v0, _More... __vs)
{
    static_assert((true && ... && is_same_v<_From, _More>));
    if constexpr (__is_vectorizable_v<_From>) {
        if constexpr (__is_vector_type_v<_To>) {
            return __make_builtin(__v0, __vs...);
        } else {
            using _T = typename _To::value_type;
            return __make_storage<_T>(__v0, __vs...);
        }
    } else if constexpr (!__is_vector_type_v<_From>) {
        return __convert<_To>(__v0._M_data, __vs._M_data...);
    } else if constexpr (!__is_vector_type_v<_To>) {
        return _To(__convert<typename _To::register_type>(__v0, __vs...));
    } else if constexpr (__is_vectorizable_v<_To>) {
        return __convert<__vector_type_t<_To, (__vector_traits<_From>::_S_width *
                                            (1 + sizeof...(_More)))>>(__v0, __vs...)
            ._M_data;
    } else {
        static_assert(sizeof...(_More) == 0 ||
                          __vector_traits<_To>::_S_width >=
                              (1 + sizeof...(_More)) * __vector_traits<_From>::_S_width,
                      "__convert(...) requires the input to fit into the output");
        return __vector_convert<_To>(__v0, __vs...);
    }
}

// }}}
// __convert_all{{{
template <typename _To, typename _From> _GLIBCXX_SIMD_INTRINSIC auto __convert_all(_From __v)
{
    static_assert(__is_vector_type_v<_To>);
    if constexpr (__is_vector_type_v<_From>) {
        using _Trait = __vector_traits<_From>;
        using S = __storage<typename _Trait::value_type, _Trait::_S_width>;
        return __convert_all<_To>(S(__v));
    } else if constexpr (_From::_S_width > __vector_traits<_To>::_S_width) {
        constexpr size_t _N = _From::_S_width / __vector_traits<_To>::_S_width;
        return __generate_from_n_evaluations<_N, std::array<_To, _N>>([&](auto __i) {
            auto part = __extract_part<decltype(__i)::value, _N>(__v);
            return __convert<_To>(part);
        });
    } else {
        return __convert<_To>(__v);
    }
}

// }}}
// __converts_via_decomposition{{{
// This lists all cases where a __vector_convert needs to fall back to conversion of
// individual scalars (i.e. decompose the input vector into scalars, convert, compose
// output vector). In those cases, masked_load & masked_store prefer to use the
// __bit_iteration implementation.
template <class _From, class _To, size_t _ToSize> struct __converts_via_decomposition {
private:
    static constexpr bool __i_to_i = is_integral_v<_From> && is_integral_v<_To>;
    static constexpr bool __f_to_i = is_floating_point_v<_From> && is_integral_v<_To>;
    static constexpr bool __f_to_f = is_floating_point_v<_From> && is_floating_point_v<_To>;
    static constexpr bool __i_to_f = is_integral_v<_From> && is_floating_point_v<_To>;

    template <size_t _A, size_t _B>
    static constexpr bool __sizes = sizeof(_From) == _A && sizeof(_To) == _B;

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
// __plus{{{
template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, _N> __plus(__storage<_T, _N> __a,
                                                         __storage<_T, _N> __b)
{
    return __a._M_data + __b._M_data;
}

//}}}
// __minus{{{
template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, _N> __minus(__storage<_T, _N> __a,
                                                          __storage<_T, _N> __b)
{
    return __a._M_data - __b._M_data;
}

//}}}
// __multiplies{{{
template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, _N> __multiplies(__storage<_T, _N> __a,
                                                               __storage<_T, _N> __b)
{
    if constexpr (sizeof(_T) == 1) {
        return __vector_bitcast<_T>(
            ((__vector_bitcast<short>(__a) * __vector_bitcast<short>(__b)) &
             __vector_bitcast<short>(~__vector_type_t<ushort, _N / 2>() >> 8)) |
            (((__vector_bitcast<short>(__a) >> 8) * (__vector_bitcast<short>(__b) >> 8))
             << 8));
    }
    return __a._M_data * __b._M_data;
}

//}}}
// __abs{{{
template <class _T, size_t _N>
_GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, _N> __abs(__storage<_T, _N> __v)
{
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85572
    if constexpr (!__have_avx512vl && std::is_integral_v<_T> && sizeof(_T) == 8 && _N <= 4) {
        // positive value:
        //   negative == 0
        //   __a unchanged after xor
        //   __a - 0 -> __a
        // negative value:
        //   negative == ~0 == -1
        //   __a xor ~0    -> -__a - 1
        //   -__a - 1 - -1 -> -__a
        if constexpr(__have_sse4_2) {
            const auto negative = reinterpret_cast<__vector_type_t<_T, _N>>(__v._M_data < 0);
            return (__v._M_data ^ negative) - negative;
        } else {
            // arithmetic right shift doesn't exist for 64-bit integers, use the following
            // instead:
            // >>63: negative ->  1, positive ->  0
            //  -  : negative -> -1, positive ->  0
            const auto negative = -reinterpret_cast<__vector_type_t<_T, _N>>(
                reinterpret_cast<__vector_type_t<__ullong, _N>>(__v._M_data) >> 63);
            return (__v._M_data ^ negative) - negative;
        }
    } else
#endif
        if constexpr (std::is_floating_point_v<_T>) {
        // this workaround is only required because __builtin_abs is not a constant
        // expression
        using _I = std::make_unsigned_t<__int_for_sizeof_t<_T>>;
        return __vector_bitcast<_T>(__vector_bitcast<_I>(__v._M_data) & __vector_broadcast<_N, _I>(~_I() >> 1));
    } else {
        return __v._M_data < 0 ? -__v._M_data : __v._M_data;
    }
}

//}}}
// interleave (__lo/__hi/128) {{{
template <class _A, class _B, class _T = std::common_type_t<_A, _B>,
          class _Trait = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T __interleave_lo(_A _a, _B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const _T __a(_a);
    const _T __b(_b);
    if constexpr (sizeof(_T) == 16 && needs_intrinsics) {
        if constexpr (_Trait::_S_width == 2) {
            if constexpr (std::is_integral_v<typename _Trait::value_type>) {
                return reinterpret_cast<_T>(
                    _mm_unpacklo_epi64(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
            } else {
                return reinterpret_cast<_T>(
                    _mm_unpacklo_pd(__vector_bitcast<double>(__a), __vector_bitcast<double>(__b)));
            }
        } else if constexpr (_Trait::_S_width == 4) {
            if constexpr (std::is_integral_v<typename _Trait::value_type>) {
                return reinterpret_cast<_T>(
                    _mm_unpacklo_epi32(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
            } else {
                return reinterpret_cast<_T>(
                    _mm_unpacklo_ps(__vector_bitcast<float>(__a), __vector_bitcast<float>(__b)));
            }
        } else if constexpr (_Trait::_S_width == 8) {
            return reinterpret_cast<_T>(
                _mm_unpacklo_epi16(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
        } else if constexpr (_Trait::_S_width == 16) {
            return reinterpret_cast<_T>(
                _mm_unpacklo_epi8(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
        }
    } else if constexpr (_Trait::_S_width == 2) {
        return _T{__a[0], __b[0]};
    } else if constexpr (_Trait::_S_width == 4) {
        return _T{__a[0], __b[0], __a[1], __b[1]};
    } else if constexpr (_Trait::_S_width == 8) {
        return _T{__a[0], __b[0], __a[1], __b[1], __a[2], __b[2], __a[3], __b[3]};
    } else if constexpr (_Trait::_S_width == 16) {
        return _T{__a[0], __b[0], __a[1], __b[1], __a[2], __b[2], __a[3], __b[3],
                 __a[4], __b[4], __a[5], __b[5], __a[6], __b[6], __a[7], __b[7]};
    } else if constexpr (_Trait::_S_width == 32) {
        return _T{__a[0],  __b[0],  __a[1],  __b[1],  __a[2],  __b[2],  __a[3],  __b[3],
                 __a[4],  __b[4],  __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],
                 __a[8],  __b[8],  __a[9],  __b[9],  __a[10], __b[10], __a[11], __b[11],
                 __a[12], __b[12], __a[13], __b[13], __a[14], __b[14], __a[15], __b[15]};
    } else if constexpr (_Trait::_S_width == 64) {
        return _T{__a[0],  __b[0],  __a[1],  __b[1],  __a[2],  __b[2],  __a[3],  __b[3],  __a[4],  __b[4],
                 __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],  __a[8],  __b[8],  __a[9],  __b[9],
                 __a[10], __b[10], __a[11], __b[11], __a[12], __b[12], __a[13], __b[13], __a[14], __b[14],
                 __a[15], __b[15], __a[16], __b[16], __a[17], __b[17], __a[18], __b[18], __a[19], __b[19],
                 __a[20], __b[20], __a[21], __b[21], __a[22], __b[22], __a[23], __b[23], __a[24], __b[24],
                 __a[25], __b[25], __a[26], __b[26], __a[27], __b[27], __a[28], __b[28], __a[29], __b[29],
                 __a[30], __b[30], __a[31], __b[31]};
    } else {
        __assert_unreachable<_T>();
    }
}

template <class _A, class _B, class _T = std::common_type_t<_A, _B>,
          class _Trait = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T __interleave_hi(_A _a, _B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const _T __a(_a);
    const _T __b(_b);
    if constexpr (sizeof(_T) == 16 && needs_intrinsics) {
        if constexpr (_Trait::_S_width == 2) {
            if constexpr (std::is_integral_v<typename _Trait::value_type>) {
                return reinterpret_cast<_T>(
                    _mm_unpackhi_epi64(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
            } else {
                return reinterpret_cast<_T>(
                    _mm_unpackhi_pd(__vector_bitcast<double>(__a), __vector_bitcast<double>(__b)));
            }
        } else if constexpr (_Trait::_S_width == 4) {
            if constexpr (std::is_integral_v<typename _Trait::value_type>) {
                return reinterpret_cast<_T>(
                    _mm_unpackhi_epi32(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
            } else {
                return reinterpret_cast<_T>(
                    _mm_unpackhi_ps(__vector_bitcast<float>(__a), __vector_bitcast<float>(__b)));
            }
        } else if constexpr (_Trait::_S_width == 8) {
            return reinterpret_cast<_T>(
                _mm_unpackhi_epi16(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
        } else if constexpr (_Trait::_S_width == 16) {
            return reinterpret_cast<_T>(
                _mm_unpackhi_epi8(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
        }
    } else if constexpr (_Trait::_S_width == 2) {
        return _T{__a[1], __b[1]};
    } else if constexpr (_Trait::_S_width == 4) {
        return _T{__a[2], __b[2], __a[3], __b[3]};
    } else if constexpr (_Trait::_S_width == 8) {
        return _T{__a[4], __b[4], __a[5], __b[5], __a[6], __b[6], __a[7], __b[7]};
    } else if constexpr (_Trait::_S_width == 16) {
        return _T{__a[8],  __b[8],  __a[9],  __b[9],  __a[10], __b[10], __a[11], __b[11],
                 __a[12], __b[12], __a[13], __b[13], __a[14], __b[14], __a[15], __b[15]};
    } else if constexpr (_Trait::_S_width == 32) {
        return _T{__a[16], __b[16], __a[17], __b[17], __a[18], __b[18], __a[19], __b[19],
                 __a[20], __b[20], __a[21], __b[21], __a[22], __b[22], __a[23], __b[23],
                 __a[24], __b[24], __a[25], __b[25], __a[26], __b[26], __a[27], __b[27],
                 __a[28], __b[28], __a[29], __b[29], __a[30], __b[30], __a[31], __b[31]};
    } else if constexpr (_Trait::_S_width == 64) {
        return _T{__a[32], __b[32], __a[33], __b[33], __a[34], __b[34], __a[35], __b[35],
                 __a[36], __b[36], __a[37], __b[37], __a[38], __b[38], __a[39], __b[39],
                 __a[40], __b[40], __a[41], __b[41], __a[42], __b[42], __a[43], __b[43],
                 __a[44], __b[44], __a[45], __b[45], __a[46], __b[46], __a[47], __b[47],
                 __a[48], __b[48], __a[49], __b[49], __a[50], __b[50], __a[51], __b[51],
                 __a[52], __b[52], __a[53], __b[53], __a[54], __b[54], __a[55], __b[55],
                 __a[56], __b[56], __a[57], __b[57], __a[58], __b[58], __a[59], __b[59],
                 __a[60], __b[60], __a[61], __b[61], __a[62], __b[62], __a[63], __b[63]};
    } else {
        __assert_unreachable<_T>();
    }
}

template <class _A, class _B, class _T = std::common_type_t<_A, _B>,
          class _Trait = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T interleave128_lo(_A _a, _B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const _T __a(_a);
    const _T __b(_b);
    if constexpr (sizeof(_T) == 16) {
        return __interleave_lo(__a, __b);
    } else if constexpr (sizeof(_T) == 32 && needs_intrinsics) {
        if constexpr (_Trait::_S_width == 4) {
            return reinterpret_cast<_T>(
                _mm256_unpacklo_pd(__vector_bitcast<double>(__a), __vector_bitcast<double>(__b)));
        } else if constexpr (_Trait::_S_width == 8) {
            return reinterpret_cast<_T>(
                _mm256_unpacklo_ps(__vector_bitcast<float>(__a), __vector_bitcast<float>(__b)));
        } else if constexpr (_Trait::_S_width == 16) {
            return reinterpret_cast<_T>(
                _mm256_unpacklo_epi16(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
        } else if constexpr (_Trait::_S_width == 32) {
            return reinterpret_cast<_T>(
                _mm256_unpacklo_epi8(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
        }
    } else if constexpr (sizeof(_T) == 32) {
        if constexpr (_Trait::_S_width == 4) {
            return _T{__a[0], __b[0], __a[2], __b[2]};
        } else if constexpr (_Trait::_S_width == 8) {
            return _T{__a[0], __b[0], __a[1], __b[1], __a[4], __b[4], __a[5], __b[5]};
        } else if constexpr (_Trait::_S_width == 16) {
            return _T{__a[0], __b[0], __a[1], __b[1], __a[2],  __b[2],  __a[3],  __b[3],
                     __a[8], __b[8], __a[9], __b[9], __a[10], __b[10], __a[11], __b[11]};
        } else if constexpr (_Trait::_S_width == 32) {
            return _T{__a[0],  __b[0],  __a[1],  __b[1],  __a[2],  __b[2],  __a[3],  __b[3],
                     __a[4],  __b[4],  __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],
                     __a[16], __b[16], __a[17], __b[17], __a[18], __b[18], __a[19], __b[19],
                     __a[20], __b[20], __a[21], __b[21], __a[22], __b[22], __a[23], __b[23]};
        } else if constexpr (_Trait::_S_width == 64) {
            return _T{__a[0],  __b[0],  __a[1],  __b[1],  __a[2],  __b[2],  __a[3],  __b[3],  __a[4],  __b[4],
                     __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],  __a[8],  __b[8],  __a[9],  __b[9],
                     __a[10], __b[10], __a[11], __b[11], __a[12], __b[12], __a[13], __b[13], __a[14], __b[14],
                     __a[15], __b[15], __a[32], __b[32], __a[33], __b[33], __a[34], __b[34], __a[35], __b[35],
                     __a[36], __b[36], __a[37], __b[37], __a[38], __b[38], __a[39], __b[39], __a[40], __b[40],
                     __a[41], __b[41], __a[42], __b[42], __a[43], __b[43], __a[44], __b[44], __a[45], __b[45],
                     __a[46], __b[46], __a[47], __b[47]};
        } else {
            __assert_unreachable<_T>();
        }
    } else if constexpr (sizeof(_T) == 64 && needs_intrinsics) {
        if constexpr (_Trait::_S_width == 8) {
            return reinterpret_cast<_T>(
                _mm512_unpacklo_pd(__vector_bitcast<double>(__a), __vector_bitcast<double>(__b)));
        } else if constexpr (_Trait::_S_width == 16) {
            return reinterpret_cast<_T>(
                _mm512_unpacklo_ps(__vector_bitcast<float>(__a), __vector_bitcast<float>(__b)));
        } else if constexpr (_Trait::_S_width == 32) {
            return reinterpret_cast<_T>(
                _mm512_unpacklo_epi16(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
        } else if constexpr (_Trait::_S_width == 64) {
            return reinterpret_cast<_T>(
                _mm512_unpacklo_epi8(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
        }
    } else if constexpr (sizeof(_T) == 64) {
        if constexpr (_Trait::_S_width == 8) {
            return _T{__a[0], __b[0], __a[2], __b[2], __a[4], __b[4], __a[6], __b[6]};
        } else if constexpr (_Trait::_S_width == 16) {
            return _T{__a[0], __b[0], __a[1], __b[1], __a[4],  __b[4],  __a[5],  __b[5],
                     __a[8], __b[8], __a[9], __b[9], __a[12], __b[12], __a[13], __b[13]};
        } else if constexpr (_Trait::_S_width == 32) {
            return _T{__a[0],  __b[0],  __a[1],  __b[1],  __a[2],  __b[2],  __a[3],  __b[3],
                     __a[8],  __b[8],  __a[9],  __b[9],  __a[10], __b[10], __a[11], __b[11],
                     __a[16], __b[16], __a[17], __b[17], __a[18], __b[18], __a[19], __b[19],
                     __a[24], __b[24], __a[25], __b[25], __a[26], __b[26], __a[27], __b[27]};
        } else if constexpr (_Trait::_S_width == 64) {
            return _T{__a[0],  __b[0],  __a[1],  __b[1],  __a[2],  __b[2],  __a[3],  __b[3],  __a[4],  __b[4],
                     __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],  __a[16], __b[16], __a[17], __b[17],
                     __a[18], __b[18], __a[19], __b[19], __a[20], __b[20], __a[21], __b[21], __a[22], __b[22],
                     __a[23], __b[23], __a[32], __b[32], __a[33], __b[33], __a[34], __b[34], __a[35], __b[35],
                     __a[36], __b[36], __a[37], __b[37], __a[38], __b[38], __a[39], __b[39], __a[48], __b[48],
                     __a[49], __b[49], __a[50], __b[50], __a[51], __b[51], __a[52], __b[52], __a[53], __b[53],
                     __a[54], __b[54], __a[55], __b[55]};
        } else {
            __assert_unreachable<_T>();
        }
    }
}

template <class _A, class _B, class _T = std::common_type_t<_A, _B>,
          class _Trait = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T interleave128_hi(_A _a, _B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const _T __a(_a);
    const _T __b(_b);
    if constexpr (sizeof(_T) == 16) {
        return __interleave_hi(__a, __b);
    } else if constexpr (sizeof(_T) == 32 && needs_intrinsics) {
        if constexpr (_Trait::_S_width == 4) {
            return reinterpret_cast<_T>(
                _mm256_unpackhi_pd(__vector_bitcast<double>(__a), __vector_bitcast<double>(__b)));
        } else if constexpr (_Trait::_S_width == 8) {
            return reinterpret_cast<_T>(
                _mm256_unpackhi_ps(__vector_bitcast<float>(__a), __vector_bitcast<float>(__b)));
        } else if constexpr (_Trait::_S_width == 16) {
            return reinterpret_cast<_T>(
                _mm256_unpackhi_epi16(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
        } else if constexpr (_Trait::_S_width == 32) {
            return reinterpret_cast<_T>(
                _mm256_unpackhi_epi8(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
        }
    } else if constexpr (sizeof(_T) == 32) {
        if constexpr (_Trait::_S_width == 4) {
            return _T{__a[1], __b[1], __a[3], __b[3]};
        } else if constexpr (_Trait::_S_width == 8) {
            return _T{__a[2], __b[2], __a[3], __b[3], __a[6], __b[6], __a[7], __b[7]};
        } else if constexpr (_Trait::_S_width == 16) {
            return _T{__a[4],  __b[4],  __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],
                     __a[12], __b[12], __a[13], __b[13], __a[14], __b[14], __a[15], __b[15]};
        } else if constexpr (_Trait::_S_width == 32) {
            return _T{__a[8],  __b[8],  __a[9],  __b[9],  __a[10], __b[10], __a[11], __b[11],
                     __a[12], __b[12], __a[13], __b[13], __a[14], __b[14], __a[15], __b[15],
                     __a[24], __b[24], __a[25], __b[25], __a[26], __b[26], __a[27], __b[27],
                     __a[28], __b[28], __a[29], __b[29], __a[30], __b[30], __a[31], __b[31]};
        } else if constexpr (_Trait::_S_width == 64) {
            return _T{__a[16], __b[16], __a[17], __b[17], __a[18], __b[18], __a[19], __b[19], __a[20], __b[20],
                     __a[21], __b[21], __a[22], __b[22], __a[23], __b[23], __a[24], __b[24], __a[25], __b[25],
                     __a[26], __b[26], __a[27], __b[27], __a[28], __b[28], __a[29], __b[29], __a[30], __b[30],
                     __a[31], __b[31], __a[48], __b[48], __a[49], __b[49], __a[50], __b[50], __a[51], __b[51],
                     __a[52], __b[52], __a[53], __b[53], __a[54], __b[54], __a[55], __b[55], __a[56], __b[56],
                     __a[57], __b[57], __a[58], __b[58], __a[59], __b[59], __a[60], __b[60], __a[61], __b[61],
                     __a[62], __b[62], __a[63], __b[63]};
        } else {
            __assert_unreachable<_T>();
        }
    } else if constexpr (sizeof(_T) == 64 && needs_intrinsics) {
        if constexpr (_Trait::_S_width == 8) {
            return reinterpret_cast<_T>(
                _mm512_unpackhi_pd(__vector_bitcast<double>(__a), __vector_bitcast<double>(__b)));
        } else if constexpr (_Trait::_S_width == 16) {
            return reinterpret_cast<_T>(
                _mm512_unpackhi_ps(__vector_bitcast<float>(__a), __vector_bitcast<float>(__b)));
        } else if constexpr (_Trait::_S_width == 32) {
            return reinterpret_cast<_T>(
                _mm512_unpackhi_epi16(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
        } else if constexpr (_Trait::_S_width == 64) {
            return reinterpret_cast<_T>(
                _mm512_unpackhi_epi8(__vector_bitcast<__llong>(__a), __vector_bitcast<__llong>(__b)));
        }
    } else if constexpr (sizeof(_T) == 64) {
        if constexpr (_Trait::_S_width == 8) {
            return _T{__a[1], __b[1], __a[3], __b[3], __a[5], __b[5], __a[7], __b[7]};
        } else if constexpr (_Trait::_S_width == 16) {
            return _T{__a[2],  __b[2],  __a[3],  __b[3],  __a[6],  __b[6],  __a[7],  __b[7],
                     __a[10], __b[10], __a[11], __b[11], __a[14], __b[14], __a[15], __b[15]};
        } else if constexpr (_Trait::_S_width == 32) {
            return _T{__a[4],  __b[4],  __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],
                     __a[12], __b[12], __a[13], __b[13], __a[14], __b[14], __a[15], __b[15],
                     __a[20], __b[20], __a[21], __b[21], __a[22], __b[22], __a[23], __b[23],
                     __a[28], __b[28], __a[29], __b[29], __a[30], __b[30], __a[31], __b[31]};
        } else if constexpr (_Trait::_S_width == 64) {
            return _T{__a[8],  __b[8],  __a[9],  __b[9],  __a[10], __b[10], __a[11], __b[11], __a[12], __b[12],
                     __a[13], __b[13], __a[14], __b[14], __a[15], __b[15], __a[24], __b[24], __a[25], __b[25],
                     __a[26], __b[26], __a[27], __b[27], __a[28], __b[28], __a[29], __b[29], __a[30], __b[30],
                     __a[31], __b[31], __a[40], __b[40], __a[41], __b[41], __a[42], __b[42], __a[43], __b[43],
                     __a[44], __b[44], __a[45], __b[45], __a[46], __b[46], __a[47], __b[47], __a[56], __b[56],
                     __a[57], __b[57], __a[58], __b[58], __a[59], __b[59], __a[60], __b[60], __a[61], __b[61],
                     __a[62], __b[62], __a[63], __b[63]};
        } else {
            __assert_unreachable<_T>();
        }
    }
}

template <class _T> struct interleaved_pair {
    _T __lo, __hi;
};

template <class _A, class _B, class _T = std::common_type_t<_A, _B>,
          class _Trait = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr interleaved_pair<_T> interleave(_A __a, _B __b)
{
    return {__interleave_lo(__a, __b), __interleave_hi(__a, __b)};
}

template <class _A, class _B, class _T = std::common_type_t<_A, _B>,
          class _Trait = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr interleaved_pair<_T> interleave128(_A __a, _B __b)
{
    return {interleave128_lo(__a, __b), interleave128_hi(__a, __b)};
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
template <class _To, class _From> inline _To __convert_mask(_From __k) {
    if constexpr (std::is_same_v<_To, _From>) {  // also covers bool -> bool
        return __k;
    } else if constexpr (std::is_unsigned_v<_From> && std::is_unsigned_v<_To>) {
        // bits -> bits
        return __k;  // zero-extends or truncates
    } else if constexpr (__is_bitset_v<_From>) {
        // from std::bitset {{{
        static_assert(__k.size() <= sizeof(__ullong) * CHAR_BIT);
        using _T = std::conditional_t<
            (__k.size() <= sizeof(ushort) * CHAR_BIT),
            std::conditional_t<(__k.size() <= CHAR_BIT), __uchar, ushort>,
            std::conditional_t<(__k.size() <= sizeof(uint) * CHAR_BIT), uint, __ullong>>;
        return __convert_mask<_To>(static_cast<_T>(__k.to_ullong()));
        // }}}
    } else if constexpr (__is_bitset_v<_To>) {
        // to std::bitset {{{
        static_assert(_To().size() <= sizeof(__ullong) * CHAR_BIT);
        using _T = std::conditional_t<
            (_To().size() <= sizeof(ushort) * CHAR_BIT),
            std::conditional_t<(_To().size() <= CHAR_BIT), __uchar, ushort>,
            std::conditional_t<(_To().size() <= sizeof(uint) * CHAR_BIT), uint, __ullong>>;
        return __convert_mask<_T>(__k);
        // }}}
    } else if constexpr (__is_storage_v<_From>) {
        return __convert_mask<_To>(__k._M_data);
    } else if constexpr (__is_storage_v<_To>) {
        return __convert_mask<typename _To::register_type>(__k);
    } else if constexpr (std::is_unsigned_v<_From> && __is_vector_type_v<_To>) {
        // bits -> vector {{{
        using _Trait = __vector_traits<_To>;
        constexpr size_t N_in = sizeof(_From) * CHAR_BIT;
        using _ToT = typename _Trait::value_type;
        constexpr size_t N_out = _Trait::_S_width;
        constexpr size_t _N = std::min(N_in, N_out);
        constexpr size_t bytes_per_output_element = sizeof(_ToT);
        if constexpr (__have_avx512f) {
            if constexpr (bytes_per_output_element == 1 && sizeof(_To) == 16) {
                if constexpr (__have_avx512bw_vl) {
                    return __vector_bitcast<_ToT>(_mm_movm_epi8(__k));
                } else if constexpr (__have_avx512bw) {
                    return __vector_bitcast<_ToT>(__lo128(_mm512_movm_epi8(__k)));
                } else {
                    auto as32bits = _mm512_maskz_mov_epi32(__k, ~__m512i());
                    auto as16bits = __xzyw(
                        _mm256_packs_epi32(__lo256(as32bits), __hi256(as32bits)));
                    return __vector_bitcast<_ToT>(
                        _mm_packs_epi16(__lo128(as16bits), __hi128(as16bits)));
                }
            } else if constexpr (bytes_per_output_element == 1 && sizeof(_To) == 32) {
                if constexpr (__have_avx512bw_vl) {
                    return __vector_bitcast<_ToT>(_mm256_movm_epi8(__k));
                } else if constexpr (__have_avx512bw) {
                    return __vector_bitcast<_ToT>(__lo256(_mm512_movm_epi8(__k)));
                } else {
                    auto as16bits =  // 0 16 1 17 ... 15 31
                        _mm512_srli_epi32(_mm512_maskz_mov_epi32(__k, ~__m512i()), 16) |
                        _mm512_slli_epi32(_mm512_maskz_mov_epi32(__k >> 16, ~__m512i()),
                                          16);
                    auto _0_16_1_17 = __xzyw(_mm256_packs_epi16(
                        __lo256(as16bits),
                        __hi256(as16bits))  // 0 16 1 17 2 18 3 19 8 24 9 25 ...
                    );
                    // deinterleave:
                    return __vector_bitcast<_ToT>(__xzyw(_mm256_shuffle_epi8(
                        _0_16_1_17,  // 0 16 1 17 2 ...
                        _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13,
                                         15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11,
                                         13, 15))));  // 0-7 16-23 8-15 24-31 -> xzyw
                                                      // 0-3  8-11 16-19 24-27
                                                      // 4-7 12-15 20-23 28-31
                }
            } else if constexpr (bytes_per_output_element == 1 && sizeof(_To) == 64) {
                return reinterpret_cast<__vector_type_t<__schar, 64>>(_mm512_movm_epi8(__k));
            } else if constexpr (bytes_per_output_element == 2 && sizeof(_To) == 16) {
                if constexpr (__have_avx512bw_vl) {
                    return __vector_bitcast<_ToT>(_mm_movm_epi16(__k));
                } else if constexpr (__have_avx512bw) {
                    return __vector_bitcast<_ToT>(__lo128(_mm512_movm_epi16(__k)));
                } else {
                    __m256i as32bits;
                    if constexpr (__have_avx512vl) {
                        as32bits = _mm256_maskz_mov_epi32(__k, ~__m256i());
                    } else {
                        as32bits = __lo256(_mm512_maskz_mov_epi32(__k, ~__m512i()));
                    }
                    return __vector_bitcast<_ToT>(
                        _mm_packs_epi32(__lo128(as32bits), __hi128(as32bits)));
                }
            } else if constexpr (bytes_per_output_element == 2 && sizeof(_To) == 32) {
                if constexpr (__have_avx512bw_vl) {
                    return __vector_bitcast<_ToT>(_mm256_movm_epi16(__k));
                } else if constexpr (__have_avx512bw) {
                    return __vector_bitcast<_ToT>(__lo256(_mm512_movm_epi16(__k)));
                } else {
                    auto as32bits = _mm512_maskz_mov_epi32(__k, ~__m512i());
                    return __vector_bitcast<_ToT>(__xzyw(
                        _mm256_packs_epi32(__lo256(as32bits), __hi256(as32bits))));
                }
            } else if constexpr (bytes_per_output_element == 2 && sizeof(_To) == 64) {
                return __vector_bitcast<_ToT>(_mm512_movm_epi16(__k));
            } else if constexpr (bytes_per_output_element == 4 && sizeof(_To) == 16) {
                return __vector_bitcast<_ToT>(
                    __have_avx512dq_vl
                        ? _mm_movm_epi32(__k)
                        : __have_avx512dq
                              ? __lo128(_mm512_movm_epi32(__k))
                              : __have_avx512vl
                                    ? _mm_maskz_mov_epi32(__k, ~__m128i())
                                    : __lo128(_mm512_maskz_mov_epi32(__k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 4 && sizeof(_To) == 32) {
                return __vector_bitcast<_ToT>(
                    __have_avx512dq_vl
                        ? _mm256_movm_epi32(__k)
                        : __have_avx512dq
                              ? __lo256(_mm512_movm_epi32(__k))
                              : __have_avx512vl
                                    ? _mm256_maskz_mov_epi32(__k, ~__m256i())
                                    : __lo256(_mm512_maskz_mov_epi32(__k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 4 && sizeof(_To) == 64) {
                return __vector_bitcast<_ToT>(__have_avx512dq
                                             ? _mm512_movm_epi32(__k)
                                             : _mm512_maskz_mov_epi32(__k, ~__m512i()));
            } else if constexpr (bytes_per_output_element == 8 && sizeof(_To) == 16) {
                return __vector_bitcast<_ToT>(
                    __have_avx512dq_vl
                        ? _mm_movm_epi64(__k)
                        : __have_avx512dq
                              ? __lo128(_mm512_movm_epi64(__k))
                              : __have_avx512vl
                                    ? _mm_maskz_mov_epi64(__k, ~__m128i())
                                    : __lo128(_mm512_maskz_mov_epi64(__k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 8 && sizeof(_To) == 32) {
                return __vector_bitcast<_ToT>(
                    __have_avx512dq_vl
                        ? _mm256_movm_epi64(__k)
                        : __have_avx512dq
                              ? __lo256(_mm512_movm_epi64(__k))
                              : __have_avx512vl
                                    ? _mm256_maskz_mov_epi64(__k, ~__m256i())
                                    : __lo256(_mm512_maskz_mov_epi64(__k, ~__m512i())));
            } else if constexpr (bytes_per_output_element == 8 && sizeof(_To) == 64) {
                return __vector_bitcast<_ToT>(__have_avx512dq
                                             ? _mm512_movm_epi64(__k)
                                             : _mm512_maskz_mov_epi64(__k, ~__m512i()));
            } else {
                __assert_unreachable<_To>();
            }
        } else if constexpr (__have_sse) {
            using _U = std::make_unsigned_t<__int_for_sizeof_t<_ToT>>;
            using _V = __vector_type_t<_U, _N>;  // simd<_U, _Abi>;
            static_assert(sizeof(_V) <= 32);  // can't be AVX512
            constexpr size_t bits_per_element = sizeof(_U) * CHAR_BIT;
            if constexpr (!__have_avx2 && __have_avx && sizeof(_V) == 32) {
                if constexpr (_N == 8) {
                    return _mm256_cmp_ps(
                        _mm256_and_ps(
                            _mm256_castsi256_ps(_mm256_set1_epi32(__k)),
                            _mm256_castsi256_ps(_mm256_setr_epi32(
                                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80))),
                        _mm256_setzero_ps(), _CMP_NEQ_UQ);
                } else if constexpr (_N == 4) {
                    return _mm256_cmp_pd(
                        _mm256_and_pd(
                            _mm256_castsi256_pd(_mm256_set1_epi64x(__k)),
                            _mm256_castsi256_pd(
                                _mm256_setr_epi64x(0x01, 0x02, 0x04, 0x08))),
                        _mm256_setzero_pd(), _CMP_NEQ_UQ);
                } else {
                    __assert_unreachable<_To>();
                }
            } else if constexpr (bits_per_element >= _N) {
                constexpr auto bitmask = __generate_builtin<__vector_type_t<_U, _N>>(
                    [](auto __i) -> _U { return 1ull << __i; });
                return __vector_bitcast<_ToT>(
                    (__vector_broadcast<_N, _U>(__k) & bitmask) != 0);
            } else if constexpr (sizeof(_V) == 16 && sizeof(_ToT) == 1 && __have_ssse3) {
                const auto bitmask = __to_intrin(__make_builtin<__uchar>(
                    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128));
                return __vector_bitcast<_ToT>(
                    __vector_bitcast<_ToT>(
                        _mm_shuffle_epi8(
                            __to_intrin(__vector_type_t<__ullong, 2>{__k}),
                            _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                                          1)) &
                        bitmask) != 0);
            } else if constexpr (sizeof(_V) == 32 && sizeof(_ToT) == 1 && __have_avx2) {
                const auto bitmask =
                    _mm256_broadcastsi128_si256(__to_intrin(__make_builtin<__uchar>(
                        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128)));
                return __vector_bitcast<_ToT>(
                    __vector_bitcast<_ToT>(_mm256_shuffle_epi8(
                                        _mm256_broadcastsi128_si256(__to_intrin(
                                            __vector_type_t<__ullong, 2>{__k})),
                                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                                                         1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                                         2, 2, 3, 3, 3, 3, 3, 3, 3, 3)) &
                                    bitmask) != 0);
                /* TODO:
                } else if constexpr (sizeof(_V) == 32 && sizeof(_ToT) == 2 && __have_avx2) {
                    constexpr auto bitmask = _mm256_broadcastsi128_si256(
                        _mm_setr_epi8(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000)); return
                __vector_bitcast<_ToT>( _mm256_shuffle_epi8(
                                   _mm256_broadcastsi128_si256(__m128i{__k}),
                                   _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)) & bitmask) != 0;
                */
            } else {
                const _V tmp = __generate_builtin<_V>([&](auto __i) {
                                  return static_cast<_U>(
                                      __k >> (bits_per_element * (__i / bits_per_element)));
                              }) &
                              __generate_builtin<_V>([](auto __i) {
                                  return static_cast<_U>(1ull << (__i % bits_per_element));
                              });  // mask bit index
                return __vector_bitcast<_ToT>(tmp != _V());
            }
        } else {
            __assert_unreachable<_To>();
        } // }}}
    } else if constexpr (__is_vector_type_v<_From> && std::is_unsigned_v<_To>) {
        // vector -> bits {{{
        using _Trait = __vector_traits<_From>;
        using _T = typename _Trait::value_type;
        constexpr size_t FromN = _Trait::_S_width;
        constexpr size_t cvt_id = FromN * 10 + sizeof(_T);
        constexpr bool __have_avx512_int = __have_avx512f && std::is_integral_v<_T>;
        [[maybe_unused]]  // PR85827
        const auto intrin = __to_intrin(__k);

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
        else if constexpr (cvt_id ==  4'4 && __have_avx512dq_vl) { return    _mm_movepi32_mask(__vector_bitcast<__llong>(__k)); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512dq   ) { return _mm512_movepi32_mask(__zero_extend(__vector_bitcast<__llong>(__k))); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512vl   ) { return    _mm_cmp_epi32_mask(__vector_bitcast<__llong>(__k), __m128i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512_int ) { return _mm512_cmp_epi32_mask(__zero_extend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'4                    ) { return    _mm_movemask_ps(__k); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512dq_vl) { return _mm256_movepi32_mask(__vector_bitcast<__llong>(__k)); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512dq   ) { return _mm512_movepi32_mask(__zero_extend(__vector_bitcast<__llong>(__k))); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512vl   ) { return _mm256_cmp_epi32_mask(__vector_bitcast<__llong>(__k), __m256i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512_int ) { return _mm512_cmp_epi32_mask(__zero_extend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  8'4                    ) { return _mm256_movemask_ps(__k); }
        else if constexpr (cvt_id == 16'4 && __have_avx512dq   ) { return _mm512_movepi32_mask(__vector_bitcast<__llong>(__k)); }
        else if constexpr (cvt_id == 16'4                    ) { return _mm512_cmp_epi32_mask(__vector_bitcast<__llong>(__k), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512dq_vl) { return    _mm_movepi64_mask(__vector_bitcast<__llong>(__k)); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512dq   ) { return _mm512_movepi64_mask(__zero_extend(__vector_bitcast<__llong>(__k))); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512vl   ) { return    _mm_cmp_epi64_mask(__vector_bitcast<__llong>(__k), __m128i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512_int ) { return _mm512_cmp_epi64_mask(__zero_extend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8                    ) { return    _mm_movemask_pd(__k); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512dq_vl) { return _mm256_movepi64_mask(__vector_bitcast<__llong>(__k)); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512dq   ) { return _mm512_movepi64_mask(__zero_extend(__vector_bitcast<__llong>(__k))); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512vl   ) { return _mm256_cmp_epi64_mask(__vector_bitcast<__llong>(__k), __m256i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512_int ) { return _mm512_cmp_epi64_mask(__zero_extend(intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'8                    ) { return _mm256_movemask_pd(__k); }
        else if constexpr (cvt_id ==  8'8 && __have_avx512dq   ) { return _mm512_movepi64_mask(__vector_bitcast<__llong>(__k)); }
        else if constexpr (cvt_id ==  8'8                    ) { return _mm512_cmp_epi64_mask(__vector_bitcast<__llong>(__k), __m512i(), _MM_CMPINT_LT); }
        else { __assert_unreachable<_To>(); }
        // }}}
    } else if constexpr (__is_vector_type_v<_From> && __is_vector_type_v<_To>) {
        // vector -> vector {{{
        using ToTrait = __vector_traits<_To>;
        using FromTrait = __vector_traits<_From>;
        using _ToT = typename ToTrait::value_type;
        using _T = typename FromTrait::value_type;
        constexpr size_t FromN = FromTrait::_S_width;
        constexpr size_t ToN = ToTrait::_S_width;
        constexpr int FromBytes = sizeof(_T);
        constexpr int ToBytes = sizeof(_ToT);

        if constexpr (FromN == ToN && sizeof(_From) == sizeof(_To)) {
            // reinterpret the bits
            return reinterpret_cast<_To>(__k);
        } else if constexpr (sizeof(_To) == 16 && sizeof(__k) == 16) {
            // SSE -> SSE {{{
            if constexpr (FromBytes == 4 && ToBytes == 8) {
                if constexpr(std::is_integral_v<_T>) {
                    return __vector_bitcast<_ToT>(interleave128_lo(__k, __k));
                } else {
                    return __vector_bitcast<_ToT>(interleave128_lo(__k, __k));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 8) {
                const auto __y = __vector_bitcast<int>(interleave128_lo(__k, __k));
                return __vector_bitcast<_ToT>(interleave128_lo(__y, __y));
            } else if constexpr (FromBytes == 1 && ToBytes == 8) {
                auto __y = __vector_bitcast<short>(interleave128_lo(__k, __k));
                auto __z = __vector_bitcast<int>(interleave128_lo(__y, __y));
                return __vector_bitcast<_ToT>(interleave128_lo(__z, __z));
            } else if constexpr (FromBytes == 8 && ToBytes == 4) {
                if constexpr (std::is_floating_point_v<_T>) {
                    return __vector_bitcast<_ToT>(_mm_shuffle_ps(__vector_bitcast<float>(__k), __m128(),
                                                     __make_immediate<4>(1, 3, 1, 3)));
                } else {
                    auto __y = __vector_bitcast<__llong>(__k);
                    return __vector_bitcast<_ToT>(_mm_packs_epi32(__y, __m128i()));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 4) {
                return __vector_bitcast<_ToT>(interleave128_lo(__k, __k));
            } else if constexpr (FromBytes == 1 && ToBytes == 4) {
                const auto __y = __vector_bitcast<short>(interleave128_lo(__k, __k));
                return __vector_bitcast<_ToT>(interleave128_lo(__y, __y));
            } else if constexpr (FromBytes == 8 && ToBytes == 2) {
                if constexpr(__have_ssse3) {
                    return __vector_bitcast<_ToT>(
                        _mm_shuffle_epi8(__vector_bitcast<__llong>(__k),
                                         _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    const auto __y = _mm_packs_epi32(__vector_bitcast<__llong>(__k), __m128i());
                    return __vector_bitcast<_ToT>(_mm_packs_epi32(__y, __m128i()));
                }
            } else if constexpr (FromBytes == 4 && ToBytes == 2) {
                return __vector_bitcast<_ToT>(
                    _mm_packs_epi32(__vector_bitcast<__llong>(__k), __m128i()));
            } else if constexpr (FromBytes == 1 && ToBytes == 2) {
                return __vector_bitcast<_ToT>(interleave128_lo(__k, __k));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {
                if constexpr(__have_ssse3) {
                    return __vector_bitcast<_ToT>(
                        _mm_shuffle_epi8(__vector_bitcast<__llong>(__k),
                                         _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    auto __y = _mm_packs_epi32(__vector_bitcast<__llong>(__k), __m128i());
                    __y = _mm_packs_epi32(__y, __m128i());
                    return __vector_bitcast<_ToT>(_mm_packs_epi16(__y, __m128i()));
                }
            } else if constexpr (FromBytes == 4 && ToBytes == 1) {
                if constexpr(__have_ssse3) {
                    return __vector_bitcast<_ToT>(
                        _mm_shuffle_epi8(__vector_bitcast<__llong>(__k),
                                         _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    const auto __y = _mm_packs_epi32(__vector_bitcast<__llong>(__k), __m128i());
                    return __vector_bitcast<_ToT>(_mm_packs_epi16(__y, __m128i()));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 1) {
                return __vector_bitcast<_ToT>(_mm_packs_epi16(__vector_bitcast<__llong>(__k), __m128i()));
            } else {
                static_assert(!std::is_same_v<_T, _T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(_To) == 32 && sizeof(__k) == 32) {
            // AVX -> AVX {{{
            if constexpr (FromBytes == ToBytes) {  // keep low 1/2
                static_assert(!std::is_same_v<_T, _T>, "should be unreachable");
            } else if constexpr (FromBytes == ToBytes * 2) {
                const auto __y = __vector_bitcast<__llong>(__k);
                return __vector_bitcast<_ToT>(
                    _mm256_castsi128_si256(_mm_packs_epi16(__lo128(__y), __hi128(__y))));
            } else if constexpr (FromBytes == ToBytes * 4) {
                const auto __y = __vector_bitcast<__llong>(__k);
                return __vector_bitcast<_ToT>(_mm256_castsi128_si256(
                    _mm_packs_epi16(_mm_packs_epi16(__lo128(__y), __hi128(__y)), __m128i())));
            } else if constexpr (FromBytes == ToBytes * 8) {
                const auto __y = __vector_bitcast<__llong>(__k);
                return __vector_bitcast<_ToT>(_mm256_castsi128_si256(
                    _mm_shuffle_epi8(_mm_packs_epi16(__lo128(__y), __hi128(__y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1))));
            } else if constexpr (FromBytes * 2 == ToBytes) {
                auto __y = __xzyw(__to_intrin(__k));
                if constexpr(std::is_floating_point_v<_T>) {
                    return __vector_bitcast<_ToT>(_mm256_unpacklo_ps(__y, __y));
                } else {
                    return __vector_bitcast<_ToT>(_mm256_unpacklo_epi8(__y, __y));
                }
            } else if constexpr (FromBytes * 4 == ToBytes) {
                auto __y = _mm_unpacklo_epi8(__lo128(__vector_bitcast<__llong>(__k)),
                                           __lo128(__vector_bitcast<__llong>(__k)));  // drops 3/4 of input
                return __vector_bitcast<_ToT>(
                    __concat(_mm_unpacklo_epi16(__y, __y), _mm_unpackhi_epi16(__y, __y)));
            } else if constexpr (FromBytes == 1 && ToBytes == 8) {
                auto __y = _mm_unpacklo_epi8(__lo128(__vector_bitcast<__llong>(__k)),
                                           __lo128(__vector_bitcast<__llong>(__k)));  // drops 3/4 of input
                __y = _mm_unpacklo_epi16(__y, __y);  // drops another 1/2 => 7/8 total
                return __vector_bitcast<_ToT>(
                    __concat(_mm_unpacklo_epi32(__y, __y), _mm_unpackhi_epi32(__y, __y)));
            } else {
                static_assert(!std::is_same_v<_T, _T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(_To) == 32 && sizeof(__k) == 16) {
            // SSE -> AVX {{{
            if constexpr (FromBytes == ToBytes) {
                return __vector_bitcast<_ToT>(
                    __intrinsic_type_t<_T, 32 / sizeof(_T)>(__zero_extend(__to_intrin(__k))));
            } else if constexpr (FromBytes * 2 == ToBytes) {  // keep all
                return __vector_bitcast<_ToT>(__concat(_mm_unpacklo_epi8(__vector_bitcast<__llong>(__k), __vector_bitcast<__llong>(__k)),
                                         _mm_unpackhi_epi8(__vector_bitcast<__llong>(__k), __vector_bitcast<__llong>(__k))));
            } else if constexpr (FromBytes * 4 == ToBytes) {
                if constexpr (__have_avx2) {
                    return __vector_bitcast<_ToT>(_mm256_shuffle_epi8(
                        __concat(__vector_bitcast<__llong>(__k), __vector_bitcast<__llong>(__k)),
                        _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7,
                                         7)));
                } else {
                    return __vector_bitcast<_ToT>(
                        __concat(_mm_shuffle_epi8(__vector_bitcast<__llong>(__k),
                                                _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2,
                                                              2, 2, 2, 3, 3, 3, 3)),
                               _mm_shuffle_epi8(__vector_bitcast<__llong>(__k),
                                                _mm_setr_epi8(4, 4, 4, 4, 5, 5, 5, 5, 6,
                                                              6, 6, 6, 7, 7, 7, 7))));
                }
            } else if constexpr (FromBytes * 8 == ToBytes) {
                if constexpr (__have_avx2) {
                    return __vector_bitcast<_ToT>(_mm256_shuffle_epi8(
                        __concat(__vector_bitcast<__llong>(__k), __vector_bitcast<__llong>(__k)),
                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                                         3)));
                } else {
                    return __vector_bitcast<_ToT>(
                        __concat(_mm_shuffle_epi8(__vector_bitcast<__llong>(__k),
                                                _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                              1, 1, 1, 1, 1, 1, 1)),
                               _mm_shuffle_epi8(__vector_bitcast<__llong>(__k),
                                                _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3,
                                                              3, 3, 3, 3, 3, 3, 3))));
                }
            } else if constexpr (FromBytes == ToBytes * 2) {
                return __vector_bitcast<_ToT>(
                    __m256i(__zero_extend(_mm_packs_epi16(__vector_bitcast<__llong>(__k), __m128i()))));
            } else if constexpr (FromBytes == 8 && ToBytes == 2) {
                return __vector_bitcast<_ToT>(__m256i(__zero_extend(
                    _mm_shuffle_epi8(__vector_bitcast<__llong>(__k),
                                     _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else if constexpr (FromBytes == 4 && ToBytes == 1) {
                return __vector_bitcast<_ToT>(__m256i(__zero_extend(
                    _mm_shuffle_epi8(__vector_bitcast<__llong>(__k),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {
                return __vector_bitcast<_ToT>(__m256i(__zero_extend(
                    _mm_shuffle_epi8(__vector_bitcast<__llong>(__k),
                                     _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else {
                static_assert(!std::is_same_v<_T, _T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(_To) == 16 && sizeof(__k) == 32) {
            // AVX -> SSE {{{
            if constexpr (FromBytes == ToBytes) {  // keep low 1/2
                return __vector_bitcast<_ToT>(__lo128(__k));
            } else if constexpr (FromBytes == ToBytes * 2) {  // keep all
                auto __y = __vector_bitcast<__llong>(__k);
                return __vector_bitcast<_ToT>(_mm_packs_epi16(__lo128(__y), __hi128(__y)));
            } else if constexpr (FromBytes == ToBytes * 4) {  // add 1/2 undef
                auto __y = __vector_bitcast<__llong>(__k);
                return __vector_bitcast<_ToT>(
                    _mm_packs_epi16(_mm_packs_epi16(__lo128(__y), __hi128(__y)), __m128i()));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {  // add 3/4 undef
                auto __y = __vector_bitcast<__llong>(__k);
                return __vector_bitcast<_ToT>(
                    _mm_shuffle_epi8(_mm_packs_epi16(__lo128(__y), __hi128(__y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)));
            } else if constexpr (FromBytes * 2 == ToBytes) {  // keep low 1/4
                auto __y = __lo128(__vector_bitcast<__llong>(__k));
                return __vector_bitcast<_ToT>(_mm_unpacklo_epi8(__y, __y));
            } else if constexpr (FromBytes * 4 == ToBytes) {  // keep low 1/8
                auto __y = __lo128(__vector_bitcast<__llong>(__k));
                __y = _mm_unpacklo_epi8(__y, __y);
                return __vector_bitcast<_ToT>(_mm_unpacklo_epi8(__y, __y));
            } else if constexpr (FromBytes * 8 == ToBytes) {  // keep low 1/16
                auto __y = __lo128(__vector_bitcast<__llong>(__k));
                __y = _mm_unpacklo_epi8(__y, __y);
                __y = _mm_unpacklo_epi8(__y, __y);
                return __vector_bitcast<_ToT>(_mm_unpacklo_epi8(__y, __y));
            } else {
                static_assert(!std::is_same_v<_T, _T>, "should be unreachable");
            }
            // }}}
        }
        // }}}
    } else {
        __assert_unreachable<_To>();
    }
}

// }}}

template <class _Abi> struct __simd_math_fallback {  //{{{
    template <class _T> simd<_T, _Abi> __acos(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::acos(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __asin(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::asin(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __atan(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::atan(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __atan2(const simd<_T, _Abi> &__x, const simd<_T, _Abi> &__y)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::atan2(__x[__i], __y[__i]); });
    }

    template <class _T> simd<_T, _Abi> __cos(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::cos(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __sin(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::sin(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __tan(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::tan(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __acosh(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::acosh(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __asinh(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::asinh(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __atanh(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::atanh(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __cosh(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::cosh(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __sinh(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::sinh(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __tanh(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::tanh(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __exp(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::exp(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __exp2(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::exp2(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __expm1(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::expm1(__x[__i]); });
    }

    template <class _T>
    simd<_T, _Abi> __frexp(const simd<_T, _Abi> &__x,
                         fixed_size_simd<int, simd_size_v<_T, _Abi>> &exp)
    {
        return simd<_T, _Abi>([&](auto __i) {
            int tmp;
            _T __r = std::frexp(__x[__i], &tmp);
            exp[__i] = tmp;
            return __r;
        });
    }

    template <class _T>
    simd<_T, _Abi> __ldexp(const simd<_T, _Abi> &__x,
                         const fixed_size_simd<int, simd_size_v<_T, _Abi>> &exp)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::ldexp(__x[__i], exp[__i]); });
    }

    template <class _T>
    fixed_size_simd<int, simd_size_v<_T, _Abi>> __ilogb(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::ilogb(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __log(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::log(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __log10(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::log10(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __log1p(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::log1p(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __log2(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::log2(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __logb(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::logb(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __modf(const simd<_T, _Abi> &__x, simd<_T, _Abi> &__y)
    {
        return simd<_T, _Abi>([&](auto __i) {
            _T tmp;
            _T __r = std::modf(__x[__i], &tmp);
            __y[__i] = tmp;
            return __r;
        });
    }

    template <class _T>
    simd<_T, _Abi> __scalbn(const simd<_T, _Abi> &__x,
                          const fixed_size_simd<int, simd_size_v<_T, _Abi>> &__y)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::scalbn(__x[__i], __y[__i]); });
    }

    template <class _T>
    simd<_T, _Abi> __scalbln(const simd<_T, _Abi> &__x,
                           const fixed_size_simd<long, simd_size_v<_T, _Abi>> &__y)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::scalbln(__x[__i], __y[__i]); });
    }

    template <class _T> simd<_T, _Abi> __cbrt(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::cbrt(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __abs(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::abs(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __fabs(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::fabs(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __pow(const simd<_T, _Abi> &__x, const simd<_T, _Abi> &__y)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::pow(__x[__i], __y[__i]); });
    }

    template <class _T> simd<_T, _Abi> __sqrt(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::sqrt(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __erf(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::erf(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __erfc(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::erfc(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __lgamma(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::lgamma(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __tgamma(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::tgamma(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __ceil(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::ceil(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __floor(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::floor(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __nearbyint(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::nearbyint(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __rint(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::rint(__x[__i]); });
    }

    template <class _T>
    fixed_size_simd<long, simd_size_v<_T, _Abi>> __lrint(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::lrint(__x[__i]); });
    }

    template <class _T>
    fixed_size_simd<long long, simd_size_v<_T, _Abi>> __llrint(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::llrint(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __round(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::round(__x[__i]); });
    }

    template <class _T>
    fixed_size_simd<long, simd_size_v<_T, _Abi>> __lround(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::lround(__x[__i]); });
    }

    template <class _T>
    fixed_size_simd<long long, simd_size_v<_T, _Abi>> __llround(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::llround(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __trunc(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::trunc(__x[__i]); });
    }

    template <class _T> simd<_T, _Abi> __fmod(const simd<_T, _Abi> &__x, const simd<_T, _Abi> &__y)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::fmod(__x[__i], __y[__i]); });
    }

    template <class _T>
    simd<_T, _Abi> __remainder(const simd<_T, _Abi> &__x, const simd<_T, _Abi> &__y)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::remainder(__x[__i], __y[__i]); });
    }

    template <class _T>
    simd<_T, _Abi> __remquo(const simd<_T, _Abi> &__x, const simd<_T, _Abi> &__y,
                          fixed_size_simd<int, simd_size_v<_T, _Abi>> &__z)
    {
        return simd<_T, _Abi>([&](auto __i) {
            int tmp;
            _T __r = std::remquo(__x[__i], __y[__i], &tmp);
            __z[__i] = tmp;
            return __r;
        });
    }

    template <class _T>
    simd<_T, _Abi> __copysign(const simd<_T, _Abi> &__x, const simd<_T, _Abi> &__y)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::copysign(__x[__i], __y[__i]); });
    }

    template <class _T>
    simd<_T, _Abi> __nextafter(const simd<_T, _Abi> &__x, const simd<_T, _Abi> &__y)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::nextafter(__x[__i], __y[__i]); });
    }

    template <class _T> simd<_T, _Abi> __fdim(const simd<_T, _Abi> &__x, const simd<_T, _Abi> &__y)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::fdim(__x[__i], __y[__i]); });
    }

    template <class _T> simd<_T, _Abi> __fmax(const simd<_T, _Abi> &__x, const simd<_T, _Abi> &__y)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::fmax(__x[__i], __y[__i]); });
    }

    template <class _T> simd<_T, _Abi> __fmin(const simd<_T, _Abi> &__x, const simd<_T, _Abi> &__y)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::fmin(__x[__i], __y[__i]); });
    }

    template <class _T>
    simd<_T, _Abi> __fma(const simd<_T, _Abi> &__x, const simd<_T, _Abi> &__y,
                       const simd<_T, _Abi> &__z)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::fma(__x[__i], __y[__i], __z[__i]); });
    }

    template <class _T>
    fixed_size_simd<int, simd_size_v<_T, _Abi>> __fpclassify(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::fpclassify(__x[__i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isfinite(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::isfinite(__x[__i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isinf(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::isinf(__x[__i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isnan(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::isnan(__x[__i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isnormal(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::isnormal(__x[__i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __signbit(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::signbit(__x[__i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isgreater(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::isgreater(__x[__i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isgreaterequal(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::isgreaterequal(__x[__i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isless(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::isless(__x[__i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __islessequal(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::islessequal(__x[__i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __islessgreater(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::islessgreater(__x[__i]); });
    }

    template <class _T> simd_mask<_T, _Abi> __isunordered(const simd<_T, _Abi> &__x)
    {
        return simd<_T, _Abi>([&](auto __i) { return std::isunordered(__x[__i]); });
    }
};  // }}}
// __scalar_simd_impl {{{
struct __scalar_simd_impl : __simd_math_fallback<simd_abi::scalar> {
    // member types {{{2
    using abi = std::experimental::simd_abi::scalar;
    using _Mask_member_type = bool;
    template <class _T> using _Simd_member_type = _T;
    template <class _T> using simd = std::experimental::simd<_T, abi>;
    template <class _T> using simd_mask = std::experimental::simd_mask<_T, abi>;
    template <class _T> using __type_tag = _T *;

    // broadcast {{{2
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static constexpr _T __broadcast(_T __x) noexcept
    {
        return __x;
    }

    // generator {{{2
    template <class _F, class _T>
    _GLIBCXX_SIMD_INTRINSIC static _T generator(_F &&gen, __type_tag<_T>)
    {
        return gen(__size_constant<0>());
    }

    // load {{{2
    template <class _T, class _U, class _F>
    static inline _T load(const _U *mem, _F, __type_tag<_T>) noexcept
    {
        return static_cast<_T>(mem[0]);
    }

    // masked load {{{2
    template <class _T, class _U, class _F>
    static inline _T masked_load(_T merge, bool __k, const _U *mem, _F) noexcept
    {
        if (__k) {
            merge = static_cast<_T>(mem[0]);
        }
        return merge;
    }

    // store {{{2
    template <class _T, class _U, class _F>
    static inline void store(_T __v, _U *mem, _F, __type_tag<_T>) noexcept
    {
        mem[0] = static_cast<_T>(__v);
    }

    // masked store {{{2
    template <class _T, class _U, class _F>
    static inline void masked_store(const _T __v, _U *mem, _F, const bool __k) noexcept
    {
        if (__k) {
            mem[0] = __v;
        }
    }

    // negation {{{2
    template <class _T> static inline bool negate(_T __x) noexcept { return !__x; }

    // reductions {{{2
    template <class _T, class _BinaryOperation>
    static inline _T reduce(const simd<_T> &__x, _BinaryOperation &)
    {
        return __x._M_data;
    }

    // min, max, clamp {{{2
    template <class _T> static inline _T min(const _T __a, const _T __b)
    {
        return std::min(__a, __b);
    }

    template <class _T> static inline _T max(const _T __a, const _T __b)
    {
        return std::max(__a, __b);
    }

    // complement {{{2
    template <class _T> static inline _T complement(_T __x) noexcept
    {
        return static_cast<_T>(~__x);
    }

    // unary minus {{{2
    template <class _T> static inline _T unary_minus(_T __x) noexcept
    {
        return static_cast<_T>(-__x);
    }

    // arithmetic operators {{{2
    template <class _T> static inline _T plus(_T __x, _T __y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(__x) +
                              __promote_preserving_unsigned(__y));
    }

    template <class _T> static inline _T minus(_T __x, _T __y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(__x) -
                              __promote_preserving_unsigned(__y));
    }

    template <class _T> static inline _T multiplies(_T __x, _T __y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(__x) *
                              __promote_preserving_unsigned(__y));
    }

    template <class _T> static inline _T divides(_T __x, _T __y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(__x) /
                              __promote_preserving_unsigned(__y));
    }

    template <class _T> static inline _T modulus(_T __x, _T __y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(__x) %
                              __promote_preserving_unsigned(__y));
    }

    template <class _T> static inline _T bit_and(_T __x, _T __y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(__x) &
                              __promote_preserving_unsigned(__y));
    }
    static inline float bit_and(float __x, float __y)
    {
        static_assert(sizeof(float) == sizeof(uint), "");
        const uint __r = reinterpret_cast<const __may_alias<uint> &>(__x) &
                       reinterpret_cast<const __may_alias<uint> &>(__y);
        return reinterpret_cast<const __may_alias<float> &>(__r);
    }
    static inline double bit_and(double __x, double __y)
    {
        static_assert(sizeof(double) == sizeof(__ullong), "");
        const __ullong __r = reinterpret_cast<const __may_alias<__ullong> &>(__x) &
                         reinterpret_cast<const __may_alias<__ullong> &>(__y);
        return reinterpret_cast<const __may_alias<double> &>(__r);
    }

    template <class _T> static inline _T bit_or(_T __x, _T __y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(__x) |
                              __promote_preserving_unsigned(__y));
    }
    static inline float bit_or(float __x, float __y)
    {
        static_assert(sizeof(float) == sizeof(uint), "");
        const uint __r = reinterpret_cast<const __may_alias<uint> &>(__x) |
                       reinterpret_cast<const __may_alias<uint> &>(__y);
        return reinterpret_cast<const __may_alias<float> &>(__r);
    }
    static inline double bit_or(double __x, double __y)
    {
        static_assert(sizeof(double) == sizeof(__ullong), "");
        const __ullong __r = reinterpret_cast<const __may_alias<__ullong> &>(__x) |
                         reinterpret_cast<const __may_alias<__ullong> &>(__y);
        return reinterpret_cast<const __may_alias<double> &>(__r);
    }


    template <class _T> static inline _T bit_xor(_T __x, _T __y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(__x) ^
                              __promote_preserving_unsigned(__y));
    }

    template <class _T> static inline _T bit_shift_left(_T __x, int __y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(__x) << __y);
    }

    template <class _T> static inline _T bit_shift_right(_T __x, int __y)
    {
        return static_cast<_T>(__promote_preserving_unsigned(__x) >> __y);
    }

    // math {{{2
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static _T __abs(_T __x) { return _T(std::abs(__x)); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static _T __sqrt(_T __x) { return std::sqrt(__x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static _T __trunc(_T __x) { return std::trunc(__x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static _T __floor(_T __x) { return std::floor(__x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static _T __ceil(_T __x) { return std::ceil(__x); }

    template <class _T> _GLIBCXX_SIMD_INTRINSIC static __simd_tuple<int, abi> __fpclassify(_T __x)
    {
        return {std::fpclassify(__x)};
    }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static bool __isfinite(_T __x) { return std::isfinite(__x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static bool __isinf(_T __x) { return std::isinf(__x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static bool __isnan(_T __x) { return std::isnan(__x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static bool __isnormal(_T __x) { return std::isnormal(__x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static bool __signbit(_T __x) { return std::signbit(__x); }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static bool __isunordered(_T __x, _T __y) { return std::isunordered(__x, __y); }

    // __increment & __decrement{{{2
    template <class _T> static inline void __increment(_T &__x) { ++__x; }
    template <class _T> static inline void __decrement(_T &__x) { --__x; }

    // compares {{{2
    template <class _T> static bool equal_to(_T __x, _T __y) { return __x == __y; }
    template <class _T> static bool not_equal_to(_T __x, _T __y) { return __x != __y; }
    template <class _T> static bool less(_T __x, _T __y) { return __x < __y; }
    template <class _T> static bool greater(_T __x, _T __y) { return __x > __y; }
    template <class _T> static bool less_equal(_T __x, _T __y) { return __x <= __y; }
    template <class _T> static bool greater_equal(_T __x, _T __y) { return __x >= __y; }

    // smart_reference access {{{2
    template <class _T, class _U> static void set(_T &__v, int __i, _U &&__x) noexcept
    {
        _GLIBCXX_SIMD_ASSERT(__i == 0);
        __unused(__i);
        __v = std::forward<_U>(__x);
    }

    // masked_assign {{{2
    template <typename _T> _GLIBCXX_SIMD_INTRINSIC static void masked_assign(bool __k, _T &lhs, _T rhs)
    {
        if (__k) {
            lhs = rhs;
        }
    }

    // __masked_cassign {{{2
    template <template <typename> class _Op, typename _T>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_cassign(const bool __k, _T &lhs, const _T rhs)
    {
        if (__k) {
            lhs = _Op<_T>{}(lhs, rhs);
        }
    }

    // masked_unary {{{2
    template <template <typename> class _Op, typename _T>
    _GLIBCXX_SIMD_INTRINSIC static _T masked_unary(const bool __k, const _T __v)
    {
        return static_cast<_T>(__k ? _Op<_T>{}(__v) : __v);
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
    template <class _F>
    _GLIBCXX_SIMD_INTRINSIC static bool masked_load(bool merge, bool mask, const bool *mem,
                                         _F) noexcept
    {
        if (mask) {
            merge = mem[0];
        }
        return merge;
    }

    // store {{{2
    template <class _F> _GLIBCXX_SIMD_INTRINSIC static void store(bool __v, bool *mem, _F) noexcept
    {
        mem[0] = __v;
    }

    // masked store {{{2
    template <class _F>
    _GLIBCXX_SIMD_INTRINSIC static void masked_store(const bool __v, bool *mem, _F,
                                          const bool __k) noexcept
    {
        if (__k) {
            mem[0] = __v;
        }
    }

    // logical and bitwise operators {{{2
    static constexpr bool logical_and(bool __x, bool __y) { return __x && __y; }
    static constexpr bool logical_or(bool __x, bool __y) { return __x || __y; }
    static constexpr bool bit_and(bool __x, bool __y) { return __x && __y; }
    static constexpr bool bit_or(bool __x, bool __y) { return __x || __y; }
    static constexpr bool bit_xor(bool __x, bool __y) { return __x != __y; }

    // smart_reference access {{{2
    static void set(bool &__k, int __i, bool __x) noexcept
    {
        _GLIBCXX_SIMD_ASSERT(__i == 0);
        __unused(__i);
        __k = __x;
    }

    // masked_assign {{{2
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(bool __k, bool &lhs, bool rhs)
    {
        if (__k) {
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
    using _Simd_member_type = typename _Abi::template __traits<_T>::_Simd_member_type;
    template <class _T>
    using _Mask_member_type = typename _Abi::template __traits<_T>::_Mask_member_type;
    template <class _T> static constexpr size_t full_size = _Simd_member_type<_T>::_S_width;

    // make_simd(__storage/__intrinsic_type_t) {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static simd<_T, _Abi> make_simd(__storage<_T, _N> __x)
    {
        return {__private_init, __x};
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static simd<_T, _Abi> make_simd(__intrinsic_type_t<_T, _N> __x)
    {
        return {__private_init, __vector_bitcast<_T>(__x)};
    }

    // broadcast {{{2
    template <class _T>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Simd_member_type<_T> __broadcast(_T __x) noexcept
    {
        return __vector_broadcast<full_size<_T>>(__x);
    }

    // generator {{{2
    template <class _F, class _T>
    _GLIBCXX_SIMD_INTRINSIC static _Simd_member_type<_T> generator(_F &&gen, __type_tag<_T>)
    {
        return __generate_storage<_T, full_size<_T>>(std::forward<_F>(gen));
    }

    // load {{{2
    template <class _T, class _U, class _F>
    _GLIBCXX_SIMD_INTRINSIC static _Simd_member_type<_T> load(const _U *mem, _F,
                                                 __type_tag<_T>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        constexpr size_t _N = _Simd_member_type<_T>::_S_width;
        constexpr size_t max_load_size =
            (sizeof(_U) >= 4 && __have_avx512f) || __have_avx512bw
                ? 64
                : (std::is_floating_point_v<_U> && __have_avx) || __have_avx2 ? 32 : 16;
        if constexpr (sizeof(_U) > 8) {
            return __generate_storage<_T, _N>(
                [&](auto __i) { return static_cast<_T>(mem[__i]); });
        } else if constexpr (std::is_same_v<_U, _T>) {
            return __vector_load<_U, _N>(mem, _F());
        } else if constexpr (sizeof(_U) * _N < 16) {
            return __convert<_Simd_member_type<_T>>(
                __vector_load16<_U, sizeof(_U) * _N>(mem, _F()));
        } else if constexpr (sizeof(_U) * _N <= max_load_size) {
            return __convert<_Simd_member_type<_T>>(__vector_load<_U, _N>(mem, _F()));
        } else if constexpr (sizeof(_U) * _N == 2 * max_load_size) {
            return __convert<_Simd_member_type<_T>>(
                __vector_load<_U, _N / 2>(mem, _F()),
                __vector_load<_U, _N / 2>(mem + _N / 2, _F()));
        } else if constexpr (sizeof(_U) * _N == 4 * max_load_size) {
            return __convert<_Simd_member_type<_T>>(
                __vector_load<_U, _N / 4>(mem, _F()),
                __vector_load<_U, _N / 4>(mem + 1 * _N / 4, _F()),
                __vector_load<_U, _N / 4>(mem + 2 * _N / 4, _F()),
                __vector_load<_U, _N / 4>(mem + 3 * _N / 4, _F()));
        } else if constexpr (sizeof(_U) * _N == 8 * max_load_size) {
            return __convert<_Simd_member_type<_T>>(
                __vector_load<_U, _N / 8>(mem, _F()),
                __vector_load<_U, _N / 8>(mem + 1 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(mem + 2 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(mem + 3 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(mem + 4 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(mem + 5 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(mem + 6 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(mem + 7 * _N / 8, _F()));
        } else {
            __assert_unreachable<_T>();
        }
    }

    // masked load {{{2
    template <class _T, size_t _N, class _U, class _F>
    static inline __storage<_T, _N> masked_load(__storage<_T, _N> __merge,
                                                _Mask_member_type<_T> __k,
                                                const _U *__mem,
                                                _F) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        if constexpr (std::is_same_v<_T, _U> ||  // no conversion
                      (sizeof(_T) == sizeof(_U) &&
                       std::is_integral_v<_T> ==
                           std::is_integral_v<_U>)  // conversion via bit reinterpretation
        ) {
            [[maybe_unused]] const auto __intrin = __to_intrin(__merge);
            constexpr bool __have_avx512bw_vl_or_zmm =
                __have_avx512bw_vl || (__have_avx512bw && sizeof(__merge) == 64);
            if constexpr (__have_avx512bw_vl_or_zmm && sizeof(_T) == 1) {
                if constexpr (sizeof(__merge) == 16) {
                    __merge = __vector_bitcast<_T>(
                        _mm_mask_loadu_epi8(__intrin, _mm_movemask_epi8(__to_intrin(__k)), __mem));
                } else if constexpr (sizeof(__merge) == 32) {
                    __merge = __vector_bitcast<_T>(_mm256_mask_loadu_epi8(
                        __intrin, _mm256_movemask_epi8(__to_intrin(__k)), __mem));
                } else if constexpr (sizeof(__merge) == 64) {
                    __merge = __vector_bitcast<_T>(
                        _mm512_mask_loadu_epi8(__intrin, __k, __mem));
                } else {
                    __assert_unreachable<_T>();
                }
            } else if constexpr (__have_avx512bw_vl_or_zmm && sizeof(_T) == 2) {
                if constexpr (sizeof(__merge) == 16) {
                    __merge = __vector_bitcast<_T>(
                        _mm_mask_loadu_epi16(__intrin, movemask_epi16(__to_intrin(__k)), __mem));
                } else if constexpr (sizeof(__merge) == 32) {
                    __merge = __vector_bitcast<_T>(
                        _mm256_mask_loadu_epi16(__intrin, movemask_epi16(__to_intrin(__k)), __mem));
                } else if constexpr (sizeof(__merge) == 64) {
                    __merge = __vector_bitcast<_T>(
                        _mm512_mask_loadu_epi16(__intrin, __k, __mem));
                } else {
                    __assert_unreachable<_T>();
                }
            } else if constexpr (__have_avx2 && sizeof(_T) == 4 &&
                                 std::is_integral_v<_U>) {
                if constexpr (sizeof(__merge) == 16) {
                    __merge = (~__k._M_data & __merge._M_data) |
                              __vector_bitcast<_T>(_mm_maskload_epi32(
                                  reinterpret_cast<const int *>(__mem), __to_intrin(__k)));
                } else if constexpr (sizeof(__merge) == 32) {
                    __merge = (~__k._M_data & __merge._M_data) |
                              __vector_bitcast<_T>(_mm256_maskload_epi32(
                                  reinterpret_cast<const int *>(__mem), __to_intrin(__k)));
                } else if constexpr (__have_avx512f && sizeof(__merge) == 64) {
                    __merge = __vector_bitcast<_T>(
                        _mm512_mask_loadu_epi32(__intrin, __k, __mem));
                } else {
                    __assert_unreachable<_T>();
                }
            } else if constexpr (__have_avx && sizeof(_T) == 4) {
                if constexpr (sizeof(__merge) == 16) {
                    __merge = __or(__andnot(__k._M_data, __merge._M_data),
                                   __vector_bitcast<_T>(_mm_maskload_ps(
                                       reinterpret_cast<const float *>(__mem),
                                       __vector_bitcast<__llong>(__k))));
                } else if constexpr (sizeof(__merge) == 32) {
                    __merge =
                        __or(__andnot(__k._M_data, __merge._M_data),
                             _mm256_maskload_ps(reinterpret_cast<const float *>(__mem),
                                                __vector_bitcast<__llong>(__k)));
                } else if constexpr (__have_avx512f && sizeof(__merge) == 64) {
                    __merge =
                        __vector_bitcast<_T>(_mm512_mask_loadu_ps(__intrin, __k, __mem));
                } else {
                    __assert_unreachable<_T>();
                }
            } else if constexpr (__have_avx2 && sizeof(_T) == 8 &&
                                 std::is_integral_v<_U>) {
                if constexpr (sizeof(__merge) == 16) {
                    __merge = (~__k._M_data & __merge._M_data) |
                              __vector_bitcast<_T>(_mm_maskload_epi64(
                                  reinterpret_cast<const __llong *>(__mem), __to_intrin(__k)));
                } else if constexpr (sizeof(__merge) == 32) {
                    __merge = (~__k._M_data & __merge._M_data) |
                              __vector_bitcast<_T>(_mm256_maskload_epi64(
                                  reinterpret_cast<const __llong *>(__mem), __to_intrin(__k)));
                } else if constexpr (__have_avx512f && sizeof(__merge) == 64) {
                    __merge = __vector_bitcast<_T>(
                        _mm512_mask_loadu_epi64(__intrin, __k, __mem));
                } else {
                    __assert_unreachable<_T>();
                }
            } else if constexpr (__have_avx && sizeof(_T) == 8) {
                if constexpr (sizeof(__merge) == 16) {
                    __merge = __or(__andnot(__k._M_data, __merge._M_data),
                                   __vector_bitcast<_T>(_mm_maskload_pd(
                                       reinterpret_cast<const double *>(__mem),
                                       __vector_bitcast<__llong>(__k))));
                } else if constexpr (sizeof(__merge) == 32) {
                    __merge =
                        __or(__andnot(__k._M_data, __merge._M_data),
                             _mm256_maskload_pd(reinterpret_cast<const double *>(__mem),
                                                __vector_bitcast<__llong>(__k)));
                } else if constexpr (__have_avx512f && sizeof(__merge) == 64) {
                    __merge =
                        __vector_bitcast<_T>(_mm512_mask_loadu_pd(__intrin, __k, __mem));
                } else {
                    __assert_unreachable<_T>();
                }
            } else {
                __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(), [&](auto __i) {
                    __merge.set(__i, static_cast<_T>(__mem[__i]));
                });
            }
        } else if constexpr (sizeof(_U) <= 8 &&  // no long double
                             !__converts_via_decomposition_v<
                                 _U, _T, sizeof(__merge)>  // conversion via decomposition
                                                           // is better handled via the
                                                           // bit_iteration fallback below
        ) {
            // TODO: copy pattern from masked_store, which doesn't resort to fixed_size
            using _A = simd_abi::deduce_t<
                _U, std::max(_N, 16 / sizeof(_U))  // _N or more, so that at least a 16
                                                   // Byte vector is used instead of a
                                                   // fixed_size filled with scalars
                >;
            using _ATraits = __simd_traits<_U, _A>;
            using AImpl = typename _ATraits::_Simd_impl_type;
            typename _ATraits::_Simd_member_type uncvted{};
            typename _ATraits::_Mask_member_type kk;
            if constexpr (__is_fixed_size_abi_v<_A>) {
                kk = __vector_to_bitset(__k._M_data);
            } else {
                kk = __convert_mask<typename _ATraits::_Mask_member_type>(__k);
            }
            uncvted = AImpl::masked_load(uncvted, kk, __mem, _F());
            __simd_converter<_U, _A, _T, _Abi> converter;
            masked_assign(__k, __merge, converter(uncvted));
        } else {
            __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(),
                            [&](auto __i) { __merge.set(__i, static_cast<_T>(__mem[__i])); });
        }
        return __merge;
    }

    // store {{{2
    template <class _T, class _U, class _F>
    _GLIBCXX_SIMD_INTRINSIC static void store(_Simd_member_type<_T> __v, _U *mem, _F,
                                   __type_tag<_T>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        // TODO: converting int -> "smaller int" can be optimized with AVX512
        constexpr size_t _N = _Simd_member_type<_T>::_S_width;
        constexpr size_t max_store_size =
            (sizeof(_U) >= 4 && __have_avx512f) || __have_avx512bw
                ? 64
                : (std::is_floating_point_v<_U> && __have_avx) || __have_avx2 ? 32 : 16;
        if constexpr (sizeof(_U) > 8) {
            __execute_n_times<_N>([&](auto __i) { mem[__i] = __v[__i]; });
        } else if constexpr (std::is_same_v<_U, _T>) {
            __vector_store(__v._M_data, mem, _F());
        } else if constexpr (sizeof(_U) * _N < 16) {
            __vector_store<sizeof(_U) * _N>(__convert<__vector_type16_t<_U>>(__v),
                                                 mem, _F());
        } else if constexpr (sizeof(_U) * _N <= max_store_size) {
            __vector_store(__convert<__vector_type_t<_U, _N>>(__v), mem, _F());
        } else {
            constexpr size_t VSize = max_store_size / sizeof(_U);
            constexpr size_t stores = _N / VSize;
            using _V = __vector_type_t<_U, VSize>;
            const std::array<_V, stores> converted = __convert_all<_V>(__v);
            __execute_n_times<stores>([&](auto __i) {
                __vector_store(converted[__i], mem + __i * VSize, _F());
            });
        }
    }

    // masked store {{{2
    template <class _T, size_t _N, class _U, class _F>
    static inline void masked_store(const __storage<_T, _N> __v, _U *__mem, _F,
                                    const _Mask_member_type<_T> __k) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        [[maybe_unused]] const auto __vi = __to_intrin(__v);
        constexpr size_t max_store_size =
            (sizeof(_U) >= 4 && __have_avx512f) || __have_avx512bw
                ? 64
                : (std::is_floating_point_v<_U> && __have_avx) || __have_avx2 ? 32 : 16;
        if constexpr (std::is_same_v<_T, _U> ||
                      (std::is_integral_v<_T> && std::is_integral_v<_U> &&
                       sizeof(_T) == sizeof(_U))) {
            // bitwise or no conversion, reinterpret:
            const auto kk = [&]() {
                if constexpr (__is_bitmask_v<decltype(__k)>) {
                    return _Mask_member_type<_U>(__k._M_data);
                } else {
                    return __storage_bitcast<_U>(__k);
                }
            }();
            __maskstore(__storage_bitcast<_U>(__v), __mem, _F(), kk);
        } else if constexpr (std::is_integral_v<_T> && std::is_integral_v<_U> &&
                             sizeof(_T) > sizeof(_U) && __have_avx512f &&
                             (sizeof(_T) >= 4 || __have_avx512bw) &&
                             (sizeof(__v) == 64 || __have_avx512vl)) {  // truncating store
            const auto kk = [&]() {
                if constexpr (__is_bitmask_v<decltype(__k)>) {
                    return __k;
                } else {
                    return __convert_mask<__storage<bool, _N>>(__k);
                }
            }();
            if constexpr (sizeof(_T) == 8 && sizeof(_U) == 4) {
                if constexpr (sizeof(__vi) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi32(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi32(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 16) {
                    _mm_mask_cvtepi64_storeu_epi32(__mem, kk, __vi);
                }
            } else if constexpr (sizeof(_T) == 8 && sizeof(_U) == 2) {
                if constexpr (sizeof(__vi) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi16(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi16(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 16) {
                    _mm_mask_cvtepi64_storeu_epi16(__mem, kk, __vi);
                }
            } else if constexpr (sizeof(_T) == 8 && sizeof(_U) == 1) {
                if constexpr (sizeof(__vi) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi8(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi8(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 16) {
                    _mm_mask_cvtepi64_storeu_epi8(__mem, kk, __vi);
                }
            } else if constexpr (sizeof(_T) == 4 && sizeof(_U) == 2) {
                if constexpr (sizeof(__vi) == 64) {
                    _mm512_mask_cvtepi32_storeu_epi16(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 32) {
                    _mm256_mask_cvtepi32_storeu_epi16(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 16) {
                    _mm_mask_cvtepi32_storeu_epi16(__mem, kk, __vi);
                }
            } else if constexpr (sizeof(_T) == 4 && sizeof(_U) == 1) {
                if constexpr (sizeof(__vi) == 64) {
                    _mm512_mask_cvtepi32_storeu_epi8(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 32) {
                    _mm256_mask_cvtepi32_storeu_epi8(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 16) {
                    _mm_mask_cvtepi32_storeu_epi8(__mem, kk, __vi);
                }
            } else if constexpr (sizeof(_T) == 2 && sizeof(_U) == 1) {
                if constexpr (sizeof(__vi) == 64) {
                    _mm512_mask_cvtepi16_storeu_epi8(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 32) {
                    _mm256_mask_cvtepi16_storeu_epi8(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 16) {
                    _mm_mask_cvtepi16_storeu_epi8(__mem, kk, __vi);
                }
            } else {
                __assert_unreachable<_T>();
            }
        } else if constexpr (sizeof(_U) <= 8 &&  // no long double
                             !__converts_via_decomposition_v<
                                 _T, _U, max_store_size>  // conversion via decomposition
                                                          // is better handled via the
                                                          // bit_iteration fallback below
        ) {
            using _VV = __storage<_U, std::clamp(_N, 16 / sizeof(_U), max_store_size / sizeof(_U))>;
            using _V = typename _VV::register_type;
            constexpr bool prefer_bitmask =
                (__have_avx512f && sizeof(_U) >= 4) || __have_avx512bw;
            using _M = __storage<std::conditional_t<prefer_bitmask, bool, _U>, _VV::_S_width>;
            constexpr size_t VN = __vector_traits<_V>::_S_width;

            if constexpr (VN >= _N) {
                __maskstore(_VV(__convert<_V>(__v)), __mem,
                               // careful, if _V has more elements than the input __v (_N),
                               // vector_aligned is incorrect:
                               std::conditional_t<(__vector_traits<_V>::_S_width > _N),
                                                  overaligned_tag<sizeof(_U) * _N>, _F>(),
                               __convert_mask<_M>(__k));
            } else if constexpr (VN * 2 == _N) {
                const std::array<_V, 2> converted = __convert_all<_V>(__v);
                __maskstore(_VV(converted[0]), __mem, _F(), __convert_mask<_M>(__extract_part<0, 2>(__k)));
                __maskstore(_VV(converted[1]), __mem + _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<1, 2>(__k)));
            } else if constexpr (VN * 4 == _N) {
                const std::array<_V, 4> converted = __convert_all<_V>(__v);
                __maskstore(_VV(converted[0]), __mem, _F(), __convert_mask<_M>(__extract_part<0, 4>(__k)));
                __maskstore(_VV(converted[1]), __mem + 1 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<1, 4>(__k)));
                __maskstore(_VV(converted[2]), __mem + 2 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<2, 4>(__k)));
                __maskstore(_VV(converted[3]), __mem + 3 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<3, 4>(__k)));
            } else if constexpr (VN * 8 == _N) {
                const std::array<_V, 8> converted = __convert_all<_V>(__v);
                __maskstore(_VV(converted[0]), __mem, _F(), __convert_mask<_M>(__extract_part<0, 8>(__k)));
                __maskstore(_VV(converted[1]), __mem + 1 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<1, 8>(__k)));
                __maskstore(_VV(converted[2]), __mem + 2 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<2, 8>(__k)));
                __maskstore(_VV(converted[3]), __mem + 3 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<3, 8>(__k)));
                __maskstore(_VV(converted[4]), __mem + 4 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<4, 8>(__k)));
                __maskstore(_VV(converted[5]), __mem + 5 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<5, 8>(__k)));
                __maskstore(_VV(converted[6]), __mem + 6 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<6, 8>(__k)));
                __maskstore(_VV(converted[7]), __mem + 7 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<7, 8>(__k)));
            } else {
                __assert_unreachable<_T>();
            }
        } else {
            __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(),
                            [&](auto __i) { __mem[__i] = static_cast<_U>(__v[__i]); });
        }
    }

    // complement {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> complement(__storage<_T, _N> __x) noexcept
    {
        return ~__x._M_data;
    }

    // unary minus {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> unary_minus(__storage<_T, _N> __x) noexcept
    {
        // GCC doesn't use the psign instructions, but pxor & psub seem to be just as good
        // a choice as pcmpeqd & psign. So meh.
        return -__x._M_data;
    }

    // arithmetic operators {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> plus(__storage<_T, _N> __x,
                                                                    __storage<_T, _N> __y)
    {
        return __plus(__x, __y);
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> minus(__storage<_T, _N> __x,
                                                                     __storage<_T, _N> __y)
    {
        return __minus(__x, __y);
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> multiplies(
        __storage<_T, _N> __x, __storage<_T, _N> __y)
    {
        return __multiplies(__x, __y);
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> divides(
        __storage<_T, _N> __x, __storage<_T, _N> __y)
    {
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX4
        return __divides(__x._M_data, __y._M_data);
#else
        return __x._M_data / __y._M_data;
#endif
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> modulus(__storage<_T, _N> __x, __storage<_T, _N> __y)
    {
        static_assert(std::is_integral<_T>::value, "modulus is only supported for integral types");
        return __x._M_data % __y._M_data;
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_and(__storage<_T, _N> __x, __storage<_T, _N> __y)
    {
        return __vector_bitcast<_T>(__vector_bitcast<__llong>(__x._M_data) & __vector_bitcast<__llong>(__y._M_data));
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_or(__storage<_T, _N> __x, __storage<_T, _N> __y)
    {
        return __vector_bitcast<_T>(__vector_bitcast<__llong>(__x._M_data) | __vector_bitcast<__llong>(__y._M_data));
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_xor(__storage<_T, _N> __x, __storage<_T, _N> __y)
    {
        return __vector_bitcast<_T>(__vector_bitcast<__llong>(__x._M_data) ^ __vector_bitcast<__llong>(__y._M_data));
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> bit_shift_left(__storage<_T, _N> __x, __storage<_T, _N> __y)
    {
        return __x._M_data << __y._M_data;
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> bit_shift_right(__storage<_T, _N> __x, __storage<_T, _N> __y)
    {
        return __x._M_data >> __y._M_data;
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_shift_left(__storage<_T, _N> __x, int __y)
    {
        return __x._M_data << __y;
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_shift_right(__storage<_T, _N> __x,
                                                                         int __y)
    {
        return __x._M_data >> __y;
    }

    // compares {{{2
    // equal_to {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Mask_member_type<_T> equal_to(
        __storage<_T, _N> __x, __storage<_T, _N> __y)
    {
        [[maybe_unused]] const auto __xi = __to_intrin(__x);
        [[maybe_unused]] const auto __yi = __to_intrin(__y);
        if constexpr (sizeof(__x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<_T>) {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmp_pd_mask(__xi, __yi, _CMP_EQ_OQ);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmp_ps_mask(__xi, __yi, _CMP_EQ_OQ);
                } else { __assert_unreachable<_T>(); }
            } else {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmpeq_epi64_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmpeq_epi32_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 2) { return _mm512_cmpeq_epi16_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 1) { return _mm512_cmpeq_epi8_mask(__xi, __yi);
                } else { __assert_unreachable<_T>(); }
            }
        } else {
            return _To_storage(__x._M_data == __y._M_data);
        }
    }

    // not_equal_to {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Mask_member_type<_T> not_equal_to(
        __storage<_T, _N> __x, __storage<_T, _N> __y)
    {
        [[maybe_unused]] const auto __xi = __to_intrin(__x);
        [[maybe_unused]] const auto __yi = __to_intrin(__y);
        if constexpr (sizeof(__x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<_T>) {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmp_pd_mask(__xi, __yi, _CMP_NEQ_UQ);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmp_ps_mask(__xi, __yi, _CMP_NEQ_UQ);
                } else { __assert_unreachable<_T>(); }
            } else {
                       if constexpr (sizeof(_T) == 8) { return ~_mm512_cmpeq_epi64_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 4) { return ~_mm512_cmpeq_epi32_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 2) { return ~_mm512_cmpeq_epi16_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 1) { return ~_mm512_cmpeq_epi8_mask(__xi, __yi);
                } else { __assert_unreachable<_T>(); }
            }
        } else {
            return _To_storage(__x._M_data != __y._M_data);
        }
    }

    // less {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Mask_member_type<_T> less(__storage<_T, _N> __x,
                                                           __storage<_T, _N> __y)
    {
        [[maybe_unused]] const auto __xi = __to_intrin(__x);
        [[maybe_unused]] const auto __yi = __to_intrin(__y);
        if constexpr (sizeof(__x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<_T>) {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmp_pd_mask(__xi, __yi, _CMP_LT_OS);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmp_ps_mask(__xi, __yi, _CMP_LT_OS);
                } else { __assert_unreachable<_T>(); }
            } else if constexpr (std::is_signed_v<_T>) {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmplt_epi64_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmplt_epi32_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 2) { return _mm512_cmplt_epi16_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 1) { return _mm512_cmplt_epi8_mask(__xi, __yi);
                } else { __assert_unreachable<_T>(); }
            } else {
                static_assert(std::is_unsigned_v<_T>);
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmplt_epu64_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmplt_epu32_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 2) { return _mm512_cmplt_epu16_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 1) { return _mm512_cmplt_epu8_mask(__xi, __yi);
                } else { __assert_unreachable<_T>(); }
            }
        } else {
            return _To_storage(__x._M_data < __y._M_data);
        }
    }

    // less_equal {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Mask_member_type<_T> less_equal(__storage<_T, _N> __x,
                                                                 __storage<_T, _N> __y)
    {
        [[maybe_unused]] const auto __xi = __to_intrin(__x);
        [[maybe_unused]] const auto __yi = __to_intrin(__y);
        if constexpr (sizeof(__x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<_T>) {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmp_pd_mask(__xi, __yi, _CMP_LE_OS);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmp_ps_mask(__xi, __yi, _CMP_LE_OS);
                } else { __assert_unreachable<_T>(); }
            } else if constexpr (std::is_signed_v<_T>) {
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmple_epi64_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmple_epi32_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 2) { return _mm512_cmple_epi16_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 1) { return _mm512_cmple_epi8_mask(__xi, __yi);
                } else { __assert_unreachable<_T>(); }
            } else {
                static_assert(std::is_unsigned_v<_T>);
                       if constexpr (sizeof(_T) == 8) { return _mm512_cmple_epu64_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 4) { return _mm512_cmple_epu32_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 2) { return _mm512_cmple_epu16_mask(__xi, __yi);
                } else if constexpr (sizeof(_T) == 1) { return _mm512_cmple_epu8_mask(__xi, __yi);
                } else { __assert_unreachable<_T>(); }
            }
        } else {
            return _To_storage(__x._M_data <= __y._M_data);
        }
    }

    // negation {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Mask_member_type<_T> negate(__storage<_T, _N> __x) noexcept
    {
        if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
            return equal_to(__x, _Simd_member_type<_T>());
        } else {
            return _To_storage(!__x._M_data);
        }
    }

    // min, max, clamp {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> min(__storage<_T, _N> __a,
                                                                   __storage<_T, _N> __b)
    {
        return __a._M_data < __b._M_data ? __a._M_data : __b._M_data;
    }
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> max(__storage<_T, _N> __a,
                                                                   __storage<_T, _N> __b)
    {
        return __a._M_data > __b._M_data ? __a._M_data : __b._M_data;
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr std::pair<__storage<_T, _N>, __storage<_T, _N>>
    minmax(__storage<_T, _N> __a, __storage<_T, _N> __b)
    {
        return {__a._M_data < __b._M_data ? __a._M_data : __b._M_data, __a._M_data < __b._M_data ? __b._M_data : __a._M_data};
    }

    // reductions {{{2
    template <class _T, class _BinaryOperation>
    _GLIBCXX_SIMD_INTRINSIC static _T reduce(simd<_T, _Abi> __x,
                                             _BinaryOperation&& __binary_op)
    {
        constexpr size_t _N = simd_size_v<_T, _Abi>;
        if constexpr (sizeof(__x) > 16) {
            using _A = simd_abi::deduce_t<_T, _N / 2>;
            using _V = std::experimental::simd<_T, _A>;
            return __simd_traits<_T, _A>::_Simd_impl_type::reduce(
                __binary_op(_V(__private_init, __extract<0, 2>(__data(__x)._M_data)),
                            _V(__private_init, __extract<1, 2>(__data(__x)._M_data))),
                std::forward<_BinaryOperation>(__binary_op));
        } else {
            auto __intrin = __to_intrin(__x._M_data);
            if constexpr (_N == 16) {
                __x =
                    __binary_op(make_simd<_T, _N>(_mm_unpacklo_epi8(__intrin, __intrin)),
                                make_simd<_T, _N>(_mm_unpackhi_epi8(__intrin, __intrin)));
                __intrin = __to_intrin(__x._M_data);
            }
            if constexpr (_N >= 8) {
                __x = __binary_op(
                    make_simd<_T, _N>(_mm_unpacklo_epi16(__intrin, __intrin)),
                    make_simd<_T, _N>(_mm_unpackhi_epi16(__intrin, __intrin)));
                __intrin = __to_intrin(__x._M_data);
            }
            if constexpr (_N >= 4) {
                using _U = std::conditional_t<std::is_floating_point_v<_T>, float, int>;
                const auto __y = __vector_bitcast<_U>(__intrin);
                __x            = __binary_op(
                    __x, make_simd<_T, _N>(_To_storage(
                             __vector_type_t<_U, 4>{__y[3], __y[2], __y[1], __y[0]})));
                __intrin = __to_intrin(__x._M_data);
            }
            const auto __y = __vector_bitcast<__llong>(__intrin);
            return __binary_op(__x, make_simd<_T, _N>(_To_storage(
                                        __vector_type_t<__llong, 2>{__y[1], __y[1]})))[0];
        }
    }

    // math {{{2
    // sqrt {{{3
    template <class _T, size_t _N> _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> __sqrt(__storage<_T, _N> __x)
    {
               if constexpr (__is_sse_ps   <_T, _N>()) { return _mm_sqrt_ps(__x);
        } else if constexpr (__is_sse_pd   <_T, _N>()) { return _mm_sqrt_pd(__x);
        } else if constexpr (__is_avx_ps   <_T, _N>()) { return _mm256_sqrt_ps(__x);
        } else if constexpr (__is_avx_pd   <_T, _N>()) { return _mm256_sqrt_pd(__x);
        } else if constexpr (__is_avx512_ps<_T, _N>()) { return _mm512_sqrt_ps(__x);
        } else if constexpr (__is_avx512_pd<_T, _N>()) { return _mm512_sqrt_pd(__x);
        } else { __assert_unreachable<_T>(); }
    }

    // abs {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> __abs(__storage<_T, _N> __x) noexcept
    {
        return std::experimental::parallelism_v2::__abs(__x);
    }

    // trunc {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> __trunc(__storage<_T, _N> __x)
    {
        if constexpr (__is_avx512_ps<_T, _N>()) {
            return _mm512_roundscale_round_ps(__x, 0x03, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx512_pd<_T, _N>()) {
            return _mm512_roundscale_round_pd(__x, 0x03, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx_ps<_T, _N>()) {
            return _mm256_round_ps(__x, 0x3);
        } else if constexpr (__is_avx_pd<_T, _N>()) {
            return _mm256_round_pd(__x, 0x3);
        } else if constexpr (__have_sse4_1 && __is_sse_ps<_T, _N>()) {
            return _mm_round_ps(__x, 0x3);
        } else if constexpr (__have_sse4_1 && __is_sse_pd<_T, _N>()) {
            return _mm_round_pd(__x, 0x3);
        } else if constexpr (__is_sse_ps<_T, _N>()) {
            auto truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(__x));
            const auto no_fractional_values = __vector_bitcast<float>(
                __vector_bitcast<int>(__vector_bitcast<uint>(__x._M_data) & 0x7f800000u) <
                0x4b000000);  // the exponent is so large that no mantissa bits signify
                              // fractional values (0x3f8 + 23*8 = 0x4b0)
            return __blend(no_fractional_values, __x, truncated);
        } else if constexpr (__is_sse_pd<_T, _N>()) {
            const auto abs_x = __abs(__x)._M_data;
            const auto min_no_fractional_bits = __vector_bitcast<double>(
                __vector_broadcast<2>(0x4330'0000'0000'0000ull));  // 0x3ff + 52 = 0x433
            __vector_type16_t<double> truncated =
                (abs_x + min_no_fractional_bits) - min_no_fractional_bits;
            // due to rounding, the result can be too large. In this case `truncated >
            // abs(__x)` holds, so subtract 1 to truncated if `abs(__x) < truncated`
            truncated -=
                __and(__vector_bitcast<double>(abs_x < truncated), __vector_broadcast<2>(1.));
            // finally, fix the sign bit:
            return __or(
                __and(__vector_bitcast<double>(__vector_broadcast<2>(0x8000'0000'0000'0000ull)),
                     __x),
                truncated);
        } else {
            __assert_unreachable<_T>();
        }
    }

    // floor {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> __floor(__storage<_T, _N> __x)
    {
        if constexpr (__is_avx512_ps<_T, _N>()) {
            return _mm512_roundscale_round_ps(__x, 0x01, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx512_pd<_T, _N>()) {
            return _mm512_roundscale_round_pd(__x, 0x01, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx_ps<_T, _N>()) {
            return _mm256_round_ps(__x, 0x1);
        } else if constexpr (__is_avx_pd<_T, _N>()) {
            return _mm256_round_pd(__x, 0x1);
        } else if constexpr (__have_sse4_1 && __is_sse_ps<_T, _N>()) {
            return _mm_floor_ps(__x);
        } else if constexpr (__have_sse4_1 && __is_sse_pd<_T, _N>()) {
            return _mm_floor_pd(__x);
        } else {
            const auto __y = __trunc(__x)._M_data;
            const auto negative_input = __vector_bitcast<_T>(__x._M_data < __vector_broadcast<_N, _T>(0));
            const auto mask = __andnot(__vector_bitcast<_T>(__y == __x._M_data), negative_input);
            return __or(__andnot(mask, __y), __and(mask, __y - __vector_broadcast<_N, _T>(1)));
        }
    }

    // ceil {{{3
    template <class _T, size_t _N> _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> __ceil(__storage<_T, _N> __x)
    {
        if constexpr (__is_avx512_ps<_T, _N>()) {
            return _mm512_roundscale_round_ps(__x, 0x02, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx512_pd<_T, _N>()) {
            return _mm512_roundscale_round_pd(__x, 0x02, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx_ps<_T, _N>()) {
            return _mm256_round_ps(__x, 0x2);
        } else if constexpr (__is_avx_pd<_T, _N>()) {
            return _mm256_round_pd(__x, 0x2);
        } else if constexpr (__have_sse4_1 && __is_sse_ps<_T, _N>()) {
            return _mm_ceil_ps(__x);
        } else if constexpr (__have_sse4_1 && __is_sse_pd<_T, _N>()) {
            return _mm_ceil_pd(__x);
        } else {
            const auto __y = __trunc(__x)._M_data;
            const auto negative_input = __vector_bitcast<_T>(__x._M_data < __vector_broadcast<_N, _T>(0));
            const auto inv_mask = __or(__vector_bitcast<_T>(__y == __x._M_data), negative_input);
            return __or(__and(inv_mask, __y),
                       __andnot(inv_mask, __y + __vector_broadcast<_N, _T>(1)));
        }
    }

    // isnan {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_T> __isnan(__storage<_T, _N> __x)
    {
             if constexpr (__is_sse_ps   <_T, _N>()) { return _mm_cmpunord_ps(__x, __x); }
        else if constexpr (__is_avx_ps   <_T, _N>()) { return _mm256_cmp_ps(__x, __x, _CMP_UNORD_Q); }
        else if constexpr (__is_avx512_ps<_T, _N>()) { return _mm512_cmp_ps_mask(__x, __x, _CMP_UNORD_Q); }
        else if constexpr (__is_sse_pd   <_T, _N>()) { return _mm_cmpunord_pd(__x, __x); }
        else if constexpr (__is_avx_pd   <_T, _N>()) { return _mm256_cmp_pd(__x, __x, _CMP_UNORD_Q); }
        else if constexpr (__is_avx512_pd<_T, _N>()) { return _mm512_cmp_pd_mask(__x, __x, _CMP_UNORD_Q); }
        else { __assert_unreachable<_T>(); }
    }

    // isfinite {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_T> __isfinite(__storage<_T, _N> __x)
    {
        return __cmpord(__x._M_data, __x._M_data * _T());
    }

    // isunordered {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_T> __isunordered(__storage<_T, _N> __x,
                                                          __storage<_T, _N> __y)
    {
        return __cmpunord(__x._M_data, __y._M_data);
    }

    // signbit {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_T> __signbit(__storage<_T, _N> __x)
    {
        using _I = __int_for_sizeof_t<_T>;
        if constexpr (__have_avx512dq && __is_avx512_ps<_T, _N>()) {
            return _mm512_movepi32_mask(__vector_bitcast<__llong>(__x));
        } else if constexpr (__have_avx512dq && __is_avx512_pd<_T, _N>()) {
            return _mm512_movepi64_mask(__vector_bitcast<__llong>(__x));
        } else if constexpr (sizeof(__x) == 64) {
            const auto signmask = __vector_broadcast<_N>(std::numeric_limits<_I>::min());
            return equal_to(__storage<_I, _N>(__vector_bitcast<_I>(__x._M_data) & signmask),
                            __storage<_I, _N>(signmask));
        } else {
            const auto __xx = __vector_bitcast<_I>(__x._M_data);
            constexpr _I signmask = std::numeric_limits<_I>::min();
            if constexpr ((sizeof(_T) == 4 && (__have_avx2 || sizeof(__x) == 16)) ||
                          __have_avx512vl) {
                (void)signmask;
                return __vector_bitcast<_T>(__xx >> std::numeric_limits<_I>::digits);
            } else if constexpr ((__have_avx2 || (__have_ssse3 && sizeof(__x) == 16))) {
                return __vector_bitcast<_T>((__xx & signmask) == signmask);
            } else {  // SSE2/3 or AVX (w/o AVX2)
                constexpr auto one = __vector_broadcast<_N, _T>(1);
                return __vector_bitcast<_T>(
                    __vector_bitcast<_T>((__xx & signmask) | __vector_bitcast<_I>(one))  // -1 or 1
                    != one);
            }
        }
    }

    // isnonzerovalue (isnormal | is subnormal == !isinf & !isnan & !is zero) {{{3
    template <class _T> _GLIBCXX_SIMD_INTRINSIC static auto isnonzerovalue(_T __x)
    {
        using _U = typename __vector_traits<_T>::value_type;
        return __cmpord(__x * std::numeric_limits<_U>::infinity(),  // NaN if __x == 0
                        __x * _U()                                  // NaN if __x == inf
        );
    }

    template <class _T> _GLIBCXX_SIMD_INTRINSIC static auto isnonzerovalue_mask(_T __x)
    {
        using _U = typename __vector_traits<_T>::value_type;
        constexpr size_t _N = __vector_traits<_T>::_S_width;
        const auto __a = __x * std::numeric_limits<_U>::infinity();  // NaN if __x == 0
        const auto __b = __x * _U();                                 // NaN if __x == inf
        if constexpr (__have_avx512vl && __is_sse_ps<_U, _N>()) {
            return _mm_cmp_ps_mask(__a, __b, _CMP_ORD_Q);
        } else if constexpr (__have_avx512f && __is_sse_ps<_U, _N>()) {
            return __mmask8(0xf &
                            _mm512_cmp_ps_mask(__auto_bitcast(__a), __auto_bitcast(__b), _CMP_ORD_Q));
        } else if constexpr (__have_avx512vl && __is_sse_pd<_U, _N>()) {
            return _mm_cmp_pd_mask(__a, __b, _CMP_ORD_Q);
        } else if constexpr (__have_avx512f && __is_sse_pd<_U, _N>()) {
            return __mmask8(0x3 &
                            _mm512_cmp_pd_mask(__auto_bitcast(__a), __auto_bitcast(__b), _CMP_ORD_Q));
        } else if constexpr (__have_avx512vl && __is_avx_ps<_U, _N>()) {
            return _mm256_cmp_ps_mask(__a, __b, _CMP_ORD_Q);
        } else if constexpr (__have_avx512f && __is_avx_ps<_U, _N>()) {
            return __mmask8(_mm512_cmp_ps_mask(__auto_bitcast(__a), __auto_bitcast(__b), _CMP_ORD_Q));
        } else if constexpr (__have_avx512vl && __is_avx_pd<_U, _N>()) {
            return _mm256_cmp_pd_mask(__a, __b, _CMP_ORD_Q);
        } else if constexpr (__have_avx512f && __is_avx_pd<_U, _N>()) {
            return __mmask8(0xf &
                            _mm512_cmp_pd_mask(__auto_bitcast(__a), __auto_bitcast(__b), _CMP_ORD_Q));
        } else if constexpr (__is_avx512_ps<_U, _N>()) {
            return _mm512_cmp_ps_mask(__a, __b, _CMP_ORD_Q);
        } else if constexpr (__is_avx512_pd<_U, _N>()) {
            return _mm512_cmp_pd_mask(__a, __b, _CMP_ORD_Q);
        } else {
            __assert_unreachable<_T>();
        }
    }

    // isinf {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_T> __isinf(__storage<_T, _N> __x)
    {
        if constexpr (__is_avx512_pd<_T, _N>()) {
            if constexpr (__have_avx512dq) {
                return _mm512_fpclass_pd_mask(__x, 0x08) | _mm512_fpclass_pd_mask(__x, 0x10);
            } else {
                return _mm512_cmp_epi64_mask(__vector_bitcast<__llong>(__abs(__x)),
                                             __vector_broadcast<_N>(0x7ff0000000000000ll),
                                             _CMP_EQ_OQ);
            }
        } else if constexpr (__is_avx512_ps<_T, _N>()) {
            if constexpr (__have_avx512dq) {
                return _mm512_fpclass_ps_mask(__x, 0x08) | _mm512_fpclass_ps_mask(__x, 0x10);
            } else {
                return _mm512_cmp_epi32_mask(__vector_bitcast<__llong>(__abs(__x)),
                                             __auto_bitcast(__vector_broadcast<_N>(0x7f800000u)),
                                             _CMP_EQ_OQ);
            }
        } else if constexpr (__have_avx512dq_vl) {
            if constexpr (__is_sse_pd<_T, _N>()) {
                return __vector_bitcast<double>(_mm_movm_epi64(_mm_fpclass_pd_mask(__x, 0x08) |
                                                           _mm_fpclass_pd_mask(__x, 0x10)));
            } else if constexpr (__is_avx_pd<_T, _N>()) {
                return __vector_bitcast<double>(_mm256_movm_epi64(
                    _mm256_fpclass_pd_mask(__x, 0x08) | _mm256_fpclass_pd_mask(__x, 0x10)));
            } else if constexpr (__is_sse_ps<_T, _N>()) {
                return __vector_bitcast<float>(_mm_movm_epi32(_mm_fpclass_ps_mask(__x, 0x08) |
                                                          _mm_fpclass_ps_mask(__x, 0x10)));
            } else if constexpr (__is_avx_ps<_T, _N>()) {
                return __vector_bitcast<float>(_mm256_movm_epi32(
                    _mm256_fpclass_ps_mask(__x, 0x08) | _mm256_fpclass_ps_mask(__x, 0x10)));
            } else {
                __assert_unreachable<_T>();
            }
        } else {
            // compares to inf using the corresponding integer type
            return __vector_bitcast<_T>(__vector_bitcast<__int_for_sizeof_t<_T>>(__abs(__x)._M_data) ==
                                   __vector_bitcast<__int_for_sizeof_t<_T>>(__vector_broadcast<_N>(
                                       std::numeric_limits<_T>::infinity())));
            // alternative:
            //return __vector_bitcast<_T>(__abs(__x)._M_data > std::numeric_limits<_T>::max());
        }
    }
    // isnormal {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_T> __isnormal(__storage<_T, _N> __x)
    {
        // subnormals -> 0
        // 0 -> 0
        // inf -> inf
        // -inf -> inf
        // nan -> inf
        // normal value -> positive value / not 0
        return isnonzerovalue(
            __and(__x._M_data, __vector_broadcast<_N>(std::numeric_limits<_T>::infinity())));
    }

    // fpclassify {{{3
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __fixed_size_storage<int, _N> __fpclassify(__storage<_T, _N> __x)
    {
        if constexpr (__is_avx512_pd<_T, _N>()) {
            // AVX512 is special because we want to use an __mmask to blend int vectors
            // (coming from double vectors). GCC doesn't allow this combination on the
            // ternary operator. Thus, resort to intrinsics:
            if constexpr (__have_avx512vl) {
                auto &&__b = [](int __y) { return __to_intrin(__vector_broadcast<_N>(__y)); };
                return {_mm256_mask_mov_epi32(
                    _mm256_mask_mov_epi32(
                        _mm256_mask_mov_epi32(__b(FP_NORMAL), __isnan(__x), __b(FP_NAN)),
                        __isinf(__x), __b(FP_INFINITE)),
                    _mm512_cmp_pd_mask(
                        __abs(__x),
                        __vector_broadcast<_N>(std::numeric_limits<double>::min()),
                        _CMP_LT_OS),
                    _mm256_mask_mov_epi32(
                        __b(FP_SUBNORMAL),
                        _mm512_cmp_pd_mask(__x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                        __b(FP_ZERO)))};
            } else {
                auto &&__b = [](int __y) {
                    return _mm512_castsi256_si512(__to_intrin(__vector_broadcast<_N>(__y)));
                };
                return {__lo256(_mm512_mask_mov_epi32(
                    _mm512_mask_mov_epi32(
                        _mm512_mask_mov_epi32(__b(FP_NORMAL), __isnan(__x), __b(FP_NAN)),
                        __isinf(__x), __b(FP_INFINITE)),
                    _mm512_cmp_pd_mask(
                        __abs(__x),
                        __vector_broadcast<_N>(std::numeric_limits<double>::min()),
                        _CMP_LT_OS),
                    _mm512_mask_mov_epi32(
                        __b(FP_SUBNORMAL),
                        _mm512_cmp_pd_mask(__x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                        __b(FP_ZERO))))};
            }
        } else {
            constexpr auto fp_normal =
                __vector_bitcast<_T>(__vector_broadcast<_N, __int_for_sizeof_t<_T>>(FP_NORMAL));
            constexpr auto fp_nan =
                __vector_bitcast<_T>(__vector_broadcast<_N, __int_for_sizeof_t<_T>>(FP_NAN));
            constexpr auto fp_infinite =
                __vector_bitcast<_T>(__vector_broadcast<_N, __int_for_sizeof_t<_T>>(FP_INFINITE));
            constexpr auto fp_subnormal =
                __vector_bitcast<_T>(__vector_broadcast<_N, __int_for_sizeof_t<_T>>(FP_SUBNORMAL));
            constexpr auto fp_zero =
                __vector_bitcast<_T>(__vector_broadcast<_N, __int_for_sizeof_t<_T>>(FP_ZERO));

            const auto tmp = __vector_bitcast<__llong>(
                __abs(__x)._M_data < std::numeric_limits<_T>::min()
                    ? (__x._M_data == 0 ? fp_zero : fp_subnormal)
                    : __blend(__isinf(__x)._M_data, __blend(__isnan(__x)._M_data, fp_normal, fp_nan),
                                 fp_infinite));
            if constexpr (std::is_same_v<_T, float>) {
                if constexpr (__fixed_size_storage<int, _N>::tuple_size == 1) {
                    return {__vector_bitcast<int>(tmp)};
                } else if constexpr (__fixed_size_storage<int, _N>::tuple_size == 2) {
                    return {__extract<0, 2>(__vector_bitcast<int>(tmp)),
                            __extract<1, 2>(__vector_bitcast<int>(tmp))};
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
    template <class _T, size_t _N> _GLIBCXX_SIMD_INTRINSIC static void __increment(__storage<_T, _N> &__x)
    {
        __x = plus<_T, _N>(__x, __vector_broadcast<_N, _T>(1));
    }
    template <class _T, size_t _N> _GLIBCXX_SIMD_INTRINSIC static void __decrement(__storage<_T, _N> &__x)
    {
        __x = minus<_T, _N>(__x, __vector_broadcast<_N, _T>(1));
    }

    // smart_reference access {{{2
    template <class _T, size_t _N, class _U>
    _GLIBCXX_SIMD_INTRINSIC static void set(__storage<_T, _N> &__v, int __i, _U &&__x) noexcept
    {
        __v.set(__i, std::forward<_U>(__x));
    }

    // masked_assign{{{2
    template <class _T, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(__storage<_K, _N> __k,
                                                      __storage<_T, _N> &lhs,
                                                      __id<__storage<_T, _N>> rhs)
    {
        lhs = __blend(__k._M_data, lhs._M_data, rhs._M_data);
    }

    template <class _T, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(__storage<_K, _N> __k, __storage<_T, _N> &lhs,
                                           __id<_T> rhs)
    {
        if (__builtin_constant_p(rhs) && rhs == 0 && std::is_same<_K, _T>::value) {
            if constexpr (!__is_bitmask(__k)) {
                // the __andnot optimization only makes sense if __k._M_data is a vector register
                lhs._M_data = __andnot(__k._M_data, lhs._M_data);
                return;
            } else {
                // for AVX512/__mmask, a _mm512_maskz_mov is best
                lhs._M_data = __auto_bitcast(__blend(__k, lhs, __intrinsic_type_t<_T, _N>()));
                return;
            }
        }
        lhs._M_data = __blend(__k._M_data, lhs._M_data, __vector_broadcast<_N>(rhs));
    }

    // __masked_cassign {{{2
    template <template <typename> class _Op, class _T, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_cassign(const __storage<_K, _N> __k, __storage<_T, _N> &lhs,
                                            const __id<__storage<_T, _N>> rhs)
    {
        lhs._M_data = __blend(__k._M_data, lhs._M_data, _Op<void>{}(lhs._M_data, rhs._M_data));
    }

    template <template <typename> class _Op, class _T, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_cassign(const __storage<_K, _N> __k, __storage<_T, _N> &lhs,
                                            const __id<_T> rhs)
    {
        lhs._M_data = __blend(__k._M_data, lhs._M_data, _Op<void>{}(lhs._M_data, __vector_broadcast<_N>(rhs)));
    }

    // masked_unary {{{2
    template <template <typename> class _Op, class _T, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_T, _N> masked_unary(const __storage<_K, _N> __k,
                                                            const __storage<_T, _N> __v)
    {
        auto __vv = make_simd(__v);
        _Op<decltype(__vv)> op;
        return __blend(__k, __v, __data(op(__vv)));
    }

    //}}}2
};

// __generic_mask_impl {{{1
template <class _Abi> struct __generic_mask_impl {
    // member types {{{2
    template <class _T> using __type_tag = _T *;
    template <class _T> using simd_mask = std::experimental::simd_mask<_T, _Abi>;
    template <class _T>
    using _Simd_member_type = typename _Abi::template __traits<_T>::_Simd_member_type;
    template <class _T>
    using _Mask_member_type = typename _Abi::template __traits<_T>::_Mask_member_type;

    // masked load {{{2
    template <class _T, size_t _N, class _F>
    static inline __storage<_T, _N> masked_load(__storage<_T, _N> merge, __storage<_T, _N> mask,
                                            const bool *mem, _F) noexcept
    {
        if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
            if constexpr (__have_avx512bw_vl) {
                if constexpr (_N == 8) {
                    const auto __a = _mm_mask_loadu_epi8(__m128i(), mask, mem);
                    return (merge & ~mask) | _mm_test_epi8_mask(__a, __a);
                } else if constexpr (_N == 16) {
                    const auto __a = _mm_mask_loadu_epi8(__m128i(), mask, mem);
                    return (merge & ~mask) | _mm_test_epi8_mask(__a, __a);
                } else if constexpr (_N == 32) {
                    const auto __a = _mm256_mask_loadu_epi8(__m256i(), mask, mem);
                    return (merge & ~mask) | _mm256_test_epi8_mask(__a, __a);
                } else if constexpr (_N == 64) {
                    const auto __a = _mm512_mask_loadu_epi8(__m512i(), mask, mem);
                    return (merge & ~mask) | _mm512_test_epi8_mask(__a, __a);
                } else {
                    __assert_unreachable<_T>();
                }
            } else {
                __bit_iteration(mask, [&](auto __i) { merge.set(__i, mem[__i]); });
                return merge;
            }
        } else if constexpr (__have_avx512bw_vl && _N == 32 && sizeof(_T) == 1) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(
                _mm256_mask_sub_epi8(__vector_bitcast<__llong>(merge), __k, __m256i(),
                                     _mm256_mask_loadu_epi8(__m256i(), __k, mem)));
        } else if constexpr (__have_avx512bw_vl && _N == 16 && sizeof(_T) == 1) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm_mask_sub_epi8(__vector_bitcast<__llong>(merge), __k, __m128i(),
                                                 _mm_mask_loadu_epi8(__m128i(), __k, mem)));
        } else if constexpr (__have_avx512bw_vl && _N == 16 && sizeof(_T) == 2) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm256_mask_sub_epi16(
                __vector_bitcast<__llong>(merge), __k, __m256i(),
                _mm256_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), __k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 8 && sizeof(_T) == 2) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm_mask_sub_epi16(
                __vector_bitcast<__llong>(merge), __k, __m128i(),
                _mm_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), __k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 8 && sizeof(_T) == 4) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm256_mask_sub_epi32(
                __vector_bitcast<__llong>(merge), __k, __m256i(),
                _mm256_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), __k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 4 && sizeof(_T) == 4) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm_mask_sub_epi32(
                __vector_bitcast<__llong>(merge), __k, __m128i(),
                _mm_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), __k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 4 && sizeof(_T) == 8) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm256_mask_sub_epi64(
                __vector_bitcast<__llong>(merge), __k, __m256i(),
                _mm256_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), __k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 2 && sizeof(_T) == 8) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm_mask_sub_epi64(
                __vector_bitcast<__llong>(merge), __k, __m128i(),
                _mm_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), __k, mem))));
        } else {
            // AVX(2) has 32/64 bit maskload, but nothing at 8 bit granularity
            auto tmp = __storage_bitcast<__int_for_sizeof_t<_T>>(merge);
            __bit_iteration(__vector_to_bitset(mask._M_data).to_ullong(),
                                  [&](auto __i) { tmp.set(__i, -mem[__i]); });
            merge = __storage_bitcast<_T>(tmp);
        }
        return merge;
    }

    // store {{{2
    template <class _T, size_t _N, class _F>
    _GLIBCXX_SIMD_INTRINSIC static void store(__storage<_T, _N> __v, bool *__mem, _F) noexcept
    {
        if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>()) {
            if constexpr (_N == 2 && __have_sse2) {
                const auto __k = __vector_bitcast<int>(__v);
                __mem[0] = -__k[1];
                __mem[1] = -__k[3];
            } else if constexpr (_N == 4 && __have_sse2) {
                const unsigned bool4 =
                    __vector_bitcast<uint>(_mm_packs_epi16(
                        _mm_packs_epi32(__vector_bitcast<__llong>(__v), __m128i()), __m128i()))[0] &
                    0x01010101u;
                std::memcpy(__mem, &bool4, 4);
            } else if constexpr (std::is_same_v<_T, float> && __have_mmx) {
                const __m128 __k = __to_intrin(__v);
                const __m64 kk = _mm_cvtps_pi8(__and(__k, _mm_set1_ps(1.f)));
                __vector_store<4>(kk, __mem, _F());
                _mm_empty();
            } else if constexpr (_N == 8 && __have_sse2) {
                __vector_store<8>(
                    _mm_packs_epi16(__to_intrin(__vector_bitcast<ushort>(__v) >> 15),
                                    __m128i()),
                    __mem, _F());
            } else if constexpr (_N == 16 && __have_sse2) {
                __vector_store(__v._M_data & 1, __mem, _F());
            } else {
                __assert_unreachable<_T>();
            }
        } else if constexpr (__is_abi<_Abi, simd_abi::__avx_abi>()) {
            if constexpr (_N == 4 && __have_avx) {
                auto __k = __vector_bitcast<__llong>(__v);
                int bool4;
                if constexpr (__have_avx2) {
                    bool4 = _mm256_movemask_epi8(__k);
                } else {
                    bool4 = (_mm_movemask_epi8(__lo128(__k)) |
                             (_mm_movemask_epi8(__hi128(__k)) << 16));
                }
                bool4 &= 0x01010101;
                std::memcpy(__mem, &bool4, 4);
            } else if constexpr (_N == 8 && __have_avx) {
                const auto __k = __vector_bitcast<__llong>(__v);
                const auto k2 = _mm_srli_epi16(_mm_packs_epi16(__lo128(__k), __hi128(__k)), 15);
                const auto k3 = _mm_packs_epi16(k2, __m128i());
                __vector_store<8>(k3, __mem, _F());
            } else if constexpr (_N == 16 && __have_avx2) {
                const auto __x = _mm256_srli_epi16(__to_intrin(__v), 15);
                const auto bools = _mm_packs_epi16(__lo128(__x), __hi128(__x));
                __vector_store<16>(bools, __mem, _F());
            } else if constexpr (_N == 16 && __have_avx) {
                const auto bools = 1 & __vector_bitcast<__uchar>(_mm_packs_epi16(
                                           __lo128(__to_intrin(__v)), __hi128(__to_intrin(__v))));
                __vector_store<16>(bools, __mem, _F());
            } else if constexpr (_N == 32 && __have_avx) {
                __vector_store<32>(1 & __v._M_data, __mem, _F());
            } else {
                __assert_unreachable<_T>();
            }
        } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
            if constexpr (_N == 8) {
                __vector_store<8>(
#if _GLIBCXX_SIMD_HAVE_AVX512VL && _GLIBCXX_SIMD_HAVE_AVX512BW
                    _mm_maskz_set1_epi8(__v._M_data, 1),
#elif defined __x86_64__
                    __make_storage<__ullong>(_pdep_u64(__v._M_data, 0x0101010101010101ULL), 0ull),
#else
                    __make_storage<uint>(_pdep_u32(__v._M_data, 0x01010101U),
                                       _pdep_u32(__v._M_data >> 4, 0x01010101U)),
#endif
                    __mem, _F());
            } else if constexpr (_N == 16 && __have_avx512bw_vl) {
                __vector_store(_mm_maskz_set1_epi8(__v._M_data, 1), __mem, _F());
            } else if constexpr (_N == 16 && __have_avx512f) {
                _mm512_mask_cvtepi32_storeu_epi8(__mem, ~__mmask16(),
                                                 _mm512_maskz_set1_epi32(__v._M_data, 1));
            } else if constexpr (_N == 32 && __have_avx512bw_vl) {
                __vector_store(_mm256_maskz_set1_epi8(__v._M_data, 1), __mem, _F());
            } else if constexpr (_N == 32 && __have_avx512bw) {
                __vector_store(__lo256(_mm512_maskz_set1_epi8(__v._M_data, 1)), __mem, _F());
            } else if constexpr (_N == 64 && __have_avx512bw) {
                __vector_store(_mm512_maskz_set1_epi8(__v._M_data, 1), __mem, _F());
            } else {
                __assert_unreachable<_T>();
            }
        } else {
            __assert_unreachable<_T>();
        }
    }

    // masked store {{{2
    template <class _T, size_t _N, class _F>
    static inline void masked_store(const __storage<_T, _N> __v, bool *__mem, _F,
                                    const __storage<_T, _N> __k) noexcept
    {
        if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
            if constexpr (_N == 8 && __have_avx512bw_vl) {
                _mm_mask_cvtepi16_storeu_epi8(__mem, __k, _mm_maskz_set1_epi16(__v, 1));
            } else if constexpr (_N == 8 && __have_avx512vl) {
                _mm256_mask_cvtepi32_storeu_epi8(__mem, __k, _mm256_maskz_set1_epi32(__v, 1));
            } else if constexpr (_N == 8) {
                // we rely on __k < 0x100:
                _mm512_mask_cvtepi32_storeu_epi8(__mem, __k, _mm512_maskz_set1_epi32(__v, 1));
            } else if constexpr (_N == 16 && __have_avx512bw_vl) {
                _mm_mask_storeu_epi8(__mem, __k, _mm_maskz_set1_epi8(__v, 1));
            } else if constexpr (_N == 16) {
                _mm512_mask_cvtepi32_storeu_epi8(__mem, __k, _mm512_maskz_set1_epi32(__v, 1));
            } else if constexpr (_N == 32 && __have_avx512bw_vl) {
                _mm256_mask_storeu_epi8(__mem, __k, _mm256_maskz_set1_epi8(__v, 1));
            } else if constexpr (_N == 32 && __have_avx512bw) {
                _mm256_mask_storeu_epi8(__mem, __k, __lo256(_mm512_maskz_set1_epi8(__v, 1)));
            } else if constexpr (_N == 64 && __have_avx512bw) {
                _mm512_mask_storeu_epi8(__mem, __k, _mm512_maskz_set1_epi8(__v, 1));
            } else {
                __assert_unreachable<_T>();
            }
        } else {
            __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(), [&](auto __i) { __mem[__i] = __v[__i]; });
        }
    }

    // __from_bitset{{{2
    template <size_t _N, class _T>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_T> __from_bitset(std::bitset<_N> __bits, __type_tag<_T>)
    {
        return __convert_mask<typename _Mask_member_type<_T>::register_type>(__bits);
    }

    // logical and bitwise operators {{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> logical_and(const __storage<_T, _N> &__x,
                                                            const __storage<_T, _N> &__y)
    {
        if constexpr (std::is_same_v<_T, bool>) {
            return __x._M_data & __y._M_data;
        } else {
            return __and(__x._M_data, __y._M_data);
        }
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> logical_or(const __storage<_T, _N> &__x,
                                                           const __storage<_T, _N> &__y)
    {
        if constexpr (std::is_same_v<_T, bool>) {
            return __x._M_data | __y._M_data;
        } else {
            return __or(__x._M_data, __y._M_data);
        }
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_and(const __storage<_T, _N> &__x,
                                                        const __storage<_T, _N> &__y)
    {
        if constexpr (std::is_same_v<_T, bool>) {
            return __x._M_data & __y._M_data;
        } else {
            return __and(__x._M_data, __y._M_data);
        }
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_or(const __storage<_T, _N> &__x,
                                                       const __storage<_T, _N> &__y)
    {
        if constexpr (std::is_same_v<_T, bool>) {
            return __x._M_data | __y._M_data;
        } else {
            return __or(__x._M_data, __y._M_data);
        }
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_T, _N> bit_xor(const __storage<_T, _N> &__x,
                                                        const __storage<_T, _N> &__y)
    {
        if constexpr (std::is_same_v<_T, bool>) {
            return __x._M_data ^ __y._M_data;
        } else {
            return __xor(__x._M_data, __y._M_data);
        }
    }

    // smart_reference access {{{2
    template <class _T, size_t _N> static void set(__storage<_T, _N> &__k, int __i, bool __x) noexcept
    {
        if constexpr (std::is_same_v<_T, bool>) {
            __k.set(__i, __x);
        } else {
            using int_t = __vector_type_t<__int_for_sizeof_t<_T>, _N>;
            auto tmp = reinterpret_cast<int_t>(__k._M_data);
            tmp[__i] = -__x;
            __k._M_data = __auto_bitcast(tmp);
        }
    }
    // masked_assign{{{2
    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(__storage<_T, _N> __k, __storage<_T, _N> &lhs,
                                           __id<__storage<_T, _N>> rhs)
    {
        lhs = __blend(__k._M_data, lhs._M_data, rhs._M_data);
    }

    template <class _T, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(__storage<_T, _N> __k, __storage<_T, _N> &lhs, bool rhs)
    {
        if (__builtin_constant_p(rhs)) {
            if (rhs == false) {
                lhs = __andnot(__k._M_data, lhs._M_data);
            } else {
                lhs = __or(__k._M_data, lhs._M_data);
            }
            return;
        }
        lhs = __blend(__k, lhs, __data(simd_mask<_T>(rhs)));
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
 *  - alignment of `simd<_T, _N>` is `_N * sizeof(_T)` if _N is __a power-of-2 value,
 *    otherwise `__next_power_of_2(_N * sizeof(_T))` (Note: if the alignment were to
 *    exceed the system/compiler maximum, it is bounded to that maximum)
 *  - simd_mask objects are passed like std::bitset<_N>
 *  - memory layout of `simd_mask<_T, _N>` is equivalent to `std::bitset<_N>`
 *  - alignment of `simd_mask<_T, _N>` is equal to the alignment of `std::bitset<_N>`
 */
// __autocvt_to_simd {{{
template <class _T, bool = std::is_arithmetic_v<std::decay_t<_T>>>
struct __autocvt_to_simd {
    _T _M_data;
    using TT = std::decay_t<_T>;
    operator TT() { return _M_data; }
    operator TT &()
    {
        static_assert(std::is_lvalue_reference<_T>::value, "");
        static_assert(!std::is_const<_T>::value, "");
        return _M_data;
    }
    operator TT *()
    {
        static_assert(std::is_lvalue_reference<_T>::value, "");
        static_assert(!std::is_const<_T>::value, "");
        return &_M_data;
    }

    constexpr inline __autocvt_to_simd(_T dd) : _M_data(dd) {}

    template <class _Abi> operator simd<typename TT::value_type, _Abi>()
    {
        return {__private_init, _M_data};
    }

    template <class _Abi> operator simd<typename TT::value_type, _Abi> &()
    {
        return *reinterpret_cast<simd<typename TT::value_type, _Abi> *>(&_M_data);
    }

    template <class _Abi> operator simd<typename TT::value_type, _Abi> *()
    {
        return reinterpret_cast<simd<typename TT::value_type, _Abi> *>(&_M_data);
    }
};
template <class _T> __autocvt_to_simd(_T &&)->__autocvt_to_simd<_T>;

template <class _T> struct __autocvt_to_simd<_T, true> {
    using TT = std::decay_t<_T>;
    _T _M_data;
    fixed_size_simd<TT, 1> fd;

    constexpr inline __autocvt_to_simd(_T dd) : _M_data(dd), fd(_M_data) {}
    ~__autocvt_to_simd()
    {
        _M_data = __data(fd).first;
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
    using _M = typename _V::mask_type;
    using _T = typename _V::value_type;
    constexpr _T *__type_tag = nullptr;
    return __get_impl_t<_M>::__from_bitset(
        std::bitset<_V::size()>(shifted_bits.to_ullong()), __type_tag);
}

// __fixed_size_simd_impl {{{1
template <int _N> struct __fixed_size_simd_impl {
    // member types {{{2
    using _Mask_member_type = std::bitset<_N>;
    template <class _T> using _Simd_member_type = __fixed_size_storage<_T, _N>;
    template <class _T>
    static constexpr std::size_t tuple_size = _Simd_member_type<_T>::tuple_size;
    template <class _T>
    static constexpr std::make_index_sequence<_Simd_member_type<_T>::tuple_size> index_seq = {};
    template <class _T> using simd = std::experimental::simd<_T, simd_abi::fixed_size<_N>>;
    template <class _T> using simd_mask = std::experimental::simd_mask<_T, simd_abi::fixed_size<_N>>;
    template <class _T> using __type_tag = _T *;

    // broadcast {{{2
    template <class _T> static constexpr inline _Simd_member_type<_T> __broadcast(_T __x) noexcept
    {
        return _Simd_member_type<_T>::generate(
            [&](auto meta) { return meta.__broadcast(__x); });
    }

    // generator {{{2
    template <class _F, class _T>
    _GLIBCXX_SIMD_INTRINSIC static _Simd_member_type<_T> generator(_F &&gen, __type_tag<_T>)
    {
        return _Simd_member_type<_T>::generate([&gen](auto meta) {
            return meta.generator(
                [&](auto i_) {
                    return gen(__size_constant<meta.offset + decltype(i_)::value>());
                },
                __type_tag<_T>());
        });
    }

    // load {{{2
    template <class _T, class _U, class _F>
    static inline _Simd_member_type<_T> load(const _U *mem, _F __f,
                                              __type_tag<_T>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        return _Simd_member_type<_T>::generate(
            [&](auto meta) { return meta.load(&mem[meta.offset], __f, __type_tag<_T>()); });
    }

    // masked load {{{2
    template <class _T, class... As, class _U, class _F>
    static inline __simd_tuple<_T, As...> masked_load(__simd_tuple<_T, As...> merge,
                                                   const _Mask_member_type __bits,
                                                   const _U *mem,
                                                   _F __f) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        __for_each(merge, [&](auto meta, auto &native) {
            native = meta.masked_load(native, meta.make_mask(__bits), &mem[meta.offset], __f);
        });
        return merge;
    }

    // store {{{2
    template <class _T, class _U, class _F>
    static inline void store(const _Simd_member_type<_T> __v, _U *mem, _F __f,
                             __type_tag<_T>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        __for_each(__v, [&](auto meta, auto native) {
            meta.store(native, &mem[meta.offset], __f, __type_tag<_T>());
        });
    }

    // masked store {{{2
    template <class _T, class... As, class _U, class _F>
    static inline void masked_store(const __simd_tuple<_T, As...> __v, _U *mem, _F __f,
                                    const _Mask_member_type __bits) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        __for_each(__v, [&](auto meta, auto native) {
            meta.masked_store(native, &mem[meta.offset], __f, meta.make_mask(__bits));
        });
    }

    // negation {{{2
    template <class _T, class... As>
    static inline _Mask_member_type negate(__simd_tuple<_T, As...> __x) noexcept
    {
        _Mask_member_type __bits = 0;
        __for_each(__x, [&__bits](auto meta, auto native) {
            __bits |= meta.mask_to_shifted_ullong(meta.negate(native));
        });
        return __bits;
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
    static inline _T reduce(const simd<_T> &__x, const _BinaryOperation &__binary_op)
    {
        using ranges = __n_abis_in_tuple<_Simd_member_type<_T>>;
        return __fixed_size_simd_impl::reduce(__x._M_data, __binary_op, typename ranges::__counts(),
                                               typename ranges::__begins());
    }

    // min, max, clamp {{{2
    template <class _T, class... As>
    static inline __simd_tuple<_T, As...> min(const __simd_tuple<_T, As...> __a,
                                              const __simd_tuple<_T, As...> __b)
    {
        return __simd_tuple_apply(
            [](auto __impl, auto aa, auto bb) { return __impl.min(aa, bb); }, __a, __b);
    }

    template <class _T, class... As>
    static inline __simd_tuple<_T, As...> max(const __simd_tuple<_T, As...> __a,
                                              const __simd_tuple<_T, As...> __b)
    {
        return __simd_tuple_apply(
            [](auto __impl, auto aa, auto bb) { return __impl.max(aa, bb); }, __a, __b);
    }

    // complement {{{2
    template <class _T, class... As>
    static inline __simd_tuple<_T, As...> complement(__simd_tuple<_T, As...> __x) noexcept
    {
        return __simd_tuple_apply([](auto __impl, auto __xx) { return __impl.complement(__xx); },
                                __x);
    }

    // unary minus {{{2
    template <class _T, class... As>
    static inline __simd_tuple<_T, As...> unary_minus(__simd_tuple<_T, As...> __x) noexcept
    {
        return __simd_tuple_apply([](auto __impl, auto __xx) { return __impl.unary_minus(__xx); },
                                __x);
    }

    // arithmetic operators {{{2

#define _GLIBCXX_SIMD_FIXED_OP(name_, op_)                                               \
    template <class _T, class... As>                                                     \
    static inline __simd_tuple<_T, As...> name_(__simd_tuple<_T, As...> __x,               \
                                                __simd_tuple<_T, As...> __y)               \
    {                                                                                    \
        return __simd_tuple_apply(                                                       \
            [](auto __impl, auto __xx, auto yy) { return __impl.name_(__xx, yy); }, __x, __y);   \
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
    static inline __simd_tuple<_T, As...> bit_shift_left(__simd_tuple<_T, As...> __x, int __y)
    {
        return __simd_tuple_apply(
            [__y](auto __impl, auto __xx) { return __impl.bit_shift_left(__xx, __y); }, __x);
    }

    template <class _T, class... As>
    static inline __simd_tuple<_T, As...> bit_shift_right(__simd_tuple<_T, As...> __x,
                                                          int __y)
    {
        return __simd_tuple_apply(
            [__y](auto __impl, auto __xx) { return __impl.bit_shift_right(__xx, __y); }, __x);
    }

    // math {{{2
#define _GLIBCXX_SIMD_APPLY_ON_TUPLE_(name_)                                             \
    template <class _T, class... As>                                                     \
    static inline __simd_tuple<_T, As...> __##name_(__simd_tuple<_T, As...> __x) noexcept  \
    {                                                                                    \
        return __simd_tuple_apply(                                                       \
            [](auto __impl, auto __xx) {                                                   \
                using _V = typename decltype(__impl)::simd_type;                         \
                return __data(name_(_V(__private_init, __xx)));                            \
            },                                                                           \
            __x);                                                                          \
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
    static inline __simd_tuple<_T, As...> __frexp(const __simd_tuple<_T, As...> &__x,
                                             __fixed_size_storage<int, _N> &exp) noexcept
    {
        return __simd_tuple_apply(
            [](auto __impl, const auto &__a, auto &__b) {
                return __data(
                    __impl.__frexp(typename decltype(__impl)::simd_type(__private_init, __a),
                                 __autocvt_to_simd(__b)));
            },
            __x, exp);
    }

    template <class _T, class... As>
    static inline __fixed_size_storage<int, _N> __fpclassify(__simd_tuple<_T, As...> __x) noexcept
    {
        return __optimize_simd_tuple(__x.template apply_r<int>(
            [](auto __impl, auto __xx) { return __impl.__fpclassify(__xx); }));
    }

#define _GLIBCXX_SIMD_TEST_ON_TUPLE_(name_)                                              \
    template <class _T, class... As>                                                     \
    static inline _Mask_member_type __##name_(__simd_tuple<_T, As...> __x) noexcept         \
    {                                                                                    \
        return test([](auto __impl, auto __xx) { return __impl.__##name_(__xx); }, __x);           \
    }
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isinf)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isfinite)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isnan)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isnormal)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(signbit)
#undef _GLIBCXX_SIMD_TEST_ON_TUPLE_

    // __increment & __decrement{{{2
    template <class... _Ts> static inline void __increment(__simd_tuple<_Ts...> &__x)
    {
        __for_each(__x, [](auto meta, auto &native) { meta.__increment(native); });
    }

    template <class... _Ts> static inline void __decrement(__simd_tuple<_Ts...> &__x)
    {
        __for_each(__x, [](auto meta, auto &native) { meta.__decrement(native); });
    }

    // compares {{{2
#define _GLIBCXX_SIMD_CMP_OPERATIONS(cmp_)                                               \
    template <class _T, class... As>                                                     \
    static inline _Mask_member_type cmp_(const __simd_tuple<_T, As...>& __x,            \
                                          const __simd_tuple<_T, As...>& __y)            \
    {                                                                                    \
        _Mask_member_type __bits = 0;                                                   \
        __for_each(__x, __y, [&__bits](auto meta, auto native_x, auto native_y) {        \
            __bits |= meta.mask_to_shifted_ullong(meta.cmp_(native_x, native_y));        \
        });                                                                              \
        return __bits;                                                                   \
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
    _GLIBCXX_SIMD_INTRINSIC static void set(__simd_tuple<_T, As...> &__v, int __i, _U &&__x) noexcept
    {
        __v.set(__i, std::forward<_U>(__x));
    }

    // masked_assign {{{2
    template <typename _T, class... As>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(
        const _Mask_member_type __bits, __simd_tuple<_T, As...> &lhs,
        const __id<__simd_tuple<_T, As...>> rhs)
    {
        __for_each(lhs, rhs, [&](auto meta, auto &native_lhs, auto native_rhs) {
            meta.masked_assign(meta.make_mask(__bits), native_lhs, native_rhs);
        });
    }

    // Optimization for the case where the RHS is a scalar. No need to broadcast the
    // scalar to a simd first.
    template <typename _T, class... As>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(const _Mask_member_type __bits,
                                           __simd_tuple<_T, As...> &lhs,
                                           const __id<_T> rhs)
    {
        __for_each(lhs, [&](auto meta, auto &native_lhs) {
            meta.masked_assign(meta.make_mask(__bits), native_lhs, rhs);
        });
    }

    // __masked_cassign {{{2
    template <template <typename> class _Op, typename _T, class... As>
    static inline void __masked_cassign(const _Mask_member_type __bits,
                                      __simd_tuple<_T, As...> &lhs,
                                      const __simd_tuple<_T, As...> rhs)
    {
        __for_each(lhs, rhs, [&](auto meta, auto &native_lhs, auto native_rhs) {
            meta.template __masked_cassign<_Op>(meta.make_mask(__bits), native_lhs,
                                             native_rhs);
        });
    }

    // Optimization for the case where the RHS is a scalar. No need to broadcast the
    // scalar to a simd
    // first.
    template <template <typename> class _Op, typename _T, class... As>
    static inline void __masked_cassign(const _Mask_member_type __bits,
                                      __simd_tuple<_T, As...> &lhs, const _T rhs)
    {
        __for_each(lhs, [&](auto meta, auto &native_lhs) {
            meta.template __masked_cassign<_Op>(meta.make_mask(__bits), native_lhs, rhs);
        });
    }

    // masked_unary {{{2
    template <template <typename> class _Op, class _T, class... As>
    static inline __simd_tuple<_T, As...> masked_unary(
        const _Mask_member_type __bits,
        const __simd_tuple<_T, As...> __v)  // TODO: const-ref __v?
    {
        return __v.apply_wrapped([&__bits](auto meta, auto native) {
            return meta.template masked_unary<_Op>(meta.make_mask(__bits), native);
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
    using _Mask_member_type = std::bitset<_N>;
    template <class _T> using simd_mask = std::experimental::simd_mask<_T, simd_abi::fixed_size<_N>>;
    template <class _T> using __type_tag = _T *;

    // __from_bitset {{{2
    template <class _T>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type __from_bitset(const _Mask_member_type &bs,
                                                     __type_tag<_T>) noexcept
    {
        return bs;
    }

    // load {{{2
    template <class _F> static inline _Mask_member_type load(const bool *mem, _F __f) noexcept
    {
        // TODO: __uchar is not necessarily the best type to use here. For smaller _N ushort,
        // uint, __ullong, float, and double can be more efficient.
        __ullong __r = 0;
        using Vs = __fixed_size_storage<__uchar, _N>;
        __for_each(Vs{}, [&](auto meta, auto) {
            __r |= meta.mask_to_shifted_ullong(
                meta.simd_mask.load(&mem[meta.offset], __f, __size_constant<meta.size()>()));
        });
        return __r;
    }

    // masked load {{{2
    template <class _F>
    static inline _Mask_member_type masked_load(_Mask_member_type merge,
                                               _Mask_member_type mask, const bool *mem,
                                               _F) noexcept
    {
        __bit_iteration(mask.to_ullong(), [&](auto __i) { merge[__i] = mem[__i]; });
        return merge;
    }

    // store {{{2
    template <class _F>
    static inline void store(_Mask_member_type bs, bool *mem, _F __f) noexcept
    {
#if _GLIBCXX_SIMD_HAVE_AVX512BW
        const __m512i bool64 = _mm512_movm_epi8(bs.to_ullong()) & 0x0101010101010101ULL;
        __vector_store<_N>(bool64, mem, __f);
#elif _GLIBCXX_SIMD_HAVE_BMI2
#ifdef __x86_64__
        __unused(__f);
        __execute_n_times<_N / 8>([&](auto __i) {
            constexpr size_t offset = __i * 8;
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
        __unused(__f);
        __execute_n_times<_N / 4>([&](auto __i) {
            constexpr size_t offset = __i * 4;
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
        __ullong __bits = bs.to_ullong();
        __execute_n_times<(_N + 15) / 16>([&](auto __i) {
            constexpr size_t offset = __i * 16;
            constexpr size_t remaining = _N - offset;
            if constexpr (remaining == 1) {
                mem[offset] = static_cast<bool>(__bits >> offset);
            } else if constexpr (remaining <= 4) {
                const uint bool4 = ((__bits >> offset) * 0x00204081U) & 0x01010101U;
                std::memcpy(&mem[offset], &bool4, remaining);
            } else if constexpr (remaining <= 7) {
                const __ullong bool8 =
                    ((__bits >> offset) * 0x40810204081ULL) & 0x0101010101010101ULL;
                std::memcpy(&mem[offset], &bool8, remaining);
            } else if constexpr (__have_sse2) {
                auto tmp = _mm_cvtsi32_si128(__bits >> offset);
                tmp = _mm_unpacklo_epi8(tmp, tmp);
                tmp = _mm_unpacklo_epi16(tmp, tmp);
                tmp = _mm_unpacklo_epi32(tmp, tmp);
                _V tmp2(tmp);
                tmp2 &= _V([](auto __j) {
                    return static_cast<__uchar>(1 << (__j % CHAR_BIT));
                });  // mask bit index
                const __m128i bool16 = __intrin_bitcast<__m128i>(
                    __vector_bitcast<__uchar>(__data(tmp2 == 0)) +
                    1);  // 0xff -> 0x00 | 0x00 -> 0x01
                if constexpr (remaining >= 16) {
                    __vector_store<16>(bool16, &mem[offset], __f);
                } else if constexpr (remaining & 3) {
                    constexpr int to_shift = 16 - int(remaining);
                    _mm_maskmoveu_si128(bool16,
                                        _mm_srli_si128(__allbits<__m128i>, to_shift),
                                        reinterpret_cast<char *>(&mem[offset]));
                } else  // at this point: 8 < remaining < 16
                    if constexpr (remaining >= 8) {
                    __vector_store<8>(bool16, &mem[offset], __f);
                    if constexpr (remaining == 12) {
                        __vector_store<4>(_mm_unpackhi_epi64(bool16, bool16),
                                         &mem[offset + 8], __f);
                    }
                }
            } else {
                __assert_unreachable<_F>();
            }
        });
#else
        // TODO: __uchar is not necessarily the best type to use here. For smaller _N ushort,
        // uint, __ullong, float, and double can be more efficient.
        using Vs = __fixed_size_storage<__uchar, _N>;
        __for_each(Vs{}, [&](auto meta, auto) {
            meta.store(meta.make_mask(bs), &mem[meta.offset], __f);
        });
//#else
        //__execute_n_times<_N>([&](auto __i) { mem[__i] = bs[__i]; });
#endif  // _GLIBCXX_SIMD_HAVE_BMI2
    }

    // masked store {{{2
    template <class _F>
    static inline void masked_store(const _Mask_member_type __v, bool *mem, _F,
                                    const _Mask_member_type __k) noexcept
    {
        __bit_iteration(__k, [&](auto __i) { mem[__i] = __v[__i]; });
    }

    // logical and bitwise operators {{{2
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type logical_and(const _Mask_member_type &__x,
                                                     const _Mask_member_type &__y) noexcept
    {
        return __x & __y;
    }

    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type logical_or(const _Mask_member_type &__x,
                                                    const _Mask_member_type &__y) noexcept
    {
        return __x | __y;
    }

    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type bit_and(const _Mask_member_type &__x,
                                                 const _Mask_member_type &__y) noexcept
    {
        return __x & __y;
    }

    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type bit_or(const _Mask_member_type &__x,
                                                const _Mask_member_type &__y) noexcept
    {
        return __x | __y;
    }

    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type bit_xor(const _Mask_member_type &__x,
                                                 const _Mask_member_type &__y) noexcept
    {
        return __x ^ __y;
    }

    // smart_reference access {{{2
    _GLIBCXX_SIMD_INTRINSIC static void set(_Mask_member_type &__k, int __i, bool __x) noexcept
    {
        __k.set(__i, __x);
    }

    // masked_assign {{{2
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(const _Mask_member_type __k,
                                           _Mask_member_type &lhs,
                                           const _Mask_member_type rhs)
    {
        lhs = (lhs & ~__k) | (rhs & __k);
    }

    // Optimization for the case where the RHS is a scalar.
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(const _Mask_member_type __k,
                                           _Mask_member_type &lhs, const bool rhs)
    {
        if (rhs) {
            lhs |= __k;
        } else {
            lhs &= ~__k;
        }
    }

    // }}}2
};
// }}}1

// __simd_converter scalar -> scalar {{{
template <class _T> struct __simd_converter<_T, simd_abi::scalar, _T, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC _T operator()(_T __a) { return __a; }
};
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::scalar, _To, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC _To operator()(_From __a)
    {
        return static_cast<_To>(__a);
    }
};

// }}}
// __simd_converter __sse -> scalar {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::__sse, _To, simd_abi::scalar> {
    using Arg = __sse_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC std::array<_To, Arg::_S_width> __all(Arg __a)
    {
        return __impl(std::make_index_sequence<Arg::_S_width>(), __a);
    }

    template <size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC std::array<_To, Arg::_S_width> __impl(std::index_sequence<_Indexes...>, Arg __a)
    {
        return {static_cast<_To>(__a[_Indexes])...};
    }
};

// }}}1
// __simd_converter scalar -> __sse {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::scalar, _To, simd_abi::__sse> {
    using _R = __sse_simd_member_type<_To>;
    template <class... _More> _GLIBCXX_SIMD_INTRINSIC constexpr _R operator()(_From __a, _More... __b)
    {
        static_assert(sizeof...(_More) + 1 == _R::_S_width);
        static_assert(std::conjunction_v<std::is_same<_From, _More>...>);
        return __vector_type16_t<_To>{static_cast<_To>(__a), static_cast<_To>(__b)...};
    }
};

// }}}1
// __simd_converter __sse -> __sse {{{1
template <class _T> struct __simd_converter<_T, simd_abi::__sse, _T, simd_abi::__sse> {
    using Arg = __sse_simd_member_type<_T>;
    _GLIBCXX_SIMD_INTRINSIC const Arg &operator()(const Arg &__x) { return __x; }
};

template <class _From, class _To>
struct __simd_converter<_From, simd_abi::__sse, _To, simd_abi::__sse> {
    using Arg = __sse_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(Arg __a)
    {
        return __convert_all<__vector_type16_t<_To>>(__a);
    }

    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(Arg __a)
    {
        return __convert<__vector_type16_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(Arg __a, Arg __b)
    {
        static_assert(sizeof(_From) >= 2 * sizeof(_To), "");
        return __convert<__sse_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(Arg __a, Arg __b, Arg __c, Arg __d)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__sse_simd_member_type<_To>>(__a, __b, __c, __d);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(Arg __a, Arg __b, Arg __c, Arg __d, Arg __e,
                                                     Arg __f, Arg __g, Arg __h)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__sse_simd_member_type<_To>>(__a, __b, __c, __d, __e, __f, __g, __h);
    }
};

// }}}1
// __simd_converter __avx -> scalar {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::__avx, _To, simd_abi::scalar> {
    using Arg = __avx_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC std::array<_To, Arg::_S_width> __all(Arg __a)
    {
        return __impl(std::make_index_sequence<Arg::_S_width>(), __a);
    }

    template <size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC std::array<_To, Arg::_S_width> __impl(std::index_sequence<_Indexes...>, Arg __a)
    {
        return {static_cast<_To>(__a[_Indexes])...};
    }
};

// }}}1
// __simd_converter scalar -> __avx {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::scalar, _To, simd_abi::__avx> {
    using _R = __avx_simd_member_type<_To>;
    template <class... _More> _GLIBCXX_SIMD_INTRINSIC constexpr _R operator()(_From __a, _More... __b)
    {
        static_assert(sizeof...(_More) + 1 == _R::_S_width);
        static_assert(std::conjunction_v<std::is_same<_From, _More>...>);
        return __vector_type32_t<_To>{static_cast<_To>(__a), static_cast<_To>(__b)...};
    }
};

// }}}1
// __simd_converter __sse -> __avx {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::__sse, _To, simd_abi::__avx> {
    using Arg = __sse_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(Arg __a) { return __convert_all<__vector_type32_t<_To>>(__a); }

    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(Arg __a)
    {
        return __convert<__vector_type32_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(Arg __a, Arg __b)
    {
        static_assert(sizeof(_From) >= 1 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(Arg __a, Arg __b, Arg __c, Arg __d)
    {
        static_assert(sizeof(_From) >= 2 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b, __c, __d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(Arg __x0, Arg __x1, Arg __x2, Arg __x3,
                                                     Arg __x4, Arg __x5, Arg __x6, Arg __x7)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__x0, __x1, __x2, __x3, __x4, __x5, __x6, __x7);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(Arg __x0, Arg __x1, Arg __x2, Arg __x3,
                                                     Arg __x4, Arg __x5, Arg __x6, Arg __x7,
                                                     Arg __x8, Arg __x9, Arg x10, Arg x11,
                                                     Arg x12, Arg x13, Arg x14, Arg x15)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__x0, __x1, __x2, __x3, __x4, __x5, __x6, __x7, __x8,
                                                      __x9, x10, x11, x12, x13, x14, x15);
    }
};

// }}}1
// __simd_converter __avx -> __sse {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::__avx, _To, simd_abi::__sse> {
    using Arg = __avx_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(Arg __a) { return __convert_all<__vector_type16_t<_To>>(__a); }

    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(Arg __a)
    {
        return __convert<__vector_type16_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(Arg __a, Arg __b)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__sse_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(Arg __a, Arg __b, Arg __c, Arg __d)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__sse_simd_member_type<_To>>(__a, __b, __c, __d);
    }
};

// }}}1
// __simd_converter __avx -> __avx {{{1
template <class _T> struct __simd_converter<_T, simd_abi::__avx, _T, simd_abi::__avx> {
    using Arg = __avx_simd_member_type<_T>;
    _GLIBCXX_SIMD_INTRINSIC const Arg &operator()(const Arg &__x) { return __x; }
};

template <class _From, class _To>
struct __simd_converter<_From, simd_abi::__avx, _To, simd_abi::__avx> {
    using Arg = __avx_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(Arg __a) { return __convert_all<__vector_type32_t<_To>>(__a); }

    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(Arg __a)
    {
        return __convert<__vector_type32_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(Arg __a, Arg __b)
    {
        static_assert(sizeof(_From) >= 2 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(Arg __a, Arg __b, Arg __c, Arg __d)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b, __c, __d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(Arg __a, Arg __b, Arg __c, Arg __d, Arg __e,
                                                     Arg __f, Arg __g, Arg __h)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b, __c, __d, __e, __f, __g, __h);
    }
};

// }}}1
// __simd_converter __avx512 -> scalar {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::__avx512, _To, simd_abi::scalar> {
    using Arg = __avx512_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC std::array<_To, Arg::_S_width> __all(Arg __a)
    {
        return __impl(std::make_index_sequence<Arg::_S_width>(), __a);
    }

    template <size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC std::array<_To, Arg::_S_width> __impl(std::index_sequence<_Indexes...>, Arg __a)
    {
        return {static_cast<_To>(__a[_Indexes])...};
    }
};

// }}}1
// __simd_converter scalar -> __avx512 {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::scalar, _To, simd_abi::__avx512> {
    using _R = __avx512_simd_member_type<_To>;

    _GLIBCXX_SIMD_INTRINSIC _R operator()(_From __a)
    {
        _R __r{};
        __r.set(0, static_cast<_To>(__a));
        return __r;
    }
    _GLIBCXX_SIMD_INTRINSIC _R operator()(_From __a, _From __b)
    {
        _R __r{};
        __r.set(0, static_cast<_To>(__a));
        __r.set(1, static_cast<_To>(__b));
        return __r;
    }
    _GLIBCXX_SIMD_INTRINSIC _R operator()(_From __a, _From __b, _From __c, _From __d)
    {
        _R __r{};
        __r.set(0, static_cast<_To>(__a));
        __r.set(1, static_cast<_To>(__b));
        __r.set(2, static_cast<_To>(__c));
        __r.set(3, static_cast<_To>(__d));
        return __r;
    }
    _GLIBCXX_SIMD_INTRINSIC _R operator()(_From __a, _From __b, _From __c, _From __d, _From __e, _From __f, _From __g,
                              _From __h)
    {
        _R __r{};
        __r.set(0, static_cast<_To>(__a));
        __r.set(1, static_cast<_To>(__b));
        __r.set(2, static_cast<_To>(__c));
        __r.set(3, static_cast<_To>(__d));
        __r.set(4, static_cast<_To>(__e));
        __r.set(5, static_cast<_To>(__f));
        __r.set(6, static_cast<_To>(__g));
        __r.set(7, static_cast<_To>(__h));
        return __r;
    }
    _GLIBCXX_SIMD_INTRINSIC _R operator()(_From __x0, _From __x1, _From __x2, _From __x3, _From __x4, _From __x5,
                              _From __x6, _From __x7, _From __x8, _From __x9, _From x10, _From x11,
                              _From x12, _From x13, _From x14, _From x15)
    {
        _R __r{};
        __r.set(0, static_cast<_To>(__x0));
        __r.set(1, static_cast<_To>(__x1));
        __r.set(2, static_cast<_To>(__x2));
        __r.set(3, static_cast<_To>(__x3));
        __r.set(4, static_cast<_To>(__x4));
        __r.set(5, static_cast<_To>(__x5));
        __r.set(6, static_cast<_To>(__x6));
        __r.set(7, static_cast<_To>(__x7));
        __r.set(8, static_cast<_To>(__x8));
        __r.set(9, static_cast<_To>(__x9));
        __r.set(10, static_cast<_To>(x10));
        __r.set(11, static_cast<_To>(x11));
        __r.set(12, static_cast<_To>(x12));
        __r.set(13, static_cast<_To>(x13));
        __r.set(14, static_cast<_To>(x14));
        __r.set(15, static_cast<_To>(x15));
        return __r;
    }
    _GLIBCXX_SIMD_INTRINSIC _R operator()(_From __x0, _From __x1, _From __x2, _From __x3, _From __x4, _From __x5,
                              _From __x6, _From __x7, _From __x8, _From __x9, _From x10, _From x11,
                              _From x12, _From x13, _From x14, _From x15, _From x16, _From x17,
                              _From x18, _From x19, _From x20, _From x21, _From x22, _From x23,
                              _From x24, _From x25, _From x26, _From x27, _From x28, _From x29,
                              _From x30, _From x31)
    {
        _R __r{};
        __r.set(0, static_cast<_To>(__x0));
        __r.set(1, static_cast<_To>(__x1));
        __r.set(2, static_cast<_To>(__x2));
        __r.set(3, static_cast<_To>(__x3));
        __r.set(4, static_cast<_To>(__x4));
        __r.set(5, static_cast<_To>(__x5));
        __r.set(6, static_cast<_To>(__x6));
        __r.set(7, static_cast<_To>(__x7));
        __r.set(8, static_cast<_To>(__x8));
        __r.set(9, static_cast<_To>(__x9));
        __r.set(10, static_cast<_To>(x10));
        __r.set(11, static_cast<_To>(x11));
        __r.set(12, static_cast<_To>(x12));
        __r.set(13, static_cast<_To>(x13));
        __r.set(14, static_cast<_To>(x14));
        __r.set(15, static_cast<_To>(x15));
        __r.set(16, static_cast<_To>(x16));
        __r.set(17, static_cast<_To>(x17));
        __r.set(18, static_cast<_To>(x18));
        __r.set(19, static_cast<_To>(x19));
        __r.set(20, static_cast<_To>(x20));
        __r.set(21, static_cast<_To>(x21));
        __r.set(22, static_cast<_To>(x22));
        __r.set(23, static_cast<_To>(x23));
        __r.set(24, static_cast<_To>(x24));
        __r.set(25, static_cast<_To>(x25));
        __r.set(26, static_cast<_To>(x26));
        __r.set(27, static_cast<_To>(x27));
        __r.set(28, static_cast<_To>(x28));
        __r.set(29, static_cast<_To>(x29));
        __r.set(30, static_cast<_To>(x30));
        __r.set(31, static_cast<_To>(x31));
        return __r;
    }
    _GLIBCXX_SIMD_INTRINSIC _R operator()(_From __x0, _From __x1, _From __x2, _From __x3, _From __x4, _From __x5,
                              _From __x6, _From __x7, _From __x8, _From __x9, _From x10, _From x11,
                              _From x12, _From x13, _From x14, _From x15, _From x16, _From x17,
                              _From x18, _From x19, _From x20, _From x21, _From x22, _From x23,
                              _From x24, _From x25, _From x26, _From x27, _From x28, _From x29,
                              _From x30, _From x31, _From x32, _From x33, _From x34, _From x35,
                              _From x36, _From x37, _From x38, _From x39, _From x40, _From x41,
                              _From x42, _From x43, _From x44, _From x45, _From x46, _From x47,
                              _From x48, _From x49, _From x50, _From x51, _From x52, _From x53,
                              _From x54, _From x55, _From x56, _From x57, _From x58, _From x59,
                              _From x60, _From x61, _From x62, _From x63)
    {
        return _R(static_cast<_To>(__x0), static_cast<_To>(__x1), static_cast<_To>(__x2),
                 static_cast<_To>(__x3), static_cast<_To>(__x4), static_cast<_To>(__x5),
                 static_cast<_To>(__x6), static_cast<_To>(__x7), static_cast<_To>(__x8),
                 static_cast<_To>(__x9), static_cast<_To>(x10), static_cast<_To>(x11),
                 static_cast<_To>(x12), static_cast<_To>(x13), static_cast<_To>(x14),
                 static_cast<_To>(x15), static_cast<_To>(x16), static_cast<_To>(x17),
                 static_cast<_To>(x18), static_cast<_To>(x19), static_cast<_To>(x20),
                 static_cast<_To>(x21), static_cast<_To>(x22), static_cast<_To>(x23),
                 static_cast<_To>(x24), static_cast<_To>(x25), static_cast<_To>(x26),
                 static_cast<_To>(x27), static_cast<_To>(x28), static_cast<_To>(x29),
                 static_cast<_To>(x30), static_cast<_To>(x31), static_cast<_To>(x32),
                 static_cast<_To>(x33), static_cast<_To>(x34), static_cast<_To>(x35),
                 static_cast<_To>(x36), static_cast<_To>(x37), static_cast<_To>(x38),
                 static_cast<_To>(x39), static_cast<_To>(x40), static_cast<_To>(x41),
                 static_cast<_To>(x42), static_cast<_To>(x43), static_cast<_To>(x44),
                 static_cast<_To>(x45), static_cast<_To>(x46), static_cast<_To>(x47),
                 static_cast<_To>(x48), static_cast<_To>(x49), static_cast<_To>(x50),
                 static_cast<_To>(x51), static_cast<_To>(x52), static_cast<_To>(x53),
                 static_cast<_To>(x54), static_cast<_To>(x55), static_cast<_To>(x56),
                 static_cast<_To>(x57), static_cast<_To>(x58), static_cast<_To>(x59),
                 static_cast<_To>(x60), static_cast<_To>(x61), static_cast<_To>(x62),
                 static_cast<_To>(x63));
    }
};

// }}}1
// __simd_converter __sse -> __avx512 {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::__sse, _To, simd_abi::__avx512> {
    using Arg = __sse_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(Arg __a)
    {
        return __convert_all<__vector_type64_t<_To>>(__a);
    }

    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __a)
    {
        return __convert<__vector_type64_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __a, Arg __b)
    {
        static_assert(2 * sizeof(_From) >= sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __a, Arg __b, Arg __c, Arg __d)
    {
        static_assert(sizeof(_From) >= 1 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b, __c, __d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __x0, Arg __x1, Arg __x2, Arg __x3,
                                                     Arg __x4, Arg __x5, Arg __x6, Arg __x7)
    {
        static_assert(sizeof(_From) >= 2 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__x0, __x1, __x2, __x3, __x4, __x5, __x6,
                                                           __x7);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __x0, Arg __x1, Arg __x2, Arg __x3,
                                                     Arg __x4, Arg __x5, Arg __x6, Arg __x7,
                                                     Arg __x8, Arg __x9, Arg x10, Arg x11,
                                                     Arg x12, Arg x13, Arg x14, Arg x15)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(
            __x0, __x1, __x2, __x3, __x4, __x5, __x6, __x7, __x8, __x9, x10, x11, x12, x13, x14, x15);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(
        Arg __x0, Arg __x1, Arg __x2, Arg __x3, Arg __x4, Arg __x5, Arg __x6, Arg __x7, Arg __x8, Arg __x9,
        Arg x10, Arg x11, Arg x12, Arg x13, Arg x14, Arg x15, Arg x16, Arg x17, Arg x18,
        Arg x19, Arg x20, Arg x21, Arg x22, Arg x23, Arg x24, Arg x25, Arg x26, Arg x27,
        Arg x28, Arg x29, Arg x30, Arg x31)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(
            __x0, __x1, __x2, __x3, __x4, __x5, __x6, __x7, __x8, __x9, x10, x11, x12, x13, x14, x15, x16,
            x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31);
    }
};

// }}}1
// __simd_converter __avx512 -> __sse {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::__avx512, _To, simd_abi::__sse> {
    using Arg = __avx512_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(Arg __a)
    {
        return __convert_all<__vector_type16_t<_To>>(__a);
    }

    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(Arg __a)
    {
        return __convert<__vector_type16_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(Arg __a, Arg __b)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__sse_simd_member_type<_To>>(__a, __b);
    }
};

// }}}1
// __simd_converter __avx -> __avx512 {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::__avx, _To, simd_abi::__avx512> {
    using Arg = __avx_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(Arg __a)
    {
        return __convert_all<__vector_type64_t<_To>>(__a);
    }

    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __a)
    {
        return __convert<__vector_type64_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __a, Arg __b)
    {
        static_assert(sizeof(_From) >= 1 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __a, Arg __b, Arg __c, Arg __d)
    {
        static_assert(sizeof(_From) >= 2 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b, __c, __d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __x0, Arg __x1, Arg __x2, Arg __x3,
                                                     Arg __x4, Arg __x5, Arg __x6, Arg __x7)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__x0, __x1, __x2, __x3, __x4, __x5, __x6,
                                                           __x7);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __x0, Arg __x1, Arg __x2, Arg __x3,
                                                     Arg __x4, Arg __x5, Arg __x6, Arg __x7,
                                                     Arg __x8, Arg __x9, Arg x10, Arg x11,
                                                     Arg x12, Arg x13, Arg x14, Arg x15)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(
            __x0, __x1, __x2, __x3, __x4, __x5, __x6, __x7, __x8, __x9, x10, x11, x12, x13, x14, x15);
    }
};

// }}}1
// __simd_converter __avx512 -> __avx {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::__avx512, _To, simd_abi::__avx> {
    using Arg = __avx512_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(Arg __a)
    {
        return __convert_all<__vector_type32_t<_To>>(__a);
    }

    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(Arg __a)
    {
        return __convert<__vector_type32_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(Arg __a, Arg __b)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(Arg __a, Arg __b, Arg __c, Arg __d)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b, __c, __d);
    }
};

// }}}1
// __simd_converter __avx512 -> __avx512 {{{1
template <class _T> struct __simd_converter<_T, simd_abi::__avx512, _T, simd_abi::__avx512> {
    using Arg = __avx512_simd_member_type<_T>;
    _GLIBCXX_SIMD_INTRINSIC const Arg &operator()(const Arg &__x) { return __x; }
};

template <class _From, class _To>
struct __simd_converter<_From, simd_abi::__avx512, _To, simd_abi::__avx512> {
    using Arg = __avx512_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(Arg __a)
    {
        return __convert_all<__vector_type64_t<_To>>(__a);
    }

    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __a)
    {
        return __convert<__vector_type64_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __a, Arg __b)
    {
        static_assert(sizeof(_From) >= 2 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __a, Arg __b, Arg __c, Arg __d)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b, __c, __d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(Arg __a, Arg __b, Arg __c, Arg __d, Arg __e,
                                                     Arg __f, Arg __g, Arg __h)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b, __c, __d, __e, __f, __g, __h);
    }
};

// }}}1
// __simd_converter scalar -> fixed_size<1> {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::scalar, _To, simd_abi::fixed_size<1>> {
    __simd_tuple<_To, simd_abi::scalar> operator()(_From __x) { return {static_cast<_To>(__x)}; }
};

// __simd_converter fixed_size<1> -> scalar {{{1
template <class _From, class _To>
struct __simd_converter<_From, simd_abi::fixed_size<1>, _To, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC _To operator()(__simd_tuple<_From, simd_abi::scalar> __x)
    {
        return {static_cast<_To>(__x.first)};
    }
};

// __simd_converter fixed_size<_N> -> fixed_size<_N> {{{1
template <class _T, int _N>
struct __simd_converter<_T, simd_abi::fixed_size<_N>, _T, simd_abi::fixed_size<_N>> {
    using arg = __fixed_size_storage<_T, _N>;
    _GLIBCXX_SIMD_INTRINSIC const arg &operator()(const arg &__x) { return __x; }
};

template <size_t ChunkSize, class _T> struct determine_required_input_chunks;

template <class _T, class... _Abis>
struct determine_required_input_chunks<0, __simd_tuple<_T, _Abis...>>
    : public std::integral_constant<size_t, 0> {
};

template <size_t ChunkSize, class _T, class _Abi0, class... _Abis>
struct determine_required_input_chunks<ChunkSize, __simd_tuple<_T, _Abi0, _Abis...>>
    : public std::integral_constant<
          size_t, determine_required_input_chunks<ChunkSize - simd_size_v<_T, _Abi0>,
                                                  __simd_tuple<_T, _Abis...>>::value> {
};

template <class _From, class _To> struct __fixed_size_converter {
    struct OneToMultipleChunks {
    };
    template <int _N> struct MultipleToOneChunk {
    };
    struct EqualChunks {
    };
    template <class _FromAbi, class _ToAbi, size_t _ToSize = simd_size_v<_To, _ToAbi>,
              size_t FromSize = simd_size_v<_From, _FromAbi>>
    using _ChunkRelation = std::conditional_t<
        (_ToSize < FromSize), OneToMultipleChunks,
        std::conditional_t<(_ToSize == FromSize), EqualChunks,
                           MultipleToOneChunk<int(_ToSize / FromSize)>>>;

    template <class... _Abis>
    using __return_type = __fixed_size_storage<_To, __simd_tuple<_From, _Abis...>::size()>;


protected:
    // OneToMultipleChunks {{{2
    template <class _A0>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0> __impl(OneToMultipleChunks, const __simd_tuple<_From, _A0> &__x)
    {
        using _R = __return_type<_A0>;
        __simd_converter<_From, _A0, _To, typename _R::_First_abi> __native_cvt;
        auto &&multiple_return_chunks = __native_cvt.__all(__x.first);
        return __to_simd_tuple<_To, typename _R::_First_abi>(multiple_return_chunks);
    }

    template <class... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_Abis...> __impl(OneToMultipleChunks,
                                           const __simd_tuple<_From, _Abis...> &__x)
    {
        using _R = __return_type<_Abis...>;
        using arg = __simd_tuple<_From, _Abis...>;
        constexpr size_t first_chunk = simd_size_v<_From, typename arg::_First_abi>;
        __simd_converter<_From, typename arg::_First_abi, _To, typename _R::_First_abi>
            __native_cvt;
        auto &&multiple_return_chunks = __native_cvt.__all(__x.first);
        constexpr size_t n_output_chunks =
            first_chunk / simd_size_v<_To, typename _R::_First_abi>;
        return __simd_tuple_concat(
            __to_simd_tuple<_To, typename _R::_First_abi>(multiple_return_chunks),
            __impl(_ChunkRelation<typename arg::_Second_type::_First_abi,
                               typename __simd_tuple_element<n_output_chunks, _R>::type::abi_type>(),
                 __x.second));
    }

    // MultipleToOneChunk {{{2
    template <int _N, class _A0, class... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0, _Abis...> __impl(MultipleToOneChunk<_N>,
                                               const __simd_tuple<_From, _A0, _Abis...> &__x)
    {
        return impl_mto(std::integral_constant<bool, sizeof...(_Abis) + 1 == _N>(),
                        std::make_index_sequence<_N>(), __x);
    }

    template <size_t... _Indexes, class _A0, class... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0, _Abis...> impl_mto(true_type,
                                                   std::index_sequence<_Indexes...>,
                                                   const __simd_tuple<_From, _A0, _Abis...> &__x)
    {
        using _R = __return_type<_A0, _Abis...>;
        __simd_converter<_From, _A0, _To, typename _R::_First_abi> __native_cvt;
        return {__native_cvt(__get_tuple_at<_Indexes>(__x)...)};
    }

    template <size_t... _Indexes, class _A0, class... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0, _Abis...> impl_mto(false_type,
                                                   std::index_sequence<_Indexes...>,
                                                   const __simd_tuple<_From, _A0, _Abis...> &__x)
    {
        using _R = __return_type<_A0, _Abis...>;
        __simd_converter<_From, _A0, _To, typename _R::_First_abi> __native_cvt;
        return {
            __native_cvt(__get_tuple_at<_Indexes>(__x)...),
            __impl(
                _ChunkRelation<
                    typename __simd_tuple_element<sizeof...(_Indexes),
                                           __simd_tuple<_From, _A0, _Abis...>>::type::abi_type,
                    typename _R::_Second_type::_First_abi>(),
                __simd_tuple_pop_front(__size_constant<sizeof...(_Indexes)>(), __x))};
    }

    // EqualChunks {{{2
    template <class _A0>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0> __impl(EqualChunks, const __simd_tuple<_From, _A0> &__x)
    {
        __simd_converter<_From, _A0, _To, typename __return_type<_A0>::_First_abi> __native_cvt;
        return {__native_cvt(__x.first)};
    }

    template <class _A0, class _A1, class... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0, _A1, _Abis...> __impl(
        EqualChunks, const __simd_tuple<_From, _A0, _A1, _Abis...> &__x)
    {
        using _R = __return_type<_A0, _A1, _Abis...>;
        using _Rem = typename _R::_Second_type;
        __simd_converter<_From, _A0, _To, typename _R::_First_abi> __native_cvt;
        return {__native_cvt(__x.first),
                __impl(_ChunkRelation<_A1, typename _Rem::_First_abi>(), __x.second)};
    }

    //}}}2
};

template <class _From, class _To, int _N>
struct __simd_converter<_From, simd_abi::fixed_size<_N>, _To, simd_abi::fixed_size<_N>>
    : public __fixed_size_converter<_From, _To> {
    using __base = __fixed_size_converter<_From, _To>;
    using __return_type = __fixed_size_storage<_To, _N>;
    using __arg = __fixed_size_storage<_From, _N>;

    _GLIBCXX_SIMD_INTRINSIC __return_type operator()(const __arg &__x)
    {
        using _CR =
            typename __base::template _ChunkRelation<typename __arg::_First_abi,
                                                     typename __return_type::_First_abi>;
        return __base::__impl(_CR(), __x);
    }
};

// __simd_converter "native" -> fixed_size<_N> {{{1
// i.e. 1 register to ? registers
template <class _From, class _A, class _To, int _N>
struct __simd_converter<_From, _A, _To, simd_abi::fixed_size<_N>> {
    using __traits = __simd_traits<_From, _A>;
    using arg = typename __traits::_Simd_member_type;
    using __return_type = __fixed_size_storage<_To, _N>;
    static_assert(_N == simd_size_v<_From, _A>,
                  "__simd_converter to fixed_size only works for equal element counts");

    _GLIBCXX_SIMD_INTRINSIC __return_type operator()(arg __x)
    {
        return __impl(std::make_index_sequence<__return_type::tuple_size>(), __x);
    }

private:
    __return_type __impl(std::index_sequence<0>, arg __x)
    {
        __simd_converter<_From, _A, _To, typename __return_type::_First_abi> __native_cvt;
        return {__native_cvt(__x)};
    }
    template <size_t... _Indexes> __return_type __impl(std::index_sequence<_Indexes...>, arg __x)
    {
        __simd_converter<_From, _A, _To, typename __return_type::_First_abi> __native_cvt;
        const auto &tmp = __native_cvt.__all(__x);
        return {tmp[_Indexes]...};
    }
};

// __simd_converter fixed_size<_N> -> "native" {{{1
// i.e. ? register to 1 registers
template <class _From, int _N, class _To, class _A>
struct __simd_converter<_From, simd_abi::fixed_size<_N>, _To, _A> {
    using __traits = __simd_traits<_To, _A>;
    using __return_type = typename __traits::_Simd_member_type;
    using arg = __fixed_size_storage<_From, _N>;
    static_assert(_N == simd_size_v<_To, _A>,
                  "__simd_converter to fixed_size only works for equal element counts");

    _GLIBCXX_SIMD_INTRINSIC __return_type operator()(arg __x)
    {
        return __impl(std::make_index_sequence<arg::tuple_size>(), __x);
    }

private:
    template <size_t... _Indexes> __return_type __impl(std::index_sequence<_Indexes...>, arg __x)
    {
        __simd_converter<_From, typename arg::_First_abi, _To, _A> __native_cvt;
        return __native_cvt(__get_tuple_at<_Indexes>(__x)...);
    }
};

// }}}1
_GLIBCXX_SIMD_END_NAMESPACE
#endif  // __cplusplus >= 201703L
#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_ABIS_H_
// vim: foldmethod=marker
