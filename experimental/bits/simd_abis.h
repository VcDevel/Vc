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
template <typename _Tp> _Tp __subscript_read(_Vectorizable<_Tp> __x, size_t) noexcept
{
    return __x;
}
template <typename _Tp>
void __subscript_write(_Vectorizable<_Tp> &__x, size_t, __id<_Tp> __y) noexcept
{
    __x = __y;
}

template <typename _Tp>
typename _Tp::value_type __subscript_read(const _Tp &__x, size_t __i) noexcept
{
    return __x[__i];
}
template <typename _Tp>
void __subscript_write(_Tp &__x, size_t __i, typename _Tp::value_type __y) noexcept
{
    return __x.set(__i, __y);
}

// __simd_tuple_element {{{1
template <size_t _I, typename _Tp> struct __simd_tuple_element;
template <typename _Tp, typename _A0, typename... _As>
struct __simd_tuple_element<0, __simd_tuple<_Tp, _A0, _As...>> {
    using type = std::experimental::simd<_Tp, _A0>;
};
template <size_t _I, typename _Tp, typename _A0, typename... _As>
struct __simd_tuple_element<_I, __simd_tuple<_Tp, _A0, _As...>> {
    using type = typename __simd_tuple_element<_I - 1, __simd_tuple<_Tp, _As...>>::type;
};
template <size_t _I, typename _Tp>
using __simd_tuple_element_t = typename __simd_tuple_element<_I, _Tp>::type;

// __simd_tuple_concat {{{1
template <typename _Tp, typename... _A1s>
_GLIBCXX_SIMD_INTRINSIC constexpr __simd_tuple<_Tp, _A1s...>
  __simd_tuple_concat(const __simd_tuple<_Tp>&,
		      const __simd_tuple<_Tp, _A1s...>& __right)
{
  return __right;
}

template <typename _Tp,
	  typename _A00,
	  typename... _A0s,
	  typename _A10,
	  typename... _A1s>
_GLIBCXX_SIMD_INTRINSIC constexpr __simd_tuple<_Tp,
					       _A00,
					       _A0s...,
					       _A10,
					       _A1s...>
  __simd_tuple_concat(const __simd_tuple<_Tp, _A00, _A0s...>& __left,
		      const __simd_tuple<_Tp, _A10, _A1s...>& __right)
{
  return {__left.first, __simd_tuple_concat(__left.second, __right)};
}

template <typename _Tp, typename _A00, typename... _A0s>
_GLIBCXX_SIMD_INTRINSIC constexpr __simd_tuple<_Tp, _A00, _A0s...>
  __simd_tuple_concat(const __simd_tuple<_Tp, _A00, _A0s...>& __left,
		      const __simd_tuple<_Tp>&)
{
  return __left;
}

template <typename _Tp, typename _A10, typename... _A1s>
_GLIBCXX_SIMD_INTRINSIC constexpr __simd_tuple<_Tp,
					       simd_abi::scalar,
					       _A10,
					       _A1s...>
  __simd_tuple_concat(const _Tp&                              __left,
		      const __simd_tuple<_Tp, _A10, _A1s...>& __right)
{
  return {__left, __right};
}

// __simd_tuple_pop_front {{{1
template <typename _Tp>
_GLIBCXX_SIMD_INTRINSIC constexpr const _Tp &__simd_tuple_pop_front(_SizeConstant<0>,
                                                                   const _Tp &__x)
{
    return __x;
}
template <typename _Tp>
_GLIBCXX_SIMD_INTRINSIC constexpr _Tp &__simd_tuple_pop_front(_SizeConstant<0>, _Tp &__x)
{
    return __x;
}
template <size_t _K, typename _Tp>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto &__simd_tuple_pop_front(_SizeConstant<_K>,
                                                                     const _Tp &__x)
{
    return __simd_tuple_pop_front(_SizeConstant<_K - 1>(), __x.second);
}
template <size_t _K, typename _Tp>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__simd_tuple_pop_front(_SizeConstant<_K>, _Tp &__x)
{
    return __simd_tuple_pop_front(_SizeConstant<_K - 1>(), __x.second);
}

// __get_simd_at<_N> {{{1
struct __as_simd {};
struct __as_simd_tuple {};
template <typename _Tp, typename _A0, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr simd<_Tp, _A0> __simd_tuple_get_impl(
    __as_simd, const __simd_tuple<_Tp, _A0, _Abis...> &__t, _SizeConstant<0>)
{
    return {__private_init, __t.first};
}
template <typename _Tp, typename _A0, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto &__simd_tuple_get_impl(
    __as_simd_tuple, const __simd_tuple<_Tp, _A0, _Abis...> &__t, _SizeConstant<0>)
{
    return __t.first;
}
template <typename _Tp, typename _A0, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__simd_tuple_get_impl(
    __as_simd_tuple, __simd_tuple<_Tp, _A0, _Abis...> &__t, _SizeConstant<0>)
{
    return __t.first;
}

template <typename _R, size_t _N, typename _Tp, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __simd_tuple_get_impl(
    _R, const __simd_tuple<_Tp, _Abis...> &__t, _SizeConstant<_N>)
{
    return __simd_tuple_get_impl(_R(), __t.second, _SizeConstant<_N - 1>());
}
template <size_t _N, typename _Tp, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__simd_tuple_get_impl(
    __as_simd_tuple, __simd_tuple<_Tp, _Abis...> &__t, _SizeConstant<_N>)
{
    return __simd_tuple_get_impl(__as_simd_tuple(), __t.second, _SizeConstant<_N - 1>());
}

template <size_t _N, typename _Tp, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __get_simd_at(const __simd_tuple<_Tp, _Abis...> &__t)
{
    return __simd_tuple_get_impl(__as_simd(), __t, _SizeConstant<_N>());
}

// }}}
// __get_tuple_at<_N> {{{
template <size_t _N, typename _Tp, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __get_tuple_at(const __simd_tuple<_Tp, _Abis...> &__t)
{
    return __simd_tuple_get_impl(__as_simd_tuple(), __t, _SizeConstant<_N>());
}

template <size_t _N, typename _Tp, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__get_tuple_at(__simd_tuple<_Tp, _Abis...> &__t)
{
    return __simd_tuple_get_impl(__as_simd_tuple(), __t, _SizeConstant<_N>());
}

// __how_many_to_extract {{{1
template <size_t _LeftN, typename _RightT> constexpr size_t __tuple_elements_for () {
    if constexpr (_LeftN == 0) {
        return 0;
    } else {
        return 1 + __tuple_elements_for<_LeftN - _RightT::_S_first_size,
                                        typename _RightT::_Second_type>();
    }
}
template <size_t _LeftN, typename _RightT, bool = (_RightT::_S_first_size < _LeftN)>
struct __how_many_to_extract;
template <size_t _LeftN, typename _RightT> struct __how_many_to_extract<_LeftN, _RightT, true> {
    static constexpr std::make_index_sequence<__tuple_elements_for<_LeftN, _RightT>()> tag()
    {
        return {};
    }
};
template <typename _Tp, size_t _Offset, size_t _Length, bool _Done, typename _IndexSeq>
struct chunked {
};
template <size_t _LeftN, typename _RightT> struct __how_many_to_extract<_LeftN, _RightT, false> {
    static_assert(_LeftN != _RightT::_S_first_size, "");
    static constexpr chunked<typename _RightT::_First_type, 0, _LeftN, false,
                             std::make_index_sequence<_LeftN>>
    tag()
    {
        return {};
    }
};

// __tuple_element_meta {{{1
template <typename _Tp, typename _Abi, size_t _Offset>
struct __tuple_element_meta : public _Abi::_Simd_impl_type {
    using value_type = _Tp;
    using abi_type = _Abi;
    using __traits = __simd_traits<_Tp, _Abi>;
    using maskimpl = typename __traits::_Mask_impl_type;
    using __member_type = typename __traits::_Simd_member_type;
    using _Mask_member_type = typename __traits::_Mask_member_type;
    using simd_type = std::experimental::simd<_Tp, _Abi>;
    static constexpr size_t offset = _Offset;
    static constexpr size_t size() { return simd_size<_Tp, _Abi>::value; }
    static constexpr maskimpl simd_mask = {};

    template <size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type make_mask(std::bitset<_N> __bits)
    {
        constexpr _Tp *__type_tag = nullptr;
        return maskimpl::__from_bitset(std::bitset<size()>((__bits >> _Offset).to_ullong()),
                                     __type_tag);
    }

    _GLIBCXX_SIMD_INTRINSIC static _ULLong mask_to_shifted_ullong(_Mask_member_type __k)
    {
        return __vector_to_bitset(__k).to_ullong() << _Offset;
    }
};

template <size_t _Offset, typename _Tp, typename _Abi, typename... _As>
__tuple_element_meta<_Tp, _Abi, _Offset> make_meta(const __simd_tuple<_Tp, _Abi, _As...> &)
{
    return {};
}

// __simd_tuple specializations {{{1
// empty {{{2
template <typename _Tp> struct __simd_tuple<_Tp> {
    using value_type = _Tp;
    static constexpr size_t tuple_size = 0;
    static constexpr size_t size() { return 0; }
};

// 1 member {{{2
template <typename _Tp, typename _Abi0> struct __simd_tuple<_Tp, _Abi0> {
    using value_type = _Tp;
    using _First_type = typename __simd_traits<_Tp, _Abi0>::_Simd_member_type;
    using _Second_type = __simd_tuple<_Tp>;
    using _First_abi = _Abi0;
    static constexpr size_t tuple_size = 1;
    static constexpr size_t size() { return simd_size_v<_Tp, _Abi0>; }
    static constexpr size_t _S_first_size = simd_size_v<_Tp, _Abi0>;
    /*alignas(__next_power_of_2(sizeof(_First_type)))*/ _First_type first;
    static constexpr _Second_type second = {};

    template <size_t _Offset = 0, typename _F>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple generate(_F &&__gen, _SizeConstant<_Offset> = {})
    {
        return {__gen(__tuple_element_meta<_Tp, _Abi0, _Offset>())};
    }

    template <size_t _Offset = 0, typename _F, typename... _More>
    _GLIBCXX_SIMD_INTRINSIC __simd_tuple apply_wrapped(_F &&__fun, const _More &... __more) const
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return {__fun(make_meta<_Offset>(*this), first, __more.first...)};
    }

    template <typename _F, typename... _More>
    _GLIBCXX_SIMD_INTRINSIC constexpr friend __simd_tuple
      __simd_tuple_apply(_F&& __fun, const __simd_tuple& __x, _More&&... __more)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return __simd_tuple::apply_impl(
            __bool_constant<conjunction<__is_equal<
                size_t, _S_first_size, std::decay_t<_More>::_S_first_size>...>::value>(),
            std::forward<_F>(__fun), __x, std::forward<_More>(__more)...);
    }

  private:
    template <typename _F, typename... _More>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __simd_tuple
      apply_impl(true_type, // _S_first_size is equal for all arguments
		 _F&&                __fun,
		 const __simd_tuple& __x,
		 _More&&... __more)
    {
      _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
      //_GLIBCXX_SIMD_DEBUG_DEFERRED("__more.first = ", __more.first..., "__more
      //=
      //", __more...);
      return {__fun(__tuple_element_meta<_Tp, _Abi0, 0>(), __x.first,
		    __more.first...)};
    }

    template <typename _F, typename _More>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __simd_tuple apply_impl(
      false_type, // at least one argument in _More has different _S_first_size,
		  // __x has only one member, so _More has 2 or more
      _F&&                __fun,
      const __simd_tuple& __x,
      _More&&             __y)
    {
      _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
      //_GLIBCXX_SIMD_DEBUG_DEFERRED("__y = ", __y);
      return apply_impl(
	std::make_index_sequence<std::decay_t<_More>::tuple_size>(),
	std::forward<_F>(__fun), __x, std::forward<_More>(__y));
    }

    template <typename _F, typename _More, size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __simd_tuple
      apply_impl(std::index_sequence<_Indexes...>,
		 _F&&                __fun,
		 const __simd_tuple& __x,
		 _More&&             __y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("__y = ", __y);
        auto tmp = std::experimental::concat(__get_simd_at<_Indexes>(__y)...);
        const auto first = __fun(__tuple_element_meta<_Tp, _Abi0, 0>(), __x.first, tmp);
        if constexpr (std::is_lvalue_reference<_More>::value &&
                      !std::is_const<_More>::value) {
            // if __y is non-const lvalue ref, assume write back is necessary
            const auto __tup =
                std::experimental::split<__simd_tuple_element_t<_Indexes, std::decay_t<_More>>::size()...>(tmp);
            auto &&ignore = {
                (__get_tuple_at<_Indexes>(__y) = __data(std::get<_Indexes>(__tup)), 0)...};
            __unused(ignore);
        }
        return {first};
    }

  public:
    // apply_impl2 can only be called from a 2-element __simd_tuple
    template <typename _Tuple, size_t _Offset, typename _F2>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __simd_tuple
      __extract(_SizeConstant<_Offset>,
		_SizeConstant<std::decay_t<_Tuple>::_S_first_size - _Offset>,
		_Tuple&& __tup,
		_F2&&    __fun2)
    {
        static_assert(_Offset > 0, "");
        auto __splitted =
            split<_Offset, std::decay_t<_Tuple>::_S_first_size - _Offset>(__get_simd_at<0>(__tup));
        __simd_tuple __r = __fun2(__data(std::get<1>(__splitted)));
        // if __tup is non-const lvalue ref, write __get_tuple_at<0>(__splitted) back
        __tup.first = __data(concat(std::get<0>(__splitted), std::get<1>(__splitted)));
        return __r;
    }

    template <typename _F,
	      typename _More,
	      typename _U,
	      size_t _Length,
	      size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __simd_tuple
      apply_impl2(chunked<_U,
			  std::decay_t<_More>::_S_first_size,
			  _Length,
			  true,
			  std::index_sequence<_Indexes...>>,
		  _F&&                __fun,
		  const __simd_tuple& __x,
		  _More&&             __y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return __simd_tuple_apply(std::forward<_F>(__fun), __x, __y.second);
    }

    template <class _F,
	      class _More,
	      class _U,
	      size_t _Offset,
	      size_t _Length,
	      size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __simd_tuple apply_impl2(
      chunked<_U, _Offset, _Length, false, std::index_sequence<_Indexes...>>,
      _F&&                __fun,
      const __simd_tuple& __x,
      _More&&             __y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        static_assert(_Offset < std::decay_t<_More>::_S_first_size, "");
        static_assert(_Offset > 0, "");
        return __extract(_SizeConstant<_Offset>(), _SizeConstant<_Length>(), __y,
                       [&](auto &&__yy) -> __simd_tuple {
                           return {__fun(__tuple_element_meta<_Tp, _Abi0, 0>(), __x.first, __yy)};
                       });
    }

    template <class _R = _Tp, class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC __fixed_size_storage<_R, size()> apply_r(_F &&__fun,
                                                       const _More &... __more) const
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return {__fun(__tuple_element_meta<_Tp, _Abi0, 0>(), first, __more.first...)};
    }

    template <class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC friend std::bitset<size()> test(_F &&__fun, const __simd_tuple &__x,
                                                 const _More &... __more)
    {
        return __vector_to_bitset(
            __fun(__tuple_element_meta<_Tp, _Abi0, 0>(), __x.first, __more.first...));
    }

    _Tp operator[](size_t __i) const noexcept { return __subscript_read(first, __i); }
    void set(size_t __i, _Tp val) noexcept { __subscript_write(first, __i, val); }
};

// 2 or more {{{2
template <class _Tp, class _Abi0, class... _Abis> struct __simd_tuple<_Tp, _Abi0, _Abis...> {
    using value_type = _Tp;
    using _First_type = typename __simd_traits<_Tp, _Abi0>::_Simd_member_type;
    using _First_abi = _Abi0;
    using _Second_type = __simd_tuple<_Tp, _Abis...>;
    static constexpr size_t tuple_size = sizeof...(_Abis) + 1;
    static constexpr size_t size() { return simd_size_v<_Tp, _Abi0> + _Second_type::size(); }
    static constexpr size_t _S_first_size = simd_size_v<_Tp, _Abi0>;
    //static constexpr size_t alignment = __next_power_of_2(sizeof(_Tp) * size());
    //alignas(alignment)
    _First_type first;
    _Second_type second;

    template <size_t _Offset = 0, class _F>
    _GLIBCXX_SIMD_INTRINSIC static __simd_tuple generate(_F &&__gen, _SizeConstant<_Offset> = {})
    {
        return {__gen(__tuple_element_meta<_Tp, _Abi0, _Offset>()),
                _Second_type::generate(
                    std::forward<_F>(__gen),
                    _SizeConstant<_Offset + simd_size_v<_Tp, _Abi0>>())};
    }

    template <size_t _Offset = 0, class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC __simd_tuple apply_wrapped(_F &&__fun, const _More &... __more) const
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return {__fun(make_meta<_Offset>(*this), first, __more.first...),
                second.template apply_wrapped<_Offset + simd_size_v<_Tp, _Abi0>>(
                    std::forward<_F>(__fun), __more.second...)};
    }

    template <class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC constexpr friend __simd_tuple
      __simd_tuple_apply(_F&& __fun, const __simd_tuple& __x, _More&&... __more)
    {
      _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
      //_GLIBCXX_SIMD_DEBUG_DEFERRED("__more = ", __more...);
      return __simd_tuple::apply_impl(
	__bool_constant<conjunction<
	  __is_equal<size_t, _S_first_size,
		     std::decay_t<_More>::_S_first_size>...>::value>(),
	std::forward<_F>(__fun), __x, std::forward<_More>(__more)...);
    }

  private:
    template <class _F, class... _More>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __simd_tuple
      apply_impl(true_type, // _S_first_size is equal for all arguments
		 _F&&                __fun,
		 const __simd_tuple& __x,
		 _More&&... __more)
    {
      _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
      return {__fun(__tuple_element_meta<_Tp, _Abi0, 0>(), __x.first,
		    __more.first...),
	      __simd_tuple_apply(std::forward<_F>(__fun), __x.second,
				 __more.second...)};
    }

    template <class _F, class _More>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __simd_tuple apply_impl(
      false_type, // at least one argument in _More has different _S_first_size
      _F&&                __fun,
      const __simd_tuple& __x,
      _More&&             __y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("__y = ", __y);
        return apply_impl2(__how_many_to_extract<_S_first_size, std::decay_t<_More>>::tag(),
                           std::forward<_F>(__fun), __x, __y);
    }

    template <class _F, class _More, size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __simd_tuple
      apply_impl2(std::index_sequence<_Indexes...>,
		  _F&&                __fun,
		  const __simd_tuple& __x,
		  _More&&             __y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        //_GLIBCXX_SIMD_DEBUG_DEFERRED("__y = ", __y);
        auto tmp = std::experimental::concat(__get_simd_at<_Indexes>(__y)...);
        const auto first = __fun(__tuple_element_meta<_Tp, _Abi0, 0>(), __x.first, tmp);
        if constexpr (std::is_lvalue_reference<_More>::value &&
                      !std::is_const<_More>::value) {
            // if __y is non-const lvalue ref, assume write back is necessary
            const auto __tup =
                std::experimental::split<__simd_tuple_element_t<_Indexes, std::decay_t<_More>>::size()...>(tmp);
            [](std::initializer_list<int>) {
            }({(__get_tuple_at<_Indexes>(__y) = __data(std::get<_Indexes>(__tup)), 0)...});
        }
        return {first, __simd_tuple_apply(
                           std::forward<_F>(__fun), __x.second,
                           __simd_tuple_pop_front(_SizeConstant<sizeof...(_Indexes)>(), __y))};
    }

  public:
    template <typename _F,
	      typename _More,
	      typename _U,
	      size_t _Length,
	      size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __simd_tuple
      apply_impl2(chunked<_U,
			  std::decay_t<_More>::_S_first_size,
			  _Length,
			  true,
			  std::index_sequence<_Indexes...>>,
		  _F&&                __fun,
		  const __simd_tuple& __x,
		  _More&&             __y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return __simd_tuple_apply(std::forward<_F>(__fun), __x, __y.second);
    }

    template <typename _Tuple, size_t _Length, typename _F2>
    _GLIBCXX_SIMD_INTRINSIC static auto __extract(_SizeConstant<0>, _SizeConstant<_Length>, _Tuple &&__tup,
                                     _F2 &&__fun2)
    {
        auto __splitted =
            split<_Length, std::decay_t<_Tuple>::_S_first_size - _Length>(__get_simd_at<0>(__tup));
        auto __r = __fun2(__data(std::get<0>(__splitted)));
        // if __tup is non-const lvalue ref, write __get_tuple_at<0>(__splitted) back
        __tup.first = __data(concat(std::get<0>(__splitted), std::get<1>(__splitted)));
        return __r;
    }

    template <typename _Tuple, size_t _Offset, typename _F2>
    _GLIBCXX_SIMD_INTRINSIC static auto __extract(
        _SizeConstant<_Offset>, _SizeConstant<std::decay_t<_Tuple>::_S_first_size - _Offset>,
        _Tuple &&__tup, _F2 &&__fun2)
    {
        auto __splitted =
            split<_Offset, std::decay_t<_Tuple>::_S_first_size - _Offset>(__get_simd_at<0>(__tup));
        auto __r = __fun2(__data(std::get<1>(__splitted)));
        // if __tup is non-const lvalue ref, write __get_tuple_at<0>(__splitted) back
        __tup.first = __data(concat(std::get<0>(__splitted), std::get<1>(__splitted)));
        return __r;
    }

    template <
        typename _Tuple, size_t _Offset, size_t _Length, typename _F2,
        typename = enable_if_t<(_Offset + _Length < std::decay_t<_Tuple>::_S_first_size)>>
    _GLIBCXX_SIMD_INTRINSIC static auto __extract(_SizeConstant<_Offset>, _SizeConstant<_Length>,
                                     _Tuple &&__tup, _F2 &&__fun2)
    {
        static_assert(_Offset + _Length < std::decay_t<_Tuple>::_S_first_size, "");
        auto __splitted =
            split<_Offset, _Length, std::decay_t<_Tuple>::_S_first_size - _Offset - _Length>(
                __get_simd_at<0>(__tup));
        auto __r = __fun2(__data(std::get<1>(__splitted)));
        // if __tup is non-const lvalue ref, write __get_tuple_at<0>(__splitted) back
        __tup.first = __data(
            concat(std::get<0>(__splitted), std::get<1>(__splitted), std::get<2>(__splitted)));
        return __r;
    }

    template <typename _F,
	      typename _More,
	      typename _U,
	      size_t _Offset,
	      size_t _Length,
	      size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __simd_tuple apply_impl2(
      chunked<_U, _Offset, _Length, false, std::index_sequence<_Indexes...>>,
      _F&&                __fun,
      const __simd_tuple& __x,
      _More&&             __y)
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        static_assert(_Offset < std::decay_t<_More>::_S_first_size, "");
        return {__extract(_SizeConstant<_Offset>(), _SizeConstant<_Length>(), __y,
                        [&](auto &&__yy) {
                            return __fun(__tuple_element_meta<_Tp, _Abi0, 0>(), __x.first, __yy);
                        }),
                _Second_type::apply_impl2(
                    chunked<_U, _Offset + _Length, _Length,
                            _Offset + _Length == std::decay_t<_More>::_S_first_size,
                            std::index_sequence<_Indexes...>>(),
                    std::forward<_F>(__fun), __x.second, __y)};
    }

    template <typename _R = _Tp, typename _F, typename... _More>
    _GLIBCXX_SIMD_INTRINSIC auto apply_r(_F &&__fun, const _More &... __more) const
    {
        _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
        return __simd_tuple_concat<_R>(
            __fun(__tuple_element_meta<_Tp, _Abi0, 0>(), first, __more.first...),
            second.template apply_r<_R>(std::forward<_F>(__fun), __more.second...));
    }

    template <typename _F, typename... _More>
    _GLIBCXX_SIMD_INTRINSIC friend std::bitset<size()> test(_F &&__fun, const __simd_tuple &__x,
                                                 const _More &... __more)
    {
        return __vector_to_bitset(
                   __fun(__tuple_element_meta<_Tp, _Abi0, 0>(), __x.first, __more.first...))
                   .to_ullong() |
               (test(__fun, __x.second, __more.second...).to_ullong() << simd_size_v<_Tp, _Abi0>);
    }

    template <typename _U, _U _I>
    _GLIBCXX_SIMD_INTRINSIC constexpr _Tp operator[](std::integral_constant<_U, _I>) const noexcept
    {
        if constexpr (_I < simd_size_v<_Tp, _Abi0>) {
            return __subscript_read(first, _I);
        } else {
            return second[std::integral_constant<_U, _I - simd_size_v<_Tp, _Abi0>>()];
        }
    }

    _Tp operator[](size_t __i) const noexcept
    {
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        return reinterpret_cast<const __may_alias<_Tp> *>(this)[__i];
#else
      if constexpr (__is_abi<_Abi0, simd_abi::scalar>())
	{
	  const _Tp* ptr = &first;
	  return ptr[__i];
	}
      else
	{
	  return __i < simd_size_v<_Tp, _Abi0>
		   ? __subscript_read(first, __i)
		   : second[__i - simd_size_v<_Tp, _Abi0>];
	}
#endif
    }
    void set(size_t __i, _Tp val) noexcept
    {
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        reinterpret_cast<__may_alias<_Tp> *>(this)[__i] = val;
#else
        if (__i < simd_size_v<_Tp, _Abi0>) {
            __subscript_write(first, __i, val);
        } else {
            second.set(__i - simd_size_v<_Tp, _Abi0>, val);
        }
#endif
    }
};

// __make_simd_tuple {{{1
template <typename _Tp, typename _A0>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_Tp, _A0> __make_simd_tuple(
    std::experimental::simd<_Tp, _A0> __x0)
{
    return {__data(__x0)};
}
template <typename _Tp, typename _A0, typename... _As>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_Tp, _A0, _As...> __make_simd_tuple(
    const std::experimental::simd<_Tp, _A0> &__x0,
    const std::experimental::simd<_Tp, _As> &... __xs)
{
    return {__data(__x0), __make_simd_tuple(__xs...)};
}

template <typename _Tp, typename _A0>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_Tp, _A0> __make_simd_tuple(
    const typename __simd_traits<_Tp, _A0>::_Simd_member_type &arg0)
{
    return {arg0};
}

template <typename _Tp, typename _A0, typename _A1, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_Tp, _A0, _A1, _Abis...> __make_simd_tuple(
    const typename __simd_traits<_Tp, _A0>::_Simd_member_type &arg0,
    const typename __simd_traits<_Tp, _A1>::_Simd_member_type &arg1,
    const typename __simd_traits<_Tp, _Abis>::_Simd_member_type &... args)
{
    return {arg0, __make_simd_tuple<_Tp, _A1, _Abis...>(arg1, args...)};
}

// __to_simd_tuple {{{1
template <size_t, class _Tp> using __to_tuple_helper = _Tp;
template <class _Tp, class _A0, size_t... _Indexes>
_GLIBCXX_SIMD_INTRINSIC __simd_tuple<_Tp, __to_tuple_helper<_Indexes, _A0>...>
__to_simd_tuple_impl(std::index_sequence<_Indexes...>,
                     const std::array<__vector_type_t<_Tp, simd_size_v<_Tp, _A0>>,
                                      sizeof...(_Indexes)> &args)
{
    return __make_simd_tuple<_Tp, __to_tuple_helper<_Indexes, _A0>...>(args[_Indexes]...);
}

template <class _Tp, class _A0, size_t _N>
_GLIBCXX_SIMD_INTRINSIC auto __to_simd_tuple(
    const std::array<__vector_type_t<_Tp, simd_size_v<_Tp, _A0>>, _N> &args)
{
    return __to_simd_tuple_impl<_Tp, _A0>(std::make_index_sequence<_N>(), args);
}

// __optimize_simd_tuple {{{1
template <class _Tp> _GLIBCXX_SIMD_INTRINSIC __simd_tuple<_Tp> __optimize_simd_tuple(const __simd_tuple<_Tp>)
{
    return {};
}

template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC const __simd_tuple<_Tp, _A> &__optimize_simd_tuple(const __simd_tuple<_Tp, _A> &__x)
{
    return __x;
}

template <class _Tp, class _A0, class _A1, class... _Abis,
          class _R = __fixed_size_storage<_Tp, __simd_tuple<_Tp, _A0, _A1, _Abis...>::size()>>
_GLIBCXX_SIMD_INTRINSIC _R __optimize_simd_tuple(const __simd_tuple<_Tp, _A0, _A1, _Abis...> &__x)
{
    using _Tup = __simd_tuple<_Tp, _A0, _A1, _Abis...>;
    if constexpr (std::is_same_v<_R, _Tup>)
      {
	return __x;
      }
    else if constexpr (_R::_S_first_size == simd_size_v<_Tp, _A0>)
      {
	return __simd_tuple_concat(__simd_tuple<_Tp, typename _R::_First_abi>{__x.first},
                            __optimize_simd_tuple(__x.second));
      }
    else if constexpr (_R::_S_first_size ==
		       simd_size_v<_Tp, _A0> + simd_size_v<_Tp, _A1>)
      {
	return __simd_tuple_concat(__simd_tuple<_Tp, typename _R::_First_abi>{__data(
                                std::experimental::concat(__get_simd_at<0>(__x), __get_simd_at<1>(__x)))},
                            __optimize_simd_tuple(__x.second.second));
      }
    else if constexpr (_R::_S_first_size ==
		       4 * __simd_tuple_element_t<0, _Tup>::size())
      {
	return __simd_tuple_concat(
	  __simd_tuple<_Tp, typename _R::_First_abi>{
	    __data(concat(__get_simd_at<0>(__x), __get_simd_at<1>(__x),
			  __get_simd_at<2>(__x), __get_simd_at<3>(__x)))},
	  __optimize_simd_tuple(__x.second.second.second.second));
      }
    else if constexpr (_R::_S_first_size ==
		       8 * __simd_tuple_element_t<0, _Tup>::size())
      {
	return __simd_tuple_concat(
	  __simd_tuple<_Tp, typename _R::_First_abi>{__data(concat(
	    __get_simd_at<0>(__x), __get_simd_at<1>(__x), __get_simd_at<2>(__x),
	    __get_simd_at<3>(__x), __get_simd_at<4>(__x), __get_simd_at<5>(__x),
	    __get_simd_at<6>(__x), __get_simd_at<7>(__x)))},
	  __optimize_simd_tuple(
	    __x.second.second.second.second.second.second.second.second));
      }
    else if constexpr (_R::_S_first_size ==
		       16 * __simd_tuple_element_t<0, _Tup>::size())
      {
	return __simd_tuple_concat(
	  __simd_tuple<_Tp, typename _R::_First_abi>{__data(concat(
	    __get_simd_at<0>(__x), __get_simd_at<1>(__x), __get_simd_at<2>(__x),
	    __get_simd_at<3>(__x), __get_simd_at<4>(__x), __get_simd_at<5>(__x),
	    __get_simd_at<6>(__x), __get_simd_at<7>(__x), __get_simd_at<8>(__x),
	    __get_simd_at<9>(__x), __get_simd_at<10>(__x),
	    __get_simd_at<11>(__x), __get_simd_at<12>(__x),
	    __get_simd_at<13>(__x), __get_simd_at<14>(__x),
	    __get_simd_at<15>(__x)))},
	  __optimize_simd_tuple(
	    __x.second.second.second.second.second.second.second.second.second
	      .second.second.second.second.second.second.second));
      }
    else
      {
	return __x;
      }
}

// __for_each(const __simd_tuple &, Fun) {{{1
template <size_t _Offset = 0, class _Tp, class _A0, class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(const __simd_tuple<_Tp, _A0>& __t, _F&& __fun)
{
  std::forward<_F>(__fun)(make_meta<_Offset>(__t), __t.first);
}
template <size_t _Offset = 0,
	  class _Tp,
	  class _A0,
	  class _A1,
	  class... _As,
	  class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(const __simd_tuple<_Tp, _A0, _A1, _As...>& __t, _F&& __fun)
{
  __fun(make_meta<_Offset>(__t), __t.first);
  __for_each<_Offset + simd_size<_Tp, _A0>::value>(__t.second,
						   std::forward<_F>(__fun));
}

// __for_each(__simd_tuple &, Fun) {{{1
template <size_t _Offset = 0, class _Tp, class _A0, class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(__simd_tuple<_Tp, _A0>& __t, _F&& __fun)
{
  std::forward<_F>(__fun)(make_meta<_Offset>(__t), __t.first);
}
template <size_t _Offset = 0,
	  class _Tp,
	  class _A0,
	  class _A1,
	  class... _As,
	  class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(__simd_tuple<_Tp, _A0, _A1, _As...>& __t, _F&& __fun)
{
  __fun(make_meta<_Offset>(__t), __t.first);
  __for_each<_Offset + simd_size<_Tp, _A0>::value>(__t.second,
						   std::forward<_F>(__fun));
}

// __for_each(__simd_tuple &, const __simd_tuple &, Fun) {{{1
template <size_t _Offset = 0, class _Tp, class _A0, class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(__simd_tuple<_Tp, _A0>&       __a,
	     const __simd_tuple<_Tp, _A0>& __b,
	     _F&&                          __fun)
{
  std::forward<_F>(__fun)(make_meta<_Offset>(__a), __a.first, __b.first);
}
template <size_t _Offset = 0,
	  class _Tp,
	  class _A0,
	  class _A1,
	  class... _As,
	  class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(__simd_tuple<_Tp, _A0, _A1, _As...>&       __a,
	     const __simd_tuple<_Tp, _A0, _A1, _As...>& __b,
	     _F&&                                      __fun)
{
  __fun(make_meta<_Offset>(__a), __a.first, __b.first);
  __for_each<_Offset + simd_size<_Tp, _A0>::value>(__a.second, __b.second,
						   std::forward<_F>(__fun));
}

// __for_each(const __simd_tuple &, const __simd_tuple &, Fun) {{{1
template <size_t _Offset = 0, class _Tp, class _A0, class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(const __simd_tuple<_Tp, _A0>& __a,
	     const __simd_tuple<_Tp, _A0>& __b,
	     _F&&                          __fun)
{
  std::forward<_F>(__fun)(make_meta<_Offset>(__a), __a.first, __b.first);
}
template <size_t _Offset = 0,
	  class _Tp,
	  class _A0,
	  class _A1,
	  class... _As,
	  class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(const __simd_tuple<_Tp, _A0, _A1, _As...>& __a,
	     const __simd_tuple<_Tp, _A0, _A1, _As...>& __b,
	     _F&&                                      __fun)
{
  __fun(make_meta<_Offset>(__a), __a.first, __b.first);
  __for_each<_Offset + simd_size<_Tp, _A0>::value>(__a.second, __b.second,
						   std::forward<_F>(__fun));
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
template <unsigned __shift, class _Tp, class _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC _Tp __shift16_right(_Tp __v)
{
    static_assert(__shift <= sizeof(_Tp));
    static_assert(sizeof(_Tp) == 16);
    if constexpr (__shift == 0) {
        return __v;
    } else if constexpr(__shift == sizeof(_Tp)) {
        return _Tp();
    } else if constexpr (__have_sse && __shift == 8 && _TVT::template is<float, 4>) {
        return _mm_movehl_ps(__v, __v);
    } else if constexpr (__have_sse2 && __shift == 8 && _TVT::template is<double, 2>) {
        return _mm_unpackhi_pd(__v, __v);
    } else if constexpr (__have_sse2 && sizeof(_Tp) == 16) {
        return __intrin_bitcast<_Tp>(
            _mm_srli_si128(__intrin_bitcast<__m128i>(__v), __shift));
/*
    } else if constexpr (__shift == 16 && sizeof(_Tp) == 32) {
        if constexpr (__have_avx && _TVT::template is<double, 4>) {
            return _mm256_permute2f128_pd(__v, __v, 0x81);
        } else if constexpr (__have_avx && _TVT::template is<float, 8>) {
            return _mm256_permute2f128_ps(__v, __v, 0x81);
        } else if constexpr (__have_avx) {
            return _mm256_permute2f128_si256(__v, __v, 0x81);
        } else {
            return __auto_bitcast(__hi128(__v));
        }
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 32) {
        const auto __vi = __intrin_bitcast<__m256i>(__v);
        return __intrin_bitcast<_Tp>(_mm256_srli_si256(
            __shift < 16 ? __vi : _mm256_permute2x128_si256(__vi, __vi, 0x81),
            __shift % 16));
    } else if constexpr (sizeof(_Tp) == 32) {
        __shift % 16
        return __intrin_bitcast<_Tp>(
        __extract<_shift/16, 2>(__v)
        );
    } else if constexpr (__have512f && sizeof(_Tp) == 64) {
        if constexpr (__shift % 8 == 0) {
            return __mm512_alignr_epi64(__m512i(), __intrin_bitcast<__m512i>(__v),
                                        __shift / 8);
        } else if constexpr (__shift % 4 == 0) {
            return __mm512_alignr_epi32(__m512i(), __intrin_bitcast<__m512i>(__v),
                                        __shift / 4);
        } else {
            const auto __shifted = __mm512_alignr_epi8(
                __m512i(), __intrin_bitcast<__m512i>(__v), __shift % 16);
            return __intrin_bitcast<_Tp>(
                __shift < 16
                    ? __shifted
                    : _mm512_shuffle_i32x4(__shifted, __shifted, 0xe4 + (__shift / 16)));
        }
    } else if constexpr (__shift == 32 && sizeof(_Tp) == 64) {
        return __auto_bitcast(__hi256(__v));
    } else if constexpr (__shift % 16 == 0 && sizeof(_Tp) == 64) {
        return __auto_bitcast(__extract<__shift / 16, 4>(__v));
*/
    } else {
        constexpr int __chunksize =
            __shift % 8 == 0 ? 8 : __shift % 4 == 0 ? 4 : __shift % 2 == 0 ? 2 : 1;
        auto __w = __vector_bitcast<__int_with_sizeof_t<__chunksize>>(__v);
        return __intrin_bitcast<_Tp>(decltype(__w){__v[__shift / __chunksize], 0});
    }
}

// }}}
// __cmpord{{{
template <class _Tp, class _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC auto __cmpord(_Tp __x, _Tp __y)
{
    static_assert(is_floating_point_v<typename _TVT::value_type>);
    if constexpr (__have_sse && _TVT::template is<float, 4>) {
        return __intrin_bitcast<_Tp>(_mm_cmpord_ps(__x, __y));
    } else if constexpr (__have_sse2 && _TVT::template is<double, 2>) {
        return __intrin_bitcast<_Tp>(_mm_cmpord_pd(__x, __y));
    } else if constexpr (__have_avx && _TVT::template is<float, 8>) {
        return __intrin_bitcast<_Tp>(_mm256_cmp_ps(__x, __y, _CMP_ORD_Q));
    } else if constexpr (__have_avx && _TVT::template is<double, 4>) {
        return __intrin_bitcast<_Tp>(_mm256_cmp_pd(__x, __y, _CMP_ORD_Q));
    } else if constexpr (__have_avx512f && _TVT::template is<float, 16>) {
        return _mm512_cmp_ps_mask(__x, __y, _CMP_ORD_Q);
    } else if constexpr (__have_avx512f && _TVT::template is<double, 8>) {
        return _mm512_cmp_pd_mask(__x, __y, _CMP_ORD_Q);
    } else {
        _Tp __r;
        __execute_n_times<_TVT::_S_width>(
            [&](auto __i) { __r[__i] = (!isnan(__x[__i]) && !isnan(__y[__i])) ? -1 : 0; });
        return __r;
    }
}

// }}}
// __cmpunord{{{
template <class _Tp, class _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC auto __cmpunord(_Tp __x, _Tp __y)
{
    static_assert(is_floating_point_v<typename _TVT::value_type>);
    if constexpr (__have_sse && _TVT::template is<float, 4>) {
        return __intrin_bitcast<_Tp>(_mm_cmpunord_ps(__x, __y));
    } else if constexpr (__have_sse2 && _TVT::template is<double, 2>) {
        return __intrin_bitcast<_Tp>(_mm_cmpunord_pd(__x, __y));
    } else if constexpr (__have_avx && _TVT::template is<float, 8>) {
        return __intrin_bitcast<_Tp>(_mm256_cmp_ps(__x, __y, _CMP_UNORD_Q));
    } else if constexpr (__have_avx && _TVT::template is<double, 4>) {
        return __intrin_bitcast<_Tp>(_mm256_cmp_pd(__x, __y, _CMP_UNORD_Q));
    } else if constexpr (__have_avx512f && _TVT::template is<float, 16>) {
        return _mm512_cmp_ps_mask(__x, __y, _CMP_UNORD_Q);
    } else if constexpr (__have_avx512f && _TVT::template is<double, 8>) {
        return _mm512_cmp_pd_mask(__x, __y, _CMP_UNORD_Q);
    } else {
        _Tp __r;
        __execute_n_times<_TVT::_S_width>(
            [&](auto __i) { __r[__i] = isunordered(__x[__i], __y[__i]) ? -1 : 0; });
        return __r;
    }
}

// }}}
// __maskstore (non-converting; with optimizations for SSE2-AVX512BWVL) {{{
template <class _Tp, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage64_t<_Tp> __v, _Tp* __mem, _F,
                                         __storage<bool, __storage64_t<_Tp>::_S_width> __k)
{
    [[maybe_unused]] const auto __vi = __to_intrin(__v);
    static_assert(sizeof(__v) == 64 && __have_avx512f);
    if constexpr (__have_avx512bw && sizeof(_Tp) == 1) {
        _mm512_mask_storeu_epi8(__mem, __k, __vi);
    } else if constexpr (__have_avx512bw && sizeof(_Tp) == 2) {
        _mm512_mask_storeu_epi16(__mem, __k, __vi);
    } else if constexpr (__have_avx512f && sizeof(_Tp) == 4) {
        if constexpr (__is_aligned_v<_F, 64> && std::is_integral_v<_Tp>) {
            _mm512_mask_store_epi32(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 64> && std::is_floating_point_v<_Tp>) {
            _mm512_mask_store_ps(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_Tp>) {
            _mm512_mask_storeu_epi32(__mem, __k, __vi);
        } else {
            _mm512_mask_storeu_ps(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512f && sizeof(_Tp) == 8) {
        if constexpr (__is_aligned_v<_F, 64> && std::is_integral_v<_Tp>) {
            _mm512_mask_store_epi64(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 64> && std::is_floating_point_v<_Tp>) {
            _mm512_mask_store_pd(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_Tp>) {
            _mm512_mask_storeu_epi64(__mem, __k, __vi);
        } else {
            _mm512_mask_storeu_pd(__mem, __k, __vi);
        }
    } else if constexpr (__have_sse2) {
        constexpr int _N = 16 / sizeof(_Tp);
        using _M          = __vector_type_t<_Tp, _N>;
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
        __assert_unreachable<_Tp>();
    }
}

template <class _Tp, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage32_t<_Tp> __v, _Tp* __mem, _F,
                                         __storage32_t<_Tp> __k)
{
    [[maybe_unused]] const auto __vi = __vector_bitcast<_LLong>(__v);
    [[maybe_unused]] const auto __ki = __vector_bitcast<_LLong>(__k);
    if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 1) {
        _mm256_mask_storeu_epi8(__mem, _mm256_movepi8_mask(__ki), __vi);
    } else if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 2) {
        _mm256_mask_storeu_epi16(__mem, _mm256_movepi16_mask(__ki), __vi);
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 4 && std::is_integral_v<_Tp>) {
        _mm256_maskstore_epi32(reinterpret_cast<int*>(__mem), __ki, __vi);
    } else if constexpr (sizeof(_Tp) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__mem), __ki,
                            __vector_bitcast<float>(__v));
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 8 && std::is_integral_v<_Tp>) {
        _mm256_maskstore_epi64(reinterpret_cast<_LLong*>(__mem), __ki, __vi);
    } else if constexpr (__have_avx && sizeof(_Tp) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__mem), __ki,
                            __vector_bitcast<double>(__v));
    } else if constexpr (__have_sse2) {
        _mm_maskmoveu_si128(__lo128(__vi), __lo128(__ki), reinterpret_cast<char*>(__mem));
        _mm_maskmoveu_si128(__hi128(__vi), __hi128(__ki),
                            reinterpret_cast<char*>(__mem) + 16);
    } else {
        __assert_unreachable<_Tp>();
    }
}

template <class _Tp, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage32_t<_Tp> __v, _Tp* __mem, _F,
                                         __storage<bool, __storage32_t<_Tp>::_S_width> __k)
{
    [[maybe_unused]] const auto __vi = __to_intrin(__v);
    if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 1) {
        _mm256_mask_storeu_epi8(__mem, __k, __vi);
    } else if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 2) {
        _mm256_mask_storeu_epi16(__mem, __k, __vi);
    } else if constexpr (__have_avx512vl && sizeof(_Tp) == 4) {
        if constexpr (__is_aligned_v<_F, 32> && std::is_integral_v<_Tp>) {
            _mm256_mask_store_epi32(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 32> && std::is_floating_point_v<_Tp>) {
            _mm256_mask_store_ps(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_Tp>) {
            _mm256_mask_storeu_epi32(__mem, __k, __vi);
        } else {
            _mm256_mask_storeu_ps(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512vl && sizeof(_Tp) == 8) {
        if constexpr (__is_aligned_v<_F, 32> && std::is_integral_v<_Tp>) {
            _mm256_mask_store_epi64(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 32> && std::is_floating_point_v<_Tp>) {
            _mm256_mask_store_pd(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_Tp>) {
            _mm256_mask_storeu_epi64(__mem, __k, __vi);
        } else {
            _mm256_mask_storeu_pd(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512f && (sizeof(_Tp) >= 4 || __have_avx512bw)) {
        // use a 512-bit maskstore, using zero-extension of the bitmask
        __maskstore(
            __storage64_t<_Tp>(__intrin_bitcast<__vector_type64_t<_Tp>>(__v._M_data)),
            __mem,
            // careful, vector_aligned has a stricter meaning in the 512-bit maskstore:
            std::conditional_t<std::is_same_v<_F, vector_aligned_tag>,
                               overaligned_tag<32>, _F>(),
            __storage<bool, 64 / sizeof(_Tp)>(__k._M_data));
    } else {
        __maskstore(
            __v, __mem, _F(),
            __storage32_t<_Tp>(__convert_mask<__vector_type_t<_Tp, 32 / sizeof(_Tp)>>(__k)));
    }
}

template <class _Tp, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage16_t<_Tp> __v, _Tp* __mem, _F,
                                         __storage16_t<_Tp> __k)
{
    [[maybe_unused]] const auto __vi = __vector_bitcast<_LLong>(__v);
    [[maybe_unused]] const auto __ki = __vector_bitcast<_LLong>(__k);
    if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 1) {
        _mm_mask_storeu_epi8(__mem, _mm_movepi8_mask(__ki), __vi);
    } else if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 2) {
        _mm_mask_storeu_epi16(__mem, _mm_movepi16_mask(__ki), __vi);
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 4 && std::is_integral_v<_Tp>) {
        _mm_maskstore_epi32(reinterpret_cast<int*>(__mem), __ki, __vi);
    } else if constexpr (__have_avx && sizeof(_Tp) == 4) {
        _mm_maskstore_ps(reinterpret_cast<float*>(__mem), __ki,
                         __vector_bitcast<float>(__v));
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 8 && std::is_integral_v<_Tp>) {
        _mm_maskstore_epi64(reinterpret_cast<_LLong*>(__mem), __ki, __vi);
    } else if constexpr (__have_avx && sizeof(_Tp) == 8) {
        _mm_maskstore_pd(reinterpret_cast<double*>(__mem), __ki,
                         __vector_bitcast<double>(__v));
    } else if constexpr (__have_sse2) {
        _mm_maskmoveu_si128(__vi, __ki, reinterpret_cast<char*>(__mem));
    } else {
        __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(),
                        [&](auto __i) { __mem[__i] = __v[__i]; });
    }
}

template <class _Tp, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(__storage16_t<_Tp> __v, _Tp* __mem, _F,
                                         __storage<bool, __storage16_t<_Tp>::_S_width> __k)
{
    [[maybe_unused]] const auto __vi = __to_intrin(__v);
    if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 1) {
        _mm_mask_storeu_epi8(__mem, __k, __vi);
    } else if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 2) {
        _mm_mask_storeu_epi16(__mem, __k, __vi);
    } else if constexpr (__have_avx512vl && sizeof(_Tp) == 4) {
        if constexpr (__is_aligned_v<_F, 16> && std::is_integral_v<_Tp>) {
            _mm_mask_store_epi32(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 16> && std::is_floating_point_v<_Tp>) {
            _mm_mask_store_ps(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_Tp>) {
            _mm_mask_storeu_epi32(__mem, __k, __vi);
        } else {
            _mm_mask_storeu_ps(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512vl && sizeof(_Tp) == 8) {
        if constexpr (__is_aligned_v<_F, 16> && std::is_integral_v<_Tp>) {
            _mm_mask_store_epi64(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 16> && std::is_floating_point_v<_Tp>) {
            _mm_mask_store_pd(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_Tp>) {
            _mm_mask_storeu_epi64(__mem, __k, __vi);
        } else {
            _mm_mask_storeu_pd(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512f && (sizeof(_Tp) >= 4 || __have_avx512bw)) {
        // use a 512-bit maskstore, using zero-extension of the bitmask
        __maskstore(
            __storage64_t<_Tp>(__intrin_bitcast<__intrinsic_type64_t<_Tp>>(__v._M_data)), __mem,
            // careful, vector_aligned has a stricter meaning in the 512-bit maskstore:
            std::conditional_t<std::is_same_v<_F, vector_aligned_tag>,
                               overaligned_tag<16>, _F>(),
            __storage<bool, 64 / sizeof(_Tp)>(__k._M_data));
    } else {
        __maskstore(
            __v, __mem, _F(),
            __storage16_t<_Tp>(__convert_mask<__vector_type_t<_Tp, 16 / sizeof(_Tp)>>(__k)));
    }
}

// }}}
// __xzyw{{{
// shuffles the complete vector, swapping the inner two quarters. Often useful for AVX for
// fixing up a shuffle result.
template <class _Tp, class _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC _Tp __xzyw(_Tp __a)
{
    if constexpr (sizeof(_Tp) == 16) {
        static_assert(sizeof(float) == 4 && sizeof(int) == 4);
        const auto __x = __vector_bitcast<
            conditional_t<is_floating_point_v<typename _TVT::value_type>, float, int>>(
            __a);
        return reinterpret_cast<_Tp>(decltype(__x){__x[0], __x[2], __x[1], __x[3]});
    } else if constexpr (sizeof(_Tp) == 32) {
        static_assert(sizeof(double) == 8 && sizeof(_LLong) == 8);
        const auto __x =
            __vector_bitcast<conditional_t<is_floating_point_v<typename _TVT::value_type>,
                                           double, _LLong>>(__a);
        return reinterpret_cast<_Tp>(decltype(__x){__x[0], __x[2], __x[1], __x[3]});
    } else if constexpr (sizeof(_Tp) == 64) {
        static_assert(sizeof(double) == 8 && sizeof(_LLong) == 8);
        const auto __x =
            __vector_bitcast<conditional_t<is_floating_point_v<typename _TVT::value_type>,
                                           double, _LLong>>(__a);
        return reinterpret_cast<_Tp>(decltype(__x){__x[0], __x[1], __x[4], __x[5], __x[2],
                                                  __x[3], __x[6], __x[7]});
    } else {
        __assert_unreachable<_Tp>();
    }
}

// }}}
// __extract_part(__storage<_Tp, _N>) {{{
template <size_t _Index, size_t _Total, class _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST
    __vector_type_t<_Tp, std::max(16 / sizeof(_Tp), _N / _Total)>
    __extract_part(__storage<_Tp, _N> __x)
{
    constexpr size_t _NewN = _N / _Total;
    static_assert(_Total > _Index, "_Total must be greater than _Index");
    static_assert(_NewN * _Total == _N, "_N must be divisible by _Total");
    if constexpr (_Index == 0 && _Total == 1) {
        return __x._M_data;
    } else if constexpr (sizeof(_Tp) * _NewN >= 16) {
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
template <int _Index, int _Parts, class _Tp, class _A0, class... _As>
_GLIBCXX_SIMD_INTRINSIC auto  // __vector_type_t or __simd_tuple
__extract_part(const __simd_tuple<_Tp, _A0, _As...> &__x)
{
    // worst cases:
    // (a) 4, 4, 4 => 3, 3, 3, 3 (_Parts = 4)
    // (b) 2, 2, 2 => 3, 3       (_Parts = 2)
    // (c) 4, 2 => 2, 2, 2       (_Parts = 3)
    using _Tuple = __simd_tuple<_Tp, _A0, _As...>;
    static_assert(_Index < _Parts && _Index >= 0 && _Parts >= 1);
    constexpr size_t _N = _Tuple::size();
    static_assert(_N >= _Parts && _N % _Parts == 0);
    constexpr size_t values_per_part = _N / _Parts;
    if constexpr (_Parts == 1) {
        if constexpr (_Tuple::tuple_size == 1) {
            return __x.first._M_data;
        } else {
            return __x;
        }
    } else if constexpr (simd_size_v<_Tp, _A0> % values_per_part != 0) {
        // nasty case: The requested partition does not match the partition of the
        // __simd_tuple. Fall back to construction via scalar copies.
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        const __may_alias<_Tp> *const element_ptr =
            reinterpret_cast<const __may_alias<_Tp> *>(&__x) + _Index * values_per_part;
        return __data(simd<_Tp, simd_abi::deduce_t<_Tp, values_per_part>>(
                          [&](auto __i) { return element_ptr[__i]; }))
            ._M_data;
#else
        constexpr size_t offset = _Index * values_per_part;
        __unused(offset);  // not really
        return __data(simd<_Tp, simd_abi::deduce_t<_Tp, values_per_part>>([&](auto __i) {
                   constexpr _SizeConstant<__i + offset> __k;
                   return __x[__k];
               }))
            ._M_data;
#endif
    } else if constexpr (values_per_part * _Index >= simd_size_v<_Tp, _A0>) {  // recurse
        constexpr int parts_in_first = simd_size_v<_Tp, _A0> / values_per_part;
        return __extract_part<_Index - parts_in_first, _Parts - parts_in_first>(__x.second);
    } else {  // at this point we know that all of the return values are in __x.first
        static_assert(values_per_part * (1 + _Index) <= simd_size_v<_Tp, _A0>);
        if constexpr (simd_size_v<_Tp, _A0> == values_per_part) {
            return __x.first._M_data;
        } else {
            return __extract_part<_Index, simd_size_v<_Tp, _A0> / values_per_part>(
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
            using _Tp = typename _To::value_type;
            return __make_storage<_Tp>(__v0, __vs...);
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
        using _S = __storage<typename _Trait::value_type, _Trait::_S_width>;
        return __convert_all<_To>(_S(__v));
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
template <class _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_Tp, _N> __plus(__storage<_Tp, _N> __a,
                                                         __storage<_Tp, _N> __b)
{
    return __a._M_data + __b._M_data;
}

//}}}
// __minus{{{
template <class _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_Tp, _N> __minus(__storage<_Tp, _N> __a,
                                                          __storage<_Tp, _N> __b)
{
    return __a._M_data - __b._M_data;
}

//}}}
// __multiplies{{{
template <class _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_Tp, _N> __multiplies(__storage<_Tp, _N> __a,
                                                               __storage<_Tp, _N> __b)
{
    if constexpr (sizeof(_Tp) == 1) {
        return __vector_bitcast<_Tp>(
            ((__vector_bitcast<short>(__a) * __vector_bitcast<short>(__b)) &
             __vector_bitcast<short>(~__vector_type_t<ushort, _N / 2>() >> 8)) |
            (((__vector_bitcast<short>(__a) >> 8) * (__vector_bitcast<short>(__b) >> 8))
             << 8));
    }
    return __a._M_data * __b._M_data;
}

//}}}
// __abs{{{
template <class _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_Tp, _N> __abs(__storage<_Tp, _N> __v)
{
//X   if (__builtin_is_constant_evaluated())
//X     {
//X       return __v._M_data < 0 ? -__v._M_data : __v._M_data;
//X     }
if constexpr (std::is_floating_point_v<_Tp>)
  {
    // `v < 0 ? -v : v` cannot compile to the efficient implementation of
    // masking the signbit off because it must consider v == -0
    using _I = std::make_unsigned_t<__int_for_sizeof_t<_Tp>>;
    return __vector_bitcast<_Tp>(__vector_bitcast<_I>(__v._M_data) &
				 __vector_broadcast<_N, _I>(~_I() >> 1));
  }
else
  {
    return __v._M_data < 0 ? -__v._M_data : __v._M_data;
  }
}

//}}}
// interleave (__lo/__hi/128) {{{
template <class _A, class _B, class _Tp = std::common_type_t<_A, _B>,
          class _Trait = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC constexpr _Tp __interleave_lo(_A _a, _B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const _Tp __a(_a);
    const _Tp __b(_b);
    if constexpr (sizeof(_Tp) == 16 && needs_intrinsics) {
        if constexpr (_Trait::_S_width == 2) {
            if constexpr (std::is_integral_v<typename _Trait::value_type>) {
                return reinterpret_cast<_Tp>(
                    _mm_unpacklo_epi64(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
            } else {
                return reinterpret_cast<_Tp>(
                    _mm_unpacklo_pd(__vector_bitcast<double>(__a), __vector_bitcast<double>(__b)));
            }
        } else if constexpr (_Trait::_S_width == 4) {
            if constexpr (std::is_integral_v<typename _Trait::value_type>) {
                return reinterpret_cast<_Tp>(
                    _mm_unpacklo_epi32(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
            } else {
                return reinterpret_cast<_Tp>(
                    _mm_unpacklo_ps(__vector_bitcast<float>(__a), __vector_bitcast<float>(__b)));
            }
        } else if constexpr (_Trait::_S_width == 8) {
            return reinterpret_cast<_Tp>(
                _mm_unpacklo_epi16(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
        } else if constexpr (_Trait::_S_width == 16) {
            return reinterpret_cast<_Tp>(
                _mm_unpacklo_epi8(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
        }
    } else if constexpr (_Trait::_S_width == 2) {
        return _Tp{__a[0], __b[0]};
    } else if constexpr (_Trait::_S_width == 4) {
        return _Tp{__a[0], __b[0], __a[1], __b[1]};
    } else if constexpr (_Trait::_S_width == 8) {
        return _Tp{__a[0], __b[0], __a[1], __b[1], __a[2], __b[2], __a[3], __b[3]};
    } else if constexpr (_Trait::_S_width == 16) {
        return _Tp{__a[0], __b[0], __a[1], __b[1], __a[2], __b[2], __a[3], __b[3],
                 __a[4], __b[4], __a[5], __b[5], __a[6], __b[6], __a[7], __b[7]};
    } else if constexpr (_Trait::_S_width == 32) {
        return _Tp{__a[0],  __b[0],  __a[1],  __b[1],  __a[2],  __b[2],  __a[3],  __b[3],
                 __a[4],  __b[4],  __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],
                 __a[8],  __b[8],  __a[9],  __b[9],  __a[10], __b[10], __a[11], __b[11],
                 __a[12], __b[12], __a[13], __b[13], __a[14], __b[14], __a[15], __b[15]};
    } else if constexpr (_Trait::_S_width == 64) {
        return _Tp{__a[0],  __b[0],  __a[1],  __b[1],  __a[2],  __b[2],  __a[3],  __b[3],  __a[4],  __b[4],
                 __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],  __a[8],  __b[8],  __a[9],  __b[9],
                 __a[10], __b[10], __a[11], __b[11], __a[12], __b[12], __a[13], __b[13], __a[14], __b[14],
                 __a[15], __b[15], __a[16], __b[16], __a[17], __b[17], __a[18], __b[18], __a[19], __b[19],
                 __a[20], __b[20], __a[21], __b[21], __a[22], __b[22], __a[23], __b[23], __a[24], __b[24],
                 __a[25], __b[25], __a[26], __b[26], __a[27], __b[27], __a[28], __b[28], __a[29], __b[29],
                 __a[30], __b[30], __a[31], __b[31]};
    } else {
        __assert_unreachable<_Tp>();
    }
}

template <class _A, class _B, class _Tp = std::common_type_t<_A, _B>,
          class _Trait = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC constexpr _Tp __interleave_hi(_A _a, _B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const _Tp __a(_a);
    const _Tp __b(_b);
    if constexpr (sizeof(_Tp) == 16 && needs_intrinsics) {
        if constexpr (_Trait::_S_width == 2) {
            if constexpr (std::is_integral_v<typename _Trait::value_type>) {
                return reinterpret_cast<_Tp>(
                    _mm_unpackhi_epi64(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
            } else {
                return reinterpret_cast<_Tp>(
                    _mm_unpackhi_pd(__vector_bitcast<double>(__a), __vector_bitcast<double>(__b)));
            }
        } else if constexpr (_Trait::_S_width == 4) {
            if constexpr (std::is_integral_v<typename _Trait::value_type>) {
                return reinterpret_cast<_Tp>(
                    _mm_unpackhi_epi32(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
            } else {
                return reinterpret_cast<_Tp>(
                    _mm_unpackhi_ps(__vector_bitcast<float>(__a), __vector_bitcast<float>(__b)));
            }
        } else if constexpr (_Trait::_S_width == 8) {
            return reinterpret_cast<_Tp>(
                _mm_unpackhi_epi16(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
        } else if constexpr (_Trait::_S_width == 16) {
            return reinterpret_cast<_Tp>(
                _mm_unpackhi_epi8(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
        }
    } else if constexpr (_Trait::_S_width == 2) {
        return _Tp{__a[1], __b[1]};
    } else if constexpr (_Trait::_S_width == 4) {
        return _Tp{__a[2], __b[2], __a[3], __b[3]};
    } else if constexpr (_Trait::_S_width == 8) {
        return _Tp{__a[4], __b[4], __a[5], __b[5], __a[6], __b[6], __a[7], __b[7]};
    } else if constexpr (_Trait::_S_width == 16) {
        return _Tp{__a[8],  __b[8],  __a[9],  __b[9],  __a[10], __b[10], __a[11], __b[11],
                 __a[12], __b[12], __a[13], __b[13], __a[14], __b[14], __a[15], __b[15]};
    } else if constexpr (_Trait::_S_width == 32) {
        return _Tp{__a[16], __b[16], __a[17], __b[17], __a[18], __b[18], __a[19], __b[19],
                 __a[20], __b[20], __a[21], __b[21], __a[22], __b[22], __a[23], __b[23],
                 __a[24], __b[24], __a[25], __b[25], __a[26], __b[26], __a[27], __b[27],
                 __a[28], __b[28], __a[29], __b[29], __a[30], __b[30], __a[31], __b[31]};
    } else if constexpr (_Trait::_S_width == 64) {
        return _Tp{__a[32], __b[32], __a[33], __b[33], __a[34], __b[34], __a[35], __b[35],
                 __a[36], __b[36], __a[37], __b[37], __a[38], __b[38], __a[39], __b[39],
                 __a[40], __b[40], __a[41], __b[41], __a[42], __b[42], __a[43], __b[43],
                 __a[44], __b[44], __a[45], __b[45], __a[46], __b[46], __a[47], __b[47],
                 __a[48], __b[48], __a[49], __b[49], __a[50], __b[50], __a[51], __b[51],
                 __a[52], __b[52], __a[53], __b[53], __a[54], __b[54], __a[55], __b[55],
                 __a[56], __b[56], __a[57], __b[57], __a[58], __b[58], __a[59], __b[59],
                 __a[60], __b[60], __a[61], __b[61], __a[62], __b[62], __a[63], __b[63]};
    } else {
        __assert_unreachable<_Tp>();
    }
}

template <class _A, class _B, class _Tp = std::common_type_t<_A, _B>,
          class _Trait = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC constexpr _Tp __interleave128_lo(_A _a, _B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const _Tp __a(_a);
    const _Tp __b(_b);
    if constexpr (sizeof(_Tp) == 16) {
        return __interleave_lo(__a, __b);
    } else if constexpr (sizeof(_Tp) == 32 && needs_intrinsics) {
        if constexpr (_Trait::_S_width == 4) {
            return reinterpret_cast<_Tp>(
                _mm256_unpacklo_pd(__vector_bitcast<double>(__a), __vector_bitcast<double>(__b)));
        } else if constexpr (_Trait::_S_width == 8) {
            return reinterpret_cast<_Tp>(
                _mm256_unpacklo_ps(__vector_bitcast<float>(__a), __vector_bitcast<float>(__b)));
        } else if constexpr (_Trait::_S_width == 16) {
            return reinterpret_cast<_Tp>(
                _mm256_unpacklo_epi16(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
        } else if constexpr (_Trait::_S_width == 32) {
            return reinterpret_cast<_Tp>(
                _mm256_unpacklo_epi8(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
        }
    } else if constexpr (sizeof(_Tp) == 32) {
        if constexpr (_Trait::_S_width == 4) {
            return _Tp{__a[0], __b[0], __a[2], __b[2]};
        } else if constexpr (_Trait::_S_width == 8) {
            return _Tp{__a[0], __b[0], __a[1], __b[1], __a[4], __b[4], __a[5], __b[5]};
        } else if constexpr (_Trait::_S_width == 16) {
            return _Tp{__a[0], __b[0], __a[1], __b[1], __a[2],  __b[2],  __a[3],  __b[3],
                     __a[8], __b[8], __a[9], __b[9], __a[10], __b[10], __a[11], __b[11]};
        } else if constexpr (_Trait::_S_width == 32) {
            return _Tp{__a[0],  __b[0],  __a[1],  __b[1],  __a[2],  __b[2],  __a[3],  __b[3],
                     __a[4],  __b[4],  __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],
                     __a[16], __b[16], __a[17], __b[17], __a[18], __b[18], __a[19], __b[19],
                     __a[20], __b[20], __a[21], __b[21], __a[22], __b[22], __a[23], __b[23]};
        } else if constexpr (_Trait::_S_width == 64) {
            return _Tp{__a[0],  __b[0],  __a[1],  __b[1],  __a[2],  __b[2],  __a[3],  __b[3],  __a[4],  __b[4],
                     __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],  __a[8],  __b[8],  __a[9],  __b[9],
                     __a[10], __b[10], __a[11], __b[11], __a[12], __b[12], __a[13], __b[13], __a[14], __b[14],
                     __a[15], __b[15], __a[32], __b[32], __a[33], __b[33], __a[34], __b[34], __a[35], __b[35],
                     __a[36], __b[36], __a[37], __b[37], __a[38], __b[38], __a[39], __b[39], __a[40], __b[40],
                     __a[41], __b[41], __a[42], __b[42], __a[43], __b[43], __a[44], __b[44], __a[45], __b[45],
                     __a[46], __b[46], __a[47], __b[47]};
        } else {
            __assert_unreachable<_Tp>();
        }
    } else if constexpr (sizeof(_Tp) == 64 && needs_intrinsics) {
        if constexpr (_Trait::_S_width == 8) {
            return reinterpret_cast<_Tp>(
                _mm512_unpacklo_pd(__vector_bitcast<double>(__a), __vector_bitcast<double>(__b)));
        } else if constexpr (_Trait::_S_width == 16) {
            return reinterpret_cast<_Tp>(
                _mm512_unpacklo_ps(__vector_bitcast<float>(__a), __vector_bitcast<float>(__b)));
        } else if constexpr (_Trait::_S_width == 32) {
            return reinterpret_cast<_Tp>(
                _mm512_unpacklo_epi16(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
        } else if constexpr (_Trait::_S_width == 64) {
            return reinterpret_cast<_Tp>(
                _mm512_unpacklo_epi8(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
        }
    } else if constexpr (sizeof(_Tp) == 64) {
        if constexpr (_Trait::_S_width == 8) {
            return _Tp{__a[0], __b[0], __a[2], __b[2], __a[4], __b[4], __a[6], __b[6]};
        } else if constexpr (_Trait::_S_width == 16) {
            return _Tp{__a[0], __b[0], __a[1], __b[1], __a[4],  __b[4],  __a[5],  __b[5],
                     __a[8], __b[8], __a[9], __b[9], __a[12], __b[12], __a[13], __b[13]};
        } else if constexpr (_Trait::_S_width == 32) {
            return _Tp{__a[0],  __b[0],  __a[1],  __b[1],  __a[2],  __b[2],  __a[3],  __b[3],
                     __a[8],  __b[8],  __a[9],  __b[9],  __a[10], __b[10], __a[11], __b[11],
                     __a[16], __b[16], __a[17], __b[17], __a[18], __b[18], __a[19], __b[19],
                     __a[24], __b[24], __a[25], __b[25], __a[26], __b[26], __a[27], __b[27]};
        } else if constexpr (_Trait::_S_width == 64) {
            return _Tp{__a[0],  __b[0],  __a[1],  __b[1],  __a[2],  __b[2],  __a[3],  __b[3],  __a[4],  __b[4],
                     __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],  __a[16], __b[16], __a[17], __b[17],
                     __a[18], __b[18], __a[19], __b[19], __a[20], __b[20], __a[21], __b[21], __a[22], __b[22],
                     __a[23], __b[23], __a[32], __b[32], __a[33], __b[33], __a[34], __b[34], __a[35], __b[35],
                     __a[36], __b[36], __a[37], __b[37], __a[38], __b[38], __a[39], __b[39], __a[48], __b[48],
                     __a[49], __b[49], __a[50], __b[50], __a[51], __b[51], __a[52], __b[52], __a[53], __b[53],
                     __a[54], __b[54], __a[55], __b[55]};
        } else {
            __assert_unreachable<_Tp>();
        }
    }
}

template <class _A, class _B, class _Tp = std::common_type_t<_A, _B>,
          class _Trait = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC constexpr _Tp __interleave128_hi(_A _a, _B _b)
{
#if defined __GNUC__ && __GNUC__ < 9
    constexpr bool needs_intrinsics = true;
#else
    constexpr bool needs_intrinsics = false;
#endif

    const _Tp __a(_a);
    const _Tp __b(_b);
    if constexpr (sizeof(_Tp) == 16) {
        return __interleave_hi(__a, __b);
    } else if constexpr (sizeof(_Tp) == 32 && needs_intrinsics) {
        if constexpr (_Trait::_S_width == 4) {
            return reinterpret_cast<_Tp>(
                _mm256_unpackhi_pd(__vector_bitcast<double>(__a), __vector_bitcast<double>(__b)));
        } else if constexpr (_Trait::_S_width == 8) {
            return reinterpret_cast<_Tp>(
                _mm256_unpackhi_ps(__vector_bitcast<float>(__a), __vector_bitcast<float>(__b)));
        } else if constexpr (_Trait::_S_width == 16) {
            return reinterpret_cast<_Tp>(
                _mm256_unpackhi_epi16(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
        } else if constexpr (_Trait::_S_width == 32) {
            return reinterpret_cast<_Tp>(
                _mm256_unpackhi_epi8(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
        }
    } else if constexpr (sizeof(_Tp) == 32) {
        if constexpr (_Trait::_S_width == 4) {
            return _Tp{__a[1], __b[1], __a[3], __b[3]};
        } else if constexpr (_Trait::_S_width == 8) {
            return _Tp{__a[2], __b[2], __a[3], __b[3], __a[6], __b[6], __a[7], __b[7]};
        } else if constexpr (_Trait::_S_width == 16) {
            return _Tp{__a[4],  __b[4],  __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],
                     __a[12], __b[12], __a[13], __b[13], __a[14], __b[14], __a[15], __b[15]};
        } else if constexpr (_Trait::_S_width == 32) {
            return _Tp{__a[8],  __b[8],  __a[9],  __b[9],  __a[10], __b[10], __a[11], __b[11],
                     __a[12], __b[12], __a[13], __b[13], __a[14], __b[14], __a[15], __b[15],
                     __a[24], __b[24], __a[25], __b[25], __a[26], __b[26], __a[27], __b[27],
                     __a[28], __b[28], __a[29], __b[29], __a[30], __b[30], __a[31], __b[31]};
        } else if constexpr (_Trait::_S_width == 64) {
            return _Tp{__a[16], __b[16], __a[17], __b[17], __a[18], __b[18], __a[19], __b[19], __a[20], __b[20],
                     __a[21], __b[21], __a[22], __b[22], __a[23], __b[23], __a[24], __b[24], __a[25], __b[25],
                     __a[26], __b[26], __a[27], __b[27], __a[28], __b[28], __a[29], __b[29], __a[30], __b[30],
                     __a[31], __b[31], __a[48], __b[48], __a[49], __b[49], __a[50], __b[50], __a[51], __b[51],
                     __a[52], __b[52], __a[53], __b[53], __a[54], __b[54], __a[55], __b[55], __a[56], __b[56],
                     __a[57], __b[57], __a[58], __b[58], __a[59], __b[59], __a[60], __b[60], __a[61], __b[61],
                     __a[62], __b[62], __a[63], __b[63]};
        } else {
            __assert_unreachable<_Tp>();
        }
    } else if constexpr (sizeof(_Tp) == 64 && needs_intrinsics) {
        if constexpr (_Trait::_S_width == 8) {
            return reinterpret_cast<_Tp>(
                _mm512_unpackhi_pd(__vector_bitcast<double>(__a), __vector_bitcast<double>(__b)));
        } else if constexpr (_Trait::_S_width == 16) {
            return reinterpret_cast<_Tp>(
                _mm512_unpackhi_ps(__vector_bitcast<float>(__a), __vector_bitcast<float>(__b)));
        } else if constexpr (_Trait::_S_width == 32) {
            return reinterpret_cast<_Tp>(
                _mm512_unpackhi_epi16(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
        } else if constexpr (_Trait::_S_width == 64) {
            return reinterpret_cast<_Tp>(
                _mm512_unpackhi_epi8(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b)));
        }
    } else if constexpr (sizeof(_Tp) == 64) {
        if constexpr (_Trait::_S_width == 8) {
            return _Tp{__a[1], __b[1], __a[3], __b[3], __a[5], __b[5], __a[7], __b[7]};
        } else if constexpr (_Trait::_S_width == 16) {
            return _Tp{__a[2],  __b[2],  __a[3],  __b[3],  __a[6],  __b[6],  __a[7],  __b[7],
                     __a[10], __b[10], __a[11], __b[11], __a[14], __b[14], __a[15], __b[15]};
        } else if constexpr (_Trait::_S_width == 32) {
            return _Tp{__a[4],  __b[4],  __a[5],  __b[5],  __a[6],  __b[6],  __a[7],  __b[7],
                     __a[12], __b[12], __a[13], __b[13], __a[14], __b[14], __a[15], __b[15],
                     __a[20], __b[20], __a[21], __b[21], __a[22], __b[22], __a[23], __b[23],
                     __a[28], __b[28], __a[29], __b[29], __a[30], __b[30], __a[31], __b[31]};
        } else if constexpr (_Trait::_S_width == 64) {
            return _Tp{__a[8],  __b[8],  __a[9],  __b[9],  __a[10], __b[10], __a[11], __b[11], __a[12], __b[12],
                     __a[13], __b[13], __a[14], __b[14], __a[15], __b[15], __a[24], __b[24], __a[25], __b[25],
                     __a[26], __b[26], __a[27], __b[27], __a[28], __b[28], __a[29], __b[29], __a[30], __b[30],
                     __a[31], __b[31], __a[40], __b[40], __a[41], __b[41], __a[42], __b[42], __a[43], __b[43],
                     __a[44], __b[44], __a[45], __b[45], __a[46], __b[46], __a[47], __b[47], __a[56], __b[56],
                     __a[57], __b[57], __a[58], __b[58], __a[59], __b[59], __a[60], __b[60], __a[61], __b[61],
                     __a[62], __b[62], __a[63], __b[63]};
        } else {
            __assert_unreachable<_Tp>();
        }
    }
}

template <class _Tp> struct __interleaved_pair {
    _Tp __lo, __hi;
};

template <class _A, class _B, class _Tp = std::common_type_t<_A, _B>,
          class _Trait = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC constexpr __interleaved_pair<_Tp> interleave(_A __a, _B __b)
{
    return {__interleave_lo(__a, __b), __interleave_hi(__a, __b)};
}

template <class _A, class _B, class _Tp = std::common_type_t<_A, _B>,
          class _Trait = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC constexpr __interleaved_pair<_Tp> interleave128(_A __a, _B __b)
{
    return {__interleave128_lo(__a, __b), __interleave128_hi(__a, __b)};
}
// }}}
// __is_bitset {{{
template <class _Tp> struct __is_bitset : false_type {};
template <size_t _N> struct __is_bitset<std::bitset<_N>> : true_type {};
template <class _Tp> inline constexpr bool __is_bitset_v = __is_bitset<_Tp>::value;

// }}}
// __is_storage {{{
template <class _Tp> struct __is_storage : false_type {};
template <class _Tp, size_t _N> struct __is_storage<__storage<_Tp, _N>> : true_type {};
template <class _Tp> inline constexpr bool __is_storage_v = __is_storage<_Tp>::value;

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
        static_assert(__k.size() <= sizeof(_ULLong) * CHAR_BIT);
        using _Tp = std::conditional_t<
            (__k.size() <= sizeof(ushort) * CHAR_BIT),
            std::conditional_t<(__k.size() <= CHAR_BIT), _UChar, ushort>,
            std::conditional_t<(__k.size() <= sizeof(_UInt) * CHAR_BIT), _UInt, _ULLong>>;
        return __convert_mask<_To>(static_cast<_Tp>(__k.to_ullong()));
        // }}}
    } else if constexpr (__is_bitset_v<_To>) {
        // to std::bitset {{{
        static_assert(_To().size() <= sizeof(_ULLong) * CHAR_BIT);
        using _Tp = std::conditional_t<
            (_To().size() <= sizeof(ushort) * CHAR_BIT),
            std::conditional_t<(_To().size() <= CHAR_BIT), _UChar, ushort>,
            std::conditional_t<(_To().size() <= sizeof(_UInt) * CHAR_BIT), _UInt, _ULLong>>;
        return __convert_mask<_Tp>(__k);
        // }}}
    } else if constexpr (__is_storage_v<_From>) {
        return __convert_mask<_To>(__k._M_data);
    } else if constexpr (__is_storage_v<_To>) {
        return __convert_mask<typename _To::register_type>(__k);
    } else if constexpr (std::is_unsigned_v<_From> && __is_vector_type_v<_To>) {
        // bits -> vector {{{
        using _Trait = __vector_traits<_To>;
        constexpr size_t _N_in = sizeof(_From) * CHAR_BIT;
        using _ToT = typename _Trait::value_type;
        constexpr size_t _N_out = _Trait::_S_width;
        constexpr size_t _N = std::min(_N_in, _N_out);
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
                return reinterpret_cast<__vector_type_t<_SChar, 64>>(_mm512_movm_epi8(__k));
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
                const auto bitmask = __to_intrin(__make_builtin<_UChar>(
                    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128));
                return __vector_bitcast<_ToT>(
                    __vector_bitcast<_ToT>(
                        _mm_shuffle_epi8(
                            __to_intrin(__vector_type_t<_ULLong, 2>{__k}),
                            _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                                          1)) &
                        bitmask) != 0);
            } else if constexpr (sizeof(_V) == 32 && sizeof(_ToT) == 1 && __have_avx2) {
                const auto bitmask =
                    _mm256_broadcastsi128_si256(__to_intrin(__make_builtin<_UChar>(
                        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128)));
                return __vector_bitcast<_ToT>(
                    __vector_bitcast<_ToT>(_mm256_shuffle_epi8(
                                        _mm256_broadcastsi128_si256(__to_intrin(
                                            __vector_type_t<_ULLong, 2>{__k})),
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
        using _Tp = typename _Trait::value_type;
        constexpr size_t _FromN = _Trait::_S_width;
        constexpr size_t cvt_id = _FromN * 10 + sizeof(_Tp);
        constexpr bool __have_avx512_int = __have_avx512f && std::is_integral_v<_Tp>;
        [[maybe_unused]]  // PR85827
        const auto __intrin = __to_intrin(__k);

             if constexpr (cvt_id == 16'1 && __have_avx512bw_vl) { return    _mm_movepi8_mask(__intrin); }
        else if constexpr (cvt_id == 16'1 && __have_avx512bw   ) { return _mm512_movepi8_mask(__zero_extend(__intrin)); }
        else if constexpr (cvt_id == 16'1                      ) { return    _mm_movemask_epi8(__intrin); }
        else if constexpr (cvt_id == 32'1 && __have_avx512bw_vl) { return _mm256_movepi8_mask(__intrin); }
        else if constexpr (cvt_id == 32'1 && __have_avx512bw   ) { return _mm512_movepi8_mask(__zero_extend(__intrin)); }
        else if constexpr (cvt_id == 32'1                      ) { return _mm256_movemask_epi8(__intrin); }
        else if constexpr (cvt_id == 64'1 && __have_avx512bw   ) { return _mm512_movepi8_mask(__intrin); }
        else if constexpr (cvt_id ==  8'2 && __have_avx512bw_vl) { return    _mm_movepi16_mask(__intrin); }
        else if constexpr (cvt_id ==  8'2 && __have_avx512bw   ) { return _mm512_movepi16_mask(__zero_extend(__intrin)); }
        else if constexpr (cvt_id ==  8'2                      ) { return movemask_epi16(__intrin); }
        else if constexpr (cvt_id == 16'2 && __have_avx512bw_vl) { return _mm256_movepi16_mask(__intrin); }
        else if constexpr (cvt_id == 16'2 && __have_avx512bw   ) { return _mm512_movepi16_mask(__zero_extend(__intrin)); }
        else if constexpr (cvt_id == 16'2                      ) { return movemask_epi16(__intrin); }
        else if constexpr (cvt_id == 32'2 && __have_avx512bw   ) { return _mm512_movepi16_mask(__intrin); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512dq_vl) { return    _mm_movepi32_mask(__vector_bitcast<_LLong>(__k)); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512dq   ) { return _mm512_movepi32_mask(__zero_extend(__vector_bitcast<_LLong>(__k))); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512vl   ) { return    _mm_cmp_epi32_mask(__vector_bitcast<_LLong>(__k), __m128i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512_int ) { return _mm512_cmp_epi32_mask(__zero_extend(__intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'4                      ) { return    _mm_movemask_ps(__k); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512dq_vl) { return _mm256_movepi32_mask(__vector_bitcast<_LLong>(__k)); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512dq   ) { return _mm512_movepi32_mask(__zero_extend(__vector_bitcast<_LLong>(__k))); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512vl   ) { return _mm256_cmp_epi32_mask(__vector_bitcast<_LLong>(__k), __m256i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512_int ) { return _mm512_cmp_epi32_mask(__zero_extend(__intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  8'4                      ) { return _mm256_movemask_ps(__k); }
        else if constexpr (cvt_id == 16'4 && __have_avx512dq   ) { return _mm512_movepi32_mask(__vector_bitcast<_LLong>(__k)); }
        else if constexpr (cvt_id == 16'4                      ) { return _mm512_cmp_epi32_mask(__vector_bitcast<_LLong>(__k), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512dq_vl) { return    _mm_movepi64_mask(__vector_bitcast<_LLong>(__k)); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512dq   ) { return _mm512_movepi64_mask(__zero_extend(__vector_bitcast<_LLong>(__k))); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512vl   ) { return    _mm_cmp_epi64_mask(__vector_bitcast<_LLong>(__k), __m128i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512_int ) { return _mm512_cmp_epi64_mask(__zero_extend(__intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8                      ) { return    _mm_movemask_pd(__k); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512dq_vl) { return _mm256_movepi64_mask(__vector_bitcast<_LLong>(__k)); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512dq   ) { return _mm512_movepi64_mask(__zero_extend(__vector_bitcast<_LLong>(__k))); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512vl   ) { return _mm256_cmp_epi64_mask(__vector_bitcast<_LLong>(__k), __m256i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512_int ) { return _mm512_cmp_epi64_mask(__zero_extend(__intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'8                      ) { return _mm256_movemask_pd(__k); }
        else if constexpr (cvt_id ==  8'8 && __have_avx512dq   ) { return _mm512_movepi64_mask(__vector_bitcast<_LLong>(__k)); }
        else if constexpr (cvt_id ==  8'8                      ) { return _mm512_cmp_epi64_mask(__vector_bitcast<_LLong>(__k), __m512i(), _MM_CMPINT_LT); }
        else { __assert_unreachable<_To>(); }
        // }}}
    } else if constexpr (__is_vector_type_v<_From> && __is_vector_type_v<_To>) {
        // vector -> vector {{{
        using _ToTrait = __vector_traits<_To>;
        using _FromTrait = __vector_traits<_From>;
        using _ToT = typename _ToTrait::value_type;
        using _Tp = typename _FromTrait::value_type;
        constexpr size_t _FromN = _FromTrait::_S_width;
        constexpr size_t _ToN = _ToTrait::_S_width;
        constexpr int _FromBytes = sizeof(_Tp);
        constexpr int _ToBytes = sizeof(_ToT);

        if constexpr (_FromN == _ToN && sizeof(_From) == sizeof(_To)) {
            // reinterpret the bits
            return reinterpret_cast<_To>(__k);
        } else if constexpr (sizeof(_To) == 16 && sizeof(__k) == 16) {
            // SSE -> SSE {{{
            if constexpr (_FromBytes == 4 && _ToBytes == 8) {
                if constexpr(std::is_integral_v<_Tp>) {
                    return __vector_bitcast<_ToT>(__interleave128_lo(__k, __k));
                } else {
                    return __vector_bitcast<_ToT>(__interleave128_lo(__k, __k));
                }
            } else if constexpr (_FromBytes == 2 && _ToBytes == 8) {
                const auto __y = __vector_bitcast<int>(__interleave128_lo(__k, __k));
                return __vector_bitcast<_ToT>(__interleave128_lo(__y, __y));
            } else if constexpr (_FromBytes == 1 && _ToBytes == 8) {
                auto __y = __vector_bitcast<short>(__interleave128_lo(__k, __k));
                auto __z = __vector_bitcast<int>(__interleave128_lo(__y, __y));
                return __vector_bitcast<_ToT>(__interleave128_lo(__z, __z));
            } else if constexpr (_FromBytes == 8 && _ToBytes == 4) {
                if constexpr (std::is_floating_point_v<_Tp>) {
                    return __vector_bitcast<_ToT>(_mm_shuffle_ps(__vector_bitcast<float>(__k), __m128(),
                                                     __make_immediate<4>(1, 3, 1, 3)));
                } else {
                    auto __y = __vector_bitcast<_LLong>(__k);
                    return __vector_bitcast<_ToT>(_mm_packs_epi32(__y, __m128i()));
                }
            } else if constexpr (_FromBytes == 2 && _ToBytes == 4) {
                return __vector_bitcast<_ToT>(__interleave128_lo(__k, __k));
            } else if constexpr (_FromBytes == 1 && _ToBytes == 4) {
                const auto __y = __vector_bitcast<short>(__interleave128_lo(__k, __k));
                return __vector_bitcast<_ToT>(__interleave128_lo(__y, __y));
            } else if constexpr (_FromBytes == 8 && _ToBytes == 2) {
                if constexpr(__have_ssse3) {
                    return __vector_bitcast<_ToT>(
                        _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                         _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    const auto __y = _mm_packs_epi32(__vector_bitcast<_LLong>(__k), __m128i());
                    return __vector_bitcast<_ToT>(_mm_packs_epi32(__y, __m128i()));
                }
            } else if constexpr (_FromBytes == 4 && _ToBytes == 2) {
                return __vector_bitcast<_ToT>(
                    _mm_packs_epi32(__vector_bitcast<_LLong>(__k), __m128i()));
            } else if constexpr (_FromBytes == 1 && _ToBytes == 2) {
                return __vector_bitcast<_ToT>(__interleave128_lo(__k, __k));
            } else if constexpr (_FromBytes == 8 && _ToBytes == 1) {
                if constexpr(__have_ssse3) {
                    return __vector_bitcast<_ToT>(
                        _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                         _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    auto __y = _mm_packs_epi32(__vector_bitcast<_LLong>(__k), __m128i());
                    __y = _mm_packs_epi32(__y, __m128i());
                    return __vector_bitcast<_ToT>(_mm_packs_epi16(__y, __m128i()));
                }
            } else if constexpr (_FromBytes == 4 && _ToBytes == 1) {
                if constexpr(__have_ssse3) {
                    return __vector_bitcast<_ToT>(
                        _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                         _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    const auto __y = _mm_packs_epi32(__vector_bitcast<_LLong>(__k), __m128i());
                    return __vector_bitcast<_ToT>(_mm_packs_epi16(__y, __m128i()));
                }
            } else if constexpr (_FromBytes == 2 && _ToBytes == 1) {
                return __vector_bitcast<_ToT>(_mm_packs_epi16(__vector_bitcast<_LLong>(__k), __m128i()));
            } else {
                static_assert(!std::is_same_v<_Tp, _Tp>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(_To) == 32 && sizeof(__k) == 32) {
            // AVX -> AVX {{{
            if constexpr (_FromBytes == _ToBytes) {  // keep low 1/2
                static_assert(!std::is_same_v<_Tp, _Tp>, "should be unreachable");
            } else if constexpr (_FromBytes == _ToBytes * 2) {
                const auto __y = __vector_bitcast<_LLong>(__k);
                return __vector_bitcast<_ToT>(
                    _mm256_castsi128_si256(_mm_packs_epi16(__lo128(__y), __hi128(__y))));
            } else if constexpr (_FromBytes == _ToBytes * 4) {
                const auto __y = __vector_bitcast<_LLong>(__k);
                return __vector_bitcast<_ToT>(_mm256_castsi128_si256(
                    _mm_packs_epi16(_mm_packs_epi16(__lo128(__y), __hi128(__y)), __m128i())));
            } else if constexpr (_FromBytes == _ToBytes * 8) {
                const auto __y = __vector_bitcast<_LLong>(__k);
                return __vector_bitcast<_ToT>(_mm256_castsi128_si256(
                    _mm_shuffle_epi8(_mm_packs_epi16(__lo128(__y), __hi128(__y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1))));
            } else if constexpr (_FromBytes * 2 == _ToBytes) {
                auto __y = __xzyw(__to_intrin(__k));
                if constexpr(std::is_floating_point_v<_Tp>) {
                    return __vector_bitcast<_ToT>(_mm256_unpacklo_ps(__y, __y));
                } else {
                    return __vector_bitcast<_ToT>(_mm256_unpacklo_epi8(__y, __y));
                }
            } else if constexpr (_FromBytes * 4 == _ToBytes) {
                auto __y = _mm_unpacklo_epi8(__lo128(__vector_bitcast<_LLong>(__k)),
                                           __lo128(__vector_bitcast<_LLong>(__k)));  // drops 3/4 of input
                return __vector_bitcast<_ToT>(
                    __concat(_mm_unpacklo_epi16(__y, __y), _mm_unpackhi_epi16(__y, __y)));
            } else if constexpr (_FromBytes == 1 && _ToBytes == 8) {
                auto __y = _mm_unpacklo_epi8(__lo128(__vector_bitcast<_LLong>(__k)),
                                           __lo128(__vector_bitcast<_LLong>(__k)));  // drops 3/4 of input
                __y = _mm_unpacklo_epi16(__y, __y);  // drops another 1/2 => 7/8 total
                return __vector_bitcast<_ToT>(
                    __concat(_mm_unpacklo_epi32(__y, __y), _mm_unpackhi_epi32(__y, __y)));
            } else {
                static_assert(!std::is_same_v<_Tp, _Tp>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(_To) == 32 && sizeof(__k) == 16) {
            // SSE -> AVX {{{
            if constexpr (_FromBytes == _ToBytes) {
                return __vector_bitcast<_ToT>(
                    __intrinsic_type_t<_Tp, 32 / sizeof(_Tp)>(__zero_extend(__to_intrin(__k))));
            } else if constexpr (_FromBytes * 2 == _ToBytes) {  // keep all
                return __vector_bitcast<_ToT>(__concat(_mm_unpacklo_epi8(__vector_bitcast<_LLong>(__k), __vector_bitcast<_LLong>(__k)),
                                         _mm_unpackhi_epi8(__vector_bitcast<_LLong>(__k), __vector_bitcast<_LLong>(__k))));
            } else if constexpr (_FromBytes * 4 == _ToBytes) {
                if constexpr (__have_avx2) {
                    return __vector_bitcast<_ToT>(_mm256_shuffle_epi8(
                        __concat(__vector_bitcast<_LLong>(__k), __vector_bitcast<_LLong>(__k)),
                        _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7,
                                         7)));
                } else {
                    return __vector_bitcast<_ToT>(
                        __concat(_mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                                _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2,
                                                              2, 2, 2, 3, 3, 3, 3)),
                               _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                                _mm_setr_epi8(4, 4, 4, 4, 5, 5, 5, 5, 6,
                                                              6, 6, 6, 7, 7, 7, 7))));
                }
            } else if constexpr (_FromBytes * 8 == _ToBytes) {
                if constexpr (__have_avx2) {
                    return __vector_bitcast<_ToT>(_mm256_shuffle_epi8(
                        __concat(__vector_bitcast<_LLong>(__k), __vector_bitcast<_LLong>(__k)),
                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                                         3)));
                } else {
                    return __vector_bitcast<_ToT>(
                        __concat(_mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                                _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                              1, 1, 1, 1, 1, 1, 1)),
                               _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                                _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3,
                                                              3, 3, 3, 3, 3, 3, 3))));
                }
            } else if constexpr (_FromBytes == _ToBytes * 2) {
                return __vector_bitcast<_ToT>(
                    __m256i(__zero_extend(_mm_packs_epi16(__vector_bitcast<_LLong>(__k), __m128i()))));
            } else if constexpr (_FromBytes == 8 && _ToBytes == 2) {
                return __vector_bitcast<_ToT>(__m256i(__zero_extend(
                    _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                     _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else if constexpr (_FromBytes == 4 && _ToBytes == 1) {
                return __vector_bitcast<_ToT>(__m256i(__zero_extend(
                    _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else if constexpr (_FromBytes == 8 && _ToBytes == 1) {
                return __vector_bitcast<_ToT>(__m256i(__zero_extend(
                    _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                     _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else {
                static_assert(!std::is_same_v<_Tp, _Tp>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(_To) == 16 && sizeof(__k) == 32) {
            // AVX -> SSE {{{
            if constexpr (_FromBytes == _ToBytes) {  // keep low 1/2
                return __vector_bitcast<_ToT>(__lo128(__k));
            } else if constexpr (_FromBytes == _ToBytes * 2) {  // keep all
                auto __y = __vector_bitcast<_LLong>(__k);
                return __vector_bitcast<_ToT>(_mm_packs_epi16(__lo128(__y), __hi128(__y)));
            } else if constexpr (_FromBytes == _ToBytes * 4) {  // add 1/2 undef
                auto __y = __vector_bitcast<_LLong>(__k);
                return __vector_bitcast<_ToT>(
                    _mm_packs_epi16(_mm_packs_epi16(__lo128(__y), __hi128(__y)), __m128i()));
            } else if constexpr (_FromBytes == 8 && _ToBytes == 1) {  // add 3/4 undef
                auto __y = __vector_bitcast<_LLong>(__k);
                return __vector_bitcast<_ToT>(
                    _mm_shuffle_epi8(_mm_packs_epi16(__lo128(__y), __hi128(__y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)));
            } else if constexpr (_FromBytes * 2 == _ToBytes) {  // keep low 1/4
                auto __y = __lo128(__vector_bitcast<_LLong>(__k));
                return __vector_bitcast<_ToT>(_mm_unpacklo_epi8(__y, __y));
            } else if constexpr (_FromBytes * 4 == _ToBytes) {  // keep low 1/8
                auto __y = __lo128(__vector_bitcast<_LLong>(__k));
                __y = _mm_unpacklo_epi8(__y, __y);
                return __vector_bitcast<_ToT>(_mm_unpacklo_epi8(__y, __y));
            } else if constexpr (_FromBytes * 8 == _ToBytes) {  // keep low 1/16
                auto __y = __lo128(__vector_bitcast<_LLong>(__k));
                __y = _mm_unpacklo_epi8(__y, __y);
                __y = _mm_unpacklo_epi8(__y, __y);
                return __vector_bitcast<_ToT>(_mm_unpacklo_epi8(__y, __y));
            } else {
                static_assert(!std::is_same_v<_Tp, _Tp>, "should be unreachable");
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
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __acos(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::acos(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __asin(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::asin(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __atan(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::atan(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __atan2(const simd<_Tp, _Abi> &__x, const simd<_Tp, _Abi> &__y)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::atan2(__x[__i], __y[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __cos(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::cos(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __sin(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::sin(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __tan(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::tan(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __acosh(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::acosh(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __asinh(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::asinh(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __atanh(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::atanh(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __cosh(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::cosh(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __sinh(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::sinh(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __tanh(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::tanh(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __exp(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::exp(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __exp2(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::exp2(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __expm1(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::expm1(__x[__i]); });
    }

    template <class _Tp>
    simd<_Tp, _Abi> __frexp(const simd<_Tp, _Abi> &__x,
                         fixed_size_simd<int, simd_size_v<_Tp, _Abi>> &exp)
    {
        return simd<_Tp, _Abi>([&](auto __i) {
            int tmp;
            _Tp __r = std::frexp(__x[__i], &tmp);
            exp[__i] = tmp;
            return __r;
        });
    }

    template <class _Tp>
    simd<_Tp, _Abi> __ldexp(const simd<_Tp, _Abi> &__x,
                         const fixed_size_simd<int, simd_size_v<_Tp, _Abi>> &exp)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::ldexp(__x[__i], exp[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC
    fixed_size_simd<int, simd_size_v<_Tp, _Abi>> __ilogb(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::ilogb(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __log(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::log(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __log10(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::log10(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __log1p(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::log1p(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __log2(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::log2(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __logb(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::logb(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __modf(const simd<_Tp, _Abi> &__x, simd<_Tp, _Abi> &__y)
    {
        return simd<_Tp, _Abi>([&](auto __i) {
            _Tp tmp;
            _Tp __r = std::modf(__x[__i], &tmp);
            __y[__i] = tmp;
            return __r;
        });
    }

    template <class _Tp>
    _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi>
			    __scalbn(const simd<_Tp, _Abi>&                              __x,
				     const fixed_size_simd<int, simd_size_v<_Tp, _Abi>>& __y)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::scalbn(__x[__i], __y[__i]); });
    }

    template <class _Tp>
    _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi>
			    __scalbln(const simd<_Tp, _Abi>&                               __x,
				      const fixed_size_simd<long, simd_size_v<_Tp, _Abi>>& __y)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::scalbln(__x[__i], __y[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __cbrt(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::cbrt(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __abs(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::abs(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __fabs(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::fabs(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __pow(const simd<_Tp, _Abi> &__x, const simd<_Tp, _Abi> &__y)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::pow(__x[__i], __y[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __sqrt(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::sqrt(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __erf(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::erf(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __erfc(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::erfc(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __lgamma(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::lgamma(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __tgamma(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::tgamma(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __ceil(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::ceil(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __floor(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::floor(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __nearbyint(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::nearbyint(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __rint(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::rint(__x[__i]); });
    }

    template <class _Tp>
    fixed_size_simd<long, simd_size_v<_Tp, _Abi>> __lrint(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::lrint(__x[__i]); });
    }

    template <class _Tp>
    fixed_size_simd<long long, simd_size_v<_Tp, _Abi>> __llrint(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::llrint(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __round(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::round(__x[__i]); });
    }

    template <class _Tp>
    fixed_size_simd<long, simd_size_v<_Tp, _Abi>> __lround(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::lround(__x[__i]); });
    }

    template <class _Tp>
    fixed_size_simd<long long, simd_size_v<_Tp, _Abi>> __llround(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::llround(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __trunc(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::trunc(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __fmod(const simd<_Tp, _Abi> &__x, const simd<_Tp, _Abi> &__y)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::fmod(__x[__i], __y[__i]); });
    }

    template <class _Tp>
    simd<_Tp, _Abi> __remainder(const simd<_Tp, _Abi> &__x, const simd<_Tp, _Abi> &__y)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::remainder(__x[__i], __y[__i]); });
    }

    template <class _Tp>
    simd<_Tp, _Abi> __remquo(const simd<_Tp, _Abi> &__x, const simd<_Tp, _Abi> &__y,
                          fixed_size_simd<int, simd_size_v<_Tp, _Abi>> &__z)
    {
        return simd<_Tp, _Abi>([&](auto __i) {
            int tmp;
            _Tp __r = std::remquo(__x[__i], __y[__i], &tmp);
            __z[__i] = tmp;
            return __r;
        });
    }

    template <class _Tp>
    simd<_Tp, _Abi> __copysign(const simd<_Tp, _Abi> &__x, const simd<_Tp, _Abi> &__y)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::copysign(__x[__i], __y[__i]); });
    }

    template <class _Tp>
    simd<_Tp, _Abi> __nextafter(const simd<_Tp, _Abi> &__x, const simd<_Tp, _Abi> &__y)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::nextafter(__x[__i], __y[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __fdim(const simd<_Tp, _Abi> &__x, const simd<_Tp, _Abi> &__y)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::fdim(__x[__i], __y[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __fmax(const simd<_Tp, _Abi> &__x, const simd<_Tp, _Abi> &__y)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::fmax(__x[__i], __y[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __fmin(const simd<_Tp, _Abi> &__x, const simd<_Tp, _Abi> &__y)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::fmin(__x[__i], __y[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC
    simd<_Tp, _Abi> __fma(const simd<_Tp, _Abi> &__x, const simd<_Tp, _Abi> &__y,
                       const simd<_Tp, _Abi> &__z)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::fma(__x[__i], __y[__i], __z[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC
    fixed_size_simd<int, simd_size_v<_Tp, _Abi>> __fpclassify(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::fpclassify(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi> __isfinite(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::isfinite(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi> __isinf(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::isinf(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi> __isnan(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::isnan(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi> __isnormal(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::isnormal(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi> __signbit(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::signbit(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi> __isgreater(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::isgreater(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi> __isgreaterequal(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::isgreaterequal(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi> __isless(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::isless(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi> __islessequal(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::islessequal(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi> __islessgreater(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::islessgreater(__x[__i]); });
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi> __isunordered(const simd<_Tp, _Abi> &__x)
    {
        return simd<_Tp, _Abi>([&](auto __i) { return std::isunordered(__x[__i]); });
    }
};  // }}}
// __scalar_simd_impl {{{
struct __scalar_simd_impl : __simd_math_fallback<simd_abi::scalar> {
    // member types {{{2
    using abi = std::experimental::simd_abi::scalar;
    using _Mask_member_type = bool;
    template <class _Tp> using _Simd_member_type = _Tp;
    template <class _Tp> using simd = std::experimental::simd<_Tp, abi>;
    template <class _Tp> using simd_mask = std::experimental::simd_mask<_Tp, abi>;
    template <class _Tp> using __type_tag = _Tp *;

    // broadcast {{{2
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static constexpr _Tp __broadcast(_Tp __x) noexcept
    {
        return __x;
    }

    // generator {{{2
    template <class _F, class _Tp>
    _GLIBCXX_SIMD_INTRINSIC static _Tp generator(_F &&__gen, __type_tag<_Tp>)
    {
        return __gen(_SizeConstant<0>());
    }

    // load {{{2
    template <class _Tp, class _U, class _F>
    static inline _Tp load(const _U *mem, _F, __type_tag<_Tp>) noexcept
    {
        return static_cast<_Tp>(mem[0]);
    }

    // masked load {{{2
    template <class _Tp, class _U, class _F>
    static inline _Tp masked_load(_Tp merge, bool __k, const _U *mem, _F) noexcept
    {
        if (__k) {
            merge = static_cast<_Tp>(mem[0]);
        }
        return merge;
    }

    // store {{{2
    template <class _Tp, class _U, class _F>
    static inline void store(_Tp __v, _U *mem, _F, __type_tag<_Tp>) noexcept
    {
        mem[0] = static_cast<_Tp>(__v);
    }

    // masked store {{{2
    template <class _Tp, class _U, class _F>
    static inline void masked_store(const _Tp __v, _U *mem, _F, const bool __k) noexcept
    {
        if (__k) {
            mem[0] = __v;
        }
    }

    // negation {{{2
    template <class _Tp> static inline bool negate(_Tp __x) noexcept { return !__x; }

    // reductions {{{2
    template <class _Tp, class _BinaryOperation>
    static inline _Tp reduce(const simd<_Tp> &__x, _BinaryOperation &)
    {
        return __x._M_data;
    }

    // min, max, clamp {{{2
    template <class _Tp> static inline _Tp min(const _Tp __a, const _Tp __b)
    {
        return std::min(__a, __b);
    }

    template <class _Tp> static inline _Tp max(const _Tp __a, const _Tp __b)
    {
        return std::max(__a, __b);
    }

    // complement {{{2
    template <class _Tp> static inline _Tp complement(_Tp __x) noexcept
    {
        return static_cast<_Tp>(~__x);
    }

    // unary minus {{{2
    template <class _Tp> static inline _Tp unary_minus(_Tp __x) noexcept
    {
        return static_cast<_Tp>(-__x);
    }

    // arithmetic operators {{{2
    template <class _Tp> static inline _Tp plus(_Tp __x, _Tp __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) +
                              __promote_preserving_unsigned(__y));
    }

    template <class _Tp> static inline _Tp minus(_Tp __x, _Tp __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) -
                              __promote_preserving_unsigned(__y));
    }

    template <class _Tp> static inline constexpr _Tp multiplies(_Tp __x, _Tp __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) *
                              __promote_preserving_unsigned(__y));
    }

    template <class _Tp> static inline _Tp divides(_Tp __x, _Tp __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) /
                              __promote_preserving_unsigned(__y));
    }

    template <class _Tp> static inline _Tp modulus(_Tp __x, _Tp __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) %
                              __promote_preserving_unsigned(__y));
    }

    template <class _Tp>
    static inline _Tp bit_and(_Tp __x, _Tp __y)
    {
      if constexpr (is_floating_point_v<_Tp>)
	{
	  using _I     = __int_for_sizeof_t<_Tp>;
	  const _I __r = reinterpret_cast<const __may_alias<_I>&>(__x) &
			 reinterpret_cast<const __may_alias<_I>&>(__y);
	  return reinterpret_cast<const __may_alias<_Tp>&>(__r);
	}
      else
	{
	  return static_cast<_Tp>(__promote_preserving_unsigned(__x) &
				 __promote_preserving_unsigned(__y));
	}
    }

    template <class _Tp>
    static inline _Tp bit_or(_Tp __x, _Tp __y)
    {
      if constexpr (is_floating_point_v<_Tp>)
	{
	  using _I     = __int_for_sizeof_t<_Tp>;
	  const _I __r = reinterpret_cast<const __may_alias<_I>&>(__x) |
			 reinterpret_cast<const __may_alias<_I>&>(__y);
	  return reinterpret_cast<const __may_alias<_Tp>&>(__r);
	}
      else
	{
	  return static_cast<_Tp>(__promote_preserving_unsigned(__x) |
				 __promote_preserving_unsigned(__y));
	}
    }

    template <class _Tp>
    static inline _Tp bit_xor(_Tp __x, _Tp __y)
    {
      if constexpr (is_floating_point_v<_Tp>)
	{
	  using _I     = __int_for_sizeof_t<_Tp>;
	  const _I __r = reinterpret_cast<const __may_alias<_I>&>(__x) ^
			 reinterpret_cast<const __may_alias<_I>&>(__y);
	  return reinterpret_cast<const __may_alias<_Tp>&>(__r);
	}
      else
	{
	  return static_cast<_Tp>(__promote_preserving_unsigned(__x) ^
				 __promote_preserving_unsigned(__y));
	}
    }

    template <class _Tp> static inline _Tp bit_shift_left(_Tp __x, int __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) << __y);
    }

    template <class _Tp> static inline _Tp bit_shift_right(_Tp __x, int __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) >> __y);
    }

    // math {{{2
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static _Tp __abs(_Tp __x) { return _Tp(std::abs(__x)); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static _Tp __sqrt(_Tp __x) { return std::sqrt(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static _Tp __trunc(_Tp __x) { return std::trunc(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static _Tp __floor(_Tp __x) { return std::floor(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static _Tp __ceil(_Tp __x) { return std::ceil(__x); }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static __simd_tuple<int, abi> __fpclassify(_Tp __x)
    {
        return {std::fpclassify(__x)};
    }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static bool __isfinite(_Tp __x) { return std::isfinite(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static bool __isinf(_Tp __x) { return std::isinf(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static bool __isnan(_Tp __x) { return std::isnan(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static bool __isnormal(_Tp __x) { return std::isnormal(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static bool __signbit(_Tp __x) { return std::signbit(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static bool __isunordered(_Tp __x, _Tp __y) { return std::isunordered(__x, __y); }

    // __increment & __decrement{{{2
    template <class _Tp> static inline void __increment(_Tp &__x) { ++__x; }
    template <class _Tp> static inline void __decrement(_Tp &__x) { --__x; }

    // compares {{{2
    template <class _Tp> static bool equal_to(_Tp __x, _Tp __y) { return __x == __y; }
    template <class _Tp> static bool not_equal_to(_Tp __x, _Tp __y) { return __x != __y; }
    template <class _Tp> static bool less(_Tp __x, _Tp __y) { return __x < __y; }
    template <class _Tp> static bool greater(_Tp __x, _Tp __y) { return __x > __y; }
    template <class _Tp> static bool less_equal(_Tp __x, _Tp __y) { return __x <= __y; }
    template <class _Tp> static bool greater_equal(_Tp __x, _Tp __y) { return __x >= __y; }

    // smart_reference access {{{2
    template <class _Tp, class _U> static void set(_Tp &__v, int __i, _U &&__x) noexcept
    {
        _GLIBCXX_DEBUG_ASSERT(__i == 0);
        __unused(__i);
        __v = std::forward<_U>(__x);
    }

    // masked_assign {{{2
    template <typename _Tp> _GLIBCXX_SIMD_INTRINSIC static void masked_assign(bool __k, _Tp &__lhs, _Tp __rhs)
    {
        if (__k) {
            __lhs = __rhs;
        }
    }

    // __masked_cassign {{{2
    template <template <typename> class _Op, typename _Tp>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_cassign(const bool __k, _Tp &__lhs, const _Tp __rhs)
    {
        if (__k) {
            __lhs = _Op<_Tp>{}(__lhs, __rhs);
        }
    }

    // masked_unary {{{2
    template <template <typename> class _Op, typename _Tp>
    _GLIBCXX_SIMD_INTRINSIC static _Tp masked_unary(const bool __k, const _Tp __v)
    {
        return static_cast<_Tp>(__k ? _Op<_Tp>{}(__v) : __v);
    }

    // }}}2
};

// }}}
// __scalar_mask_impl {{{
struct __scalar_mask_impl {
    // member types {{{2
    template <class _Tp> using simd_mask = std::experimental::simd_mask<_Tp, simd_abi::scalar>;
    template <class _Tp> using __type_tag = _Tp *;

    // __from_bitset {{{2
    template <class _Tp>
    _GLIBCXX_SIMD_INTRINSIC static bool __from_bitset(std::bitset<1> bs, __type_tag<_Tp>) noexcept
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
        _GLIBCXX_DEBUG_ASSERT(__i == 0);
        __unused(__i);
        __k = __x;
    }

    // masked_assign {{{2
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(bool __k, bool &__lhs, bool __rhs)
    {
        if (__k) {
            __lhs = __rhs;
        }
    }

    // }}}2
};

// }}}

// ISA & type detection {{{1
template <class _Tp, size_t _N> constexpr bool __is_sse_ps()
{
    return __have_sse && std::is_same_v<_Tp, float> && _N == 4;
}
template <class _Tp, size_t _N> constexpr bool __is_sse_pd()
{
    return __have_sse2 && std::is_same_v<_Tp, double> && _N == 2;
}
template <class _Tp, size_t _N> constexpr bool __is_avx_ps()
{
    return __have_avx && std::is_same_v<_Tp, float> && _N == 8;
}
template <class _Tp, size_t _N> constexpr bool __is_avx_pd()
{
    return __have_avx && std::is_same_v<_Tp, double> && _N == 4;
}
template <class _Tp, size_t _N> constexpr bool __is_avx512_ps()
{
    return __have_avx512f && std::is_same_v<_Tp, float> && _N == 16;
}
template <class _Tp, size_t _N> constexpr bool __is_avx512_pd()
{
    return __have_avx512f && std::is_same_v<_Tp, double> && _N == 8;
}

template <class _Tp, size_t _N> constexpr bool __is_neon_ps()
{
    return __have_neon && std::is_same_v<_Tp, float> && _N == 4;
}
template <class _Tp, size_t _N> constexpr bool __is_neon_pd()
{
    return __have_neon && std::is_same_v<_Tp, double> && _N == 2;
}

// __generic_simd_impl {{{1
template <class _Abi> struct __generic_simd_impl : __simd_math_fallback<_Abi> {
    // member types {{{2
    template <class _Tp> using __type_tag = _Tp *;
    template <class _Tp>
    using _Simd_member_type = typename _Abi::template __traits<_Tp>::_Simd_member_type;
    template <class _Tp>
    using _Mask_member_type = typename _Abi::template __traits<_Tp>::_Mask_member_type;
    template <class _Tp> static constexpr size_t full_size = _Simd_member_type<_Tp>::_S_width;

    // make_simd(__storage/__intrinsic_type_t) {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static simd<_Tp, _Abi> make_simd(__storage<_Tp, _N> __x)
    {
        return {__private_init, __x};
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static simd<_Tp, _Abi> make_simd(__intrinsic_type_t<_Tp, _N> __x)
    {
        return {__private_init, __vector_bitcast<_Tp>(__x)};
    }

    // broadcast {{{2
    template <class _Tp>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Simd_member_type<_Tp> __broadcast(_Tp __x) noexcept
    {
        return __vector_broadcast<full_size<_Tp>>(__x);
    }

    // generator {{{2
    template <class _F, class _Tp>
    inline static _Simd_member_type<_Tp> generator(_F &&__gen, __type_tag<_Tp>)
    {
        return __generate_storage<_Tp, full_size<_Tp>>(std::forward<_F>(__gen));
    }

    // load {{{2
    template <class _Tp, class _U, class _F>
    _GLIBCXX_SIMD_INTRINSIC static _Simd_member_type<_Tp> load(const _U *mem, _F,
                                                 __type_tag<_Tp>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        constexpr size_t _N = _Simd_member_type<_Tp>::_S_width;
        constexpr size_t max_load_size =
            (sizeof(_U) >= 4 && __have_avx512f) || __have_avx512bw
                ? 64
                : (std::is_floating_point_v<_U> && __have_avx) || __have_avx2 ? 32 : 16;
        if constexpr (sizeof(_U) > 8) {
            return __generate_storage<_Tp, _N>(
                [&](auto __i) { return static_cast<_Tp>(mem[__i]); });
        } else if constexpr (std::is_same_v<_U, _Tp>) {
            return __vector_load<_U, _N>(mem, _F());
        } else if constexpr (sizeof(_U) * _N < 16) {
            return __convert<_Simd_member_type<_Tp>>(
                __vector_load16<_U, sizeof(_U) * _N>(mem, _F()));
        } else if constexpr (sizeof(_U) * _N <= max_load_size) {
            return __convert<_Simd_member_type<_Tp>>(__vector_load<_U, _N>(mem, _F()));
        } else if constexpr (sizeof(_U) * _N == 2 * max_load_size) {
            return __convert<_Simd_member_type<_Tp>>(
                __vector_load<_U, _N / 2>(mem, _F()),
                __vector_load<_U, _N / 2>(mem + _N / 2, _F()));
        } else if constexpr (sizeof(_U) * _N == 4 * max_load_size) {
            return __convert<_Simd_member_type<_Tp>>(
                __vector_load<_U, _N / 4>(mem, _F()),
                __vector_load<_U, _N / 4>(mem + 1 * _N / 4, _F()),
                __vector_load<_U, _N / 4>(mem + 2 * _N / 4, _F()),
                __vector_load<_U, _N / 4>(mem + 3 * _N / 4, _F()));
        } else if constexpr (sizeof(_U) * _N == 8 * max_load_size) {
            return __convert<_Simd_member_type<_Tp>>(
                __vector_load<_U, _N / 8>(mem, _F()),
                __vector_load<_U, _N / 8>(mem + 1 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(mem + 2 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(mem + 3 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(mem + 4 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(mem + 5 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(mem + 6 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(mem + 7 * _N / 8, _F()));
        } else {
            __assert_unreachable<_Tp>();
        }
    }

    // masked load {{{2
    template <class _Tp, size_t _N, class _U, class _F>
    static inline __storage<_Tp, _N> masked_load(__storage<_Tp, _N> __merge,
                                                _Mask_member_type<_Tp> __k,
                                                const _U *__mem,
                                                _F) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        if constexpr (std::is_same_v<_Tp, _U> ||  // no conversion
                      (sizeof(_Tp) == sizeof(_U) &&
                       std::is_integral_v<_Tp> ==
                           std::is_integral_v<_U>)  // conversion via bit reinterpretation
        ) {
            [[maybe_unused]] const auto __intrin = __to_intrin(__merge);
            constexpr bool __have_avx512bw_vl_or_zmm =
                __have_avx512bw_vl || (__have_avx512bw && sizeof(__merge) == 64);
            if constexpr (__have_avx512bw_vl_or_zmm && sizeof(_Tp) == 1) {
                if constexpr (sizeof(__merge) == 16) {
                    __merge = __vector_bitcast<_Tp>(
                        _mm_mask_loadu_epi8(__intrin, _mm_movemask_epi8(__to_intrin(__k)), __mem));
                } else if constexpr (sizeof(__merge) == 32) {
                    __merge = __vector_bitcast<_Tp>(_mm256_mask_loadu_epi8(
                        __intrin, _mm256_movemask_epi8(__to_intrin(__k)), __mem));
                } else if constexpr (sizeof(__merge) == 64) {
                    __merge = __vector_bitcast<_Tp>(
                        _mm512_mask_loadu_epi8(__intrin, __k, __mem));
                } else {
                    __assert_unreachable<_Tp>();
                }
            } else if constexpr (__have_avx512bw_vl_or_zmm && sizeof(_Tp) == 2) {
                if constexpr (sizeof(__merge) == 16) {
                    __merge = __vector_bitcast<_Tp>(
                        _mm_mask_loadu_epi16(__intrin, movemask_epi16(__to_intrin(__k)), __mem));
                } else if constexpr (sizeof(__merge) == 32) {
                    __merge = __vector_bitcast<_Tp>(
                        _mm256_mask_loadu_epi16(__intrin, movemask_epi16(__to_intrin(__k)), __mem));
                } else if constexpr (sizeof(__merge) == 64) {
                    __merge = __vector_bitcast<_Tp>(
                        _mm512_mask_loadu_epi16(__intrin, __k, __mem));
                } else {
                    __assert_unreachable<_Tp>();
                }
            } else if constexpr (__have_avx2 && sizeof(_Tp) == 4 &&
                                 std::is_integral_v<_U>) {
                if constexpr (sizeof(__merge) == 16) {
                    __merge = (~__k._M_data & __merge._M_data) |
                              __vector_bitcast<_Tp>(_mm_maskload_epi32(
                                  reinterpret_cast<const int *>(__mem), __to_intrin(__k)));
                } else if constexpr (sizeof(__merge) == 32) {
                    __merge = (~__k._M_data & __merge._M_data) |
                              __vector_bitcast<_Tp>(_mm256_maskload_epi32(
                                  reinterpret_cast<const int *>(__mem), __to_intrin(__k)));
                } else if constexpr (__have_avx512f && sizeof(__merge) == 64) {
                    __merge = __vector_bitcast<_Tp>(
                        _mm512_mask_loadu_epi32(__intrin, __k, __mem));
                } else {
                    __assert_unreachable<_Tp>();
                }
            } else if constexpr (__have_avx && sizeof(_Tp) == 4) {
                if constexpr (sizeof(__merge) == 16) {
                    __merge = __or(__andnot(__k._M_data, __merge._M_data),
                                   __vector_bitcast<_Tp>(_mm_maskload_ps(
                                       reinterpret_cast<const float *>(__mem),
                                       __vector_bitcast<_LLong>(__k))));
                } else if constexpr (sizeof(__merge) == 32) {
                    __merge =
                        __or(__andnot(__k._M_data, __merge._M_data),
                             _mm256_maskload_ps(reinterpret_cast<const float *>(__mem),
                                                __vector_bitcast<_LLong>(__k)));
                } else if constexpr (__have_avx512f && sizeof(__merge) == 64) {
                    __merge =
                        __vector_bitcast<_Tp>(_mm512_mask_loadu_ps(__intrin, __k, __mem));
                } else {
                    __assert_unreachable<_Tp>();
                }
            } else if constexpr (__have_avx2 && sizeof(_Tp) == 8 &&
                                 std::is_integral_v<_U>) {
                if constexpr (sizeof(__merge) == 16) {
                    __merge = (~__k._M_data & __merge._M_data) |
                              __vector_bitcast<_Tp>(_mm_maskload_epi64(
                                  reinterpret_cast<const _LLong *>(__mem), __to_intrin(__k)));
                } else if constexpr (sizeof(__merge) == 32) {
                    __merge = (~__k._M_data & __merge._M_data) |
                              __vector_bitcast<_Tp>(_mm256_maskload_epi64(
                                  reinterpret_cast<const _LLong *>(__mem), __to_intrin(__k)));
                } else if constexpr (__have_avx512f && sizeof(__merge) == 64) {
                    __merge = __vector_bitcast<_Tp>(
                        _mm512_mask_loadu_epi64(__intrin, __k, __mem));
                } else {
                    __assert_unreachable<_Tp>();
                }
            } else if constexpr (__have_avx && sizeof(_Tp) == 8) {
                if constexpr (sizeof(__merge) == 16) {
                    __merge = __or(__andnot(__k._M_data, __merge._M_data),
                                   __vector_bitcast<_Tp>(_mm_maskload_pd(
                                       reinterpret_cast<const double *>(__mem),
                                       __vector_bitcast<_LLong>(__k))));
                } else if constexpr (sizeof(__merge) == 32) {
                    __merge =
                        __or(__andnot(__k._M_data, __merge._M_data),
                             _mm256_maskload_pd(reinterpret_cast<const double *>(__mem),
                                                __vector_bitcast<_LLong>(__k)));
                } else if constexpr (__have_avx512f && sizeof(__merge) == 64) {
                    __merge =
                        __vector_bitcast<_Tp>(_mm512_mask_loadu_pd(__intrin, __k, __mem));
                } else {
                    __assert_unreachable<_Tp>();
                }
            } else {
                __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(), [&](auto __i) {
                    __merge.set(__i, static_cast<_Tp>(__mem[__i]));
                });
            }
        } else if constexpr (sizeof(_U) <= 8 &&  // no long double
                             !__converts_via_decomposition_v<
                                 _U, _Tp, sizeof(__merge)>  // conversion via decomposition
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
            using _AImpl = typename _ATraits::_Simd_impl_type;
            typename _ATraits::_Simd_member_type uncvted{};
            typename _ATraits::_Mask_member_type kk;
            if constexpr (__is_fixed_size_abi_v<_A>) {
                kk = __vector_to_bitset(__k._M_data);
            } else {
                kk = __convert_mask<typename _ATraits::_Mask_member_type>(__k);
            }
            uncvted = _AImpl::masked_load(uncvted, kk, __mem, _F());
            __simd_converter<_U, _A, _Tp, _Abi> converter;
            masked_assign(__k, __merge, converter(uncvted));
        } else {
            __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(),
                            [&](auto __i) { __merge.set(__i, static_cast<_Tp>(__mem[__i])); });
        }
        return __merge;
    }

    // store {{{2
    template <class _Tp, class _U, class _F>
    _GLIBCXX_SIMD_INTRINSIC static void store(_Simd_member_type<_Tp> __v, _U *mem, _F,
                                   __type_tag<_Tp>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        // TODO: converting int -> "smaller int" can be optimized with AVX512
        constexpr size_t _N = _Simd_member_type<_Tp>::_S_width;
        constexpr size_t __max_store_size =
            (sizeof(_U) >= 4 && __have_avx512f) || __have_avx512bw
                ? 64
                : (std::is_floating_point_v<_U> && __have_avx) || __have_avx2 ? 32 : 16;
        if constexpr (sizeof(_U) > 8) {
            __execute_n_times<_N>([&](auto __i) { mem[__i] = __v[__i]; });
        } else if constexpr (std::is_same_v<_U, _Tp>) {
            __vector_store(__v._M_data, mem, _F());
        } else if constexpr (sizeof(_U) * _N < 16) {
            __vector_store<sizeof(_U) * _N>(__convert<__vector_type16_t<_U>>(__v),
                                                 mem, _F());
        } else if constexpr (sizeof(_U) * _N <= __max_store_size) {
            __vector_store(__convert<__vector_type_t<_U, _N>>(__v), mem, _F());
        } else {
            constexpr size_t __vsize = __max_store_size / sizeof(_U);
            constexpr size_t __stores = _N / __vsize;
            using _V = __vector_type_t<_U, __vsize>;
            const std::array<_V, __stores> __converted = __convert_all<_V>(__v);
            __execute_n_times<__stores>([&](auto __i) {
                __vector_store(__converted[__i], mem + __i * __vsize, _F());
            });
        }
    }

    // masked store {{{2
    template <class _Tp, size_t _N, class _U, class _F>
    static inline void masked_store(const __storage<_Tp, _N> __v, _U *__mem, _F,
                                    const _Mask_member_type<_Tp> __k) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        [[maybe_unused]] const auto __vi = __to_intrin(__v);
        constexpr size_t __max_store_size =
            (sizeof(_U) >= 4 && __have_avx512f) || __have_avx512bw
                ? 64
                : (std::is_floating_point_v<_U> && __have_avx) || __have_avx2 ? 32 : 16;
        if constexpr (std::is_same_v<_Tp, _U> ||
                      (std::is_integral_v<_Tp> && std::is_integral_v<_U> &&
                       sizeof(_Tp) == sizeof(_U))) {
            // bitwise or no conversion, reinterpret:
            const auto kk = [&]() {
                if constexpr (__is_bitmask_v<decltype(__k)>) {
                    return _Mask_member_type<_U>(__k._M_data);
                } else {
                    return __storage_bitcast<_U>(__k);
                }
            }();
            __maskstore(__storage_bitcast<_U>(__v), __mem, _F(), kk);
        } else if constexpr (std::is_integral_v<_Tp> && std::is_integral_v<_U> &&
                             sizeof(_Tp) > sizeof(_U) && __have_avx512f &&
                             (sizeof(_Tp) >= 4 || __have_avx512bw) &&
                             (sizeof(__v) == 64 || __have_avx512vl)) {  // truncating store
            const auto kk = [&]() {
                if constexpr (__is_bitmask_v<decltype(__k)>) {
                    return __k;
                } else {
                    return __convert_mask<__storage<bool, _N>>(__k);
                }
            }();
            if constexpr (sizeof(_Tp) == 8 && sizeof(_U) == 4) {
                if constexpr (sizeof(__vi) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi32(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi32(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 16) {
                    _mm_mask_cvtepi64_storeu_epi32(__mem, kk, __vi);
                }
            } else if constexpr (sizeof(_Tp) == 8 && sizeof(_U) == 2) {
                if constexpr (sizeof(__vi) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi16(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi16(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 16) {
                    _mm_mask_cvtepi64_storeu_epi16(__mem, kk, __vi);
                }
            } else if constexpr (sizeof(_Tp) == 8 && sizeof(_U) == 1) {
                if constexpr (sizeof(__vi) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi8(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi8(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 16) {
                    _mm_mask_cvtepi64_storeu_epi8(__mem, kk, __vi);
                }
            } else if constexpr (sizeof(_Tp) == 4 && sizeof(_U) == 2) {
                if constexpr (sizeof(__vi) == 64) {
                    _mm512_mask_cvtepi32_storeu_epi16(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 32) {
                    _mm256_mask_cvtepi32_storeu_epi16(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 16) {
                    _mm_mask_cvtepi32_storeu_epi16(__mem, kk, __vi);
                }
            } else if constexpr (sizeof(_Tp) == 4 && sizeof(_U) == 1) {
                if constexpr (sizeof(__vi) == 64) {
                    _mm512_mask_cvtepi32_storeu_epi8(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 32) {
                    _mm256_mask_cvtepi32_storeu_epi8(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 16) {
                    _mm_mask_cvtepi32_storeu_epi8(__mem, kk, __vi);
                }
            } else if constexpr (sizeof(_Tp) == 2 && sizeof(_U) == 1) {
                if constexpr (sizeof(__vi) == 64) {
                    _mm512_mask_cvtepi16_storeu_epi8(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 32) {
                    _mm256_mask_cvtepi16_storeu_epi8(__mem, kk, __vi);
                } else if constexpr (sizeof(__vi) == 16) {
                    _mm_mask_cvtepi16_storeu_epi8(__mem, kk, __vi);
                }
            } else {
                __assert_unreachable<_Tp>();
            }
        } else if constexpr (sizeof(_U) <= 8 &&  // no long double
                             !__converts_via_decomposition_v<
                                 _Tp, _U, __max_store_size>  // conversion via decomposition
                                                          // is better handled via the
                                                          // bit_iteration fallback below
        ) {
            using _VV = __storage<_U, std::clamp(_N, 16 / sizeof(_U), __max_store_size / sizeof(_U))>;
            using _V = typename _VV::register_type;
            constexpr bool prefer_bitmask =
                (__have_avx512f && sizeof(_U) >= 4) || __have_avx512bw;
            using _M = __storage<std::conditional_t<prefer_bitmask, bool, _U>, _VV::_S_width>;
            constexpr size_t _VN = __vector_traits<_V>::_S_width;

            if constexpr (_VN >= _N) {
                __maskstore(_VV(__convert<_V>(__v)), __mem,
                               // careful, if _V has more elements than the input __v (_N),
                               // vector_aligned is incorrect:
                               std::conditional_t<(__vector_traits<_V>::_S_width > _N),
                                                  overaligned_tag<sizeof(_U) * _N>, _F>(),
                               __convert_mask<_M>(__k));
            } else if constexpr (_VN * 2 == _N) {
                const std::array<_V, 2> converted = __convert_all<_V>(__v);
                __maskstore(_VV(converted[0]), __mem, _F(), __convert_mask<_M>(__extract_part<0, 2>(__k)));
                __maskstore(_VV(converted[1]), __mem + _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<1, 2>(__k)));
            } else if constexpr (_VN * 4 == _N) {
                const std::array<_V, 4> converted = __convert_all<_V>(__v);
                __maskstore(_VV(converted[0]), __mem, _F(), __convert_mask<_M>(__extract_part<0, 4>(__k)));
                __maskstore(_VV(converted[1]), __mem + 1 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<1, 4>(__k)));
                __maskstore(_VV(converted[2]), __mem + 2 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<2, 4>(__k)));
                __maskstore(_VV(converted[3]), __mem + 3 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<3, 4>(__k)));
            } else if constexpr (_VN * 8 == _N) {
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
                __assert_unreachable<_Tp>();
            }
        } else {
            __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(),
                            [&](auto __i) { __mem[__i] = static_cast<_U>(__v[__i]); });
        }
    }

    // complement {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> complement(__storage<_Tp, _N> __x) noexcept
    {
        return ~__x._M_data;
    }

    // unary minus {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> unary_minus(__storage<_Tp, _N> __x) noexcept
    {
        // GCC doesn't use the psign instructions, but pxor & psub seem to be just as good
        // a choice as pcmpeqd & psign. So meh.
        return -__x._M_data;
    }

    // arithmetic operators {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> plus(__storage<_Tp, _N> __x,
                                                                    __storage<_Tp, _N> __y)
    {
        return __plus(__x, __y);
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> minus(__storage<_Tp, _N> __x,
                                                                     __storage<_Tp, _N> __y)
    {
        return __minus(__x, __y);
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> multiplies(
        __storage<_Tp, _N> __x, __storage<_Tp, _N> __y)
    {
        return __multiplies(__x, __y);
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> divides(
        __storage<_Tp, _N> __x, __storage<_Tp, _N> __y)
    {
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX4
        return __divides(__x._M_data, __y._M_data);
#else
        return __x._M_data / __y._M_data;
#endif
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> modulus(__storage<_Tp, _N> __x, __storage<_Tp, _N> __y)
    {
        static_assert(std::is_integral<_Tp>::value, "modulus is only supported for integral types");
        return __x._M_data % __y._M_data;
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> bit_and(__storage<_Tp, _N> __x, __storage<_Tp, _N> __y)
    {
        return __vector_bitcast<_Tp>(__vector_bitcast<_LLong>(__x._M_data) & __vector_bitcast<_LLong>(__y._M_data));
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> bit_or(__storage<_Tp, _N> __x, __storage<_Tp, _N> __y)
    {
        return __vector_bitcast<_Tp>(__vector_bitcast<_LLong>(__x._M_data) | __vector_bitcast<_LLong>(__y._M_data));
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> bit_xor(__storage<_Tp, _N> __x, __storage<_Tp, _N> __y)
    {
        return __vector_bitcast<_Tp>(__vector_bitcast<_LLong>(__x._M_data) ^ __vector_bitcast<_LLong>(__y._M_data));
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_Tp, _N> bit_shift_left(__storage<_Tp, _N> __x, __storage<_Tp, _N> __y)
    {
        return __x._M_data << __y._M_data;
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_Tp, _N> bit_shift_right(__storage<_Tp, _N> __x, __storage<_Tp, _N> __y)
    {
        return __x._M_data >> __y._M_data;
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> bit_shift_left(__storage<_Tp, _N> __x, int __y)
    {
        return __x._M_data << __y;
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> bit_shift_right(__storage<_Tp, _N> __x,
                                                                         int __y)
    {
        return __x._M_data >> __y;
    }

    // compares {{{2
    // equal_to {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Mask_member_type<_Tp> equal_to(
        __storage<_Tp, _N> __x, __storage<_Tp, _N> __y)
    {
        [[maybe_unused]] const auto __xi = __to_intrin(__x);
        [[maybe_unused]] const auto __yi = __to_intrin(__y);
        if constexpr (sizeof(__x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<_Tp>) {
                       if constexpr (sizeof(_Tp) == 8) { return _mm512_cmp_pd_mask(__xi, __yi, _CMP_EQ_OQ);
                } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmp_ps_mask(__xi, __yi, _CMP_EQ_OQ);
                } else { __assert_unreachable<_Tp>(); }
            } else {
                       if constexpr (sizeof(_Tp) == 8) { return _mm512_cmpeq_epi64_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmpeq_epi32_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 2) { return _mm512_cmpeq_epi16_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 1) { return _mm512_cmpeq_epi8_mask(__xi, __yi);
                } else { __assert_unreachable<_Tp>(); }
            }
        } else {
            return _To_storage(__x._M_data == __y._M_data);
        }
    }

    // not_equal_to {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Mask_member_type<_Tp> not_equal_to(
        __storage<_Tp, _N> __x, __storage<_Tp, _N> __y)
    {
        [[maybe_unused]] const auto __xi = __to_intrin(__x);
        [[maybe_unused]] const auto __yi = __to_intrin(__y);
        if constexpr (sizeof(__x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<_Tp>) {
                       if constexpr (sizeof(_Tp) == 8) { return _mm512_cmp_pd_mask(__xi, __yi, _CMP_NEQ_UQ);
                } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmp_ps_mask(__xi, __yi, _CMP_NEQ_UQ);
                } else { __assert_unreachable<_Tp>(); }
            } else {
                       if constexpr (sizeof(_Tp) == 8) { return ~_mm512_cmpeq_epi64_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 4) { return ~_mm512_cmpeq_epi32_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 2) { return ~_mm512_cmpeq_epi16_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 1) { return ~_mm512_cmpeq_epi8_mask(__xi, __yi);
                } else { __assert_unreachable<_Tp>(); }
            }
        } else {
            return _To_storage(__x._M_data != __y._M_data);
        }
    }

    // less {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Mask_member_type<_Tp> less(__storage<_Tp, _N> __x,
                                                           __storage<_Tp, _N> __y)
    {
        [[maybe_unused]] const auto __xi = __to_intrin(__x);
        [[maybe_unused]] const auto __yi = __to_intrin(__y);
        if constexpr (sizeof(__x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<_Tp>) {
                       if constexpr (sizeof(_Tp) == 8) { return _mm512_cmp_pd_mask(__xi, __yi, _CMP_LT_OS);
                } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmp_ps_mask(__xi, __yi, _CMP_LT_OS);
                } else { __assert_unreachable<_Tp>(); }
            } else if constexpr (std::is_signed_v<_Tp>) {
                       if constexpr (sizeof(_Tp) == 8) { return _mm512_cmplt_epi64_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmplt_epi32_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 2) { return _mm512_cmplt_epi16_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 1) { return _mm512_cmplt_epi8_mask(__xi, __yi);
                } else { __assert_unreachable<_Tp>(); }
            } else {
                static_assert(std::is_unsigned_v<_Tp>);
                       if constexpr (sizeof(_Tp) == 8) { return _mm512_cmplt_epu64_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmplt_epu32_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 2) { return _mm512_cmplt_epu16_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 1) { return _mm512_cmplt_epu8_mask(__xi, __yi);
                } else { __assert_unreachable<_Tp>(); }
            }
        } else {
            return _To_storage(__x._M_data < __y._M_data);
        }
    }

    // less_equal {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Mask_member_type<_Tp> less_equal(__storage<_Tp, _N> __x,
                                                                 __storage<_Tp, _N> __y)
    {
        [[maybe_unused]] const auto __xi = __to_intrin(__x);
        [[maybe_unused]] const auto __yi = __to_intrin(__y);
        if constexpr (sizeof(__x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<_Tp>) {
                       if constexpr (sizeof(_Tp) == 8) { return _mm512_cmp_pd_mask(__xi, __yi, _CMP_LE_OS);
                } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmp_ps_mask(__xi, __yi, _CMP_LE_OS);
                } else { __assert_unreachable<_Tp>(); }
            } else if constexpr (std::is_signed_v<_Tp>) {
                       if constexpr (sizeof(_Tp) == 8) { return _mm512_cmple_epi64_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmple_epi32_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 2) { return _mm512_cmple_epi16_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 1) { return _mm512_cmple_epi8_mask(__xi, __yi);
                } else { __assert_unreachable<_Tp>(); }
            } else {
                static_assert(std::is_unsigned_v<_Tp>);
                       if constexpr (sizeof(_Tp) == 8) { return _mm512_cmple_epu64_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmple_epu32_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 2) { return _mm512_cmple_epu16_mask(__xi, __yi);
                } else if constexpr (sizeof(_Tp) == 1) { return _mm512_cmple_epu8_mask(__xi, __yi);
                } else { __assert_unreachable<_Tp>(); }
            }
        } else {
            return _To_storage(__x._M_data <= __y._M_data);
        }
    }

    // negation {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Mask_member_type<_Tp> negate(__storage<_Tp, _N> __x) noexcept
    {
        if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
            return equal_to(__x, _Simd_member_type<_Tp>());
        } else {
            return _To_storage(!__x._M_data);
        }
    }

    // min, max, clamp {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> min(__storage<_Tp, _N> __a,
                                                                   __storage<_Tp, _N> __b)
    {
        return __a._M_data < __b._M_data ? __a._M_data : __b._M_data;
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> max(__storage<_Tp, _N> __a,
                                                                   __storage<_Tp, _N> __b)
    {
        return __a._M_data > __b._M_data ? __a._M_data : __b._M_data;
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr std::pair<__storage<_Tp, _N>, __storage<_Tp, _N>>
    minmax(__storage<_Tp, _N> __a, __storage<_Tp, _N> __b)
    {
        return {__a._M_data < __b._M_data ? __a._M_data : __b._M_data, __a._M_data < __b._M_data ? __b._M_data : __a._M_data};
    }

    // reductions {{{2
    template <class _Tp, class _BinaryOperation>
    _GLIBCXX_SIMD_INTRINSIC static _Tp reduce(simd<_Tp, _Abi> __x,
                                             _BinaryOperation&& __binary_op)
    {
        constexpr size_t _N = simd_size_v<_Tp, _Abi>;
        if constexpr (sizeof(__x) > 16) {
            using _A = simd_abi::deduce_t<_Tp, _N / 2>;
            using _V = std::experimental::simd<_Tp, _A>;
            return __simd_traits<_Tp, _A>::_Simd_impl_type::reduce(
                __binary_op(_V(__private_init, __extract<0, 2>(__data(__x)._M_data)),
                            _V(__private_init, __extract<1, 2>(__data(__x)._M_data))),
                std::forward<_BinaryOperation>(__binary_op));
        } else {
            auto __intrin = __to_intrin(__x._M_data);
            if constexpr (_N == 16) {
                __x =
                    __binary_op(make_simd<_Tp, _N>(_mm_unpacklo_epi8(__intrin, __intrin)),
                                make_simd<_Tp, _N>(_mm_unpackhi_epi8(__intrin, __intrin)));
                __intrin = __to_intrin(__x._M_data);
            }
            if constexpr (_N >= 8) {
                __x = __binary_op(
                    make_simd<_Tp, _N>(_mm_unpacklo_epi16(__intrin, __intrin)),
                    make_simd<_Tp, _N>(_mm_unpackhi_epi16(__intrin, __intrin)));
                __intrin = __to_intrin(__x._M_data);
            }
            if constexpr (_N >= 4) {
                using _U = std::conditional_t<std::is_floating_point_v<_Tp>, float, int>;
                const auto __y = __vector_bitcast<_U>(__intrin);
                __x            = __binary_op(
                    __x, make_simd<_Tp, _N>(_To_storage(
                             __vector_type_t<_U, 4>{__y[3], __y[2], __y[1], __y[0]})));
                __intrin = __to_intrin(__x._M_data);
            }
            const auto __y = __vector_bitcast<_LLong>(__intrin);
            return __binary_op(__x, make_simd<_Tp, _N>(_To_storage(
                                        __vector_type_t<_LLong, 2>{__y[1], __y[1]})))[0];
        }
    }

    // math {{{2
    // sqrt {{{3
    template <class _Tp, size_t _N> _GLIBCXX_SIMD_INTRINSIC static __storage<_Tp, _N> __sqrt(__storage<_Tp, _N> __x)
    {
               if constexpr (__is_sse_ps   <_Tp, _N>()) { return _mm_sqrt_ps(__x);
        } else if constexpr (__is_sse_pd   <_Tp, _N>()) { return _mm_sqrt_pd(__x);
        } else if constexpr (__is_avx_ps   <_Tp, _N>()) { return _mm256_sqrt_ps(__x);
        } else if constexpr (__is_avx_pd   <_Tp, _N>()) { return _mm256_sqrt_pd(__x);
        } else if constexpr (__is_avx512_ps<_Tp, _N>()) { return _mm512_sqrt_ps(__x);
        } else if constexpr (__is_avx512_pd<_Tp, _N>()) { return _mm512_sqrt_pd(__x);
        } else { __assert_unreachable<_Tp>(); }
    }

    // abs {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_Tp, _N> __abs(__storage<_Tp, _N> __x) noexcept
    {
        return std::experimental::parallelism_v2::__abs(__x);
    }

    // trunc {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_Tp, _N> __trunc(__storage<_Tp, _N> __x)
    {
        if constexpr (__is_avx512_ps<_Tp, _N>()) {
            return _mm512_roundscale_round_ps(__x, 0x03, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx512_pd<_Tp, _N>()) {
            return _mm512_roundscale_round_pd(__x, 0x03, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx_ps<_Tp, _N>()) {
            return _mm256_round_ps(__x, 0x3);
        } else if constexpr (__is_avx_pd<_Tp, _N>()) {
            return _mm256_round_pd(__x, 0x3);
        } else if constexpr (__have_sse4_1 && __is_sse_ps<_Tp, _N>()) {
            return _mm_round_ps(__x, 0x3);
        } else if constexpr (__have_sse4_1 && __is_sse_pd<_Tp, _N>()) {
            return _mm_round_pd(__x, 0x3);
        } else if constexpr (__is_sse_ps<_Tp, _N>()) {
            auto truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(__x));
            const auto no_fractional_values = __vector_bitcast<float>(
                __vector_bitcast<int>(__vector_bitcast<_UInt>(__x._M_data) & 0x7f800000u) <
                0x4b000000);  // the exponent is so large that no mantissa bits signify
                              // fractional values (0x3f8 + 23*8 = 0x4b0)
            return __blend(no_fractional_values, __x, truncated);
        } else if constexpr (__is_sse_pd<_Tp, _N>()) {
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
            __assert_unreachable<_Tp>();
        }
    }

    // floor {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_Tp, _N> __floor(__storage<_Tp, _N> __x)
    {
        if constexpr (__is_avx512_ps<_Tp, _N>()) {
            return _mm512_roundscale_round_ps(__x, 0x01, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx512_pd<_Tp, _N>()) {
            return _mm512_roundscale_round_pd(__x, 0x01, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx_ps<_Tp, _N>()) {
            return _mm256_round_ps(__x, 0x1);
        } else if constexpr (__is_avx_pd<_Tp, _N>()) {
            return _mm256_round_pd(__x, 0x1);
        } else if constexpr (__have_sse4_1 && __is_sse_ps<_Tp, _N>()) {
            return _mm_floor_ps(__x);
        } else if constexpr (__have_sse4_1 && __is_sse_pd<_Tp, _N>()) {
            return _mm_floor_pd(__x);
        } else {
            const auto __y = __trunc(__x)._M_data;
            const auto negative_input = __vector_bitcast<_Tp>(__x._M_data < __vector_broadcast<_N, _Tp>(0));
            const auto mask = __andnot(__vector_bitcast<_Tp>(__y == __x._M_data), negative_input);
            return __or(__andnot(mask, __y), __and(mask, __y - __vector_broadcast<_N, _Tp>(1)));
        }
    }

    // ceil {{{3
    template <class _Tp, size_t _N> _GLIBCXX_SIMD_INTRINSIC static __storage<_Tp, _N> __ceil(__storage<_Tp, _N> __x)
    {
        if constexpr (__is_avx512_ps<_Tp, _N>()) {
            return _mm512_roundscale_round_ps(__x, 0x02, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx512_pd<_Tp, _N>()) {
            return _mm512_roundscale_round_pd(__x, 0x02, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (__is_avx_ps<_Tp, _N>()) {
            return _mm256_round_ps(__x, 0x2);
        } else if constexpr (__is_avx_pd<_Tp, _N>()) {
            return _mm256_round_pd(__x, 0x2);
        } else if constexpr (__have_sse4_1 && __is_sse_ps<_Tp, _N>()) {
            return _mm_ceil_ps(__x);
        } else if constexpr (__have_sse4_1 && __is_sse_pd<_Tp, _N>()) {
            return _mm_ceil_pd(__x);
        } else {
            const auto __y = __trunc(__x)._M_data;
            const auto negative_input = __vector_bitcast<_Tp>(__x._M_data < __vector_broadcast<_N, _Tp>(0));
            const auto inv_mask = __or(__vector_bitcast<_Tp>(__y == __x._M_data), negative_input);
            return __or(__and(inv_mask, __y),
                       __andnot(inv_mask, __y + __vector_broadcast<_N, _Tp>(1)));
        }
    }

    // isnan {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_Tp> __isnan(__storage<_Tp, _N> __x)
    {
             if constexpr (__is_sse_ps   <_Tp, _N>()) { return _mm_cmpunord_ps(__x, __x); }
        else if constexpr (__is_avx_ps   <_Tp, _N>()) { return _mm256_cmp_ps(__x, __x, _CMP_UNORD_Q); }
        else if constexpr (__is_avx512_ps<_Tp, _N>()) { return _mm512_cmp_ps_mask(__x, __x, _CMP_UNORD_Q); }
        else if constexpr (__is_sse_pd   <_Tp, _N>()) { return _mm_cmpunord_pd(__x, __x); }
        else if constexpr (__is_avx_pd   <_Tp, _N>()) { return _mm256_cmp_pd(__x, __x, _CMP_UNORD_Q); }
        else if constexpr (__is_avx512_pd<_Tp, _N>()) { return _mm512_cmp_pd_mask(__x, __x, _CMP_UNORD_Q); }
        else { __assert_unreachable<_Tp>(); }
    }

    // isfinite {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_Tp> __isfinite(__storage<_Tp, _N> __x)
    {
        return __cmpord(__x._M_data, __x._M_data * _Tp());
    }

    // isunordered {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_Tp> __isunordered(__storage<_Tp, _N> __x,
                                                          __storage<_Tp, _N> __y)
    {
        return __cmpunord(__x._M_data, __y._M_data);
    }

    // signbit {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_Tp> __signbit(__storage<_Tp, _N> __x)
    {
        using _I = __int_for_sizeof_t<_Tp>;
        if constexpr (__have_avx512dq && __is_avx512_ps<_Tp, _N>()) {
            return _mm512_movepi32_mask(__vector_bitcast<_LLong>(__x));
        } else if constexpr (__have_avx512dq && __is_avx512_pd<_Tp, _N>()) {
            return _mm512_movepi64_mask(__vector_bitcast<_LLong>(__x));
        } else if constexpr (sizeof(__x) == 64) {
            const auto signmask = __vector_broadcast<_N>(std::numeric_limits<_I>::min());
            return equal_to(__storage<_I, _N>(__vector_bitcast<_I>(__x._M_data) & signmask),
                            __storage<_I, _N>(signmask));
        } else {
            const auto __xx = __vector_bitcast<_I>(__x._M_data);
            constexpr _I signmask = std::numeric_limits<_I>::min();
            if constexpr ((sizeof(_Tp) == 4 && (__have_avx2 || sizeof(__x) == 16)) ||
                          __have_avx512vl) {
                (void)signmask;
                return __vector_bitcast<_Tp>(__xx >> std::numeric_limits<_I>::digits);
            } else if constexpr ((__have_avx2 || (__have_ssse3 && sizeof(__x) == 16))) {
                return __vector_bitcast<_Tp>((__xx & signmask) == signmask);
            } else {  // SSE2/3 or AVX (w/o AVX2)
                constexpr auto one = __vector_broadcast<_N, _Tp>(1);
                return __vector_bitcast<_Tp>(
                    __vector_bitcast<_Tp>((__xx & signmask) | __vector_bitcast<_I>(one))  // -1 or 1
                    != one);
            }
        }
    }

    // isnonzerovalue (isnormal | is subnormal == !isinf & !isnan & !is zero) {{{3
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static auto isnonzerovalue(_Tp __x)
    {
      using _Traits = __vector_traits<_Tp>;
      if constexpr (__have_avx512dq)
	{
	  if constexpr (__have_avx512vl && _Traits::template is<float, 4>)
	    return __vector_bitcast<float>(
	      _mm_movm_epi32(_knot_mask8(_mm_fpclass_ps_mask(__x, 0x9f))));
	  else if constexpr (__have_avx512vl && _Traits::template is<float, 8>)
	    return __vector_bitcast<float>(_mm256_movm_epi32(
	      _knot_mask8(_mm256_fpclass_ps_mask(__x, 0x9f))));
	  else if constexpr (_Traits::template is<float, 16>)
	    return _knot_mask16(_mm512_fpclass_ps_mask(__x, 0x9f));
	  else if constexpr (__have_avx512vl && _Traits::template is<double, 2>)
	    return __vector_bitcast<double>(
	      _mm_movm_epi64(_knot_mask8(_mm_fpclass_pd_mask(__x, 0x9f))));
	  else if constexpr (__have_avx512vl && _Traits::template is<double, 4>)
	    return __vector_bitcast<double>(_mm256_movm_epi64(
	      _knot_mask8(_mm256_fpclass_pd_mask(__x, 0x9f))));
	  else if constexpr (_Traits::template is<double, 8>)
	    return _knot_mask8(_mm512_fpclass_pd_mask(__x, 0x9f));
	  else
	    __assert_unreachable<_Tp>();
	}
      else
	{
	  using _U = typename _Traits::value_type;
	  return __cmpord(
	    __x * std::numeric_limits<_U>::infinity(), // NaN if __x == 0
	    __x * _U()                                 // NaN if __x == inf
	  );
	}
    }

    template <class _Tp>
    _GLIBCXX_SIMD_INTRINSIC static auto isnonzerovalue_mask(_Tp __x)
    {
      using _Traits = __vector_traits<_Tp>;
      if constexpr (__have_avx512dq_vl)
	{
	  if constexpr (_Traits::template is<float, 4>)
	    return _knot_mask8(_mm_fpclass_ps_mask(__x, 0x9f));
	  else if constexpr (_Traits::template is<float, 8>)
	    return _knot_mask8(_mm256_fpclass_ps_mask(__x, 0x9f));
	  else if constexpr (_Traits::template is<float, 16>)
	    return _knot_mask16(_mm512_fpclass_ps_mask(__x, 0x9f));
	  else if constexpr (_Traits::template is<double, 2>)
	    return _knot_mask8(_mm_fpclass_pd_mask(__x, 0x9f));
	  else if constexpr (_Traits::template is<double, 4>)
	    return _knot_mask8(_mm256_fpclass_pd_mask(__x, 0x9f));
	  else if constexpr (_Traits::template is<double, 8>)
	    return _knot_mask8(_mm512_fpclass_pd_mask(__x, 0x9f));
	  else
	    __assert_unreachable<_Tp>();
	}
      else
	{
	  using _U            = typename _Traits::value_type;
	  constexpr size_t _N = _Traits::_S_width;
	  const auto       __a =
	    __x * std::numeric_limits<_U>::infinity(); // NaN if __x == 0
	  const auto __b = __x * _U();                 // NaN if __x == inf
	  if constexpr (__have_avx512vl && __is_sse_ps<_U, _N>())
	    {
	      return _mm_cmp_ps_mask(__a, __b, _CMP_ORD_Q);
	    }
	  else if constexpr (__have_avx512f && __is_sse_ps<_U, _N>())
	    {
	      return __mmask8(0xf & _mm512_cmp_ps_mask(__auto_bitcast(__a),
						       __auto_bitcast(__b),
						       _CMP_ORD_Q));
	    }
	  else if constexpr (__have_avx512vl && __is_sse_pd<_U, _N>())
	    {
	      return _mm_cmp_pd_mask(__a, __b, _CMP_ORD_Q);
	    }
	  else if constexpr (__have_avx512f && __is_sse_pd<_U, _N>())
	    {
	      return __mmask8(0x3 & _mm512_cmp_pd_mask(__auto_bitcast(__a),
						       __auto_bitcast(__b),
						       _CMP_ORD_Q));
	    }
	  else if constexpr (__have_avx512vl && __is_avx_ps<_U, _N>())
	    {
	      return _mm256_cmp_ps_mask(__a, __b, _CMP_ORD_Q);
	    }
	  else if constexpr (__have_avx512f && __is_avx_ps<_U, _N>())
	    {
	      return __mmask8(_mm512_cmp_ps_mask(
		__auto_bitcast(__a), __auto_bitcast(__b), _CMP_ORD_Q));
	    }
	  else if constexpr (__have_avx512vl && __is_avx_pd<_U, _N>())
	    {
	      return _mm256_cmp_pd_mask(__a, __b, _CMP_ORD_Q);
	    }
	  else if constexpr (__have_avx512f && __is_avx_pd<_U, _N>())
	    {
	      return __mmask8(0xf & _mm512_cmp_pd_mask(__auto_bitcast(__a),
						       __auto_bitcast(__b),
						       _CMP_ORD_Q));
	    }
	  else if constexpr (__is_avx512_ps<_U, _N>())
	    {
	      return _mm512_cmp_ps_mask(__a, __b, _CMP_ORD_Q);
	    }
	  else if constexpr (__is_avx512_pd<_U, _N>())
	    {
	      return _mm512_cmp_pd_mask(__a, __b, _CMP_ORD_Q);
	    }
	  else
	    {
	      __assert_unreachable<_Tp>();
	    }
	}
    }

    // isinf {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_Tp> __isinf(__storage<_Tp, _N> __x)
    {
      if constexpr (__is_avx512_pd<_Tp, _N>() && __have_avx512dq)
	return _mm512_fpclass_pd_mask(__x, 0x18);
      else if constexpr (__is_avx512_ps<_Tp, _N>() && __have_avx512dq)
	return _mm512_fpclass_ps_mask(__x, 0x18);
      else if constexpr (__have_avx512dq_vl)
	{
	  if constexpr (__is_sse_pd<_Tp, _N>())
	    return __vector_bitcast<double>(
	      _mm_movm_epi64(_mm_fpclass_pd_mask(__x, 0x18)));
	  else if constexpr (__is_avx_pd<_Tp, _N>())
	    return __vector_bitcast<double>(
	      _mm256_movm_epi64(_mm256_fpclass_pd_mask(__x, 0x18)));
	  else if constexpr (__is_sse_ps<_Tp, _N>())
	    return __vector_bitcast<float>(
	      _mm_movm_epi32(_mm_fpclass_ps_mask(__x, 0x18)));
	  else if constexpr (__is_avx_ps<_Tp, _N>())
	    return __vector_bitcast<float>(
	      _mm256_movm_epi32(_mm256_fpclass_ps_mask(__x, 0x18)));
	  else
	    __assert_unreachable<_Tp>();
	}
      else
	{
	  return equal_to(__abs(__x), __storage<_Tp, _N>(__vector_broadcast<_N>(
					std::numeric_limits<_Tp>::infinity())));
	  // alternative:
	  // compare to inf using the corresponding integer type
	  /*
	  return
	  __vector_bitcast<_Tp>(__vector_bitcast<__int_for_sizeof_t<_Tp>>(__abs(__x)._M_data)
	  ==
				 __vector_bitcast<__int_for_sizeof_t<_Tp>>(__vector_broadcast<_N>(
				     std::numeric_limits<_Tp>::infinity())));
				     */
	}
    }
    // isnormal {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_Tp> __isnormal(__storage<_Tp, _N> __x)
    {
      if constexpr (__have_avx512dq)
	{
	  if constexpr (__have_avx512vl && __is_sse_ps<_Tp, _N>())
	    return __vector_bitcast<float>(
	      _mm_movm_epi32(_knot_mask8(_mm_fpclass_ps_mask(__x, 0xbf))));
	  else if constexpr (__have_avx512vl && __is_avx_ps<_Tp, _N>())
	    return __vector_bitcast<float>(_mm256_movm_epi32(
	      _knot_mask8(_mm256_fpclass_ps_mask(__x, 0xbf))));
	  else if constexpr (__is_avx512_ps<_Tp, _N>())
	    return _knot_mask16(_mm512_fpclass_ps_mask(__x, 0xbf));
	  else if constexpr (__have_avx512vl && __is_sse_pd<_Tp, _N>())
	    return __vector_bitcast<double>(
	      _mm_movm_epi64(_knot_mask8(_mm_fpclass_pd_mask(__x, 0xbf))));
	  else if constexpr (__have_avx512vl && __is_avx_pd<_Tp, _N>())
	    return __vector_bitcast<double>(_mm256_movm_epi64(
	      _knot_mask8(_mm256_fpclass_pd_mask(__x, 0xbf))));
	  else if constexpr (__is_avx512_pd<_Tp, _N>())
	    return _knot_mask8(_mm512_fpclass_pd_mask(__x, 0xbf));
	  else
	    __assert_unreachable<_Tp>();
	}
      else
	{
	  // subnormals -> 0
	  // 0 -> 0
	  // inf -> inf
	  // -inf -> inf
	  // nan -> inf
	  // normal value -> positive value / not 0
	  return isnonzerovalue(__and(
	    __x._M_data,
	    __vector_broadcast<_N>(std::numeric_limits<_Tp>::infinity())));
	}
    }

    // fpclassify {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __fixed_size_storage<int, _N> __fpclassify(__storage<_Tp, _N> __x)
    {
        if constexpr (__is_avx512_pd<_Tp, _N>()) {
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
                __vector_bitcast<_Tp>(__vector_broadcast<_N, __int_for_sizeof_t<_Tp>>(FP_NORMAL));
            constexpr auto fp_nan =
                __vector_bitcast<_Tp>(__vector_broadcast<_N, __int_for_sizeof_t<_Tp>>(FP_NAN));
            constexpr auto fp_infinite =
                __vector_bitcast<_Tp>(__vector_broadcast<_N, __int_for_sizeof_t<_Tp>>(FP_INFINITE));
            constexpr auto fp_subnormal =
                __vector_bitcast<_Tp>(__vector_broadcast<_N, __int_for_sizeof_t<_Tp>>(FP_SUBNORMAL));
            constexpr auto fp_zero =
                __vector_bitcast<_Tp>(__vector_broadcast<_N, __int_for_sizeof_t<_Tp>>(FP_ZERO));

            const auto tmp = __vector_bitcast<_LLong>(
                __abs(__x)._M_data < std::numeric_limits<_Tp>::min()
                    ? (__x._M_data == 0 ? fp_zero : fp_subnormal)
                    : __blend(__isinf(__x)._M_data, __blend(__isnan(__x)._M_data, fp_normal, fp_nan),
                                 fp_infinite));
            if constexpr (std::is_same_v<_Tp, float>) {
                if constexpr (__fixed_size_storage<int, _N>::tuple_size == 1) {
                    return {__vector_bitcast<int>(tmp)};
                } else if constexpr (__fixed_size_storage<int, _N>::tuple_size == 2) {
                    return {__extract<0, 2>(__vector_bitcast<int>(tmp)),
                            __extract<1, 2>(__vector_bitcast<int>(tmp))};
                } else {
                    __assert_unreachable<_Tp>();
                }
            } else if constexpr (__is_sse_pd<_Tp, _N>()) {
                static_assert(__fixed_size_storage<int, _N>::tuple_size == 2);
                return {_mm_cvtsi128_si32(tmp),
                        {_mm_cvtsi128_si32(_mm_unpackhi_epi64(tmp, tmp))}};
            } else if constexpr (__is_avx_pd<_Tp, _N>()) {
                static_assert(__fixed_size_storage<int, _N>::tuple_size == 1);
                return {_mm_packs_epi32(__lo128(tmp), __hi128(tmp))};
            } else {
                __assert_unreachable<_Tp>();
            }
        }
    }

    // __increment & __decrement{{{2
    template <class _Tp, size_t _N> _GLIBCXX_SIMD_INTRINSIC static void __increment(__storage<_Tp, _N> &__x)
    {
        __x = plus<_Tp, _N>(__x, __vector_broadcast<_N, _Tp>(1));
    }
    template <class _Tp, size_t _N> _GLIBCXX_SIMD_INTRINSIC static void __decrement(__storage<_Tp, _N> &__x)
    {
        __x = minus<_Tp, _N>(__x, __vector_broadcast<_N, _Tp>(1));
    }

    // smart_reference access {{{2
    template <class _Tp, size_t _N, class _U>
    _GLIBCXX_SIMD_INTRINSIC static void set(__storage<_Tp, _N> &__v, int __i, _U &&__x) noexcept
    {
        __v.set(__i, std::forward<_U>(__x));
    }

    // masked_assign{{{2
    template <class _Tp, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(__storage<_K, _N> __k,
                                                      __storage<_Tp, _N> &__lhs,
                                                      __id<__storage<_Tp, _N>> __rhs)
    {
        __lhs = __blend(__k._M_data, __lhs._M_data, __rhs._M_data);
    }

    template <class _Tp, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(__storage<_K, _N> __k, __storage<_Tp, _N> &__lhs,
                                           __id<_Tp> __rhs)
    {
        if (__builtin_constant_p(__rhs) && __rhs == 0 && std::is_same<_K, _Tp>::value) {
            if constexpr (!__is_bitmask(__k)) {
                // the __andnot optimization only makes sense if __k._M_data is a vector register
                __lhs._M_data = __andnot(__k._M_data, __lhs._M_data);
                return;
            } else {
                // for AVX512/__mmask, a _mm512_maskz_mov is best
                __lhs._M_data = __auto_bitcast(__blend(__k, __lhs, __intrinsic_type_t<_Tp, _N>()));
                return;
            }
        }
        __lhs._M_data = __blend(__k._M_data, __lhs._M_data, __vector_broadcast<_N>(__rhs));
    }

    // __masked_cassign {{{2
    template <template <typename> class _Op, class _Tp, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_cassign(const __storage<_K, _N> __k, __storage<_Tp, _N> &__lhs,
                                            const __id<__storage<_Tp, _N>> __rhs)
    {
        __lhs._M_data = __blend(__k._M_data, __lhs._M_data, _Op<void>{}(__lhs._M_data, __rhs._M_data));
    }

    template <template <typename> class _Op, class _Tp, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_cassign(const __storage<_K, _N> __k, __storage<_Tp, _N> &__lhs,
                                            const __id<_Tp> __rhs)
    {
        __lhs._M_data = __blend(__k._M_data, __lhs._M_data, _Op<void>{}(__lhs._M_data, __vector_broadcast<_N>(__rhs)));
    }

    // masked_unary {{{2
    template <template <typename> class _Op, class _Tp, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __storage<_Tp, _N> masked_unary(const __storage<_K, _N> __k,
                                                            const __storage<_Tp, _N> __v)
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
    template <class _Tp> using __type_tag = _Tp *;
    template <class _Tp> using simd_mask = std::experimental::simd_mask<_Tp, _Abi>;
    template <class _Tp>
    using _Simd_member_type = typename _Abi::template __traits<_Tp>::_Simd_member_type;
    template <class _Tp>
    using _Mask_member_type = typename _Abi::template __traits<_Tp>::_Mask_member_type;

    // masked load {{{2
    template <class _Tp, size_t _N, class _F>
    static inline __storage<_Tp, _N> masked_load(__storage<_Tp, _N> merge, __storage<_Tp, _N> mask,
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
                    __assert_unreachable<_Tp>();
                }
            } else {
                __bit_iteration(mask, [&](auto __i) { merge.set(__i, mem[__i]); });
                return merge;
            }
        } else if constexpr (__have_avx512bw_vl && _N == 32 && sizeof(_Tp) == 1) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(
                _mm256_mask_sub_epi8(__vector_bitcast<_LLong>(merge), __k, __m256i(),
                                     _mm256_mask_loadu_epi8(__m256i(), __k, mem)));
        } else if constexpr (__have_avx512bw_vl && _N == 16 && sizeof(_Tp) == 1) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm_mask_sub_epi8(__vector_bitcast<_LLong>(merge), __k, __m128i(),
                                                 _mm_mask_loadu_epi8(__m128i(), __k, mem)));
        } else if constexpr (__have_avx512bw_vl && _N == 16 && sizeof(_Tp) == 2) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm256_mask_sub_epi16(
                __vector_bitcast<_LLong>(merge), __k, __m256i(),
                _mm256_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), __k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 8 && sizeof(_Tp) == 2) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm_mask_sub_epi16(
                __vector_bitcast<_LLong>(merge), __k, __m128i(),
                _mm_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), __k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 8 && sizeof(_Tp) == 4) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm256_mask_sub_epi32(
                __vector_bitcast<_LLong>(merge), __k, __m256i(),
                _mm256_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), __k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 4 && sizeof(_Tp) == 4) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm_mask_sub_epi32(
                __vector_bitcast<_LLong>(merge), __k, __m128i(),
                _mm_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), __k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 4 && sizeof(_Tp) == 8) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm256_mask_sub_epi64(
                __vector_bitcast<_LLong>(merge), __k, __m256i(),
                _mm256_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), __k, mem))));
        } else if constexpr (__have_avx512bw_vl && _N == 2 && sizeof(_Tp) == 8) {
            const auto __k = __convert_mask<__storage<bool, _N>>(mask);
            merge = _To_storage(_mm_mask_sub_epi64(
                __vector_bitcast<_LLong>(merge), __k, __m128i(),
                _mm_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), __k, mem))));
        } else {
            // AVX(2) has 32/64 bit maskload, but nothing at 8 bit granularity
            auto tmp = __storage_bitcast<__int_for_sizeof_t<_Tp>>(merge);
            __bit_iteration(__vector_to_bitset(mask._M_data).to_ullong(),
                                  [&](auto __i) { tmp.set(__i, -mem[__i]); });
            merge = __storage_bitcast<_Tp>(tmp);
        }
        return merge;
    }

    // store {{{2
    template <class _Tp, size_t _N, class _F>
    _GLIBCXX_SIMD_INTRINSIC static void store(__storage<_Tp, _N> __v, bool *__mem, _F) noexcept
    {
        if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>()) {
            if constexpr (_N == 2 && __have_sse2) {
                const auto __k = __vector_bitcast<int>(__v);
                __mem[0] = -__k[1];
                __mem[1] = -__k[3];
            } else if constexpr (_N == 4 && __have_sse2) {
                const unsigned bool4 =
                    __vector_bitcast<_UInt>(_mm_packs_epi16(
                        _mm_packs_epi32(__vector_bitcast<_LLong>(__v), __m128i()), __m128i()))[0] &
                    0x01010101u;
                std::memcpy(__mem, &bool4, 4);
            } else if constexpr (std::is_same_v<_Tp, float> && __have_mmx) {
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
                __assert_unreachable<_Tp>();
            }
        } else if constexpr (__is_abi<_Abi, simd_abi::__avx_abi>()) {
            if constexpr (_N == 4 && __have_avx) {
                auto __k = __vector_bitcast<_LLong>(__v);
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
                const auto __k = __vector_bitcast<_LLong>(__v);
                const auto k2 = _mm_srli_epi16(_mm_packs_epi16(__lo128(__k), __hi128(__k)), 15);
                const auto k3 = _mm_packs_epi16(k2, __m128i());
                __vector_store<8>(k3, __mem, _F());
            } else if constexpr (_N == 16 && __have_avx2) {
                const auto __x = _mm256_srli_epi16(__to_intrin(__v), 15);
                const auto bools = _mm_packs_epi16(__lo128(__x), __hi128(__x));
                __vector_store<16>(bools, __mem, _F());
            } else if constexpr (_N == 16 && __have_avx) {
                const auto bools = 1 & __vector_bitcast<_UChar>(_mm_packs_epi16(
                                           __lo128(__to_intrin(__v)), __hi128(__to_intrin(__v))));
                __vector_store<16>(bools, __mem, _F());
            } else if constexpr (_N == 32 && __have_avx) {
                __vector_store<32>(1 & __v._M_data, __mem, _F());
            } else {
                __assert_unreachable<_Tp>();
            }
        } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
            if constexpr (_N == 8) {
                __vector_store<8>(
#if _GLIBCXX_SIMD_HAVE_AVX512VL && _GLIBCXX_SIMD_HAVE_AVX512BW
                    _mm_maskz_set1_epi8(__v._M_data, 1),
#elif defined __x86_64__
                    __make_storage<_ULLong>(_pdep_u64(__v._M_data, 0x0101010101010101ULL), 0ull),
#else
                    __make_storage<_UInt>(_pdep_u32(__v._M_data, 0x01010101U),
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
                __assert_unreachable<_Tp>();
            }
        } else {
            __assert_unreachable<_Tp>();
        }
    }

    // masked store {{{2
    template <class _Tp, size_t _N, class _F>
    static inline void masked_store(const __storage<_Tp, _N> __v, bool *__mem, _F,
                                    const __storage<_Tp, _N> __k) noexcept
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
                __assert_unreachable<_Tp>();
            }
        } else {
            __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(), [&](auto __i) { __mem[__i] = __v[__i]; });
        }
    }

    // __from_bitset{{{2
    template <size_t _N, class _Tp>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type<_Tp> __from_bitset(std::bitset<_N> __bits, __type_tag<_Tp>)
    {
        return __convert_mask<typename _Mask_member_type<_Tp>::register_type>(__bits);
    }

    // logical and bitwise operators {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> logical_and(const __storage<_Tp, _N> &__x,
                                                            const __storage<_Tp, _N> &__y)
    {
      if constexpr (std::is_same_v<_Tp, bool>)
	{
	  if constexpr (__have_avx512dq && _N <= 8)
	    return _kand_mask8(__x._M_data, __y._M_data);
	  else if constexpr (_N <= 16)
	    return _kand_mask16(__x._M_data, __y._M_data);
	  else if constexpr (__have_avx512bw && _N <= 32)
	    return _kand_mask32(__x._M_data, __y._M_data);
	  else if constexpr (__have_avx512bw && _N <= 64)
	    return _kand_mask64(__x._M_data, __y._M_data);
	  else
	    __assert_unreachable<_Tp>();
	}
      else
	{
	  return __and(__x._M_data, __y._M_data);
	}
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> logical_or(const __storage<_Tp, _N> &__x,
                                                           const __storage<_Tp, _N> &__y)
    {
      if constexpr (std::is_same_v<_Tp, bool>)
	{
	  if constexpr (__have_avx512dq && _N <= 8)
	    return _kor_mask8(__x._M_data, __y._M_data);
	  else if constexpr (_N <= 16)
	    return _kor_mask16(__x._M_data, __y._M_data);
	  else if constexpr (__have_avx512bw && _N <= 32)
	    return _kor_mask32(__x._M_data, __y._M_data);
	  else if constexpr (__have_avx512bw && _N <= 64)
	    return _kor_mask64(__x._M_data, __y._M_data);
	  else
	    __assert_unreachable<_Tp>();
	}
      else
	{
	  return __or(__x._M_data, __y._M_data);
	}
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> bit_and(const __storage<_Tp, _N> &__x,
                                                        const __storage<_Tp, _N> &__y)
    {
      if constexpr (std::is_same_v<_Tp, bool>)
	{
	  if constexpr (__have_avx512dq && _N <= 8)
	    return _kand_mask8(__x._M_data, __y._M_data);
	  else if constexpr (_N <= 16)
	    return _kand_mask16(__x._M_data, __y._M_data);
	  else if constexpr (__have_avx512bw && _N <= 32)
	    return _kand_mask32(__x._M_data, __y._M_data);
	  else if constexpr (__have_avx512bw && _N <= 64)
	    return _kand_mask64(__x._M_data, __y._M_data);
	  else
	    __assert_unreachable<_Tp>();
	}
      else
	{
	  return __and(__x._M_data, __y._M_data);
	}
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> bit_or(const __storage<_Tp, _N> &__x,
                                                       const __storage<_Tp, _N> &__y)
    {
      if constexpr (std::is_same_v<_Tp, bool>)
	{
	  if constexpr (__have_avx512dq && _N <= 8)
	    return _kor_mask8(__x._M_data, __y._M_data);
	  else if constexpr (_N <= 16)
	    return _kor_mask16(__x._M_data, __y._M_data);
	  else if constexpr (__have_avx512bw && _N <= 32)
	    return _kor_mask32(__x._M_data, __y._M_data);
	  else if constexpr (__have_avx512bw && _N <= 64)
	    return _kor_mask64(__x._M_data, __y._M_data);
	  else
	    __assert_unreachable<_Tp>();
	}
      else
	{
	  return __or(__x._M_data, __y._M_data);
	}
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr __storage<_Tp, _N> bit_xor(const __storage<_Tp, _N> &__x,
                                                        const __storage<_Tp, _N> &__y)
    {
      if constexpr (std::is_same_v<_Tp, bool>)
	{
	  if constexpr (__have_avx512dq && _N <= 8)
	    return _kxor_mask8(__x._M_data, __y._M_data);
	  else if constexpr (_N <= 16)
	    return _kxor_mask16(__x._M_data, __y._M_data);
	  else if constexpr (__have_avx512bw && _N <= 32)
	    return _kxor_mask32(__x._M_data, __y._M_data);
	  else if constexpr (__have_avx512bw && _N <= 64)
	    return _kxor_mask64(__x._M_data, __y._M_data);
	  else
	    __assert_unreachable<_Tp>();
	}
      else
	{
	  return __xor(__x._M_data, __y._M_data);
	}
    }

    // smart_reference access {{{2
    template <class _Tp, size_t _N> static void set(__storage<_Tp, _N> &__k, int __i, bool __x) noexcept
    {
        if constexpr (std::is_same_v<_Tp, bool>) {
            __k.set(__i, __x);
        } else {
            using _IntT = __vector_type_t<__int_for_sizeof_t<_Tp>, _N>;
            auto tmp = reinterpret_cast<_IntT>(__k._M_data);
            tmp[__i] = -__x;
            __k._M_data = __auto_bitcast(tmp);
        }
    }
    // masked_assign{{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(__storage<_Tp, _N> __k, __storage<_Tp, _N> &__lhs,
                                           __id<__storage<_Tp, _N>> __rhs)
    {
        __lhs = __blend(__k._M_data, __lhs._M_data, __rhs._M_data);
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(__storage<_Tp, _N> __k, __storage<_Tp, _N> &__lhs, bool __rhs)
    {
        if (__builtin_constant_p(__rhs)) {
            if (__rhs == false) {
                __lhs = __andnot(__k._M_data, __lhs._M_data);
            } else {
                __lhs = __or(__k._M_data, __lhs._M_data);
            }
            return;
        }
        __lhs = __blend(__k, __lhs, __data(simd_mask<_Tp>(__rhs)));
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
 *  - memory layout of `simd<_Tp, _N>` is equivalent to `std::array<_Tp, _N>`
 *  - alignment of `simd<_Tp, _N>` is `_N * sizeof(_Tp)` if _N is __a power-of-2 value,
 *    otherwise `__next_power_of_2(_N * sizeof(_Tp))` (Note: if the alignment were to
 *    exceed the system/compiler maximum, it is bounded to that maximum)
 *  - simd_mask objects are passed like std::bitset<_N>
 *  - memory layout of `simd_mask<_Tp, _N>` is equivalent to `std::bitset<_N>`
 *  - alignment of `simd_mask<_Tp, _N>` is equal to the alignment of `std::bitset<_N>`
 */
// __autocvt_to_simd {{{
template <class _Tp, bool = std::is_arithmetic_v<std::decay_t<_Tp>>>
struct __autocvt_to_simd {
    _Tp _M_data;
    using _TT = std::decay_t<_Tp>;
    operator _TT() { return _M_data; }
    operator _TT &()
    {
        static_assert(std::is_lvalue_reference<_Tp>::value, "");
        static_assert(!std::is_const<_Tp>::value, "");
        return _M_data;
    }
    operator _TT *()
    {
        static_assert(std::is_lvalue_reference<_Tp>::value, "");
        static_assert(!std::is_const<_Tp>::value, "");
        return &_M_data;
    }

    constexpr inline __autocvt_to_simd(_Tp dd) : _M_data(dd) {}

    template <class _Abi> operator simd<typename _TT::value_type, _Abi>()
    {
        return {__private_init, _M_data};
    }

    template <class _Abi> operator simd<typename _TT::value_type, _Abi> &()
    {
        return *reinterpret_cast<simd<typename _TT::value_type, _Abi> *>(&_M_data);
    }

    template <class _Abi> operator simd<typename _TT::value_type, _Abi> *()
    {
        return reinterpret_cast<simd<typename _TT::value_type, _Abi> *>(&_M_data);
    }
};
template <class _Tp> __autocvt_to_simd(_Tp &&)->__autocvt_to_simd<_Tp>;

template <class _Tp> struct __autocvt_to_simd<_Tp, true> {
    using _TT = std::decay_t<_Tp>;
    _Tp _M_data;
    fixed_size_simd<_TT, 1> fd;

    constexpr inline __autocvt_to_simd(_Tp dd) : _M_data(dd), fd(_M_data) {}
    ~__autocvt_to_simd()
    {
        _M_data = __data(fd).first;
    }

    operator fixed_size_simd<_TT, 1>()
    {
        return fd;
    }
    operator fixed_size_simd<_TT, 1> &()
    {
        static_assert(std::is_lvalue_reference<_Tp>::value, "");
        static_assert(!std::is_const<_Tp>::value, "");
        return fd;
    }
    operator fixed_size_simd<_TT, 1> *()
    {
        static_assert(std::is_lvalue_reference<_Tp>::value, "");
        static_assert(!std::is_const<_Tp>::value, "");
        return &fd;
    }
};

// }}}
// __fixed_size_storage<_Tp, _N>{{{1
template <class _Tp, int _N, class _Tuple,
          class _Next = simd<_Tp, __all_native_abis::best_abi<_Tp, _N>>,
          int _Remain = _N - int(_Next::size())>
struct __fixed_size_storage_builder;

template <class _Tp, int _N>
struct __fixed_size_storage_builder_wrapper
    : public __fixed_size_storage_builder<_Tp, _N, __simd_tuple<_Tp>> {
};

template <class _Tp, int _N, class... _As, class _Next>
struct __fixed_size_storage_builder<_Tp, _N, __simd_tuple<_Tp, _As...>, _Next, 0> {
    using type = __simd_tuple<_Tp, _As..., typename _Next::abi_type>;
};

template <class _Tp, int _N, class... _As, class _Next, int _Remain>
struct __fixed_size_storage_builder<_Tp, _N, __simd_tuple<_Tp, _As...>, _Next, _Remain> {
    using type = typename __fixed_size_storage_builder<
        _Tp, _Remain, __simd_tuple<_Tp, _As..., typename _Next::abi_type>>::type;
};

// __n_abis_in_tuple {{{1
template <class _Tp> struct __seq_op;
template <size_t _I0, size_t... _Is> struct __seq_op<std::index_sequence<_I0, _Is...>> {
    using _FirstPlusOne = std::index_sequence<_I0 + 1, _Is...>;
    using _NotFirstPlusOne = std::index_sequence<_I0, (_Is + 1)...>;
    template <size_t _First, size_t _Add>
    using _Prepend = std::index_sequence<_First, _I0 + _Add, (_Is + _Add)...>;
};

template <class _Tp> struct __n_abis_in_tuple;
template <class _Tp> struct __n_abis_in_tuple<__simd_tuple<_Tp>> {
    using __counts = std::index_sequence<0>;
    using __begins = std::index_sequence<0>;
};
template <class _Tp, class _A> struct __n_abis_in_tuple<__simd_tuple<_Tp, _A>> {
    using __counts = std::index_sequence<1>;
    using __begins = std::index_sequence<0>;
};
template <class _Tp, class _A0, class... _As>
struct __n_abis_in_tuple<__simd_tuple<_Tp, _A0, _A0, _As...>> {
    using __counts = typename __seq_op<typename __n_abis_in_tuple<
        __simd_tuple<_Tp, _A0, _As...>>::__counts>::_FirstPlusOne;
    using __begins = typename __seq_op<typename __n_abis_in_tuple<
        __simd_tuple<_Tp, _A0, _As...>>::__begins>::_NotFirstPlusOne;
};
template <class _Tp, class _A0, class _A1, class... _As>
struct __n_abis_in_tuple<__simd_tuple<_Tp, _A0, _A1, _As...>> {
    using __counts = typename __seq_op<typename __n_abis_in_tuple<
        __simd_tuple<_Tp, _A1, _As...>>::__counts>::template _Prepend<1, 0>;
    using __begins = typename __seq_op<typename __n_abis_in_tuple<
        __simd_tuple<_Tp, _A1, _As...>>::__begins>::template _Prepend<0, 1>;
};

// __tree_reduction {{{1
template <size_t _Count, size_t _Begin> struct __tree_reduction {
    static_assert(_Count > 0,
                  "__tree_reduction requires at least one simd object to work with");
    template <class _Tp, class... _As, class _BinaryOperation>
    auto operator()(const __simd_tuple<_Tp, _As...> &__tup,
                    const _BinaryOperation &__binary_op) const noexcept
    {
        constexpr size_t __left = __next_power_of_2(_Count) / 2;
        constexpr size_t __right = _Count - __left;
        return __binary_op(__tree_reduction<__left, _Begin>()(__tup, __binary_op),
                         __tree_reduction<__right, _Begin + __left>()(__tup, __binary_op));
    }
};
template <size_t _Begin> struct __tree_reduction<1, _Begin> {
    template <class _Tp, class... _As, class _BinaryOperation>
    auto operator()(const __simd_tuple<_Tp, _As...> &__tup, const _BinaryOperation &) const
        noexcept
    {
        return __get_simd_at<_Begin>(__tup);
    }
};
template <size_t _Begin> struct __tree_reduction<2, _Begin> {
    template <class _Tp, class... _As, class _BinaryOperation>
    auto operator()(const __simd_tuple<_Tp, _As...> &__tup,
                    const _BinaryOperation &__binary_op) const noexcept
    {
        return __binary_op(__get_simd_at<_Begin>(__tup),
                         __get_simd_at<_Begin + 1>(__tup));
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
template <class _Tp, class _A0, class _A1, class _BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC simd<_Tp, _A1> __vec_to_scalar_reduction_first_pair(
    const simd<_Tp, _A0> __left, const simd<_Tp, _A1> __right, const _BinaryOperation &__binary_op,
    _SizeConstant<2>) noexcept
{
    const std::array<simd<_Tp, _A1>, 2> __splitted = split<simd<_Tp, _A1>>(__left);
    return __binary_op(__binary_op(__splitted[0], __right), __splitted[1]);
}

template <class _Tp, class _A0, class _A1, class _BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC simd<_Tp, _A1> __vec_to_scalar_reduction_first_pair(
    const simd<_Tp, _A0> __left, const simd<_Tp, _A1> __right, const _BinaryOperation &__binary_op,
    _SizeConstant<4>) noexcept
{
    constexpr auto _N0 = simd_size_v<_Tp, _A0> / 2;
    const auto __left2 = split<simd<_Tp, simd_abi::deduce_t<_Tp, _N0>>>(__left);
    const std::array<simd<_Tp, _A1>, 2> __splitted =
        split<simd<_Tp, _A1>>(__binary_op(__left2[0], __left2[1]));
    return __binary_op(__binary_op(__splitted[0], __right), __splitted[1]);
}

template <class _Tp, class _A0, class _A1, class _BinaryOperation, size_t _Factor>
_GLIBCXX_SIMD_INTRINSIC simd<_Tp, simd_abi::scalar> __vec_to_scalar_reduction_first_pair(
    const simd<_Tp, _A0> __left, const simd<_Tp, _A1> __right, const _BinaryOperation &__binary_op,
    _SizeConstant<_Factor>) noexcept
{
    return __binary_op(std::experimental::reduce(__left, __binary_op), std::experimental::reduce(__right, __binary_op));
}

template <class _Tp, class _A0, class _BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC _Tp __vec_to_scalar_reduction(const __simd_tuple<_Tp, _A0> &__tup,
                                       const _BinaryOperation &__binary_op) noexcept
{
    return std::experimental::reduce(simd<_Tp, _A0>(__private_init, __tup.first), __binary_op);
}

template <class _Tp, class _A0, class _A1, class... _As, class _BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC _Tp __vec_to_scalar_reduction(const __simd_tuple<_Tp, _A0, _A1, _As...> &__tup,
                                       const _BinaryOperation &__binary_op) noexcept
{
    return __vec_to_scalar_reduction(
        __simd_tuple_concat(
            __make_simd_tuple(
                __vec_to_scalar_reduction_first_pair<_Tp, _A0, _A1, _BinaryOperation>(
                    {__private_init, __tup.first}, {__private_init, __tup.second.first},
                    __binary_op,
                    _SizeConstant<simd_size_v<_Tp, _A0> / simd_size_v<_Tp, _A1>>())),
            __tup.second.second),
        __binary_op);
}

// __partial_bitset_to_member_type {{{1
template <class _V, size_t _N>
_GLIBCXX_SIMD_INTRINSIC auto __partial_bitset_to_member_type(std::bitset<_N> shifted_bits)
{
    static_assert(_V::size() <= _N, "");
    using _M = typename _V::mask_type;
    using _Tp = typename _V::value_type;
    constexpr _Tp *__type_tag = nullptr;
    return __get_impl_t<_M>::__from_bitset(
        std::bitset<_V::size()>(shifted_bits.to_ullong()), __type_tag);
}

// __fixed_size_simd_impl {{{1
template <int _N> struct __fixed_size_simd_impl {
    // member types {{{2
    using _Mask_member_type = std::bitset<_N>;
    template <class _Tp> using _Simd_member_type = __fixed_size_storage<_Tp, _N>;
    template <class _Tp>
    static constexpr std::size_t tuple_size = _Simd_member_type<_Tp>::tuple_size;
    template <class _Tp>
    static constexpr std::make_index_sequence<_Simd_member_type<_Tp>::tuple_size> index_seq = {};
    template <class _Tp> using simd = std::experimental::simd<_Tp, simd_abi::fixed_size<_N>>;
    template <class _Tp> using simd_mask = std::experimental::simd_mask<_Tp, simd_abi::fixed_size<_N>>;
    template <class _Tp> using __type_tag = _Tp *;

    // broadcast {{{2
    template <class _Tp> static constexpr inline _Simd_member_type<_Tp> __broadcast(_Tp __x) noexcept
    {
        return _Simd_member_type<_Tp>::generate(
            [&](auto meta) { return meta.__broadcast(__x); });
    }

    // generator {{{2
    template <class _F, class _Tp>
    inline static _Simd_member_type<_Tp> generator(_F &&__gen, __type_tag<_Tp>)
    {
        return _Simd_member_type<_Tp>::generate([&__gen](auto meta) {
            return meta.generator(
                [&](auto i_) {
                    return __gen(_SizeConstant<meta.offset + decltype(i_)::value>());
                },
                __type_tag<_Tp>());
        });
    }

    // load {{{2
    template <class _Tp, class _U, class _F>
    static inline _Simd_member_type<_Tp> load(const _U *mem, _F __f,
                                              __type_tag<_Tp>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        return _Simd_member_type<_Tp>::generate(
            [&](auto meta) { return meta.load(&mem[meta.offset], __f, __type_tag<_Tp>()); });
    }

    // masked load {{{2
    template <class _Tp, class... _As, class _U, class _F>
    static inline __simd_tuple<_Tp, _As...>
      masked_load(const __simd_tuple<_Tp, _As...>& __old,
		  const _Mask_member_type          __bits,
		  const _U*                        __mem,
		  _F __f) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
      auto __merge = __old;
      __for_each(__merge, [&](auto __meta, auto& __native) {
	__native = __meta.masked_load(__native, __meta.make_mask(__bits),
				  &__mem[__meta.offset], __f);
      });
      return __merge;
    }

    // store {{{2
    template <class _Tp, class _U, class _F>
    static inline void store(const _Simd_member_type<_Tp>& __v,
			     _U*                           __mem,
			     _F                            __f,
			     __type_tag<_Tp>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
      __for_each(__v, [&](auto __meta, auto __native) {
	__meta.store(__native, &__mem[__meta.offset], __f, __type_tag<_Tp>());
      });
    }

    // masked store {{{2
    template <class _Tp, class... _As, class _U, class _F>
    static inline void masked_store(const __simd_tuple<_Tp, _As...>& __v,
				    _U*                              __mem,
				    _F                               __f,
				    const _Mask_member_type          __bits)
      _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
      __for_each(__v, [&](auto __meta, auto __native) {
	__meta.masked_store(__native, &__mem[__meta.offset], __f,
			  __meta.make_mask(__bits));
      });
    }

    // negation {{{2
    template <class _Tp, class... _As>
    static inline _Mask_member_type
      negate(const __simd_tuple<_Tp, _As...>& __x) noexcept
    {
        _Mask_member_type __bits = 0;
        __for_each(__x, [&__bits](auto __meta, auto __native) {
            __bits |= __meta.mask_to_shifted_ullong(__meta.negate(__native));
        });
        return __bits;
    }

    // reductions {{{2
private:
    template <class _Tp, class... _As, class _BinaryOperation, size_t... _Counts,
              size_t... _Begins>
    static inline _Tp reduce(const __simd_tuple<_Tp, _As...> &__tup,
                           const _BinaryOperation &__binary_op,
                           std::index_sequence<_Counts...>, std::index_sequence<_Begins...>)
    {
        // 1. reduce all tuple elements with equal ABI to a single element in the output
        // tuple
        const auto reduced_vec =
            __make_simd_tuple(__tree_reduction<_Counts, _Begins>()(__tup, __binary_op)...);
        // 2. split and reduce until a scalar results
        return __vec_to_scalar_reduction(reduced_vec, __binary_op);
    }

public:
    template <class _Tp, class _BinaryOperation>
    static inline _Tp reduce(const simd<_Tp> &__x, const _BinaryOperation &__binary_op)
    {
        using __ranges = __n_abis_in_tuple<_Simd_member_type<_Tp>>;
        return __fixed_size_simd_impl::reduce(__x._M_data, __binary_op,
                                              typename __ranges::__counts(),
                                              typename __ranges::__begins());
    }

    // min, max, clamp {{{2
    template <typename _Tp, typename... _As>
    static inline constexpr __simd_tuple<_Tp, _As...>
      min(const __simd_tuple<_Tp, _As...>& __a,
	  const __simd_tuple<_Tp, _As...>& __b)
    {
      return __simd_tuple_apply(
	[](auto __impl, auto __aa, auto __bb) {
	  return __impl.min(__aa, __bb);
	},
	__a, __b);
    }

    template <typename _Tp, typename... _As>
    static inline constexpr __simd_tuple<_Tp, _As...>
      max(const __simd_tuple<_Tp, _As...>& __a,
	  const __simd_tuple<_Tp, _As...>& __b)
    {
      return __simd_tuple_apply(
	[](auto __impl, auto __aa, auto __bb) {
	  return __impl.max(__aa, __bb);
	},
	__a, __b);
    }

    // complement {{{2
    template <typename _Tp, typename... _As>
    static inline constexpr __simd_tuple<_Tp, _As...>
      complement(const __simd_tuple<_Tp, _As...>& __x) noexcept
    {
      return __simd_tuple_apply(
	[](auto __impl, auto __xx) { return __impl.complement(__xx); }, __x);
    }

    // unary minus {{{2
    template <typename _Tp, typename... _As>
    static inline constexpr __simd_tuple<_Tp, _As...>
      unary_minus(const __simd_tuple<_Tp, _As...>& __x) noexcept
    {
      return __simd_tuple_apply(
	[](auto __impl, auto __xx) { return __impl.unary_minus(__xx); }, __x);
    }

    // arithmetic operators {{{2

#define _GLIBCXX_SIMD_FIXED_OP(name_, op_)                                     \
  template <typename _Tp, typename... _As>                                     \
  static inline constexpr __simd_tuple<_Tp, _As...> name_(                     \
    const __simd_tuple<_Tp, _As...>& __x,                                      \
    const __simd_tuple<_Tp, _As...>& __y)                                      \
  {                                                                            \
    return __simd_tuple_apply(                                                 \
      [](auto __impl, auto __xx, auto __yy) {                                  \
	return __impl.name_(__xx, __yy);                                       \
      },                                                                       \
      __x, __y);                                                               \
  }

    _GLIBCXX_SIMD_FIXED_OP(plus, +)
    _GLIBCXX_SIMD_FIXED_OP(minus, -)
    _GLIBCXX_SIMD_FIXED_OP(multiplies, *)
    _GLIBCXX_SIMD_FIXED_OP(divides, /)
    _GLIBCXX_SIMD_FIXED_OP(modulus, %)
    _GLIBCXX_SIMD_FIXED_OP(bit_and, &)
    _GLIBCXX_SIMD_FIXED_OP(bit_or, |)
    _GLIBCXX_SIMD_FIXED_OP(bit_xor, ^)
    _GLIBCXX_SIMD_FIXED_OP(bit_shift_left, <<)
    _GLIBCXX_SIMD_FIXED_OP(bit_shift_right, >>)
#undef _GLIBCXX_SIMD_FIXED_OP

    template <typename _Tp, typename... _As>
    static inline constexpr __simd_tuple<_Tp, _As...>
      bit_shift_left(const __simd_tuple<_Tp, _As...>& __x, int __y)
    {
      return __simd_tuple_apply(
	[__y](auto __impl, auto __xx) {
	  return __impl.bit_shift_left(__xx, __y);
	},
	__x);
    }

    template <typename _Tp, typename... _As>
    static inline constexpr __simd_tuple<_Tp, _As...>
      bit_shift_right(const __simd_tuple<_Tp, _As...>& __x, int __y)
    {
      return __simd_tuple_apply(
	[__y](auto __impl, auto __xx) {
	  return __impl.bit_shift_right(__xx, __y);
	},
	__x);
    }

    // math {{{2
#define _GLIBCXX_SIMD_APPLY_ON_TUPLE_(name_)                                   \
  template <typename _Tp, typename... _As>                                     \
  static inline __simd_tuple<_Tp, _As...> __##name_(                           \
    const __simd_tuple<_Tp, _As...>& __x) noexcept                             \
  {                                                                            \
    return __simd_tuple_apply(                                                 \
      [](auto __impl, auto __xx) {                                             \
	using _V = typename decltype(__impl)::simd_type;                       \
	return __data(name_(_V(__private_init, __xx)));                        \
      },                                                                       \
      __x);                                                                    \
  }
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(sqrt)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(abs)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(trunc)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(floor)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(ceil)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(sin)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE_(cos)
#undef _GLIBCXX_SIMD_APPLY_ON_TUPLE_

    template <typename _Tp, typename... _As>
    static inline __simd_tuple<_Tp, _As...> __frexp(const __simd_tuple<_Tp, _As...> &__x,
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

    template <typename _Tp, typename... _As>
    static inline __fixed_size_storage<int, _N>
      __fpclassify(const __simd_tuple<_Tp, _As...>& __x) noexcept
    {
      return __optimize_simd_tuple(__x.template apply_r<int>(
	[](auto __impl, auto __xx) { return __impl.__fpclassify(__xx); }));
    }

#define _GLIBCXX_SIMD_TEST_ON_TUPLE_(name_)                                    \
  template <typename _Tp, typename... _As>                                     \
  static inline _Mask_member_type __##name_(                                   \
    const __simd_tuple<_Tp, _As...>& __x) noexcept                             \
  {                                                                            \
    return test([](auto __impl, auto __xx) { return __impl.__##name_(__xx); }, \
		__x);                                                          \
  }
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isinf)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isfinite)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isnan)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isnormal)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(signbit)
#undef _GLIBCXX_SIMD_TEST_ON_TUPLE_

    // __increment & __decrement{{{2
    template <typename... _Ts>
    static inline constexpr void __increment(__simd_tuple<_Ts...>& __x)
    {
      __for_each(__x,
		 [](auto meta, auto& native) { meta.__increment(native); });
    }

    template <typename... _Ts>
    static inline constexpr void __decrement(__simd_tuple<_Ts...>& __x)
    {
      __for_each(__x,
		 [](auto meta, auto& native) { meta.__decrement(native); });
    }

    // compares {{{2
#define _GLIBCXX_SIMD_CMP_OPERATIONS(cmp_)                                               \
    template <typename _Tp, typename... _As>                                                     \
    static inline _Mask_member_type cmp_(const __simd_tuple<_Tp, _As...>& __x,            \
                                          const __simd_tuple<_Tp, _As...>& __y)            \
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
    template <typename _Tp, typename... _As, typename _U>
    _GLIBCXX_SIMD_INTRINSIC static void set(__simd_tuple<_Tp, _As...> &__v, int __i, _U &&__x) noexcept
    {
        __v.set(__i, std::forward<_U>(__x));
    }

    // masked_assign {{{2
    template <typename _Tp, typename... _As>
    _GLIBCXX_SIMD_INTRINSIC static void
      masked_assign(const _Mask_member_type                __bits,
		    __simd_tuple<_Tp, _As...>&             __lhs,
		    const __id<__simd_tuple<_Tp, _As...>>& __rhs)
    {
      __for_each(__lhs, __rhs,
		 [&](auto __meta, auto& __native_lhs, auto __native_rhs) {
		   __meta.masked_assign(__meta.make_mask(__bits), __native_lhs,
					__native_rhs);
		 });
    }

    // Optimization for the case where the RHS is a scalar. No need to broadcast the
    // scalar to a simd first.
    template <typename _Tp, typename... _As>
    _GLIBCXX_SIMD_INTRINSIC static void
      masked_assign(const _Mask_member_type    __bits,
		    __simd_tuple<_Tp, _As...>& __lhs,
		    const __id<_Tp>            __rhs)
    {
      __for_each(__lhs, [&](auto __meta, auto& __native_lhs) {
	__meta.masked_assign(__meta.make_mask(__bits), __native_lhs, __rhs);
      });
    }

    // __masked_cassign {{{2
    template <template <typename> class _Op, typename _Tp, typename... _As>
    static inline void __masked_cassign(const _Mask_member_type          __bits,
					__simd_tuple<_Tp, _As...>&       __lhs,
					const __simd_tuple<_Tp, _As...>& __rhs)
    {
      __for_each(__lhs, __rhs,
		 [&](auto __meta, auto& __native_lhs, auto __native_rhs) {
		   __meta.template __masked_cassign<_Op>(
		     __meta.make_mask(__bits), __native_lhs, __native_rhs);
		 });
    }

    // Optimization for the case where the RHS is a scalar. No need to broadcast
    // the scalar to a simd first.
    template <template <typename> class _Op, typename _Tp, typename... _As>
    static inline void __masked_cassign(const _Mask_member_type    __bits,
					__simd_tuple<_Tp, _As...>& __lhs,
					const _Tp&                 __rhs)
    {
      __for_each(__lhs, [&](auto __meta, auto& __native_lhs) {
	__meta.template __masked_cassign<_Op>(__meta.make_mask(__bits),
					      __native_lhs, __rhs);
      });
    }

    // masked_unary {{{2
    template <template <typename> class _Op, typename _Tp, typename... _As>
    static inline __simd_tuple<_Tp, _As...>
      masked_unary(const _Mask_member_type         __bits,
		   const __simd_tuple<_Tp, _As...> __v) // TODO: const-ref __v?
    {
      return __v.apply_wrapped([&__bits](auto __meta, auto __native) {
	return __meta.template masked_unary<_Op>(__meta.make_mask(__bits),
						 __native);
      });
    }

    // }}}2
};

// __fixed_size_mask_impl {{{1
template <int _N> struct __fixed_size_mask_impl {
    static_assert(sizeof(_ULLong) * CHAR_BIT >= _N,
                  "The fixed_size implementation relies on one "
                  "_ULLong being able to store all boolean "
                  "elements.");  // required in load & store

    // member types {{{2
    static constexpr std::make_index_sequence<_N> index_seq = {};
    using _Mask_member_type = std::bitset<_N>;
    template <typename _Tp> using simd_mask = std::experimental::simd_mask<_Tp, simd_abi::fixed_size<_N>>;
    template <typename _Tp> using __type_tag = _Tp *;

    // __from_bitset {{{2
    template <typename _Tp>
    _GLIBCXX_SIMD_INTRINSIC static _Mask_member_type __from_bitset(const _Mask_member_type &bs,
                                                     __type_tag<_Tp>) noexcept
    {
        return bs;
    }

    // load {{{2
    template <typename _F> static inline _Mask_member_type load(const bool *mem, _F __f) noexcept
    {
        // TODO: _UChar is not necessarily the best type to use here. For smaller _N ushort,
        // _UInt, _ULLong, float, and double can be more efficient.
        _ULLong __r = 0;
        using _Vs = __fixed_size_storage<_UChar, _N>;
        __for_each(_Vs{}, [&](auto meta, auto) {
            __r |= meta.mask_to_shifted_ullong(
                meta.simd_mask.load(&mem[meta.offset], __f, _SizeConstant<meta.size()>()));
        });
        return __r;
    }

    // masked load {{{2
    template <typename _F>
    static inline _Mask_member_type masked_load(_Mask_member_type merge,
                                               _Mask_member_type mask, const bool *mem,
                                               _F) noexcept
    {
        __bit_iteration(mask.to_ullong(), [&](auto __i) { merge[__i] = mem[__i]; });
        return merge;
    }

    // store {{{2
    template <typename _F>
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
            const _ULLong bool8 =
                _pdep_u64(bs.to_ullong() >> offset, 0x0101010101010101ULL);
            std::memcpy(&mem[offset], &bool8, 8);
        });
        if (_N % 8 > 0) {
            constexpr size_t offset = (_N / 8) * 8;
            const _ULLong bool8 =
                _pdep_u64(bs.to_ullong() >> offset, 0x0101010101010101ULL);
            std::memcpy(&mem[offset], &bool8, _N % 8);
        }
#else   // __x86_64__
        __unused(__f);
        __execute_n_times<_N / 4>([&](auto __i) {
            constexpr size_t offset = __i * 4;
            const _ULLong bool4 =
                _pdep_u32(bs.to_ullong() >> offset, 0x01010101U);
            std::memcpy(&mem[offset], &bool4, 4);
        });
        if (_N % 4 > 0) {
            constexpr size_t offset = (_N / 4) * 4;
            const _ULLong bool4 =
                _pdep_u32(bs.to_ullong() >> offset, 0x01010101U);
            std::memcpy(&mem[offset], &bool4, _N % 4);
        }
#endif  // __x86_64__
#elif  _GLIBCXX_SIMD_HAVE_SSE2   // !AVX512BW && !BMI2
        using _V = simd<_UChar, simd_abi::__sse>;
        _ULLong __bits = bs.to_ullong();
        __execute_n_times<(_N + 15) / 16>([&](auto __i) {
            constexpr size_t offset = __i * 16;
            constexpr size_t remaining = _N - offset;
            if constexpr (remaining == 1) {
                mem[offset] = static_cast<bool>(__bits >> offset);
            } else if constexpr (remaining <= 4) {
                const _UInt bool4 = ((__bits >> offset) * 0x00204081U) & 0x01010101U;
                std::memcpy(&mem[offset], &bool4, remaining);
            } else if constexpr (remaining <= 7) {
                const _ULLong bool8 =
                    ((__bits >> offset) * 0x40810204081ULL) & 0x0101010101010101ULL;
                std::memcpy(&mem[offset], &bool8, remaining);
            } else if constexpr (__have_sse2) {
                auto tmp = _mm_cvtsi32_si128(__bits >> offset);
                tmp = _mm_unpacklo_epi8(tmp, tmp);
                tmp = _mm_unpacklo_epi16(tmp, tmp);
                tmp = _mm_unpacklo_epi32(tmp, tmp);
                _V tmp2(tmp);
                tmp2 &= _V([](auto __j) {
                    return static_cast<_UChar>(1 << (__j % CHAR_BIT));
                });  // mask bit index
                const __m128i bool16 = __intrin_bitcast<__m128i>(
                    __vector_bitcast<_UChar>(__data(tmp2 == 0)) +
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
        // TODO: _UChar is not necessarily the best type to use here. For smaller _N ushort,
        // _UInt, _ULLong, float, and double can be more efficient.
        using _Vs = __fixed_size_storage<_UChar, _N>;
        __for_each(_Vs{}, [&](auto meta, auto) {
            meta.store(meta.make_mask(bs), &mem[meta.offset], __f);
        });
//#else
        //__execute_n_times<_N>([&](auto __i) { mem[__i] = bs[__i]; });
#endif  // _GLIBCXX_SIMD_HAVE_BMI2
    }

    // masked store {{{2
    template <typename _F>
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
                                           _Mask_member_type &__lhs,
                                           const _Mask_member_type __rhs)
    {
        __lhs = (__lhs & ~__k) | (__rhs & __k);
    }

    // Optimization for the case where the RHS is a scalar.
    _GLIBCXX_SIMD_INTRINSIC static void masked_assign(const _Mask_member_type __k,
                                           _Mask_member_type &__lhs, const bool __rhs)
    {
        if (__rhs) {
            __lhs |= __k;
        } else {
            __lhs &= ~__k;
        }
    }

    // }}}2
};
// }}}1

// __simd_converter scalar -> scalar {{{
template <typename _Tp> struct __simd_converter<_Tp, simd_abi::scalar, _Tp, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC _Tp operator()(_Tp __a) { return __a; }
};
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::scalar, _To, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC _To operator()(_From __a)
    {
        return static_cast<_To>(__a);
    }
};

// }}}
// __simd_converter __sse -> scalar {{{1
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::__sse, _To, simd_abi::scalar> {
    using _Arg = __sse_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC std::array<_To, _Arg::_S_width> __all(_Arg __a)
    {
        return __impl(std::make_index_sequence<_Arg::_S_width>(), __a);
    }

    template <size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC std::array<_To, _Arg::_S_width> __impl(std::index_sequence<_Indexes...>, _Arg __a)
    {
        return {static_cast<_To>(__a[_Indexes])...};
    }
};

// }}}1
// __simd_converter scalar -> __sse {{{1
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::scalar, _To, simd_abi::__sse> {
    using _R = __sse_simd_member_type<_To>;
    template <typename... _More> _GLIBCXX_SIMD_INTRINSIC constexpr _R operator()(_From __a, _More... __b)
    {
        static_assert(sizeof...(_More) + 1 == _R::_S_width);
        static_assert(std::conjunction_v<std::is_same<_From, _More>...>);
        return __vector_type16_t<_To>{static_cast<_To>(__a), static_cast<_To>(__b)...};
    }
};

// }}}1
// __simd_converter __sse -> __sse {{{1
template <typename _Tp> struct __simd_converter<_Tp, simd_abi::__sse, _Tp, simd_abi::__sse> {
    using _Arg = __sse_simd_member_type<_Tp>;
    _GLIBCXX_SIMD_INTRINSIC const _Arg &operator()(const _Arg &__x) { return __x; }
};

template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::__sse, _To, simd_abi::__sse> {
    using _Arg = __sse_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(_Arg __a)
    {
        return __convert_all<__vector_type16_t<_To>>(__a);
    }

    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(_Arg __a)
    {
        return __convert<__vector_type16_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(_Arg __a, _Arg __b)
    {
        static_assert(sizeof(_From) >= 2 * sizeof(_To), "");
        return __convert<__sse_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(_Arg __a, _Arg __b, _Arg __c, _Arg __d)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__sse_simd_member_type<_To>>(__a, __b, __c, __d);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(_Arg __a, _Arg __b, _Arg __c, _Arg __d, _Arg __e,
                                                     _Arg __f, _Arg __g, _Arg __h)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__sse_simd_member_type<_To>>(__a, __b, __c, __d, __e, __f, __g, __h);
    }
};

// }}}1
// __simd_converter __avx -> scalar {{{1
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::__avx, _To, simd_abi::scalar> {
    using _Arg = __avx_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC std::array<_To, _Arg::_S_width> __all(_Arg __a)
    {
        return __impl(std::make_index_sequence<_Arg::_S_width>(), __a);
    }

    template <size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC std::array<_To, _Arg::_S_width> __impl(std::index_sequence<_Indexes...>, _Arg __a)
    {
        return {static_cast<_To>(__a[_Indexes])...};
    }
};

// }}}1
// __simd_converter scalar -> __avx {{{1
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::scalar, _To, simd_abi::__avx> {
    using _R = __avx_simd_member_type<_To>;
    template <typename... _More> _GLIBCXX_SIMD_INTRINSIC constexpr _R operator()(_From __a, _More... __b)
    {
        static_assert(sizeof...(_More) + 1 == _R::_S_width);
        static_assert(std::conjunction_v<std::is_same<_From, _More>...>);
        return __vector_type32_t<_To>{static_cast<_To>(__a), static_cast<_To>(__b)...};
    }
};

// }}}1
// __simd_converter __sse -> __avx {{{1
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::__sse, _To, simd_abi::__avx> {
    using _Arg = __sse_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(_Arg __a) { return __convert_all<__vector_type32_t<_To>>(__a); }

    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(_Arg __a)
    {
        return __convert<__vector_type32_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(_Arg __a, _Arg __b)
    {
        static_assert(sizeof(_From) >= 1 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(_Arg __a, _Arg __b, _Arg __c, _Arg __d)
    {
        static_assert(sizeof(_From) >= 2 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b, __c, __d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(_Arg __x0, _Arg __x1, _Arg __x2, _Arg __x3,
                                                     _Arg __x4, _Arg __x5, _Arg __x6, _Arg __x7)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__x0, __x1, __x2, __x3, __x4, __x5, __x6, __x7);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(_Arg __x0, _Arg __x1, _Arg __x2, _Arg __x3,
                                                     _Arg __x4, _Arg __x5, _Arg __x6, _Arg __x7,
                                                     _Arg __x8, _Arg __x9, _Arg x10, _Arg x11,
                                                     _Arg x12, _Arg x13, _Arg x14, _Arg x15)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__x0, __x1, __x2, __x3, __x4, __x5, __x6, __x7, __x8,
                                                      __x9, x10, x11, x12, x13, x14, x15);
    }
};

// }}}1
// __simd_converter __avx -> __sse {{{1
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::__avx, _To, simd_abi::__sse> {
    using _Arg = __avx_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(_Arg __a) { return __convert_all<__vector_type16_t<_To>>(__a); }

    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(_Arg __a)
    {
        return __convert<__vector_type16_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(_Arg __a, _Arg __b)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__sse_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(_Arg __a, _Arg __b, _Arg __c, _Arg __d)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__sse_simd_member_type<_To>>(__a, __b, __c, __d);
    }
};

// }}}1
// __simd_converter __avx -> __avx {{{1
template <typename _Tp> struct __simd_converter<_Tp, simd_abi::__avx, _Tp, simd_abi::__avx> {
    using _Arg = __avx_simd_member_type<_Tp>;
    _GLIBCXX_SIMD_INTRINSIC const _Arg &operator()(const _Arg &__x) { return __x; }
};

template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::__avx, _To, simd_abi::__avx> {
    using _Arg = __avx_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(_Arg __a) { return __convert_all<__vector_type32_t<_To>>(__a); }

    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(_Arg __a)
    {
        return __convert<__vector_type32_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(_Arg __a, _Arg __b)
    {
        static_assert(sizeof(_From) >= 2 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(_Arg __a, _Arg __b, _Arg __c, _Arg __d)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b, __c, __d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(_Arg __a, _Arg __b, _Arg __c, _Arg __d, _Arg __e,
                                                     _Arg __f, _Arg __g, _Arg __h)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b, __c, __d, __e, __f, __g, __h);
    }
};

// }}}1
// __simd_converter __avx512 -> scalar {{{1
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::__avx512, _To, simd_abi::scalar> {
    using _Arg = __avx512_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC std::array<_To, _Arg::_S_width> __all(_Arg __a)
    {
        return __impl(std::make_index_sequence<_Arg::_S_width>(), __a);
    }

    template <size_t... _Indexes>
    _GLIBCXX_SIMD_INTRINSIC std::array<_To, _Arg::_S_width> __impl(std::index_sequence<_Indexes...>, _Arg __a)
    {
        return {static_cast<_To>(__a[_Indexes])...};
    }
};

// }}}1
// __simd_converter scalar -> __avx512 {{{1
template <typename _From, typename _To>
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
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::__sse, _To, simd_abi::__avx512> {
    using _Arg = __sse_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(_Arg __a)
    {
        return __convert_all<__vector_type64_t<_To>>(__a);
    }

    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __a)
    {
        return __convert<__vector_type64_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __a, _Arg __b)
    {
        static_assert(2 * sizeof(_From) >= sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __a, _Arg __b, _Arg __c, _Arg __d)
    {
        static_assert(sizeof(_From) >= 1 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b, __c, __d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __x0, _Arg __x1, _Arg __x2, _Arg __x3,
                                                     _Arg __x4, _Arg __x5, _Arg __x6, _Arg __x7)
    {
        static_assert(sizeof(_From) >= 2 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__x0, __x1, __x2, __x3, __x4, __x5, __x6,
                                                           __x7);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __x0, _Arg __x1, _Arg __x2, _Arg __x3,
                                                     _Arg __x4, _Arg __x5, _Arg __x6, _Arg __x7,
                                                     _Arg __x8, _Arg __x9, _Arg x10, _Arg x11,
                                                     _Arg x12, _Arg x13, _Arg x14, _Arg x15)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(
            __x0, __x1, __x2, __x3, __x4, __x5, __x6, __x7, __x8, __x9, x10, x11, x12, x13, x14, x15);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(
        _Arg __x0, _Arg __x1, _Arg __x2, _Arg __x3, _Arg __x4, _Arg __x5, _Arg __x6, _Arg __x7, _Arg __x8, _Arg __x9,
        _Arg x10, _Arg x11, _Arg x12, _Arg x13, _Arg x14, _Arg x15, _Arg x16, _Arg x17, _Arg x18,
        _Arg x19, _Arg x20, _Arg x21, _Arg x22, _Arg x23, _Arg x24, _Arg x25, _Arg x26, _Arg x27,
        _Arg x28, _Arg x29, _Arg x30, _Arg x31)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(
            __x0, __x1, __x2, __x3, __x4, __x5, __x6, __x7, __x8, __x9, x10, x11, x12, x13, x14, x15, x16,
            x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31);
    }
};

// }}}1
// __simd_converter __avx512 -> __sse {{{1
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::__avx512, _To, simd_abi::__sse> {
    using _Arg = __avx512_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(_Arg __a)
    {
        return __convert_all<__vector_type16_t<_To>>(__a);
    }

    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(_Arg __a)
    {
        return __convert<__vector_type16_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __sse_simd_member_type<_To> operator()(_Arg __a, _Arg __b)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__sse_simd_member_type<_To>>(__a, __b);
    }
};

// }}}1
// __simd_converter __avx -> __avx512 {{{1
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::__avx, _To, simd_abi::__avx512> {
    using _Arg = __avx_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(_Arg __a)
    {
        return __convert_all<__vector_type64_t<_To>>(__a);
    }

    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __a)
    {
        return __convert<__vector_type64_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __a, _Arg __b)
    {
        static_assert(sizeof(_From) >= 1 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __a, _Arg __b, _Arg __c, _Arg __d)
    {
        static_assert(sizeof(_From) >= 2 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b, __c, __d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __x0, _Arg __x1, _Arg __x2, _Arg __x3,
                                                     _Arg __x4, _Arg __x5, _Arg __x6, _Arg __x7)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__x0, __x1, __x2, __x3, __x4, __x5, __x6,
                                                           __x7);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __x0, _Arg __x1, _Arg __x2, _Arg __x3,
                                                     _Arg __x4, _Arg __x5, _Arg __x6, _Arg __x7,
                                                     _Arg __x8, _Arg __x9, _Arg x10, _Arg x11,
                                                     _Arg x12, _Arg x13, _Arg x14, _Arg x15)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(
            __x0, __x1, __x2, __x3, __x4, __x5, __x6, __x7, __x8, __x9, x10, x11, x12, x13, x14, x15);
    }
};

// }}}1
// __simd_converter __avx512 -> __avx {{{1
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::__avx512, _To, simd_abi::__avx> {
    using _Arg = __avx512_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(_Arg __a)
    {
        return __convert_all<__vector_type32_t<_To>>(__a);
    }

    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(_Arg __a)
    {
        return __convert<__vector_type32_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(_Arg __a, _Arg __b)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx_simd_member_type<_To> operator()(_Arg __a, _Arg __b, _Arg __c, _Arg __d)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__avx_simd_member_type<_To>>(__a, __b, __c, __d);
    }
};

// }}}1
// __simd_converter __avx512 -> __avx512 {{{1
template <typename _Tp> struct __simd_converter<_Tp, simd_abi::__avx512, _Tp, simd_abi::__avx512> {
    using _Arg = __avx512_simd_member_type<_Tp>;
    _GLIBCXX_SIMD_INTRINSIC const _Arg &operator()(const _Arg &__x) { return __x; }
};

template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::__avx512, _To, simd_abi::__avx512> {
    using _Arg = __avx512_simd_member_type<_From>;

    _GLIBCXX_SIMD_INTRINSIC auto __all(_Arg __a)
    {
        return __convert_all<__vector_type64_t<_To>>(__a);
    }

    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __a)
    {
        return __convert<__vector_type64_t<_To>>(__a);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __a, _Arg __b)
    {
        static_assert(sizeof(_From) >= 2 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __a, _Arg __b, _Arg __c, _Arg __d)
    {
        static_assert(sizeof(_From) >= 4 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b, __c, __d);
    }
    _GLIBCXX_SIMD_INTRINSIC __avx512_simd_member_type<_To> operator()(_Arg __a, _Arg __b, _Arg __c, _Arg __d, _Arg __e,
                                                     _Arg __f, _Arg __g, _Arg __h)
    {
        static_assert(sizeof(_From) >= 8 * sizeof(_To), "");
        return __convert<__avx512_simd_member_type<_To>>(__a, __b, __c, __d, __e, __f, __g, __h);
    }
};

// }}}1
// __simd_converter scalar -> fixed_size<1> {{{1
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::scalar, _To, simd_abi::fixed_size<1>> {
    __simd_tuple<_To, simd_abi::scalar> operator()(_From __x) { return {static_cast<_To>(__x)}; }
};

// __simd_converter fixed_size<1> -> scalar {{{1
template <typename _From, typename _To>
struct __simd_converter<_From, simd_abi::fixed_size<1>, _To, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC _To operator()(__simd_tuple<_From, simd_abi::scalar> __x)
    {
        return {static_cast<_To>(__x.first)};
    }
};

// __simd_converter fixed_size<_N> -> fixed_size<_N> {{{1
template <typename _Tp, int _N>
struct __simd_converter<_Tp, simd_abi::fixed_size<_N>, _Tp, simd_abi::fixed_size<_N>> {
    using arg = __fixed_size_storage<_Tp, _N>;
    _GLIBCXX_SIMD_INTRINSIC const arg &operator()(const arg &__x) { return __x; }
};

template <size_t _ChunkSize, typename _Tp> struct determine_required_input_chunks;

template <typename _Tp, typename... _Abis>
struct determine_required_input_chunks<0, __simd_tuple<_Tp, _Abis...>>
    : public std::integral_constant<size_t, 0> {
};

template <size_t _ChunkSize, typename _Tp, typename _Abi0, typename... _Abis>
struct determine_required_input_chunks<_ChunkSize, __simd_tuple<_Tp, _Abi0, _Abis...>>
    : public std::integral_constant<
          size_t, determine_required_input_chunks<_ChunkSize - simd_size_v<_Tp, _Abi0>,
                                                  __simd_tuple<_Tp, _Abis...>>::value> {
};

template <typename _From, typename _To> struct __fixed_size_converter {
    struct _OneToMultipleChunks {
    };
    template <int _N> struct _MultipleToOneChunk {
    };
    struct _EqualChunks {
    };
    template <typename _FromAbi, typename _ToAbi, size_t _ToSize = simd_size_v<_To, _ToAbi>,
              size_t _FromSize = simd_size_v<_From, _FromAbi>>
    using _ChunkRelation = std::conditional_t<
        (_ToSize < _FromSize), _OneToMultipleChunks,
        std::conditional_t<(_ToSize == _FromSize), _EqualChunks,
                           _MultipleToOneChunk<int(_ToSize / _FromSize)>>>;

    template <typename... _Abis>
    using __return_type = __fixed_size_storage<_To, __simd_tuple<_From, _Abis...>::size()>;


protected:
    // _OneToMultipleChunks {{{2
    template <typename _A0>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0> __impl(_OneToMultipleChunks, const __simd_tuple<_From, _A0> &__x)
    {
        using _R = __return_type<_A0>;
        __simd_converter<_From, _A0, _To, typename _R::_First_abi> __native_cvt;
        auto &&multiple_return_chunks = __native_cvt.__all(__x.first);
        return __to_simd_tuple<_To, typename _R::_First_abi>(multiple_return_chunks);
    }

    template <typename... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_Abis...> __impl(_OneToMultipleChunks,
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

    // _MultipleToOneChunk {{{2
    template <int _N, typename _A0, typename... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0, _Abis...> __impl(_MultipleToOneChunk<_N>,
                                               const __simd_tuple<_From, _A0, _Abis...> &__x)
    {
        return impl_mto(std::integral_constant<bool, sizeof...(_Abis) + 1 == _N>(),
                        std::make_index_sequence<_N>(), __x);
    }

    template <size_t... _Indexes, typename _A0, typename... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0, _Abis...> impl_mto(true_type,
                                                   std::index_sequence<_Indexes...>,
                                                   const __simd_tuple<_From, _A0, _Abis...> &__x)
    {
        using _R = __return_type<_A0, _Abis...>;
        __simd_converter<_From, _A0, _To, typename _R::_First_abi> __native_cvt;
        return {__native_cvt(__get_tuple_at<_Indexes>(__x)...)};
    }

    template <size_t... _Indexes, typename _A0, typename... _Abis>
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
                __simd_tuple_pop_front(_SizeConstant<sizeof...(_Indexes)>(), __x))};
    }

    // _EqualChunks {{{2
    template <typename _A0>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0> __impl(_EqualChunks, const __simd_tuple<_From, _A0> &__x)
    {
        __simd_converter<_From, _A0, _To, typename __return_type<_A0>::_First_abi> __native_cvt;
        return {__native_cvt(__x.first)};
    }

    template <typename _A0, typename _A1, typename... _Abis>
    _GLIBCXX_SIMD_INTRINSIC __return_type<_A0, _A1, _Abis...> __impl(
        _EqualChunks, const __simd_tuple<_From, _A0, _A1, _Abis...> &__x)
    {
        using _R = __return_type<_A0, _A1, _Abis...>;
        using _Rem = typename _R::_Second_type;
        __simd_converter<_From, _A0, _To, typename _R::_First_abi> __native_cvt;
        return {__native_cvt(__x.first),
                __impl(_ChunkRelation<_A1, typename _Rem::_First_abi>(), __x.second)};
    }

    //}}}2
};

template <typename _From, typename _To, int _N>
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
template <typename _From, typename _A, typename _To, int _N>
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
template <typename _From, int _N, typename _To, typename _A>
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
