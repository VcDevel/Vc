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

#ifndef VC_DETAIL_DATAPAR_TUPLE_H_
#define VC_DETAIL_DATAPAR_TUPLE_H_

#include "detail.h"
#include "concepts.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// subscript_read/_write {{{1
template <class T> T subscript_read(arithmetic<T> x, size_t) noexcept { return x; }
template <class T>
void subscript_write(arithmetic<T> &x, size_t, detail::id<T> y) noexcept
{
    return x = y;
}

template <class T> typename T::value_type subscript_read(const T &x, size_t i) noexcept
{
    return x[i];
}
template <class T> void subscript_write(T &x, size_t i, typename T::value_type y) noexcept
{
    return x.set(i, y);
}

// datapar_tuple {{{1
// why not std::tuple?
// 1. std::tuple gives no guarantee about the storage order, but I require storage
//    equivalent to std::array<T, N>
// 2. much less code to instantiate: I require a very small subset of std::tuple
//    functionality
// 3. direct access to the element type (first template argument)
// 4. enforces equal element type, only different Abi types are allowed

template <class T, class... Abis> struct datapar_tuple;

// tuple_element_meta {{{1
template <class T, class Abi, size_t Offset>
struct tuple_element_meta : public detail::traits<T, Abi>::datapar_impl_type {
    using traits = detail::traits<T, Abi>;
    using maskimpl = typename traits::mask_impl_type;
    using member_type = typename traits::datapar_member_type;
    using datapar_type = datapar<T, Abi>;
    static constexpr size_t offset = Offset;
    static constexpr size_t size() { return datapar_size<T, Abi>::value; }
    static constexpr size_constant<size()> size_tag = {};
    static constexpr maskimpl mask = {};

    template <size_t N>
    static Vc_INTRINSIC typename traits::mask_member_type make_mask(std::bitset<N> bits)
    {
        constexpr T *type_tag = nullptr;
        return maskimpl::from_bitset(std::bitset<size()>((bits >> Offset).to_ullong()),
                                     type_tag);
    }

    static Vc_INTRINSIC ullong mask_to_shifted_ullong(typename traits::mask_member_type k)
    {
        return maskimpl::to_bitset(k).to_ullong() << Offset;
    }
};

template <size_t Offset, class T, class Abi, class... As>
tuple_element_meta<T, Abi, Offset> make_meta(const datapar_tuple<T, Abi, As...> &)
{
    return {};
}

// datapar_tuple specializations {{{1
// empty {{{2
template <class T> struct datapar_tuple<T> {
    static constexpr size_t tuple_size = 0;
};

// 1 member {{{2
template <class T, class Abi0> struct datapar_tuple<T, Abi0> {
    using first_type = typename detail::traits<T, Abi0>::datapar_member_type;
    static constexpr size_t tuple_size = 1;
    first_type first;

    template <size_t Offset = 0, class F>
    static Vc_INTRINSIC datapar_tuple generate(F &&gen)
    {
        return {gen(tuple_element_meta<T, Abi0, Offset>())};
    }

    template <size_t Offset = 0, class F, class... More>
    Vc_INTRINSIC datapar_tuple apply_wrapped(F &&fun, const More &... more) const
    {
        return {fun(make_meta<Offset>(*this), first, more.first...)};
    }

    template <class F, class... More>
    friend Vc_INTRINSIC datapar_tuple apply(F &&fun, const datapar_tuple &x,
                                            const More &... more)
    {
        return {fun(tuple_element_meta<T, Abi0, 0>(), x.first, more.first...)};
    }

    T operator[](size_t i) const noexcept { return subscript_read(first, i); }
    void set(size_t i, T val) noexcept { subscript_write(first, i, val); }
};

// 2 or more {{{2
template <class T, class Abi0, class... Abis> struct datapar_tuple<T, Abi0, Abis...> {
    using first_type = typename detail::traits<T, Abi0>::datapar_member_type;
    using second_type = datapar_tuple<T, Abis...>;
    static constexpr size_t tuple_size = sizeof...(Abis) + 1;
    first_type first;
    second_type second;

    template <size_t Offset = 0, class F>
    static Vc_INTRINSIC datapar_tuple generate(F &&gen)
    {
        return {gen(tuple_element_meta<T, Abi0, Offset>()),
                second_type::template generate<Offset + datapar_size<T, Abi0>::value>(
                    std::forward<F>(gen))};
    }

    template <size_t Offset = 0, class F, class... More>
    Vc_INTRINSIC datapar_tuple apply_wrapped(F &&fun, const More &... more) const
    {
        return {fun(make_meta<Offset>(*this), first, more.first...),
                second.template apply_wrapped<Offset + datapar_size_v<T, Abi0>>(
                    std::forward<F>(fun), more.second...)};
    }

    template <class F, class... More>
    friend Vc_INTRINSIC datapar_tuple apply(F &&fun, const datapar_tuple &x,
                                            const More &... more)
    {
        return {fun(tuple_element_meta<T, Abi0, 0>(), x.first, more.first...),
                apply(std::forward<F>(fun), x.second, more.second...)};
    }

    T operator[](size_t i) const noexcept
    {
#ifdef __GNUC__
        return reinterpret_cast<const may_alias<T> *>(this)[i];
#else
        return i < first_type::size() ? subscript_read(first, i)
                                      : second[i - first_type::size()];
#endif
    }
    void set(size_t i, T val) noexcept
    {
#ifdef __GNUC__
        reinterpret_cast<may_alias<T> *>(this)[i] = val;
#else
        if (i < first_type::size()) {
            subscript_write(first, i, val);
        } else {
            second.set(i - first_type::size(), val);
        }
#endif
    }
};

// make_tuple {{{1
template <class T, class A0>
Vc_INTRINSIC datapar_tuple<T, A0> make_tuple(Vc::datapar<T, A0> x0)
{
    return {detail::data(x0)};
}
template <class T, class A0, class... As>
Vc_INTRINSIC datapar_tuple<T, A0, As...> make_tuple(const Vc::datapar<T, A0> &x0,
                                                    const Vc::datapar<T, As> &... xs)
{
    return {detail::data(x0), make_tuple(xs...)};
}

// get_datapar<N> {{{1
namespace datapar_tuple_impl
{
struct as_datapar;
template <class R = void, class T, class A0, class... Abis>
auto get_impl(const datapar_tuple<T, A0, Abis...> &t, size_constant<0>)
{
    return std::conditional_t<is_same<R, as_datapar>::value, datapar<T, A0>,
                              decltype(t.first)>(t.first);
}
template <class R = void, size_t N, class T, class... Abis>
auto get_impl(const datapar_tuple<T, Abis...> &t, size_constant<N>)
{
    return get_impl<R>(t.second, size_constant<N - 1>());
}
}  // namespace datapar_tuple_impl

template <size_t N, class T, class... Abis>
auto get_datapar(const datapar_tuple<T, Abis...> &t)
{
    return datapar_tuple_impl::get_impl<datapar_tuple_impl::as_datapar>(
        t, size_constant<N>());
}

template <size_t N, class T, class... Abis> auto get(const datapar_tuple<T, Abis...> &t)
{
    return datapar_tuple_impl::get_impl(t, size_constant<N>());
}

// tuple_element {{{1
template <size_t I, class T> struct tuple_element;
template <class T, class A0, class... As>
struct tuple_element<0, datapar_tuple<T, A0, As...>> {
    using type = Vc::datapar<T, A0>;
};
template <size_t I, class T, class A0, class... As>
struct tuple_element<I, datapar_tuple<T, A0, As...>> {
    using type = typename tuple_element<I - 1, datapar_tuple<T, As...>>::type;
};
template <size_t I, class T> using tuple_element_t = typename tuple_element<I, T>::type;

// number_of_preceding_elements {{{1
template <size_t I, class T> struct number_of_preceding_elements;
template <class T, class A0, class... As>
struct number_of_preceding_elements<0, datapar_tuple<T, A0, As...>>
    : public size_constant<0> {
};
template <size_t I, class T, class A0, class... As>
struct number_of_preceding_elements<I, datapar_tuple<T, A0, As...>>
    : public std::integral_constant<
          size_t,
          datapar<T, A0>::size() +
              number_of_preceding_elements<I - 1, datapar_tuple<T, As...>>::value> {
};

// for_each(const datapar_tuple &, Fun) {{{1
template <size_t Offset = 0, class T, class A0, class F>
Vc_INTRINSIC void for_each(const datapar_tuple<T, A0> &t_, F &&fun_)
{
    std::forward<F>(fun_)(make_meta<Offset>(t_), t_.first);
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
Vc_INTRINSIC void for_each(const datapar_tuple<T, A0, A1, As...> &t_, F &&fun_)
{
    fun_(make_meta<Offset>(t_), t_.first);
    for_each<Offset + datapar_size<T, A0>::value>(t_.second, std::forward<F>(fun_));
}

// for_each(datapar_tuple &, Fun) {{{1
template <size_t Offset = 0, class T, class A0, class F>
Vc_INTRINSIC void for_each(datapar_tuple<T, A0> &t_, F &&fun_)
{
    std::forward<F>(fun_)(make_meta<Offset>(t_), t_.first);
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
Vc_INTRINSIC void for_each(datapar_tuple<T, A0, A1, As...> &t_, F &&fun_)
{
    fun_(make_meta<Offset>(t_), t_.first);
    for_each<Offset + datapar_size<T, A0>::value>(t_.second, std::forward<F>(fun_));
}

// for_each(datapar_tuple &, const datapar_tuple &, Fun) {{{1
template <size_t Offset = 0, class T, class A0, class F>
Vc_INTRINSIC void for_each(datapar_tuple<T, A0> &a_, const datapar_tuple<T, A0> &b_,
                           F &&fun_)
{
    std::forward<F>(fun_)(make_meta<Offset>(a_), a_.first, b_.first);
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
Vc_INTRINSIC void for_each(datapar_tuple<T, A0, A1, As...> & a_,
                           const datapar_tuple<T, A0, A1, As...> &b_, F &&fun_)
{
    fun_(make_meta<Offset>(a_), a_.first, b_.first);
    for_each<Offset + datapar_size<T, A0>::value>(a_.second, b_.second,
                                                  std::forward<F>(fun_));
}

// for_each(const datapar_tuple &, const datapar_tuple &, Fun) {{{1
template <size_t Offset = 0, class T, class A0, class F>
Vc_INTRINSIC void for_each(const datapar_tuple<T, A0> &a_, const datapar_tuple<T, A0> &b_,
                           F &&fun_)
{
    std::forward<F>(fun_)(make_meta<Offset>(a_), a_.first, b_.first);
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
Vc_INTRINSIC void for_each(const datapar_tuple<T, A0, A1, As...> &a_,
                           const datapar_tuple<T, A0, A1, As...> &b_, F &&fun_)
{
    fun_(make_meta<Offset>(a_), a_.first, b_.first);
    for_each<Offset + datapar_size<T, A0>::value>(a_.second, b_.second,
                                                  std::forward<F>(fun_));
}

// }}}1

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_DATAPAR_TUPLE_H_

// vim: foldmethod=marker
