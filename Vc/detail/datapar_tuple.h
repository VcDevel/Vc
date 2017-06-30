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

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// datapar_tuple
// why not std::tuple?
// 1. std::tuple gives no guarantee about the storage order, but I require storage
//    equivalent to std::array<T, N>
// 2. much less code to instantiate: I require a very small subset of std::tuple
//    functionality
// 3. direct access to the element type (first template argument)
// 4. enforces equal element type, only different Abi types are allowed

template <class T, class... Abis> struct datapar_tuple;

// datapar_tuple specializations {{{1
template <class T> struct datapar_tuple<T> {
    static constexpr size_t tuple_size = 0;
    //static constexpr size_t element_count = 0;
};
template <class T, class Abi0> struct datapar_tuple<T, Abi0> {
    using first_type = Vc::datapar<T, Abi0>;
    static constexpr size_t tuple_size = 1;
    //static constexpr size_t element_count = first_type::size();
    first_type first;

    template <size_t Offset = 0, class F>
    static Vc_INTRINSIC datapar_tuple generate(F &&gen)
    {
        return {gen(first_type(), size_constant<Offset>())};
    }

    template <class F, class... More>
    friend Vc_INTRINSIC datapar_tuple apply(F &&fun, const datapar_tuple &x,
                                            const More &... more)
    {
        return {fun(x.first, more.first...)};
    }

    T operator[](size_t i) const noexcept { return first[i]; }
    void set(size_t i, T val) noexcept { first[i] = val; }
};
template <class T, class Abi0, class... Abis> struct datapar_tuple<T, Abi0, Abis...> {
    using first_type = Vc::datapar<T, Abi0>;
    using second_type = datapar_tuple<T, Abis...>;
    static constexpr size_t tuple_size = sizeof...(Abis) + 1;
    //static constexpr size_t element_count = first_type::size + second_type::element_count;
    first_type first;
    second_type second;

    template <size_t Offset = 0, class F>
    static Vc_INTRINSIC datapar_tuple generate(F &&gen)
    {
        return {gen(first_type(), size_constant<Offset>()),
                second_type::template generate<Offset + first_type::size()>(
                    std::forward<F>(gen))};
    }

    template <class F, class... More>
    friend Vc_INTRINSIC datapar_tuple apply(F &&fun, const datapar_tuple &x,
                                            const More &... more)
    {
        return {fun(x.first, more.first...),
                apply(std::forward<F>(fun), x.second, more.second...)};
    }

    T operator[](size_t i) const noexcept
    {
#ifdef __GNUC__
        return reinterpret_cast<const may_alias<T> *>(this)[i];
#else
        return i < first_type::size() ? first[i] : second[i - first_type::size()];
#endif
    }
    void set(size_t i, T val) noexcept
    {
#ifdef __GNUC__
        reinterpret_cast<may_alias<T> *>(this)[i] = val;
#else
        if (i < first_type::size()) {
            first[i] = val;
        } else {
            second.set(i - first_type::size(), val);
        }
#endif
    }
};

// make_tuple {{{1
template <class T, class A0> datapar_tuple<T, A0> make_tuple(const Vc::datapar<T, A0> &x0)
{
    return {x0};
}
template <class T, class A0, class... As>
datapar_tuple<T, A0, As...> make_tuple(const Vc::datapar<T, A0> &x0,
                                       const Vc::datapar<T, As> &... xs)
{
    return {x0, make_tuple(xs...)};
}

// get<N> {{{1
namespace datapar_tuple_impl
{
template <class T, class... Abis>
auto get_impl(const datapar_tuple<T, Abis...> &t, size_constant<0>)
{
    return t.first;
}
template <size_t N, class T, class... Abis>
auto get_impl(const datapar_tuple<T, Abis...> &t, size_constant<N>)
{
    return get_impl(t.second, size_constant<N - 1>());
}
}  // namespace datapar_tuple_impl
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
    std::forward<F>(fun_)(t_.first, size_constant<Offset>());
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
Vc_INTRINSIC void for_each(const datapar_tuple<T, A0, A1, As...> &t_, F &&fun_)
{
    fun_(t_.first, size_constant<Offset>());
    for_each<Offset + datapar_size<T, A0>::value>(t_.second, std::forward<F>(fun_));
}

// for_each(datapar_tuple &, Fun) {{{1
template <size_t Offset = 0, class T, class A0, class F>
Vc_INTRINSIC void for_each(datapar_tuple<T, A0> &t_, F &&fun_)
{
    std::forward<F>(fun_)(t_.first, size_constant<Offset>());
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
Vc_INTRINSIC void for_each(datapar_tuple<T, A0, A1, As...> &t_, F &&fun_)
{
    fun_(t_.first, size_constant<Offset>());
    for_each<Offset + datapar_size<T, A0>::value>(t_.second, std::forward<F>(fun_));
}

// for_each(datapar_tuple &, const datapar_tuple &, Fun) {{{1
template <size_t Offset = 0, class T, class A0, class F>
Vc_INTRINSIC void for_each(datapar_tuple<T, A0> &a_, const datapar_tuple<T, A0> &b_,
                           F &&fun_)
{
    std::forward<F>(fun_)(a_.first, b_.first, size_constant<Offset>());
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
Vc_INTRINSIC void for_each(datapar_tuple<T, A0, A1, As...> & a_,
                           const datapar_tuple<T, A0, A1, As...> &b_, F &&fun_)
{
    fun_(a_.first, b_.first, size_constant<Offset>());
    for_each<Offset + datapar_size<T, A0>::value>(a_.second, b_.second, std::forward<F>(fun_));
}

// for_each(const datapar_tuple &, const datapar_tuple &, Fun) {{{1
template <size_t Offset = 0, class T, class A0, class F>
Vc_INTRINSIC void for_each(const datapar_tuple<T, A0> &a_, const datapar_tuple<T, A0> &b_,
                           F &&fun_)
{
    std::forward<F>(fun_)(a_.first, b_.first, size_constant<Offset>());
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
Vc_INTRINSIC void for_each(const datapar_tuple<T, A0, A1, As...> &a_,
                           const datapar_tuple<T, A0, A1, As...> &b_, F &&fun_)
{
    fun_(a_.first, b_.first, size_constant<Offset>());
    for_each<Offset + datapar_size<T, A0>::value>(a_.second, b_.second, std::forward<F>(fun_));
}

// }}}1

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_DATAPAR_TUPLE_H_

// vim: foldmethod=marker
