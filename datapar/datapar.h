/*  This file is part of the Vc library. {{{
Copyright © 2016 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DATAPAR_DATAPAR_H_
#define VC_DATAPAR_DATAPAR_H_

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
template <class T, class A> Vc_INTRINSIC_L auto data(const datapar<T, A> &x) Vc_INTRINSIC_R;

template <class Derived> struct generic_datapar_impl;
// allow_conversion_ctor2{{{1
template <class T0, class T1, class A, bool BothIntegral> struct allow_conversion_ctor2_1;

template <class T0, class T1, class A>
struct allow_conversion_ctor2
    : public allow_conversion_ctor2_1<
          T0, T1, A, conjunction<std::is_integral<T0>, std::is_integral<T1>>::value> {
};

// disallow 2nd conversion ctor (equal Abi), if the value_types are equal (copy ctor)
template <class T, class A> struct allow_conversion_ctor2<T, T, A> : public std::false_type {};

// disallow 2nd conversion ctor (equal Abi), if the Abi is a fixed_size instance
template <class T0, class T1, int N>
struct allow_conversion_ctor2<T0, T1, datapar_abi::fixed_size<N>> : public std::false_type {};

// disallow 2nd conversion ctor (equal Abi), if both of the above are true
template <class T, int N>
struct allow_conversion_ctor2<T, T, datapar_abi::fixed_size<N>> : public std::false_type {};

// disallow 2nd conversion ctor (equal Abi), the integers only differ in sign
template <class T0, class T1, class A>
struct allow_conversion_ctor2_1<T0, T1, A, true>
    : public std::is_same<std::make_signed_t<T0>, std::make_signed_t<T1>> {
};

// disallow 2nd conversion ctor (equal Abi), any value_type is not integral
template <class T0, class T1, class A>
struct allow_conversion_ctor2_1<T0, T1, A, false> : public std::false_type {
};

// allow_conversion_ctor3{{{1
template <class T0, class A0, class T1, class A1, bool = std::is_same<A0, A1>::value>
struct allow_conversion_ctor3 : public std::false_type {
    // disallow 3rd conversion ctor if A0 is not fixed_size<datapar_size_v<T1, A1>>
};

template <class T0, class T1, class A1>
struct allow_conversion_ctor3<T0, datapar_abi::fixed_size<datapar_size_v<T1, A1>>, T1, A1,
                              false  // disallow 3rd conversion ctor if the Abi types are
                                     // equal (disambiguate copy ctor and the two
                                     // preceding conversion ctors)
                              > : public std::is_convertible<T1, T0> {
};

//}}}1
}  // namespace detail

template <class V, bool> class datapar_int_operators;

template <class V> class datapar_int_operators<V, false>
{
};

template <class V> class datapar_int_operators<V, true>
{
    using impl = detail::get_impl_t<V>;

public:
    friend V &operator %=(V &lhs, const V &x) { return lhs = lhs  % x; }
    friend V &operator &=(V &lhs, const V &x) { return lhs = lhs  & x; }
    friend V &operator |=(V &lhs, const V &x) { return lhs = lhs  | x; }
    friend V &operator ^=(V &lhs, const V &x) { return lhs = lhs  ^ x; }
    friend V &operator<<=(V &lhs, const V &x) { return lhs = lhs << x; }
    friend V &operator>>=(V &lhs, const V &x) { return lhs = lhs >> x; }

    friend V operator%(const V &x, const V &y) { return impl::modulus(x, y); }
    friend V operator&(const V &x, const V &y) { return impl::bit_and(x, y); }
    friend V operator|(const V &x, const V &y) { return impl::bit_or(x, y); }
    friend V operator^(const V &x, const V &y) { return impl::bit_xor(x, y); }
    friend V operator<<(const V &x, const V &y) { return impl::bit_shift_left(x, y); }
    friend V operator>>(const V &x, const V &y) { return impl::bit_shift_right(x, y); }
};

template <class T, class Abi>
class datapar : public datapar_int_operators<datapar<T, Abi>, std::is_integral<T>::value>
{
    using traits = detail::traits<T, Abi>;
    using impl = typename traits::datapar_impl_type;
    using member_type = typename traits::datapar_member_type;
    using cast_type = typename traits::datapar_cast_type;
    static constexpr std::integral_constant<size_t, traits::size()> size_tag = {};
    static constexpr T *type_tag = nullptr;
    friend impl;
    friend detail::generic_datapar_impl<impl>;

public:
    using value_type = T;
    using reference = detail::smart_reference<datapar, impl>;
    using mask_type = mask<T, Abi>;
    using size_type = size_t;
    using abi_type = Abi;

    static constexpr size_type size() { return traits::size(); }
    datapar() = default;
    datapar(const datapar &) = default;
    datapar(datapar &&) = default;
    datapar &operator=(const datapar &) = default;
    datapar &operator=(datapar &&) = default;

    // non-std; required to work around MSVC error C2975
    static constexpr size_type size_v = traits::size();

    // implicit broadcast constructor
    template <
        class U, class U_ = std::decay_t<U>,
        class = std::enable_if_t<conjunction<
            std::is_convertible<U, value_type>,
            disjunction<
                std::is_same<U_, value_type>, std::is_same<U_, int>,
                conjunction<std::is_same<U_, detail::uint>, std::is_unsigned<value_type>>,
                negation<detail::is_narrowing_conversion<U_, value_type>>>>::value>>
    datapar(U &&x)
        : d(impl::broadcast(static_cast<value_type>(std::forward<U>(x)), size_tag))
    {
    }

    // implicit type conversion constructor (convert from fixed_size to fixed_size)
    template <class U>
    datapar(
        const datapar<U, datapar_abi::fixed_size<size_v>> &x,
        std::enable_if_t<
            conjunction<std::is_same<datapar_abi::fixed_size<size_v>, abi_type>,
                        negation<detail::is_narrowing_conversion<U, value_type>>,
                        detail::converts_to_higher_integer_rank<U, value_type>>::value,
            void *> = nullptr)
        : datapar{static_cast<const std::array<U, size()> &>(x).data(),
                  flags::vector_aligned}
    {
    }

    // explicit type conversion constructor
    // 1st conversion ctor: convert from fixed_size<size()>
    template <class U>
    explicit datapar(
        const datapar<U, datapar_abi::fixed_size<size_v>> &x,
        std::enable_if_t<
            disjunction<conjunction<negation<std::is_same<datapar_abi::fixed_size<size_v>,
                                                          abi_type>>,
                                    std::is_convertible<U, value_type>>,
                        detail::is_narrowing_conversion<U, value_type>>::value,
            void *> = nullptr)
        : datapar{static_cast<const std::array<U, size()> &>(x).data(),
                  flags::vector_aligned}
    {
    }

    // 2nd conversion ctor: convert equal Abi, integers that only differ in signedness
    template <class U>
    explicit datapar(
        const datapar<U, Abi> &x,
        std::enable_if_t<detail::allow_conversion_ctor2<value_type, U, Abi>::value,
                         void *> = nullptr)
        : d{static_cast<cast_type>(x)}
    {
    }

    // 3rd conversion ctor: convert from non-fixed_size to fixed_size if U is convertible to
    // value_type
    template <class U, class Abi2>
    explicit datapar(
        const datapar<U, Abi2> &x,
        std::enable_if_t<detail::allow_conversion_ctor3<value_type, Abi, U, Abi2>::value,
                         void *> = nullptr)
    {
        x.memstore(d.data(), flags::overaligned<alignof(datapar)>);
    }

    // generator constructor
    template <class F>
    explicit datapar(
        F &&gen,
        enable_if<std::is_same<
            value_type, decltype(declval<F>()(
                            declval<std::integral_constant<size_t, 0> &>()))>::value> =
            nullarg)
        : d(detail::generate_from_n_evaluations<size(), member_type>(
              [&gen](auto element_idx_) { return gen(element_idx_); }))
    {
    }

    template <class U, U... Indexes>
    static datapar seq(std::integer_sequence<U, Indexes...>)
    {
        alignas(
            memory_alignment<datapar>::value) static constexpr value_type mem[size()] = {
            value_type(Indexes)...};
        return datapar(mem, flags::vector_aligned);
    }
    static datapar seq() {
        return seq(std::make_index_sequence<size()>());
    }

    // load constructor
    template <class U, class Flags>
    datapar(const U *mem, Flags f)
        : d(impl::load(mem, f, type_tag))
    {
    }
    template <class U, class Flags> datapar(const U *mem, const mask_type &k, Flags f) : d{}
    {
        impl::masked_load(d, k, mem, f);
    }

    // loads [datapar.load]
    template <class U, class Flags> void memload(const U *mem, Flags f)
    {
        d = static_cast<decltype(d)>(impl::load(mem, f, type_tag));
    }
    template <class U, class Flags> void Vc_VDECL memload(const U *mem, mask_type k, Flags f)
    {
        impl::masked_load(d, k, mem, f);
    }

    // stores [datapar.store]
    template <class U, class Flags> void memstore(U *mem, Flags f) const
    {
        impl::store(d, mem, f, type_tag);
    }
    template <class U, class Flags> void Vc_VDECL memstore(U *mem, mask_type k, Flags f) const
    {
        impl::masked_store(d, mem, f, k);
    }

    // scalar access
    reference operator[](size_type i) { return {*this, int(i)}; }
    value_type operator[](size_type i) const { return impl::get(*this, int(i)); }

    // increment and decrement:
    datapar &operator++() { impl::increment(d); return *this; }
    datapar operator++(int) { datapar r = *this; impl::increment(d); return r; }
    datapar &operator--() { impl::decrement(d); return *this; }
    datapar operator--(int) { datapar r = *this; impl::decrement(d); return r; }

    // unary operators (for integral T)
    mask_type operator!() const { return impl::negate(*this); }
    datapar operator~() const { return impl::complement(*this); }

    // unary operators (for any T)
    datapar operator+() const { return *this; }
    datapar operator-() const { return impl::unary_minus(*this); }

    // access to internal representation (suggested extension)
    explicit operator cast_type() const { return d; }
    explicit datapar(const cast_type &init) : d(init) {}

    // compound assignment [datapar.cassign]
    friend datapar &operator+=(datapar &lhs, const datapar &x) { return lhs = lhs + x; }
    friend datapar &operator-=(datapar &lhs, const datapar &x) { return lhs = lhs - x; }
    friend datapar &operator*=(datapar &lhs, const datapar &x) { return lhs = lhs * x; }
    friend datapar &operator/=(datapar &lhs, const datapar &x) { return lhs = lhs / x; }

    // binary operators [datapar.binary]
    friend datapar operator+(const datapar &x, const datapar &y)
    {
        return impl::plus(x, y);
    }
    friend datapar operator-(const datapar &x, const datapar &y)
    {
        return impl::minus(x, y);
    }
    friend datapar operator*(const datapar &x, const datapar &y)
    {
        return impl::multiplies(x, y);
    }
    friend datapar operator/(const datapar &x, const datapar &y)
    {
        return impl::divides(x, y);
    }

    // compares [datapar.comparison]
    friend mask_type operator==(const datapar &x, const datapar &y)
    {
        return impl::equal_to(x, y);
    }
    friend mask_type operator!=(const datapar &x, const datapar &y)
    {
        return impl::not_equal_to(x, y);
    }
    friend mask_type operator<(const datapar &x, const datapar &y)
    {
        return impl::less(x, y);
    }
    friend mask_type operator<=(const datapar &x, const datapar &y)
    {
        return impl::less_equal(x, y);
    }
    friend mask_type operator>(const datapar &x, const datapar &y)
    {
        return impl::less(y, x);
    }
    friend mask_type operator>=(const datapar &x, const datapar &y)
    {
        return impl::less_equal(y, x);
    }

private:
#ifdef Vc_MSVC
    // Work around "warning C4396: the inline specifier cannot be used when a friend
    // declaration refers to a specialization of a function template"
    template <class U, class A> friend auto detail::data(const datapar<U, A> &);
#else
    friend auto detail::data<value_type, abi_type>(const datapar &);
#endif
    datapar(detail::private_init_t, const member_type &init) : d(init) {}
    alignas(traits::datapar_member_alignment) member_type d = {};
};

namespace detail
{
template <class T, class A> Vc_INTRINSIC auto data(const datapar<T, A> &x) { return x.d; }
}  // namespace detail

Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_DATAPAR_H_

// vim: foldmethod=marker
