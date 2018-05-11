/*  This file is part of the Vc library. {{{
Copyright © 2016-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SIMD_SMART_REFERENCE_H_
#define VC_SIMD_SMART_REFERENCE_H_

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
template <class U, class Accessor = U, class ValueType = typename U::value_type>
class smart_reference
{
    friend Accessor;
    int index;
    U &obj;

    constexpr Vc_INTRINSIC ValueType read() const noexcept
    {
        if constexpr (std::is_arithmetic_v<U>) {
            Vc_ASSERT(index == 0);
            return obj;
        } else {
            return obj[index];
        }
    }

    template <class T> constexpr Vc_INTRINSIC void write(T &&x) const
    {
        Accessor::set(obj, index, std::forward<T>(x));
    }

public:
    Vc_INTRINSIC smart_reference(U &o, int i) noexcept : index(i), obj(o) {}

    using value_type = ValueType;

    Vc_INTRINSIC smart_reference(const smart_reference &) = delete;

    constexpr Vc_INTRINSIC operator value_type() const noexcept { return read(); }

    template <class T,
              class = detail::value_preserving_or_int<std::decay_t<T>, value_type>>
    constexpr Vc_INTRINSIC smart_reference operator=(T &&x) &&
    {
        write(std::forward<T>(x));
        return {obj, index};
    }

// TODO: improve with operator.()

#define Vc_OP_(op_)                                                                      \
    template <class T,                                                                   \
              class TT = decltype(std::declval<value_type>() op_ std::declval<T>()),     \
              class = detail::value_preserving_or_int<std::decay_t<T>, TT>,              \
              class = detail::value_preserving_or_int<TT, value_type>>                   \
        Vc_INTRINSIC smart_reference operator op_##=(T &&x) &&                           \
    {                                                                                    \
        const value_type &lhs = read();                                                  \
        write(lhs op_ x);                                                                \
        return {obj, index};                                                             \
    }
    Vc_ALL_ARITHMETICS(Vc_OP_);
    Vc_ALL_SHIFTS(Vc_OP_);
    Vc_ALL_BINARY(Vc_OP_);
#undef Vc_OP_

    template <class T = void,
              class = decltype(
                  ++std::declval<std::conditional_t<true, value_type, T> &>())>
    Vc_INTRINSIC smart_reference operator++() &&
    {
        value_type x = read();
        write(++x);
        return {obj, index};
    }

    template <class T = void,
              class = decltype(
                  std::declval<std::conditional_t<true, value_type, T> &>()++)>
    Vc_INTRINSIC value_type operator++(int) &&
    {
        const value_type r = read();
        value_type x = r;
        write(++x);
        return r;
    }

    template <class T = void,
              class = decltype(
                  --std::declval<std::conditional_t<true, value_type, T> &>())>
    Vc_INTRINSIC smart_reference operator--() &&
    {
        value_type x = read();
        write(--x);
        return {obj, index};
    }

    template <class T = void,
              class = decltype(
                  std::declval<std::conditional_t<true, value_type, T> &>()--)>
    Vc_INTRINSIC value_type operator--(int) &&
    {
        const value_type r = read();
        value_type x = r;
        write(--x);
        return r;
    }

    friend Vc_INTRINSIC void swap(smart_reference &&a, smart_reference &&b) noexcept(
        all<std::is_nothrow_constructible<value_type, smart_reference &&>,
            std::is_nothrow_assignable<smart_reference &&, value_type &&>>::value)
    {
        value_type tmp = static_cast<smart_reference &&>(a);
        static_cast<smart_reference &&>(a) = static_cast<value_type>(b);
        static_cast<smart_reference &&>(b) = std::move(tmp);
    }

    friend Vc_INTRINSIC void swap(value_type &a, smart_reference &&b) noexcept(
        all<std::is_nothrow_constructible<value_type, value_type &&>,
            std::is_nothrow_assignable<value_type &, value_type &&>,
            std::is_nothrow_assignable<smart_reference &&, value_type &&>>::value)
    {
        value_type tmp(std::move(a));
        a = static_cast<value_type>(b);
        static_cast<smart_reference &&>(b) = std::move(tmp);
    }

    friend Vc_INTRINSIC void swap(smart_reference &&a, value_type &b) noexcept(
        all<std::is_nothrow_constructible<value_type, smart_reference &&>,
            std::is_nothrow_assignable<value_type &, value_type &&>,
            std::is_nothrow_assignable<smart_reference &&, value_type &&>>::value)
    {
        value_type tmp(a);
        static_cast<smart_reference &&>(a) = std::move(b);
        b = std::move(tmp);
    }
};
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_SIMD_SMART_REFERENCE_H_

// vim: foldmethod=marker
