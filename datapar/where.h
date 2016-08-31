/*  This file is part of the Vc library. {{{
Copyright © 2013-2016 Matthias Kretz <kretz@kde.org>

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

#include <utility>
#include <functional>

#ifndef VC_DATAPAR_WHERE_H_
#define VC_DATAPAR_WHERE_H_

namespace Vc_VERSIONED_NAMESPACE::detail
{
template <typename T, typename U> inline void masked_assign(bool k, T &lhs, U &&rhs)
{
    if (k) {
        lhs = std::forward<U>(rhs);
    }
}
template <template <typename> class Op, typename T, typename U>
inline void masked_cassign(bool k, T &lhs, U &&rhs)
{
    if (k) {
        lhs = Op<void>{}(lhs, std::forward<T>(rhs));
    }
}
template <template <typename> class Op, typename T> inline auto masked_unary(bool k, T &d)
{
    return k ? Op<void>{}(d) : d;
}

template <class T = void> struct shift_left {
    constexpr T operator()(const T &a, const T &b) const { return a << b; }
};
template <> struct shift_left<void> {
    template <typename L, typename R> constexpr auto operator()(L &&a, R &&b) const
    {
        return std::forward<L>(a) << std::forward<R>(b);
    }
};
template <class T = void> struct shift_right {
    constexpr T operator()(const T &a, const T &b) const { return a >> b; }
};
template <> struct shift_right<void> {
    template <typename L, typename R> constexpr auto operator()(L &&a, R &&b) const
    {
        return std::forward<L>(a) >> std::forward<R>(b);
    }
};
template <class = void> struct pre_increment {
    template <typename T> constexpr T &operator()(T &a) const { return ++a; }
};
template <class = void> struct post_increment {
    template <typename T> constexpr T operator()(T &a) const { return a++; }
};
template <class = void> struct pre_decrement {
    template <typename T> constexpr T &operator()(T &a) const { return --a; }
};
template <class = void> struct post_decrement {
    template <typename T> constexpr T operator()(T &a) const { return a--; }
};

template <typename Mask, typename T> class where_proxy
{
public:
    where_proxy() = delete;
    where_proxy(const where_proxy &) = delete;
    where_proxy(where_proxy &&) = delete;
    where_proxy &operator=(const where_proxy &) = delete;
    where_proxy &operator=(where_proxy &&) = delete;
    constexpr where_proxy(const Mask &kk, T &dd) : k(kk), d(dd) {}
    template <class U> void operator=(U &&x) { masked_assign(k, d, std::forward<U>(x)); }
    template <class U> void operator+=(U &&x)
    {
        masked_cassign<std::plus>(k, d, std::forward<U>(x));
    }
    template <class U> void operator-=(U &&x)
    {
        masked_cassign<std::minus>(k, d, std::forward<U>(x));
    }
    template <class U> void operator*=(U &&x)
    {
        masked_cassign<std::multiplies>(k, d, std::forward<U>(x));
    }
    template <class U> void operator/=(U &&x)
    {
        masked_cassign<std::divides>(k, d, std::forward<U>(x));
    }
    template <class U> void operator%=(U &&x)
    {
        masked_cassign<std::modulus>(k, d, std::forward<U>(x));
    }
    template <class U> void operator&=(U &&x)
    {
        masked_cassign<std::bit_and>(k, d, std::forward<U>(x));
    }
    template <class U> void operator|=(U &&x)
    {
        masked_cassign<std::bit_or>(k, d, std::forward<U>(x));
    }
    template <class U> void operator^=(U &&x)
    {
        masked_cassign<std::bit_xor>(k, d, std::forward<U>(x));
    }
    template <class U> void operator<<=(U &&x)
    {
        masked_cassign<shift_left>(k, d, std::forward<U>(x));
    }
    template <class U> void operator>>=(U &&x)
    {
        masked_cassign<shift_right>(k, d, std::forward<U>(x));
    }
    T &operator++() { return masked_unary<pre_increment>(k, d); }
    T operator++(int) { return masked_unary<post_increment>(k, d); }
    T &operator--() { return masked_unary<pre_decrement>(k, d); }
    T operator--(int) { return masked_unary<post_decrement>(k, d); }
    T operator-() const { return masked_unary<std::negate>(k, d); }
    auto operator!() const { return masked_unary<std::logical_not>(k, d); }
private:
    const Mask &k;
    T &d;
};
}  // namespace Vc_VERSIONED_NAMESPACE

#endif  // VC_DATAPAR_WHERE_H_
