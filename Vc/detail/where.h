/*  This file is part of the Vc library. {{{
Copyright © 2013-2017 Matthias Kretz <kretz@kde.org>

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

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
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
template <class T = void> struct increment {
    constexpr T operator()(T a) const { return ++a; }
};
template <> struct increment<void> {
    template <typename T> constexpr T operator()(T a) const { return ++a; }
};
template <class T = void> struct decrement {
    constexpr T operator()(T a) const { return --a; }
};
template <> struct decrement<void> {
    template <typename T> constexpr T operator()(T a) const { return --a; }
};

template <class T> struct get_impl {
    static_assert(
        std::is_arithmetic<T>::value,
        "Vc chose the wrong implementation class. This should not be possible.");

    template <class U, class F>
    Vc_INTRINSIC void masked_load(T &d, bool k, const U *mem, F)
    {
        if (k) {
            d = static_cast<T>(mem[0]);
        }
    }
};
template <> struct get_impl<bool> {
    template <class F> Vc_INTRINSIC void masked_load(bool &d, bool k, const bool *mem, F)
    {
        if (k) {
            d = mem[0];
        }
    }
};

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_WHERE_H_
