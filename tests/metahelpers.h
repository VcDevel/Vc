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

#ifndef VC_TESTS_METAHELPERS_H_
#define VC_TESTS_METAHELPERS_H_

#include <vir/metahelpers.h>

// more operator objects {{{1
struct assignment {
    template <class A, class B>
    constexpr decltype(std::declval<A>() = std::declval<B>()) operator()(A &&a,
                                                                         B &&b) const
        noexcept(noexcept(std::forward<A>(a) = std::forward<B>(b)))
    {
        return std::forward<A>(a) = std::forward<B>(b);
    }
};

struct bit_shift_left {
    template <class A, class B>
    constexpr decltype(std::declval<A>() << std::declval<B>()) operator()(A &&a,
                                                                          B &&b) const
        noexcept(noexcept(std::forward<A>(a) << std::forward<B>(b)))
    {
        return std::forward<A>(a) << std::forward<B>(b);
    }
};

struct bit_shift_right {
    template <class A, class B>
    constexpr decltype(std::declval<A>() >> std::declval<B>()) operator()(A &&a,
                                                                          B &&b) const
        noexcept(noexcept(std::forward<A>(a) >> std::forward<B>(b)))
    {
        return std::forward<A>(a) >> std::forward<B>(b);
    }
};

struct assign_modulus {
    template <class A, class B>
    constexpr decltype(std::declval<A>() %= std::declval<B>()) operator()(A &&a,
                                                                          B &&b) const
        noexcept(noexcept(std::forward<A>(a) %= std::forward<B>(b)))
    {
        return std::forward<A>(a) %= std::forward<B>(b);
    }
};

struct assign_bit_and {
    template <class A, class B>
    constexpr decltype(std::declval<A>() &= std::declval<B>()) operator()(A &&a,
                                                                          B &&b) const
        noexcept(noexcept(std::forward<A>(a) &= std::forward<B>(b)))
    {
        return std::forward<A>(a) &= std::forward<B>(b);
    }
};

struct assign_bit_or {
    template <class A, class B>
    constexpr decltype(std::declval<A>() |= std::declval<B>()) operator()(A &&a,
                                                                          B &&b) const
        noexcept(noexcept(std::forward<A>(a) |= std::forward<B>(b)))
    {
        return std::forward<A>(a) |= std::forward<B>(b);
    }
};

struct assign_bit_xor {
    template <class A, class B>
    constexpr decltype(std::declval<A>() ^= std::declval<B>()) operator()(A &&a,
                                                                          B &&b) const
        noexcept(noexcept(std::forward<A>(a) ^= std::forward<B>(b)))
    {
        return std::forward<A>(a) ^= std::forward<B>(b);
    }
};

struct assign_bit_shift_left {
    template <class A, class B>
    constexpr decltype(std::declval<A>() <<= std::declval<B>()) operator()(A &&a,
                                                                          B &&b) const
        noexcept(noexcept(std::forward<A>(a) <<= std::forward<B>(b)))
    {
        return std::forward<A>(a) <<= std::forward<B>(b);
    }
};

struct assign_bit_shift_right {
    template <class A, class B>
    constexpr decltype(std::declval<A>() >>= std::declval<B>()) operator()(A &&a,
                                                                          B &&b) const
        noexcept(noexcept(std::forward<A>(a) >>= std::forward<B>(b)))
    {
        return std::forward<A>(a) >>= std::forward<B>(b);
    }
};

// operator_is_substitution_failure {{{1
template <class A, class B, class Op = std::plus<>>
constexpr bool is_substitution_failure =
    vir::test::operator_is_substitution_failure<A, B, Op>();

// sfinae_is_callable{{{1
using vir::test::sfinae_is_callable;

// traits {{{1
using vir::test::has_less_bits;

//}}}1

#endif  // VC_TESTS_METAHELPERS_H_
// vim: foldmethod=marker
