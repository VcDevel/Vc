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

#ifndef TESTS_METAHELPERS_H_
#define TESTS_METAHELPERS_H_

#include <climits>
#include <cfloat>

#if LONG_MAX == INT_MAX
#define LONG_IS_INT 1
#define LONG_IS_LLONG 0
#elif LONG_MAX == LLONG_MAX
#define LONG_IS_INT 0
#define LONG_IS_LLONG 1
#else
#error "What else could long be?"
#endif

#if LDBL_MANT_DIG == DBL_MANT_DIG
#define LDOUBLE_IS_DOUBLE 1
#else
#define LDOUBLE_IS_DOUBLE 0
#endif

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
constexpr bool operator_is_substitution_failure_impl(float)
{
    return true;
}

template <class A, class B, class Op = std::plus<>>
constexpr std::conditional_t<true, bool,
                             decltype(Op()(Vc::declval<A>(), Vc::declval<B>()))>
operator_is_substitution_failure_impl(int)
{
    return false;
}

template <class... Ts>
constexpr bool operator_is_substitution_failure =
    operator_is_substitution_failure_impl<Ts...>(int());

// sfinae_is_callable{{{1
template <class F, class... Args>
constexpr auto sfinae_is_callable(F &&f, Args &&... args)
    -> std::conditional_t<true, bool,
                          decltype(std::forward<F>(f)(std::forward<Args>(args)...))>
{
    return true;
}
constexpr bool sfinae_is_callable(...) { return false; }

// traits {{{1
template <class A, class B>
constexpr bool has_less_bits =
    std::numeric_limits<A>::digits < std::numeric_limits<B>::digits;

//}}}1

#endif  // TESTS_METAHELPERS_H_
// vim: foldmethod=marker
