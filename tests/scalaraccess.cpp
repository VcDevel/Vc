/*  This file is part of the Vc library. {{{
Copyright © 2010-2016 Matthias Kretz <kretz@kde.org>

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

#include "generators.h"
#include "unittest.h"

using namespace Vc;

TEST_TYPES(V, reads, concat<AllVectors, SimdArrayList>)
{
    typedef typename V::EntryType T;

    V a = V(0);
    const T zero = 0;
    for (size_t i = 0; i < V::Size; ++i) {
        const T x = a[i];
        COMPARE(x, zero);
    }
    a = V([](T n) { return n; });
    for (size_t i = 0; i < V::Size; ++i) {
        const T x = a[i];
        const T y = i;
        COMPARE(x, y);
    }
    // The non-const operator[] returns a smart reference that mimics an lvalue reference.
    // But since it's not an actual lvalue reference the proxy is supposed to live as
    // short as possible. To achieve this all move & copy operations must be disabled.
    VERIFY(std::is_move_constructible<decltype(a[0])>::value);
    VERIFY(!std::is_copy_constructible<decltype(a[0])>::value);
    VERIFY(!std::is_move_assignable<decltype(a[0])>::value);
    VERIFY(!std::is_copy_assignable<decltype(a[0])>::value);
    VERIFY(!std::is_reference<decltype(a[0])>::value);
    COMPARE(typeid(decltype(a[0])), typeid(typename V::reference));

    // the operator[] const overload returns an rvalue T and is therefore not constrained
    // in the same way as the non-const operator[]
    const V b = a;
    VERIFY(std::is_move_constructible<decltype(b[0])>::value);
    VERIFY(std::is_copy_constructible<decltype(b[0])>::value);
    VERIFY(std::is_move_assignable<decltype(b[0])>::value);
    VERIFY(std::is_copy_assignable<decltype(b[0])>::value);
    VERIFY(!std::is_reference<decltype(b[0])>::value);
    COMPARE(typeid(decltype(b[0])), typeid(typename V::value_type));
}

template <typename V, size_t Index> inline void readsConstantIndexTest(V a, V b)
{
    typedef typename V::EntryType T;
    {
        const T x = a[Index];
        const T zero = 0;
        COMPARE(x, zero) << Index;
    }{
        const T x = b[Index];
        const T y = Index;
        COMPARE(x, y) << Index;
    }
}

template<typename V, size_t Index>
struct ReadsConstantIndex
{
    ReadsConstantIndex(const V &a, const V &b)
    {
        readsConstantIndexTest<V, Index>(a, b);
        ReadsConstantIndex<V, Index - 1>(a, b);
    }
};


template<typename V>
struct ReadsConstantIndex<V, 0>
{
    ReadsConstantIndex(const V &a, const V &b) { readsConstantIndexTest<V, 0>(a, b); }
};

TEST_TYPES(V, readsConstantIndex, concat<AllVectors, SimdArrayList>)
{
    V a = V(0);
    V b = V([](int n) { return n; });
    ReadsConstantIndex<V, V::Size - 1>(a, b);
}

TEST_TYPES(V, writes, concat<AllVectors, SimdArrayList>)
{
    typedef typename V::EntryType T;

    V a = 0;
    std::array<T, V::size()> reference = {0};
    int i = 0;
    iterateNumericRange<T>([&](T x) {
        reference[i] = x;
        a[i] = x;
        COMPARE(a, V(&reference[0]));
        i = (i + 1) % V::size();
    });
}

template <typename T> using decayed = typename std::decay<T>::type;
template <typename V> using EntryType = typename decayed<V>::EntryType;

#define INT_OP(op, name)                                                                 \
    template <typename V> bool test_##name##_assign(V &&, float, int) { return false; }  \
    template <typename V,                                                                \
              typename = decltype(std::declval<V>()[0] op## = EntryType<V>())>           \
    bool test_##name##_assign(V &&a, int x, int y)                                       \
    {                                                                                    \
        using T = EntryType<V>;                                                          \
        COMPARE(a[0] op## = T(x), T(y));                                                 \
        COMPARE(a[0], T(y));                                                             \
        return true;                                                                     \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
INT_OP(+, plus);
INT_OP(-, minus);
INT_OP(*, times);
INT_OP(/, divide);
INT_OP(%, percent);
INT_OP(<<, lshift);
INT_OP(>>, rshift);
INT_OP(|, bor);
INT_OP(&, band);
INT_OP(^, bxor);
#undef INT_OP

#define UNARY_OP(op, name)                                                               \
    template <typename V> bool test_unary_##name(V &&, ...) { return false; }            \
    template <typename V>                                                                \
    bool test_unary_##name(V &&a, decltype(op(std::declval<V>()[0])) x)                  \
    {                                                                                    \
        COMPARE(op a[0], x);                                                             \
        return true;                                                                     \
    }                                                                                    \
    template <typename V> bool test_unary_##name##_nosubscript(V &&, ...)                \
    {                                                                                    \
        return false;                                                                    \
    }                                                                                    \
    template <typename V>                                                                \
    bool test_unary_##name##_nosubscript(V &&a, decltype(op std::declval<V>()) x)        \
    {                                                                                    \
        COMPARE(op a, x);                                                                \
        return true;                                                                     \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
UNARY_OP(!, not);
UNARY_OP(+, plus);
UNARY_OP(-, minus);
UNARY_OP(~, bitflip);
#undef UNARY_OP

template <typename V> bool test_pre_increment(V &&, float) { return false; }
template <typename V, typename = decltype(++std::declval<V>()[0])>
bool test_pre_increment(V &&a, int x)
{
    using T = EntryType<V>;
    COMPARE(++a[0], T(x));
    COMPARE(a[0], T(x));
    return true;
}
template <typename V> bool test_post_increment(V &&, float) { return false; }
template <typename V, typename = decltype(std::declval<V>()[0]++)>
bool test_post_increment(V &&a, int x)
{
    using T = EntryType<V>;
    COMPARE(a[0]++, T(x - 1));
    COMPARE(a[0], T(x));
    return true;
}
template <typename V> bool test_pre_decrement(V &&, float) { return false; }
template <typename V, typename = decltype(--std::declval<V>()[0])>
bool test_pre_decrement(V &&a, int x)
{
    using T = EntryType<V>;
    COMPARE(--a[0], T(x));
    COMPARE(a[0], T(x));
    return true;
}
template <typename V> bool test_post_decrement(V &&, float) { return false; }
template <typename V, typename = decltype(std::declval<V>()[0]--)>
bool test_post_decrement(V &&a, int x)
{
    using T = EntryType<V>;
    COMPARE(a[0]--, T(x + 1));
    COMPARE(a[0], T(x));
    return true;
}

TEST_TYPES(V, operators, concat<AllVectors, SimdArrayList>)
{
    using T = EntryType<V>;
    V a = 10;
    VERIFY(test_unary_not(a, false));
    VERIFY(test_unary_plus(a, 10));
    VERIFY(test_unary_minus(a, -10));
    COMPARE(test_unary_bitflip(a, ~10), std::is_integral<T>::value);
    VERIFY(test_plus_assign(a, 1, 11));
    VERIFY(test_minus_assign(a, 1, 10));
    VERIFY(test_times_assign(a, 2, 20));
    VERIFY(test_divide_assign(a, 2, 10));
    VERIFY(test_pre_decrement(a, 9));
    VERIFY(test_post_decrement(a, 8));
    VERIFY(test_pre_increment(a, 9));
    VERIFY(test_post_increment(a, 10));
    COMPARE(test_percent_assign(a, 6, 4), std::is_integral<T>::value);
    COMPARE(test_lshift_assign(a, 1, 8), std::is_integral<T>::value);
    COMPARE(test_rshift_assign(a, 2, 2), std::is_integral<T>::value);
    COMPARE(test_bor_assign(a, 9, 11), std::is_integral<T>::value);
    COMPARE(test_band_assign(a, 13, 9), std::is_integral<T>::value);
    COMPARE(test_bxor_assign(a, 1, 8), std::is_integral<T>::value);

    // assignment operators should never work on const ref
    a = 10;
    const auto &x = a[0];
    COMPARE(x, T(10));
    VERIFY(!test_plus_assign(x, 1, 11));
    VERIFY(!test_minus_assign(x, 1, 10));
    VERIFY(!test_times_assign(x, 2, 20));
    VERIFY(!test_divide_assign(x, 2, 10));
    VERIFY(!test_pre_decrement(x, 9));
    VERIFY(!test_post_decrement(x, 8));
    VERIFY(!test_pre_increment(x, 9));
    VERIFY(!test_post_increment(x, 10));
    VERIFY(!test_percent_assign(x, 6, 4));
    VERIFY(!test_lshift_assign(x, 1, 8));
    VERIFY(!test_rshift_assign(x, 2, 2));
    VERIFY(!test_bor_assign(x, 9, 11));
    VERIFY(!test_band_assign(x, 13, 9));
    VERIFY(!test_bxor_assign(x, 1, 8));

    // unary operators should work, though
    VERIFY(test_unary_not_nosubscript(x, false));
    VERIFY(test_unary_plus_nosubscript(x, 10));
    VERIFY(test_unary_minus_nosubscript(x, -10));
    COMPARE(test_unary_bitflip_nosubscript(x, ~10), std::is_integral<T>::value);

    // Just to make sure, the assignment operators also should not work on lvalue
    // references
    decltype(a[0]) *y = nullptr;
    VERIFY(!test_plus_assign(*y, 1, 11));
    VERIFY(!test_minus_assign(*y, 1, 10));
    VERIFY(!test_times_assign(*y, 2, 20));
    VERIFY(!test_divide_assign(*y, 2, 10));
    VERIFY(!test_pre_decrement(*y, 9));
    VERIFY(!test_post_decrement(*y, 8));
    VERIFY(!test_pre_increment(*y, 9));
    VERIFY(!test_post_increment(*y, 10));
    VERIFY(!test_percent_assign(*y, 6, 4));
    VERIFY(!test_lshift_assign(*y, 1, 8));
    VERIFY(!test_rshift_assign(*y, 2, 2));
    VERIFY(!test_bor_assign(*y, 9, 11));
    VERIFY(!test_band_assign(*y, 13, 9));
    VERIFY(!test_bxor_assign(*y, 1, 8));

    auto k0 = a == a;
    const auto &k1 = k0;
    VERIFY(test_unary_not(k0, 0));
    VERIFY(test_unary_not(k1, 0));
    k0 = a != a;
    VERIFY(test_unary_not(k0, 1));
}

TEST_TYPES(V, ensure_noexcept, concat<AllVectors, SimdArrayList>)
{
    V a{};
    {
        const V &b = a;
        EntryType<V> x = a[0]; if (x == x) {}
        VERIFY(noexcept(a[0]));
        VERIFY(noexcept(x = b[0]));
        VERIFY(noexcept(x = a[0]));
        VERIFY(noexcept(a[0] = 1));
        VERIFY(noexcept(a[0] += 1));
        VERIFY(noexcept(a[0] -= 1));
        VERIFY(noexcept(a[0] *= 1));
        VERIFY(noexcept(a[0] /= 1));
        VERIFY(noexcept(++a[0]));
        VERIFY(noexcept(--a[0]));
        VERIFY(noexcept(a[0]++));
        VERIFY(noexcept(a[0]--));
    }
    {
        auto k0 = a == a;
        const auto &k1 = k0;
        bool x = false; if (x) {}
        VERIFY(noexcept(k0[0]));
        VERIFY(noexcept(x = k1[0]));
        VERIFY(noexcept(x = k0[0]));
        VERIFY(noexcept(k0[0] = 1));
    }
}
