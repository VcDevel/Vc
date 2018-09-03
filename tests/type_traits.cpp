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

#include "unittest.h"
#include <Vc/type_traits>
#include <memory>
#include <list>
#include <map>
#include <unordered_map>

using Vc::float_v;
using Vc::double_v;
using Vc::int_v;
using Vc::uint_v;
using Vc::short_v;
using Vc::ushort_v;

TEST_TYPES(V, isIntegral, AllVectors)
{
    using T = typename V::EntryType;
    COMPARE(Vc::is_integral<V>::value, std::is_integral<T>::value);
}

TEST_TYPES(V, isFloatingPoint, AllVectors)
{
    using T = typename V::EntryType;
    COMPARE(Vc::is_floating_point<V>::value, std::is_floating_point<T>::value);
}

TEST_TYPES(V, isSigned, AllVectors)
{
    using T = typename V::EntryType;
    COMPARE(Vc::is_signed<V>::value, std::is_signed<T>::value);
    COMPARE(Vc::is_unsigned<V>::value, std::is_unsigned<T>::value);
}

TEST_TYPES(V, hasSubscript, AllVectors)
{
    VERIFY(Vc::Traits::has_subscript_operator<V>::value);
}

TEST_TYPES(V, hasMultiply, AllVectors)
{
    VERIFY(Vc::Traits::has_multiply_operator<V>::value);
}

TEST_TYPES(V, hasEquality, AllVectors)
{
    VERIFY(Vc::Traits::has_equality_operator<V>::value);
}

TEST_TYPES(V, isSimdMask, AllVectors)
{
    VERIFY(!Vc::is_simd_mask<V>::value);
    VERIFY( Vc::is_simd_mask<typename V::Mask>::value);
}

TEST_TYPES(V, isSimdVector, AllVectors)
{
    VERIFY( Vc::is_simd_vector<V>::value);
    VERIFY(!Vc::is_simd_vector<typename V::Mask>::value);
}

template <typename T> void hasContiguousStorageImpl(T &&, const char *type)
{
    VERIFY(Vc::Traits::has_contiguous_storage<T>::value) << type;
}

TEST(hasContiguousStorage)
{
    std::unique_ptr<int[]> a(new int[3]);
    int b[3] = {1, 2, 3};
    const std::unique_ptr<int[]> c(new int[3]);
    const int d[3] = {1, 2, 3};
    auto &&e = {1, 2, 3};
    const auto &f = {1, 2, 3};
    std::vector<int> g;
    std::array<int, 3> h;
    hasContiguousStorageImpl(a.get(), "T[]");
    hasContiguousStorageImpl(a, "std::unique_ptr<T[]>");
    hasContiguousStorageImpl(&a[0], "T *");
    hasContiguousStorageImpl(b, "T[3]");
    hasContiguousStorageImpl(c.get(), "const T[]");
    hasContiguousStorageImpl(c, "std::unique_ptr<const T[]>");
    hasContiguousStorageImpl(&c[0], "const T *");
    hasContiguousStorageImpl(d, "const T[3]");
    hasContiguousStorageImpl(e, "std::initializer_list 1");
    hasContiguousStorageImpl(f, "std::initializer_list 2");
    hasContiguousStorageImpl(g, "std::vector<int>");
    hasContiguousStorageImpl(g.begin(), "std::vector<int>::iterator");
    hasContiguousStorageImpl(g.cbegin(), "std::vector<int>::iterator");
    hasContiguousStorageImpl(h, "std::array<int, 3>");
    hasContiguousStorageImpl(h.begin(), "std::array<int, 3>::iterator");
    hasContiguousStorageImpl(h.cbegin(), "std::array<int, 3>::iterator");
    VERIFY(!(Vc::Traits::has_contiguous_storage<std::list<int>>::value));
    VERIFY(!(Vc::Traits::has_contiguous_storage<std::list<int>::iterator>::value));
    VERIFY(!(Vc::Traits::has_contiguous_storage<std::list<int>::const_iterator>::value));
    VERIFY(!(Vc::Traits::has_contiguous_storage<std::map<int, int>>::value));
    VERIFY(!(Vc::Traits::has_contiguous_storage<std::map<int, int>::iterator>::value));
    VERIFY(!(Vc::Traits::has_contiguous_storage<std::unordered_map<int, int>>::value));
    VERIFY(!(Vc::Traits::has_contiguous_storage<std::unordered_map<int, int>::iterator>::value));
}

struct F0 {
    template <typename T> void operator()(T &) const {}
};
struct F1 {
    template <typename T> void operator()(const T &) const {}
};
struct F2 {
    template <typename T> void operator()(T) const {}
};
struct F3 {
    template <typename T> void operator()(const T) const {}
};
struct F4 {
    // this could be a reference argument but with move semantics. Then, the caller cannot
    // see changes to the argument as the variable is in an undefined state after the
    // call.
    // But as forwarding reference T can also bind as T &, in which case the argument is
    // mutable.
    template <typename T> void operator()(T &&) const {}
};
void fun1(int &) {}
void fun2(const int &) {}

TEST(test_is_functor_argument_immutable)
{
    using Vc::Traits::is_functor_argument_immutable;
    VERIFY(!(is_functor_argument_immutable<F0, int>::value));
    VERIFY( (is_functor_argument_immutable<F1, int>::value));
    VERIFY( (is_functor_argument_immutable<F2, int>::value));
    VERIFY( (is_functor_argument_immutable<F3, int>::value));
    VERIFY(!(is_functor_argument_immutable<F4, int>::value));

    VERIFY(!(is_functor_argument_immutable<decltype(fun1), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype(fun2), int>::value));

    int x = 0;
    auto &&  int_lambda   = [&](int) { x += 1; };
    auto &&  int_lambda_l = [&](int &y) { x += 1; y += 1; };
    auto &&  int_lambda_r = [&](int &&) { x += 1; };
    auto &&c_int_lambda   = [&](const int) { x += 1; };
    auto &&c_int_lambda_l = [&](const int &) { x += 1; };
    auto &&c_int_lambda_r = [&](const int &&) { x += 1; };
    auto &&v_int_lambda   = [&](volatile int) { x += 1; };
    auto &&v_int_lambda_l = [&](volatile int &) { x += 1; };
    auto &&v_int_lambda_r = [&](volatile int &&) { x += 1; };
    auto &&cv_int_lambda   = [&](const volatile int) { x += 1; };
    auto &&cv_int_lambda_l = [&](const volatile int &) { x += 1; };
    auto &&cv_int_lambda_r = [&](const volatile int &&) { x += 1; };
    VERIFY( (is_functor_argument_immutable<decltype(   int_lambda  ), int>::value));
    VERIFY(!(is_functor_argument_immutable<decltype(   int_lambda_l), int>::value));
    VERIFY(!(is_functor_argument_immutable<decltype(   int_lambda_r), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype( c_int_lambda  ), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype( c_int_lambda_l), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype( c_int_lambda_r), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype( v_int_lambda  ), int>::value));
    VERIFY(!(is_functor_argument_immutable<decltype( v_int_lambda_l), int>::value));
    VERIFY(!(is_functor_argument_immutable<decltype( v_int_lambda_r), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype(cv_int_lambda  ), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype(cv_int_lambda_l), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype(cv_int_lambda_r), int>::value));
#ifdef Vc_CXX14
    auto &&auto_lambda = [&](auto) { x += 1; };
    auto &&auto_lambda_l = [&](auto &y) { x += 1; y += 1; };
    auto &&auto_lambda_r = [&](auto &&) { x += 1; };
    auto &&c_auto_lambda   = [&](const auto) { x += 1; };
    auto &&c_auto_lambda_l = [&](const auto &) { x += 1; };
    auto &&c_auto_lambda_r = [&](const auto &&) { x += 1; };
    auto &&v_auto_lambda   = [&](volatile auto) { x += 1; };
    auto &&v_auto_lambda_l = [&](volatile auto &) { x += 1; };
    auto &&v_auto_lambda_r = [&](volatile auto &&) { x += 1; };
    auto &&cv_auto_lambda   = [&](const volatile auto) { x += 1; };
    auto &&cv_auto_lambda_l = [&](const volatile auto &) { x += 1; };
    auto &&cv_auto_lambda_r = [&](const volatile auto &&) { x += 1; };
    VERIFY( (is_functor_argument_immutable<decltype(auto_lambda  ), int>::value));
    VERIFY(!(is_functor_argument_immutable<decltype(auto_lambda_l), int>::value));
    VERIFY(!(is_functor_argument_immutable<decltype(auto_lambda_r), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype(c_auto_lambda  ), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype(c_auto_lambda_l), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype(c_auto_lambda_r), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype(v_auto_lambda  ), int>::value));
    VERIFY(!(is_functor_argument_immutable<decltype(v_auto_lambda_l), int>::value));
    VERIFY(!(is_functor_argument_immutable<decltype(v_auto_lambda_r), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype(cv_auto_lambda  ), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype(cv_auto_lambda_l), int>::value));
    VERIFY( (is_functor_argument_immutable<decltype(cv_auto_lambda_r), int>::value));
#endif
}

TEST(test_is_output_iterator)
{
    VERIFY( (Vc::Traits::is_output_iterator<std::vector<int>::iterator>::value));
    VERIFY(!(Vc::Traits::is_output_iterator<std::vector<int>::const_iterator>::value));
    VERIFY( (Vc::Traits::is_output_iterator<std::ostream_iterator<int>>::value));
    VERIFY(!(Vc::Traits::is_output_iterator<std::istream_iterator<int>>::value));
}
