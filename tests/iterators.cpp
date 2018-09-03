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

#include "unittest.h"

TEST_TYPES(V, check_return_types, concat<AllVectors, SimdArrayList, AllMasks>)
{
    V a{};
    auto &&it = begin(a);
    using It = typename std::decay<decltype(it)>::type;

    // InputIterator:
    VERIFY((std::is_convertible<decltype(it == it), bool>::value));
    VERIFY((std::is_convertible<decltype(it != it), bool>::value));
    COMPARE(typeid(*it), typeid(typename It::reference));
    VERIFY((std::is_convertible<decltype(*it), typename It::value_type>::value));
    // can't test it->m because it points to fundamental types, which have no members
    // COMPARE(typeid(it->m), typeid());
    COMPARE(typeid(++it), typeid(It &));
    VERIFY((std::is_convertible<decltype(*it++), typename It::value_type>::value));

    // OutputIterator has no additional return type requirements

    // ForwardIterator:
    VERIFY(std::is_default_constructible<It>::value);
    // the lvalue-ref guarantee for reference does not (i.e. cannot) hold
    COMPARE(typeid(it++), typeid(It));
    COMPARE(typeid(*it++), typeid(typename It::reference));

    // BidirectionalIterator:
    COMPARE(typeid(--it), typeid(It &));
    VERIFY((std::is_convertible<decltype(it--), const It &>::value));
    COMPARE(typeid(*it--), typeid(typename It::reference));

    // RandomAccessIterator:
    COMPARE(typeid(it += 1), typeid(It &));
    COMPARE(typeid(it + 1), typeid(It));
    COMPARE(typeid(1 + it), typeid(It));
    COMPARE(typeid(it -= 1), typeid(It &));
    COMPARE(typeid(it - 1), typeid(It));
    COMPARE(typeid(it - it), typeid(typename It::difference_type));
    COMPARE(typeid(it[0]), typeid(typename It::reference));
    COMPARE(typeid(it < it), typeid(bool));
    COMPARE(typeid(it > it), typeid(bool));
    COMPARE(typeid(it >= it), typeid(bool));
    COMPARE(typeid(it <= it), typeid(bool));
}

TEST_TYPES(V, input_iterator, concat<AllVectors, SimdArrayList>)
{
    using T = typename V::EntryType;
    V a = V([](int n) { return n; });
#if defined Vc_GCC && Vc_GCC < 0x40900
    // work around miscompilation
    asm("" : "+m"(a));
#endif
    auto k = a == 0;
    {
        auto &&it = cbegin(a);
        VERIFY(it != cend(a));
        VERIFY(!(it != it));
        COMPARE(*it, T(0));
        int n = 0;
        while (++it != cend(a)) {
            COMPARE(*it, T(++n)) << a;
        }
    }
    {
        auto &&it = begin(a);
        VERIFY(it != end(a));
        VERIFY(!(it != it));
        COMPARE(*it, T(0));
        int n = 0;
        while (++it != end(a)) {
            COMPARE(*it, T(++n));
        }
    }
    {
        auto &&it = cbegin(k);
        VERIFY(it != cend(k));
        VERIFY(!(it != it));
        COMPARE(*it, true);
        int n = 0;
        while (++it != cend(k)) {
            COMPARE(*it, false) << ++n;
        }
    }
    {
        auto &&it = begin(k);
        VERIFY(it != end(k));
        VERIFY(!(it != it));
        COMPARE(*it, true);
        while (++it != end(k)) {
            COMPARE(*it, false);
        }
    }
}

TEST_TYPES(V, output_iterator, concat<AllVectors, SimdArrayList>)
{
    using T = typename V::EntryType;
    V a = V([](int n) { return n; });
#if defined Vc_GCC && Vc_GCC < 0x40900
    // work around miscompilation
    asm("" : "+m"(a));
#endif
    auto k = a == 0;
    {
        auto it = begin(a);
        int n = 0;
        for (; it != end(a); ++it) {
            COMPARE(*it, T(n++));
            *it = T(n);
        }
        n = 1;
        for (it = begin(a); it != end(a); ++it) {
            COMPARE(*it, T(n++));
        }
    }
    {
        auto it = begin(k);
        int n = 0;
        for (; it != end(k); ++it) {
            *it = n++ & 1;
        }
        n = 0;
        for (it = begin(k); it != end(k); ++it) {
            COMPARE(*it, n++ & 1);
        }
    }
}

TEST_TYPES(V, forward_iterator, concat<AllVectors, SimdArrayList>)
{
    using T = typename V::EntryType;
    V a = V([](int n) { return n; });
    auto &&it = begin(a);
    using It = typename std::decay<decltype(it)>::type;
    T value = *it;
    (void)++It(it); // increment a temporary iterator copy
    COMPARE(*it, value);
}

TEST_TYPES(V, range_for, AllVectors)
{
    typedef typename V::EntryType T;
    typedef typename V::Mask M;

    {
        V x = V(0);
        for (auto &&i : x) {
            COMPARE(i, T(0));
            VERIFY(!(std::is_assignable<decltype((i)), T>::value));
        }
        x = V([](int i) { return i; });
        int n = 0;
        for (T i : x) {
            COMPARE(i, T(n++));
            i = 0;
        }
        COMPARE(x, V([](int i) { return i; }));
    }

    {
        M m{true};
        for (auto &&i : m) {
            VERIFY(i);
            VERIFY(!(std::is_assignable<decltype((i)), bool>::value));
        }
        for (auto i : static_cast<const M &>(m)) {
            VERIFY(i);
            i = false;
            VERIFY(!i);
        }
        for (bool i : m) {
            VERIFY(i);
        }
    }

    for_all_masks(V, mask) {
        int count = 0;
        V test = V(0);
        for (size_t i : where(mask)) {
            VERIFY(i < V::Size);
            test[i] = T(1);
            ++count;
        }
        COMPARE(test == V(1), mask);
        COMPARE(count, mask.count());
    }
}

// vim: foldmethod=marker
