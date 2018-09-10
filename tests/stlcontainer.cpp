/*{{{
    Copyright (C) 2012-2015 Matthias Kretz <kretz@kde.org>

    Permission to use, copy, modify, and distribute this software
    and its documentation for any purpose and without fee is hereby
    granted, provided that the above copyright notice appear in all
    copies and that both that the copyright notice and this
    permission notice and warranty disclaimer appear in supporting
    documentation, and that the name of the author not be used in
    advertising or publicity pertaining to distribution of the
    software without specific, written prior permission.

    The author disclaim all warranties with regard to this
    software, including all implied warranties of merchantability
    and fitness.  In no event shall the author be liable for any
    special, indirect or consequential damages or any damages
    whatsoever resulting from loss of use, data or profits, whether
    in an action of contract, negligence or other tortious action,
    arising out of or in connection with the use or performance of
    this software.

}}}*/

#include "unittest.h"
#include <Vc/Allocator>
#include <vector>
#include <array>
#include <forward_list>
#include <list>
#include <deque>

#include "../Vc/common/macros.h"

template<typename Vec> size_t alignmentMask()
{
    if (Vec::Size == 1) {
        // on 32bit the maximal alignment is 4 Bytes, even for 8-Byte doubles.
        return std::min(sizeof(void*), sizeof(typename Vec::EntryType)) - 1;
    }
    // AVX::VectorAlignment is too large
    return std::min<size_t>(sizeof(Vec), Vc::VectorAlignment) - 1;
}

template<typename T> struct SomeStruct { char a; T x; };

TEST_TYPES(V, stdVectorAlignment, AllVectors)
{
    const size_t mask = alignmentMask<V>();
    const char *const null = 0;

    std::vector<V> v(11);
    for (int i = 0; i < 11; ++i) {
        COMPARE((reinterpret_cast<char *>(&v[i]) - null) & mask, 0u) << "&v[i] = " << &v[i] << ", mask = " << mask << ", i = " << i;
    }

    std::vector<SomeStruct<V>, Vc::Allocator<SomeStruct<V> > > v2(11);
    for (int i = 0; i < 11; ++i) {
        COMPARE((reinterpret_cast<char *>(&v2[i]) - null) & mask, 0u) << "&v2[i] = " << &v2[i] << ", mask = " << mask << ", i = " << i;
    }

    std::vector<V> v3(v);
    std::vector<SomeStruct<V>, Vc::Allocator<SomeStruct<V> > > v4(v2);

    typedef typename V::EntryType T;
    for (int i = 1; i < 100; ++i) {
        std::vector<T, Vc::Allocator<T>> v5(i);
        const size_t expectedAlignment = alignof(V);
        COMPARE(reinterpret_cast<std::uintptr_t>(v5.data()) & (expectedAlignment - 1), 0u)
            << "expectedAlignment: " << expectedAlignment;
    }
}

template <typename V, typename Container, std::size_t... Indexes>
void listInitializationImpl(Vc::index_sequence<Indexes...>)
{
    typedef typename V::EntryType T;
    const auto data = Vc::makeContainer<Container>({T(Indexes + 1)...});
    V reference = V([](int n) { return n + 1; });
    for (const auto &v : data) {
        reference.setZero(reference > int(sizeof...(Indexes)));
        COMPARE(v, reference) << vir::typeToString<Container>() << " -> "
                              << vir::typeToString<decltype(data)>();
        reference += int(V::size());
    }
}
TEST_TYPES(V, listInitialization, AllVectors)
{
    listInitializationImpl<V, std::vector<V>>(Vc::make_index_sequence<9>());
    listInitializationImpl<V, std::vector<V>>(Vc::make_index_sequence<3>());
    listInitializationImpl<V, std::array<V, 9>>(Vc::make_index_sequence<9>());
    listInitializationImpl<V, std::array<V, 3>>(Vc::make_index_sequence<3>());
#ifndef Vc_MSVC
    listInitializationImpl<V, std::deque<V>>(Vc::make_index_sequence<9>());
    listInitializationImpl<V, std::deque<V>>(Vc::make_index_sequence<3>());
#endif

    // The following two crash (at least with AVX). Probably unaligned memory access.
    //listInitialization<V, std::forward_list<V>>();
    //listInitialization<V, std::list<V>>();
}

#ifdef Vc_CXX14
TEST_TYPES(V, simdForEach, AllVectors)
{
    typedef typename V::EntryType T;
    std::vector<T> data;
    data.resize(100);

    for (int variant = 0; variant < 2; ++variant) {
        std::iota(data.begin(), data.end(), T(0));
        T reference = 1;
        int called_with_scalar = 0;
        int called_with_V = 0;
        int position = 1;

        auto &&test1 = [&](auto &x) {
            const auto ref = reference + x.IndexesFromZero();
            COMPARE(ref, x);
            reference += x.Size;
            x += 1;
            if (std::is_same<decltype(x), Vc::Scalar::Vector<T> &>::value) {
                ++called_with_scalar;
            }
            if (std::is_same<decltype(x), V &>::value) {
                ++called_with_V;
            }
            static_assert(std::is_same<decltype(x), Vc::Scalar::Vector<T> &>::value ||
                              std::is_same<decltype(x), V &>::value,
                          "wut?");
            for (std::size_t i = 0; i < x.Size; ++i) {
                data[position++] += T(2);  // modify the container directly - if it is not
                                           // undone by simd_for_each we have a bug
            }
        };
        auto &&test2 = [&](auto x) {
            const auto ref = reference + x.IndexesFromZero();
            COMPARE(ref, x);
            reference += x.Size;
            x += 1;
            for (std::size_t i = 0; i < x.Size; ++i) {
                data[position++] += T(2);  // modify the container directly - if it is
                                           // undone by simd_for_each we have a bug
            }
        };
        auto &&test3 = [&reference](auto x) {
            const auto ref = reference + x.IndexesFromZero();
            COMPARE(ref, x) << "if ref == x + 2 then simd_for_each wrote back the "
                               "closure argument, even though it should not have";
            reference += x.Size;
        };

        auto &&for_each = [&](auto test) {
            auto b = std::next(data.begin());
            if (variant == 0) {
                Vc::simd_for_each(b, data.end(), test);
            } else {
                Vc::simd_for_each_n(b, data.size() - 1, test);
            }
        };
        for_each(test1);
        VERIFY(called_with_scalar > 0);
        VERIFY(called_with_V > 0);
        if (Vc::Scalar::is_vector<V>::value) {
            // in this case called_with_V and called_with_scalar will have been
            // incremented both on every call
            COMPARE(called_with_V, called_with_scalar);
            COMPARE(called_with_scalar, int(data.size() - 1));
        } else {
            COMPARE(called_with_V * V::Size + called_with_scalar, data.size() - 1);
        }

        reference = 2;
        position = 1;
        for_each(test2);
        reference = 4;
        for_each(test3);
    }
}
#endif
