/*{{{
    Copyright © 2013-2015 Matthias Kretz <kretz@kde.org>

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
#include <Vc/array>
#include <Vc/vector>

template <class T> T inline seq()
{
    return T([](int n) -> typename T::value_type { return n; });
}

TEST_TYPES(V, init, AllVectors)
{
    typedef typename V::EntryType T;
    Vc::array<T, 256> data;
    Vc::vector<T> data2(256);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = i;
        data2[i] = i;
    }

    std::initializer_list<T> initList1 = {T(), T()};
    const std::initializer_list<T> initList2 = {T(), T()};

    Vc::vector<T> ctorTest1(6, T());
    Vc::vector<T> ctorTest2{T(), T()};
    Vc::vector<T> ctorTest3 = initList1;
    Vc::vector<T> ctorTest4 = initList2;
    Vc::vector<T> ctorTest5(initList1);
    Vc::vector<T> ctorTest6(initList2);
    COMPARE(ctorTest1.size(), 6u);
    COMPARE(ctorTest2.size(), 2u);
    COMPARE(ctorTest3.size(), 2u);
    COMPARE(ctorTest4.size(), 2u);
    COMPARE(ctorTest5.size(), 2u);
    COMPARE(ctorTest6.size(), 2u);
}

template <typename V>
static typename std::enable_if<Vc::is_unsigned<V>::value, V>::type positiveRandom()
{
    return V::Random();
}
template <typename V>
static typename std::enable_if<!Vc::is_unsigned<V>::value, V>::type positiveRandom()
{
    return abs(V::Random());
}

template <typename V, int Modulo> static V randomIndexes()
{
    V indexes = positiveRandom<V>() % Modulo;
    for (std::size_t i = 0; i < V::Size; ++i) {
        while ((indexes == static_cast<int>(indexes[i])).count() !=
               1) {  // the static_cast prevents an ICE with GCC 4.8.1
            indexes[i] = (indexes[i] + 1) % Modulo;
        }
    }
    if (V::size() > 1) {
        VERIFY(none_of(indexes.sorted() == indexes.sorted().rotated(1)))
            << "\nindexes must not contain duplicate values, otherwise the scatter tests "
               "will fail without actually doing anything wrong!\nindexes: " << indexes;
    }
    return indexes;
}

TEST_TYPES(V, gathers, AllVectors)
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType IT;
    Vc::array<T, 256> data;
    Vc::vector<T> data2(256);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = i;
        data2[i] = i;
    }

    static_assert(Vc::Common::is_valid_indexvector_<const IT &>::value, "");

    V test = data[seq<IT>()];
    COMPARE(test, seq<V>());
    test = data2[seq<IT>()];
    COMPARE(test, seq<V>());

    for (std::size_t repetition = 0; repetition < 1024 / V::Size; ++repetition) {
        const IT indexes = randomIndexes<IT, 256>();
        test = data[indexes];
        COMPARE(test, Vc::simd_cast<V>(indexes));
        test = data2[indexes];
        COMPARE(test, Vc::simd_cast<V>(indexes));
    }
}

template <typename T, std::size_t N>
std::ostream &operator<<(std::ostream &s, const Vc::array<T, N> &data)
{
    for (std::size_t i = 0; i < N; ++i) {
        s << data[i];
        if (i % 32 == 31) {
            s << '\n';
        } else {
            s << ' ';
        }
    }
    s << std::setw(0);
    return s;
}

TEST_TYPES(V, scatters, AllVectors)
{
    static_assert(std::is_same<decltype(V() + 1), V>::value, "");
    typedef typename V::EntryType T;
    typedef typename V::IndexType IT;
    Vc::array<T, 256> data1;
    Vc::vector<T> data2(256);
    std::fill_n(&data1[0], 256, 0);
    std::fill_n(&data2[0], 256, 0);

    data1[seq<IT>()] = seq<V>();
    data2[seq<IT>()] = seq<V>();

    for (size_t i = 0; i < V::Size; ++i) {
        COMPARE(data1[i], T(i));
        COMPARE(data2[i], T(i));
    }

    for (std::size_t repetition = 0; repetition < 1024 / V::Size; ++repetition) {
        std::fill_n(&data1[0], 256, 0);
        std::fill_n(&data2[0], 256, 0);

        const IT indexes = randomIndexes<IT, 256>();
        data1[indexes] = seq<V>() + 1;
        data2[indexes] = seq<V>() + 1;

        //std::cerr << data1 << '\n';

        for (size_t i = 0; i < V::Size; ++i) {
            COMPARE(data1[indexes[i]], T(i + 1)) << ", indexes: " << indexes
                                                 << ", i: " << i << ", data1:\n" << data1;
            COMPARE(data2[indexes[i]], T(i + 1)) << ", indexes: " << indexes
                                                 << ", i: " << i << ", data2:\n" << data2;
        }
    }
}

template <typename T, size_t Align = std::is_arithmetic<T>::value ? sizeof(T) : alignof(T)>
struct alignas(Align > alignof(double) ? Align : alignof(double)) S
{
    void operator=(int x)
    {
        a = x;
        b = x + 1;
        c = x + 2;
    }
    double x0;
    T a;
    char x1;
    T b;
    short x2;
    T c;
    char x3;
};

TEST_TYPES(V, structGathers, AllVectors)
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType IT;
    Vc::array<S<T>, 256> data1;
    Vc::array<S<S<T>>, 256> data2;
    Vc::array<S<S<S<T>>>, 256> data3;
    Vc::array<S<S<S<S<T>>>>, 256> data4;
    for (int i = 0; i < 256; ++i) {
        data1[i] = i;
        data2[i] = i;
        data3[i] = i;
        data4[i] = i;
    }

    V test;

    test = data1[seq<IT>()][&S<T>::a];
    COMPARE(test, seq<V>());
    test = data1[seq<IT>()][&S<T>::b];
    COMPARE(test, seq<V>() + 1);
    test = data1[seq<IT>()][&S<T>::c];
    COMPARE(test, seq<V>() + 2);

    test = -V(data1[seq<IT>()][&S<T>::a]);
    COMPARE(test, -seq<V>());
    test = data1[seq<IT>()][&S<T>::a] + V(1);
    COMPARE(test, seq<V>() + 1);
    // TODO: should this work? if yes, how? a gather needs to know the type it gets converted to,
    // applying a unary operator before conversion implies that all operations must be delayed until
    // conversion. => expression templates?
    //test = -data1[seq<IT>()][&S<T>::a];
    //COMPARE(test, -seq<V>());

    test = data2[seq<IT>()][&S<S<T>>::a][&S<T>::a];
    COMPARE(test, seq<V>());
    test = data2[seq<IT>()][&S<S<T>>::b][&S<T>::c];
    COMPARE(test, seq<V>() + 3);

    test = data3[seq<IT>()][&S<S<S<T>>>::a][&S<S<T>>::a][&S<T>::a];
    COMPARE(test, seq<V>());
    test = data3[seq<IT>()][&S<S<S<T>>>::b][&S<S<T>>::c][&S<T>::b];
    COMPARE(test, seq<V>() + 4);

    test = data4[seq<IT>()][&S<S<S<S<T>>>>::a][&S<S<S<T>>>::a][&S<S<T>>::a][&S<T>::a];
    COMPARE(test, seq<V>());
    test = data4[seq<IT>()][&S<S<S<S<T>>>>::a][&S<S<S<T>>>::a][&S<S<T>>::a][&S<T>::b];
    COMPARE(test, seq<V>() + 1);
    test = data4[seq<IT>()][&S<S<S<S<T>>>>::a][&S<S<S<T>>>::a][&S<S<T>>::b][&S<T>::a];
    COMPARE(test, seq<V>() + 1);
    test = data4[seq<IT>()][&S<S<S<S<T>>>>::c][&S<S<S<T>>>::b][&S<S<T>>::b][&S<T>::c];
    COMPARE(test, seq<V>() + 6);

    for (std::size_t repetition = 0; repetition < 1024 / V::Size; ++repetition) {
        const IT indexes = randomIndexes<IT, 256>();
        test = data1[indexes][&S<T>::b];
        COMPARE(test, Vc::simd_cast<V>(indexes + 1));

        test = data2[indexes][&S<S<T>>::c][&S<T>::b];
        COMPARE(test, Vc::simd_cast<V>(indexes + 3));

        test = data3[indexes][&S<S<S<T>>>::b][&S<S<T>>::c][&S<T>::b];
        COMPARE(test, Vc::simd_cast<V>(indexes + 4));

        test = data4[indexes][&S<S<S<S<T>>>>::c][&S<S<S<T>>>::b][&S<S<T>>::c][&S<T>::b];
        COMPARE(test, Vc::simd_cast<V>(indexes + 6));
    }
}

TEST_TYPES(V, subarrayGathers, AllVectors)
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType IT;
    Vc::array<Vc::array<T, 256>, 256> data1;
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 256; ++j) {
            data1[i][j] = i + j;
        }
    }

    V test;

    test = data1[seq<IT>()][0];
    COMPARE(test, seq<V>());
    test = data1[seq<IT>()][255];
    COMPARE(test, seq<V>() + 255);
    test = data1[254][seq<IT>()];
    COMPARE(test, seq<V>() + 254);

    test = data1[seq<IT>()][seq<IT>()];
    COMPARE(test, seq<V>() * 2);

    for (std::size_t repetition = 0; repetition < 1024 / V::Size; ++repetition) {
        const IT indexes1 = randomIndexes<IT, 256>();
        const IT indexes2 = randomIndexes<IT, 256>();

        test = data1[indexes1][indexes2];
        COMPARE(test, Vc::simd_cast<V>(indexes1 + indexes2));
    }

    Vc::vector<std::array<T, 256>> data2(64);
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 256; ++j) {
            data2[i][j] = i + j;
        }
    }

    test = data2[seq<IT>()][0];
    COMPARE(test, seq<V>());
}

TEST_TYPES(V, fixedWidthGatherScatter4, SimdArrays<4>)
{
    typedef typename V::EntryType T;
    Vc::array<T, 256> data;
    Vc::vector<T> data2(256);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = i;
        data2[i] = i + 1;
    }
    V test;

    const std::initializer_list<int> indexes = { 0, 5, 8, 3 };
    test = data[indexes];
    const V reference = {0, 5, 8, 3};
    COMPARE(test, reference);
    test = data2[indexes];
    COMPARE(test, reference + 1);

    const int indexes2[4] = {3, 8, 5, 0};
    test = data[indexes2];
    const V reference2 = {3, 8, 5, 0};
    COMPARE(test, reference2);
    test = data2[indexes2];
    COMPARE(test, reference2 + 1);

    /*
    test = V::Random();
    data[indexes] = test;
    data2[indexes] = test;

    std::size_t i = 0;
    for (const auto x : indexes) {
        COMPARE(data[x], test[i]);
        COMPARE(data2[x], test[i]);
        ++i;
    }
    */
}

TEST_TYPES(V, fixedWidthGatherScatter32, SimdArrays<32>)
{
    typedef typename V::EntryType T;
    Vc::array<T, 256> data;
    Vc::vector<T> data2(256);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = i;
        data2[i] = i;
    }
    V test;
    const std::initializer_list<int> indexes = {73, 17, 15, 0,  38, 74, 25, 41, 7, 28, 68,
                                                2,  23, 98, 5,  87, 72, 18, 16, 1, 39, 75,
                                                26, 42, 8,  29, 69, 3,  24, 99, 6, 88};
    test = data[indexes];
    const V reference = {73, 17, 15, 0, 38, 74, 25, 41, 7, 28, 68, 2, 23, 98, 5, 87,
                         72, 18, 16, 1, 39, 75, 26, 42, 8, 29, 69, 3, 24, 99, 6, 88};
    COMPARE(test, reference);
    test = data2[indexes];
    COMPARE(test, reference);
}

TEST(promotionOfIndexVectorType)
{
    Vc::vector<Vc::array<int, 1024>> data;
    data.resize(1024);
    int *ptr = &data[0][0];
    for (std::size_t i = 0; i < 1024 * 1024; ++i) {
        ptr[i] = i;
    }

    const Vc::short_v indexes = 1023 - seq<Vc::short_v>();
    Vc::int_v reference = Vc::simd_cast<Vc::int_v>(indexes) * 1024;
    Vc::int_v test = data[indexes][0];
    COMPARE(test, reference);
}
