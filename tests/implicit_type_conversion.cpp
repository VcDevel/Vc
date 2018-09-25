/*  This file is part of the Vc library. {{{
Copyright © 2012-2015 Matthias Kretz <kretz@kde.org>

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
#include "virtest/vir/metahelpers.h"

using namespace Vc;

static_assert(!std::is_convertible< float_v, double_v>::value, "!std::is_convertible< float_v, double_v>");
static_assert(!std::is_convertible<   int_v, double_v>::value, "!std::is_convertible<   int_v, double_v>");
static_assert(!std::is_convertible<  uint_v, double_v>::value, "!std::is_convertible<  uint_v, double_v>");
static_assert(!std::is_convertible< short_v, double_v>::value, "!std::is_convertible< short_v, double_v>");
static_assert(!std::is_convertible<ushort_v, double_v>::value, "!std::is_convertible<ushort_v, double_v>");

static_assert(!std::is_convertible<double_v,  float_v>::value, "!std::is_convertible<double_v,  float_v>");
static_assert(!std::is_convertible<   int_v,  float_v>::value, "!std::is_convertible<   int_v,  float_v>");
static_assert(!std::is_convertible<  uint_v,  float_v>::value, "!std::is_convertible<  uint_v,  float_v>");
static_assert(!std::is_convertible< short_v,  float_v>::value, "!std::is_convertible< short_v,  float_v>");
static_assert(!std::is_convertible<ushort_v,  float_v>::value, "!std::is_convertible<ushort_v,  float_v>");

static_assert(!std::is_convertible<double_v,    int_v>::value, "!std::is_convertible<double_v,    int_v>");
static_assert(!std::is_convertible< float_v,    int_v>::value, "!std::is_convertible< float_v,    int_v>");
static_assert( std::is_convertible<  uint_v,    int_v>::value, " std::is_convertible<  uint_v,    int_v>");
static_assert(!std::is_convertible< short_v,    int_v>::value, "!std::is_convertible< short_v,    int_v>");
static_assert(!std::is_convertible<ushort_v,    int_v>::value, "!std::is_convertible<ushort_v,    int_v>");

static_assert(!std::is_convertible<double_v,   uint_v>::value, "!std::is_convertible<double_v,   uint_v>");
static_assert(!std::is_convertible< float_v,   uint_v>::value, "!std::is_convertible< float_v,   uint_v>");
static_assert( std::is_convertible<   int_v,   uint_v>::value, " std::is_convertible<   int_v,   uint_v>");
static_assert(!std::is_convertible< short_v,   uint_v>::value, "!std::is_convertible< short_v,   uint_v>");
static_assert(!std::is_convertible<ushort_v,   uint_v>::value, "!std::is_convertible<ushort_v,   uint_v>");

static_assert(!std::is_convertible<double_v,  short_v>::value, "!std::is_convertible<double_v,  short_v>");
static_assert(!std::is_convertible< float_v,  short_v>::value, "!std::is_convertible< float_v,  short_v>");
static_assert(!std::is_convertible<   int_v,  short_v>::value, "!std::is_convertible<   int_v,  short_v>");
static_assert(!std::is_convertible<  uint_v,  short_v>::value, "!std::is_convertible<  uint_v,  short_v>");
static_assert( std::is_convertible<ushort_v,  short_v>::value, " std::is_convertible<ushort_v,  short_v>");

static_assert(!std::is_convertible<double_v, ushort_v>::value, "!std::is_convertible<double_v, ushort_v>");
static_assert(!std::is_convertible< float_v, ushort_v>::value, "!std::is_convertible< float_v, ushort_v>");
static_assert(!std::is_convertible<   int_v, ushort_v>::value, "!std::is_convertible<   int_v, ushort_v>");
static_assert(!std::is_convertible<  uint_v, ushort_v>::value, "!std::is_convertible<  uint_v, ushort_v>");
static_assert( std::is_convertible< short_v, ushort_v>::value, " std::is_convertible< short_v, ushort_v>");

template <typename A, typename B, typename C>
Vc::enable_if<std::is_integral<A>::value && std::is_integral<B>::value, void>
integral_tests(A a, B b, C c)
{
#if !defined(Vc_GCC) || Vc_GCC != 0x40801
    // Skipping tests involving operator& because of a bug in GCC 4.8.1
    // (http://gcc.gnu.org/bugzilla/show_bug.cgi?id=57532)
    static_assert(std::is_same<decltype(a & b), C>::value, "incorrect return type deduction");
    COMPARE(typeid(a & b), typeid(C));
#endif
    static_assert(std::is_same<decltype(a | b), C>::value, "incorrect return type deduction");
    static_assert(std::is_same<decltype(a ^ b), C>::value, "incorrect return type deduction");
    COMPARE(typeid(a | b), typeid(C));
    COMPARE(typeid(a ^ b), typeid(C));
}
template <typename A, typename B, typename C>
Vc::enable_if<!(std::is_integral<A>::value && std::is_integral<B>::value), void>
integral_tests(A, B, C)
{
}
#define _TYPE_TEST(a, b, c)                                                              \
    {                                                                                    \
        integral_tests(a(), b(), c());                                                   \
        using logical_return = typename std::conditional<                                \
            std::is_fundamental<decltype(b())>::value, decltype(!a()),                   \
            typename std::conditional<std::is_fundamental<decltype(a())>::value,         \
                                      decltype(!b()), c::Mask>::type>::type;             \
        COMPARE(typeid(a() && b()), typeid(logical_return));                             \
        COMPARE(typeid(a() || b()), typeid(logical_return));                             \
        COMPARE(typeid(a() *  b()), typeid(c));                                          \
        COMPARE(typeid(a() /  b()), typeid(c));                                          \
        COMPARE(typeid(a() +  b()), typeid(c));                                          \
        COMPARE(typeid(a() -  b()), typeid(c));                                          \
        COMPARE(typeid(a() == b()), typeid(c::Mask));                                    \
        COMPARE(typeid(a() != b()), typeid(c::Mask));                                    \
        COMPARE(typeid(a() <= b()), typeid(c::Mask));                                    \
        COMPARE(typeid(a() >= b()), typeid(c::Mask));                                    \
        COMPARE(typeid(a() <  b()), typeid(c::Mask));                                    \
        static_assert(std::is_same<decltype(a() && b()), logical_return>::value, #a " && " #b " => " #c); \
        static_assert(std::is_same<decltype(a() || b()), logical_return>::value, #a " || " #b " => " #c); \
        static_assert(std::is_same<decltype(a() *  b()), c>::value, #a " *  " #b " => " #c); \
        static_assert(std::is_same<decltype(a() /  b()), c>::value, #a " /  " #b " => " #c); \
        static_assert(std::is_same<decltype(a() +  b()), c>::value, #a " +  " #b " => " #c); \
        static_assert(std::is_same<decltype(a() -  b()), c>::value, #a " -  " #b " => " #c); \
        static_assert(std::is_same<decltype(a() == b()), c::Mask>::value, #a " == " #b " => " #c "::Mask"); \
        static_assert(std::is_same<decltype(a() != b()), c::Mask>::value, #a " != " #b " => " #c "::Mask"); \
        static_assert(std::is_same<decltype(a() <= b()), c::Mask>::value, #a " <= " #b " => " #c "::Mask"); \
        static_assert(std::is_same<decltype(a() >= b()), c::Mask>::value, #a " >= " #b " => " #c "::Mask"); \
        static_assert(std::is_same<decltype(a() <  b()), c::Mask>::value, #a " <  " #b " => " #c "::Mask"); \
    }

#define TYPE_TEST(a, b, c) \
    _TYPE_TEST(a, b, c) \
    COMPARE(typeid(a() >  b()), typeid(c::Mask))

template<typename T>
struct TestImplicitCast {
    static bool test(const T &) { return  true; }
    static bool test(   ...   ) { return false; }
};

enum SomeEnum { EnumValue = 0 };
SomeEnum Enum() { return EnumValue; }

TEST(testImplicitTypeConversions)
{
    VERIFY( TestImplicitCast<     int>::test(double()));
    VERIFY( TestImplicitCast<     int>::test( float()));
    VERIFY( TestImplicitCast<     int>::test(  Enum()));
    VERIFY( TestImplicitCast<     int>::test( short()));
    VERIFY( TestImplicitCast<     int>::test(ushort()));
    VERIFY( TestImplicitCast<     int>::test(  char()));
    VERIFY( TestImplicitCast<     int>::test(  uint()));
    VERIFY( TestImplicitCast<     int>::test(  long()));
    VERIFY( TestImplicitCast<     int>::test( ulong()));
    VERIFY( TestImplicitCast<     int>::test(  bool()));
    VERIFY( TestImplicitCast<double_v>::test(double()));
    VERIFY( TestImplicitCast<double_v>::test( float()));
    VERIFY( TestImplicitCast<double_v>::test(   int()));
    VERIFY( TestImplicitCast< float_v>::test( float()));
    VERIFY( TestImplicitCast<   int_v>::test(   int()));
    VERIFY( TestImplicitCast<  uint_v>::test(  uint()));
    VERIFY( TestImplicitCast< short_v>::test( short()));
    VERIFY( TestImplicitCast<ushort_v>::test(ushort()));
    VERIFY(!TestImplicitCast<double_v>::test(nullptr));
    VERIFY(!TestImplicitCast< float_v>::test(nullptr));
    VERIFY(!TestImplicitCast<   int_v>::test(nullptr));
    VERIFY(!TestImplicitCast<  uint_v>::test(nullptr));
    VERIFY(!TestImplicitCast< short_v>::test(nullptr));
    VERIFY(!TestImplicitCast<ushort_v>::test(nullptr));

    TYPE_TEST( double_v,    double_v, double_v);
    TYPE_TEST( double_v,      double, double_v);
    TYPE_TEST( double_v,       float, double_v);
    TYPE_TEST( double_v,       short, double_v);
    TYPE_TEST( double_v,      ushort, double_v);
    TYPE_TEST( double_v,         int, double_v);
    TYPE_TEST( double_v,        uint, double_v);
    TYPE_TEST( double_v,        long, double_v);
    TYPE_TEST( double_v,       ulong, double_v);
    TYPE_TEST( double_v,       llong, double_v);
    TYPE_TEST( double_v,      ullong, double_v);
    TYPE_TEST( double_v,        Enum, double_v);
    TYPE_TEST( double_v,        bool, double_v);
    TYPE_TEST(   double,    double_v, double_v);
    TYPE_TEST(    float,    double_v, double_v);
    TYPE_TEST(    short,    double_v, double_v);
    TYPE_TEST(   ushort,    double_v, double_v);
    TYPE_TEST(      int,    double_v, double_v);
    TYPE_TEST(     uint,    double_v, double_v);
    TYPE_TEST(     long,    double_v, double_v);
    TYPE_TEST(    ulong,    double_v, double_v);
    TYPE_TEST(    llong,    double_v, double_v);
    TYPE_TEST(   ullong,    double_v, double_v);
    TYPE_TEST(     Enum,    double_v, double_v);
    TYPE_TEST(     bool,    double_v, double_v);
    // double_v done

    TYPE_TEST(  float_v,     float_v,  float_v);
    TYPE_TEST(  float_v,       float,  float_v);
    TYPE_TEST(  float_v,       short,  float_v);
    TYPE_TEST(  float_v,      ushort,  float_v);
    TYPE_TEST(  float_v,         int,  float_v);
    TYPE_TEST(  float_v,        uint,  float_v);
    TYPE_TEST(  float_v,        long,  float_v);
    TYPE_TEST(  float_v,       ulong,  float_v);
    TYPE_TEST(  float_v,       llong,  float_v);
    TYPE_TEST(  float_v,      ullong,  float_v);
    TYPE_TEST(  float_v,        Enum,  float_v);
    TYPE_TEST(  float_v,        bool,  float_v);
    TYPE_TEST(    float,     float_v,  float_v);
    TYPE_TEST(    short,     float_v,  float_v);
    TYPE_TEST(   ushort,     float_v,  float_v);
    TYPE_TEST(      int,     float_v,  float_v);
    TYPE_TEST(     uint,     float_v,  float_v);
    TYPE_TEST(     long,     float_v,  float_v);
    TYPE_TEST(    ulong,     float_v,  float_v);
    TYPE_TEST(    llong,     float_v,  float_v);
    TYPE_TEST(   ullong,     float_v,  float_v);
    TYPE_TEST(     Enum,     float_v,  float_v);
    TYPE_TEST(     bool,     float_v,  float_v);
    // double_v + float_v done

    TYPE_TEST(  short_v,     short_v,  short_v);
    TYPE_TEST(  short_v,       short,  short_v);
    TYPE_TEST(  short_v,    ushort_v, ushort_v);
    TYPE_TEST(  short_v,      ushort, ushort_v);
    TYPE_TEST(  short_v,         int,  short_v);
    TYPE_TEST(  short_v,        uint, ushort_v);
    TYPE_TEST(  short_v,        Enum,  short_v);
    TYPE_TEST(  short_v,        bool,  short_v);
    TYPE_TEST(    short,     short_v,  short_v);
    TYPE_TEST( ushort_v,     short_v, ushort_v);
    TYPE_TEST(   ushort,     short_v, ushort_v);
    TYPE_TEST(      int,     short_v,  short_v);
    TYPE_TEST(     uint,     short_v, ushort_v);
    TYPE_TEST(     Enum,     short_v,  short_v);
    TYPE_TEST(     bool,     short_v,  short_v);

    TYPE_TEST( ushort_v,       short, ushort_v);
    TYPE_TEST( ushort_v,    ushort_v, ushort_v);
    TYPE_TEST( ushort_v,      ushort, ushort_v);
    TYPE_TEST( ushort_v,         int, ushort_v);
    TYPE_TEST( ushort_v,        uint, ushort_v);
    TYPE_TEST( ushort_v,        Enum, ushort_v);
    TYPE_TEST( ushort_v,        bool, ushort_v);
    TYPE_TEST(    short,    ushort_v, ushort_v);
    TYPE_TEST(   ushort,    ushort_v, ushort_v);
    TYPE_TEST(      int,    ushort_v, ushort_v);
    TYPE_TEST(     uint,    ushort_v, ushort_v);
    TYPE_TEST(     Enum,    ushort_v, ushort_v);
    TYPE_TEST(     bool,    ushort_v, ushort_v);

    TYPE_TEST(    int_v,      ushort,    int_v);
    TYPE_TEST(    int_v,       short,    int_v);
    TYPE_TEST(    int_v,       int_v,    int_v);
    TYPE_TEST(    int_v,         int,    int_v);
    TYPE_TEST(    int_v,      uint_v,   uint_v);
    TYPE_TEST(    int_v,        uint,   uint_v);
    TYPE_TEST(    int_v,        Enum,    int_v);
    TYPE_TEST(    int_v,        bool,    int_v);
    TYPE_TEST(   ushort,       int_v,    int_v);
    TYPE_TEST(    short,       int_v,    int_v);
    TYPE_TEST(      int,       int_v,    int_v);
    TYPE_TEST(   uint_v,       int_v,   uint_v);
    TYPE_TEST(     uint,       int_v,   uint_v);
    TYPE_TEST(     Enum,       int_v,    int_v);
    TYPE_TEST(     bool,       int_v,    int_v);

    TYPE_TEST(   uint_v,       short,   uint_v);
    TYPE_TEST(   uint_v,      ushort,   uint_v);
    TYPE_TEST(   uint_v,       int_v,   uint_v);
    TYPE_TEST(   uint_v,         int,   uint_v);
    TYPE_TEST(   uint_v,      uint_v,   uint_v);
    TYPE_TEST(   uint_v,        uint,   uint_v);
    TYPE_TEST(   uint_v,        Enum,   uint_v);
    TYPE_TEST(   uint_v,        bool,   uint_v);
    TYPE_TEST(    short,      uint_v,   uint_v);
    TYPE_TEST(   ushort,      uint_v,   uint_v);
    TYPE_TEST(    int_v,      uint_v,   uint_v);
    TYPE_TEST(      int,      uint_v,   uint_v);
    TYPE_TEST(     uint,      uint_v,   uint_v);
    TYPE_TEST(     Enum,      uint_v,   uint_v);
    TYPE_TEST(     bool,      uint_v,   uint_v);
}

struct Plus {
    template <class L, class R>
    decltype(std::declval<L>() + std::declval<R>()) operator()(L &&, R &&);
};
struct Xor {
    template <class L, class R>
    decltype(std::declval<L>() ^ std::declval<R>()) operator()(L &&, R &&);
};
struct Equal {
    template <class L, class R>
    decltype(std::declval<L>() == std::declval<R>()) operator()(L &&, R &&);
};
TEST_TYPES(Pair, failures, Typelist<
           Typelist<double_v, float_v>,
           Typelist<double_v, short_v>,
           Typelist<double_v, ushort_v>,
           Typelist<double_v, int_v>,
           Typelist<double_v, uint_v>,
           Typelist<float_v, double>,
           Typelist<float_v, short_v>,
           Typelist<float_v, ushort_v>,
           Typelist<int_v, float_v>,
           Typelist<int_v, float>,
           Typelist<int_v, double>,
           Typelist<int_v, long>,
           Typelist<int_v, llong>,
           Typelist<short_v, int_v>,
           Typelist<short_v, uint_v>,
           Typelist<short_v, float>,
           Typelist<ushort_v, int_v>,
           Typelist<ushort_v, uint_v>,
           Typelist<ushort_v, float>,
           Typelist<fixed_size_simd<int, 3>, int_v>
           >)
{
    using A = typename Pair::template at<0>;
    using B = typename Pair::template at<1>;
    VERIFY(!(vir::test::sfinae_is_callable<A, B>(Plus())));
    VERIFY(!(vir::test::sfinae_is_callable<B, A>(Plus())));
    VERIFY(!(vir::test::sfinae_is_callable<A, B>(Xor())));
    VERIFY(!(vir::test::sfinae_is_callable<B, A>(Xor())));
    VERIFY(!(vir::test::sfinae_is_callable<A, B>(Equal())));
    VERIFY(!(vir::test::sfinae_is_callable<B, A>(Equal())));
}

// vim: foldmethod=marker
