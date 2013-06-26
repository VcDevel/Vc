/*  This file is part of the Vc library. {{{

    Copyright (C) 2012-2013 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

}}}*/

#ifndef VC_COMMON_TYPES_H
#define VC_COMMON_TYPES_H

#ifdef VC_CHECK_ALIGNMENT
#include <cstdlib>
#include <cstdio>
#endif

#include <type_traits>
#include "macros.h"

Vc_PUBLIC_NAMESPACE_BEGIN
// helper type to implement sfloat_v (Vector<Vc::sfloat>)
struct sfloat {};

/* TODO: add type for half-float, something along these lines:
class half_float
{
    uint16_t data;
public:
    constexpr half_float() : data(0) {}
    constexpr half_float(const half_float &) = default;
    constexpr half_float(half_float &&) = default;
    constexpr half_float &operator=(const half_float &) = default;

    constexpr explicit half_float(float);
    constexpr explicit half_float(double);
    constexpr explicit half_float(int);
    constexpr explicit half_float(unsigned int);

    explicit operator float       () const;
    explicit operator double      () const;
    explicit operator int         () const;
    explicit operator unsigned int() const;

    bool operator==(half_float rhs) const;
    bool operator!=(half_float rhs) const;
    bool operator>=(half_float rhs) const;
    bool operator<=(half_float rhs) const;
    bool operator> (half_float rhs) const;
    bool operator< (half_float rhs) const;

    half_float operator+(half_float rhs) const;
    half_float operator-(half_float rhs) const;
    half_float operator*(half_float rhs) const;
    half_float operator/(half_float rhs) const;
};
*/

// TODO: all of the following doesn't really belong into the toplevel Vc namespace. An anonymous
// namespace might be enough:

template<typename T> struct DetermineEntryType { typedef T Type; };
template<> struct DetermineEntryType<sfloat> { typedef float Type; };

template<typename T> struct NegateTypeHelper { typedef T Type; };
template<> struct NegateTypeHelper<unsigned char > { typedef char  Type; };
template<> struct NegateTypeHelper<unsigned short> { typedef short Type; };
template<> struct NegateTypeHelper<unsigned int  > { typedef int   Type; };

namespace VectorSpecialInitializerZero { enum ZEnum { Zero = 0 }; }
namespace VectorSpecialInitializerOne { enum OEnum { One = 1 }; }
namespace VectorSpecialInitializerIndexesFromZero { enum IEnum { IndexesFromZero }; }

#ifdef VC_MSVC
#  if defined(VC_IMPL_Scalar)
namespace Scalar {
    template<typename T> class Vector;
    template<unsigned int VectorSize> class Mask;
}
#  elif defined(VC_IMPL_SSE)
namespace SSE {
    template<typename T> class Vector;
    template<unsigned int VectorSize> class Mask;
    class Float8Mask;
}
#  elif defined(VC_IMPL_AVX)
namespace AVX {
    template<typename T> class Vector;
    template<unsigned int VectorSize, size_t RegisterWidth> class Mask;
}
#  else
#    error "Sorry, MSVC is a nasty compiler and needs extra care. Please help."
#  endif
#endif

struct FlagBase {};
struct LoadStoreFlag : public FlagBase {};
static struct AlignedFlag   : public LoadStoreFlag {} Aligned;
static struct UnalignedFlag : public LoadStoreFlag {} Unaligned;
static struct StreamingFlag : public LoadStoreFlag {} Streaming;
struct Exclusive {};
struct Shared {};
#ifdef VC_IMPL_MIC
template<int L1Stride = 8, int L2Stride = 64, typename ExclusiveOrShared = void> struct PrefetchFlag : public FlagBase {};
#else
// TODO: determine a good default for typical CPU use
template<int L1Stride = 16, int L2Stride = 0, typename ExclusiveOrShared = void> struct PrefetchFlag : public FlagBase {};
#endif
static PrefetchFlag<> Prefetch;
// CombineFlags: for now we can only combine 2 flags, more doesn't make sense with the current set/*{{{*/
namespace
{
template<typename F0, typename F1> struct CombineFlags
{
    typedef F0 Flag0;
    typedef F1 Flag1;
};
} // anonymous namespace
typedef CombineFlags<StreamingFlag, UnalignedFlag> StreamingAndUnalignedFlag;
/*}}}*/

namespace
{
    template<typename T, typename... Us> struct is_contained;
    template<typename T, typename U> struct is_contained<T, U> : public std::integral_constant<bool, std::is_same<T, U>::value> {};
    template<typename T, typename U, typename... Vs> struct is_contained<T, U, Vs...>
        : public std::integral_constant<bool, is_contained<T, Vs...>::value || std::is_same<T, U>::value> {};

    //template<bool B, typename T = void> using enable_if = typename std::enable_if<B, T>::type;
    template<typename B, typename T = void> using enable_if = typename std::enable_if<B::value, T>::type;

    template<bool Test, typename T = void> struct EnableIf { typedef T Value; };
    template<typename T> struct EnableIf<false, T> {};

    template<typename T> struct IsSignedInteger    { enum { Value = 0 }; };
    template<> struct IsSignedInteger<signed char> { enum { Value = 1 }; };
    template<> struct IsSignedInteger<short>       { enum { Value = 1 }; };
    template<> struct IsSignedInteger<int>         { enum { Value = 1 }; };
    template<> struct IsSignedInteger<long>        { enum { Value = 1 }; };
    template<> struct IsSignedInteger<long long>   { enum { Value = 1 }; };

    template<typename T> struct IsUnsignedInteger           { enum { Value = 0 }; };
    template<> struct IsUnsignedInteger<unsigned char>      { enum { Value = 1 }; };
    template<> struct IsUnsignedInteger<unsigned short>     { enum { Value = 1 }; };
    template<> struct IsUnsignedInteger<unsigned int>       { enum { Value = 1 }; };
    template<> struct IsUnsignedInteger<unsigned long>      { enum { Value = 1 }; };
    template<> struct IsUnsignedInteger<unsigned long long> { enum { Value = 1 }; };

    template<typename T> struct IsInteger { enum { Value = IsSignedInteger<T>::Value | IsUnsignedInteger<T>::Value }; };

    template<typename T> struct IsReal { enum { Value = 0 }; };
    template<> struct IsReal<float>    { enum { Value = 1 }; };
    template<> struct IsReal<double>   { enum { Value = 1 }; };

    template<typename T, typename U> struct IsEqualType { enum { Value = 0 }; };
    template<typename T> struct IsEqualType<T, T> { enum { Value = 1 }; };

    template<typename T, typename List0, typename List1 = void, typename List2 = void, typename List3 = void, typename List4 = void, typename List5 = void, typename List6 = void>
        struct IsInTypelist { enum { Value = false }; };
    template<typename T, typename List1, typename List2, typename List3, typename List4, typename List5, typename List6> struct IsInTypelist<T, T, List1, List2, List3, List4, List5, List6> { enum { Value = true }; };
    template<typename T, typename List0, typename List2, typename List3, typename List4, typename List5, typename List6> struct IsInTypelist<T, List0, T, List2, List3, List4, List5, List6> { enum { Value = true }; };
    template<typename T, typename List0, typename List1, typename List3, typename List4, typename List5, typename List6> struct IsInTypelist<T, List0, List1, T, List3, List4, List5, List6> { enum { Value = true }; };
    template<typename T, typename List0, typename List1, typename List2, typename List4, typename List5, typename List6> struct IsInTypelist<T, List0, List1, List2, T, List4, List5, List6> { enum { Value = true }; };
    template<typename T, typename List0, typename List1, typename List2, typename List3, typename List5, typename List6> struct IsInTypelist<T, List0, List1, List2, List3, T, List5, List6> { enum { Value = true }; };
    template<typename T, typename List0, typename List1, typename List2, typename List3, typename List4, typename List6> struct IsInTypelist<T, List0, List1, List2, List3, List4, T, List6> { enum { Value = true }; };
    template<typename T, typename List0, typename List1, typename List2, typename List3, typename List4, typename List5> struct IsInTypelist<T, List0, List1, List2, List3, List4, List5, T> { enum { Value = true }; };

    template<typename Arg0, typename Arg1, typename T0, typename T1> struct IsCombinationOf { enum { Value = false }; };
    template<typename Arg0, typename Arg1> struct IsCombinationOf<Arg0, Arg1, Arg0, Arg1> { enum { Value = true }; };
    template<typename Arg0, typename Arg1> struct IsCombinationOf<Arg0, Arg1, Arg1, Arg0> { enum { Value = true }; };

    namespace
    {
        struct yes { char x; };
        struct  no { yes x, y; };
    } // anonymous namespace

    template<typename From, typename To> struct HasImplicitCast
    {
        template<typename F> static F makeT();
#if defined(VC_GCC) && VC_GCC < 0x40300
        // older GCCs don't do SFINAE correctly
        static yes test( To) { return yes(); }
        static  no test(...) { return  no(); }
        enum {
            Value = !!(sizeof(test(makeT<From>())) == sizeof(yes))
        };
#else
        template<typename T> static int test2(const T &);
#ifdef VC_MSVC
            // I want to test whether implicit cast works. If it works MSVC thinks it should give a warning. Wrong. Shut up.
#pragma warning(suppress : 4257 4267)
#endif
        template<typename F, typename T> static typename EnableIf<sizeof(test2<T>(makeT<F>())) == sizeof(int), yes>::Value test(int);
        template<typename, typename> static no  test(...);
        enum {
            Value = !!(sizeof(test<From, To>(0)) == sizeof(yes))
        };
#endif
    };
#if defined(VC_GCC) && VC_GCC < 0x40300
    // GCC 4.1 is very noisy because of the float->int and double->int type trait tests. We get
    // around this noise with a little specialization.
    template<> struct HasImplicitCast<float , int> { enum { Value = true }; };
    template<> struct HasImplicitCast<double, int> { enum { Value = true }; };
#endif

    template<typename T> struct CanConvertToInt : public HasImplicitCast<T, int> {};
    template<> struct CanConvertToInt<bool>     { enum { Value = 0 }; };
    //template<> struct CanConvertToInt<float>    { enum { Value = 0 }; };
    //template<> struct CanConvertToInt<double>   { enum { Value = 0 }; };

    enum TestEnum {};
    static_assert(CanConvertToInt<int>::Value == 1, "CanConvertToInt_is_broken");
    static_assert(CanConvertToInt<unsigned char>::Value == 1, "CanConvertToInt_is_broken");
    static_assert(CanConvertToInt<bool>::Value == 0, "CanConvertToInt_is_broken");
    static_assert(CanConvertToInt<float>::Value == 1, "CanConvertToInt_is_broken");
    static_assert(CanConvertToInt<double>::Value == 1, "CanConvertToInt_is_broken");
    static_assert(CanConvertToInt<float*>::Value == 0, "CanConvertToInt_is_broken");
    static_assert(CanConvertToInt<TestEnum>::Value == 1, "CanConvertToInt_is_broken");

    typedef HasImplicitCast<TestEnum, short> HasImplicitCastTest0;
    typedef HasImplicitCast<int *, void *> HasImplicitCastTest1;
    typedef HasImplicitCast<int *, const void *> HasImplicitCastTest2;
    typedef HasImplicitCast<const int *, const void *> HasImplicitCastTest3;
    typedef HasImplicitCast<const int *, int *> HasImplicitCastTest4;

    static_assert(HasImplicitCastTest0::Value ==  true, "HasImplicitCast0_is_broken");
    static_assert(HasImplicitCastTest1::Value ==  true, "HasImplicitCast1_is_broken");
    static_assert(HasImplicitCastTest2::Value ==  true, "HasImplicitCast2_is_broken");
    static_assert(HasImplicitCastTest3::Value ==  true, "HasImplicitCast3_is_broken");
    static_assert(HasImplicitCastTest4::Value == false, "HasImplicitCast4_is_broken");

    template<typename T> struct IsLikeInteger { enum { Value = !IsReal<T>::Value && CanConvertToInt<T>::Value }; };
    template<typename T> struct IsLikeSignedInteger { enum { Value = IsLikeInteger<T>::Value && !IsUnsignedInteger<T>::Value }; };
} // anonymous namespace

#ifndef VC_CHECK_ALIGNMENT
template<typename _T> static Vc_ALWAYS_INLINE void assertCorrectAlignment(const _T *){}
#else
template<typename _T> static Vc_ALWAYS_INLINE void assertCorrectAlignment(const _T *ptr)
{
    const size_t s = alignof(_T);
    if((reinterpret_cast<size_t>(ptr) & ((s ^ (s & (s - 1))) - 1)) != 0) {
        fprintf(stderr, "A vector with incorrect alignment has just been created. Look at the stacktrace to find the guilty object.\n");
        abort();
    }
}
#endif

Vc_NAMESPACE_END

#include "memoryfwd.h"
#include "undomacros.h"

#endif // VC_COMMON_TYPES_H

// vim: foldmethod=marker
