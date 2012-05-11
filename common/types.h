/*  This file is part of the Vc library. {{{

    Copyright (C) 2012 Matthias Kretz <kretz@kde.org>

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

namespace Vc
{

// helper type to implement sfloat_v (Vector<Vc::sfloat>)
struct sfloat {};

template<typename T> struct DetermineEntryType { typedef T Type; };
template<> struct DetermineEntryType<sfloat> { typedef float Type; };

template<typename T> struct NegateTypeHelper { typedef T Type; };
template<> struct NegateTypeHelper<unsigned char > { typedef char  Type; };
template<> struct NegateTypeHelper<unsigned short> { typedef short Type; };
template<> struct NegateTypeHelper<unsigned int  > { typedef int   Type; };

namespace VectorSpecialInitializerZero { enum ZEnum { Zero = 0 }; }
namespace VectorSpecialInitializerOne { enum OEnum { One = 1 }; }
namespace VectorSpecialInitializerIndexesFromZero { enum IEnum { IndexesFromZero }; }

namespace
{
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

    template<typename T> struct IsBuiltin { enum { Value = IsInteger<T>::Value | IsReal<T>::Value }; };

    template<typename T, typename U> struct IsEqualType { enum { Value = 0 }; };
    template<typename T> struct IsEqualType<T, T> { enum { Value = 1 }; };

    struct CanConvertToInt_Impl
    {
        struct yes { char x; };
        struct no  { yes x, y; };
        static yes foo(int) { return yes(); }
        static no  foo(...) { return  no(); }
    };
    template<typename T> struct CanConvertToInt { enum { Value = !!(sizeof(CanConvertToInt_Impl::foo(*static_cast<T *>(0))) == sizeof(CanConvertToInt_Impl::yes)) }; };
    template<> struct CanConvertToInt<bool>     { enum { Value = 0 }; };
    //template<> struct CanConvertToInt<float>    { enum { Value = 0 }; };
    //template<> struct CanConvertToInt<double>   { enum { Value = 0 }; };

    enum TestEnum {};
    VC_STATIC_ASSERT(CanConvertToInt<int>::Value == 1, CanConvertToInt_is_broken);
    VC_STATIC_ASSERT(CanConvertToInt<unsigned char>::Value == 1, CanConvertToInt_is_broken);
    VC_STATIC_ASSERT(CanConvertToInt<bool>::Value == 0, CanConvertToInt_is_broken);
    VC_STATIC_ASSERT(CanConvertToInt<float>::Value == 1, CanConvertToInt_is_broken);
    VC_STATIC_ASSERT(CanConvertToInt<double>::Value == 1, CanConvertToInt_is_broken);
    VC_STATIC_ASSERT(CanConvertToInt<float*>::Value == 0, CanConvertToInt_is_broken);
    VC_STATIC_ASSERT(CanConvertToInt<TestEnum>::Value == 1, CanConvertToInt_is_broken);
} // anonymous namespace

} // namespace Vc

#endif // VC_COMMON_TYPES_H
