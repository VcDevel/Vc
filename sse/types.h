/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

*/

#ifndef SSE_TYPES_H
#define SSE_TYPES_H

#include "intrinsics.h"
#include "../common/storage.h"
#include "macros.h"

namespace Vc
{
namespace SSE
{
    template<typename T> class Vector;

    // define our own long because on Windows64 long == int while on Linux long == max. register width
    // since we want to have a type that depends on 32 vs. 64 bit we need to do some special casing on Windows
#ifdef _WIN64
    typedef __int64 _long;
    typedef unsigned __int64 _ulong;
#else
    typedef long _long;
    typedef unsigned long _ulong;
#endif


    class Float8Mask;
    class Float8GatherMask;
    template<unsigned int VectorSize> class Mask;

    /*
     * Hack to create a vector object with 8 floats
     */
    class float8 {};

    class M256 {
        public:
            //inline M256() {}
            //inline M256(_M128 a, _M128 b) { d[0] = a; d[1] = b; }
            static inline M256 create(_M128 a, _M128 b) { M256 r; r.d[0] = a; r.d[1] = b; return r; }
            inline _M128 &operator[](int i) { return d[i]; }
            inline const _M128 &operator[](int i) const { return d[i]; }
        private:
            _M128 d[2];
    };

    template<typename T> struct ParameterHelper {
        typedef T ByValue;
        typedef T & Reference;
        typedef const T & ConstRef;
    };
#if defined VC_MSVC && !defined _WIN64
    // The calling convention on WIN32 can't guarantee alignment.
    // An exception are the first three arguments, which may be passed in a register.
    template<> struct ParameterHelper<M256> {
        typedef const M256 & ByValue;
        typedef M256 & Reference;
        typedef const M256 & ConstRef;
    };
#endif

    template<typename T> struct VectorHelper {};

    template<typename T> struct NegateTypeHelper { typedef T Type; };
    template<> struct NegateTypeHelper<unsigned char > { typedef char  Type; };
    template<> struct NegateTypeHelper<unsigned short> { typedef short Type; };
    template<> struct NegateTypeHelper<unsigned int  > { typedef int   Type; };

    template<unsigned int Size> struct IndexTypeHelper;
    template<> struct IndexTypeHelper<2u> { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<4u> { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<8u> { typedef unsigned short Type; };
    template<> struct IndexTypeHelper<16u>{ typedef unsigned char  Type; };

    template<typename T> struct CtorTypeHelper { typedef T Type; };
    template<> struct CtorTypeHelper<short> { typedef int Type; };
    template<> struct CtorTypeHelper<unsigned short> { typedef unsigned int Type; };
    template<> struct CtorTypeHelper<float> { typedef double Type; };

    template<typename T> struct ExpandTypeHelper { typedef T Type; };
    template<> struct ExpandTypeHelper<short> { typedef int Type; };
    template<> struct ExpandTypeHelper<unsigned short> { typedef unsigned int Type; };
    template<> struct ExpandTypeHelper<float> { typedef double Type; };

    template<typename T> struct VectorHelperSize;

    namespace VectorSpecialInitializerZero { enum ZEnum { Zero = 0 }; }
    namespace VectorSpecialInitializerOne { enum OEnum { One = 1 }; }
    namespace VectorSpecialInitializerIndexesFromZero { enum IEnum { IndexesFromZero }; }

    template<typename V = Vector<float> >
    class STRUCT_ALIGN1(16) VectorAlignedBaseT
    {
        public:
            FREE_STORE_OPERATORS_ALIGNED(16)
    } STRUCT_ALIGN2(16);

} // namespace SSE
} // namespace Vc

#endif // SSE_TYPES_H
