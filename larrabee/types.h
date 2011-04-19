/*  This file is part of the Vc library.

    Copyright (C) 2011 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_LARRABEE_TYPES_H
#define VC_LARRABEE_TYPES_H

#include "macros.h"
#include <cstdlib>

namespace Vc
{
    template<typename V, unsigned int Size> class Memory;

namespace LRBni
{
    template<typename T> class Vector;
    template<typename T> struct SwizzledVector;
    template<unsigned int VectorSize> class Mask;
    ALIGN(16) extern const char _IndexesFromZero[16];

    template<typename T> struct NegateTypeHelper { typedef T Type; };
    template<> struct NegateTypeHelper<unsigned char > { typedef char  Type; };
    template<> struct NegateTypeHelper<unsigned short> { typedef short Type; };
    template<> struct NegateTypeHelper<unsigned int  > { typedef int   Type; };

    template<typename T> struct ReturnTypeHelper { typedef char Type; };
    template<> struct ReturnTypeHelper<unsigned int> { typedef unsigned char Type; };
    template<> struct ReturnTypeHelper<int> { typedef signed char Type; };
    template<typename T> const typename ReturnTypeHelper<T>::Type *IndexesFromZeroHelper() {
        return reinterpret_cast<const typename ReturnTypeHelper<T>::Type *>(&_IndexesFromZero[0]);
    }

    namespace VectorSpecialInitializerZero { enum ZEnum { Zero = 0 }; }
    namespace VectorSpecialInitializerOne { enum OEnum { One = 1 }; }
    namespace VectorSpecialInitializerIndexesFromZero { enum IEnum { IndexesFromZero }; }

    enum { VectorAlignment = 64 };

    template<typename V = Vector<float> >
    class VectorAlignedBaseT
    {
        public:
            void *operator new(size_t size) { void *r; if (posix_memalign(&r, VectorAlignment, size)) {}; return r; }
            void *operator new[](size_t size) { void *r; if (posix_memalign(&r, VectorAlignment, size)) {}; return r; }
            void operator delete(void *ptr, size_t) { free(ptr); }
            void operator delete[](void *ptr, size_t) { free(ptr); }
    } ALIGN(64);

} // namespace LRBni
} // namespace Vc

#include "undomacros.h"

#endif // VC_LARRABEE_TYPES_H
