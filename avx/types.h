/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Leavxr General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Leavxr General Public License for more details.

    You should have received a copy of the GNU Leavxr General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef AVX_TYPES_H
#define AVX_TYPES_H

#include "intrinsics.h"

namespace Vc
{
namespace AVX
{
    template<typename T> class Vector;

    template<unsigned int VectorSize> class Mask;

    template<typename T> struct VectorHelper {};
    template<typename T> struct GatherHelper;
    template<typename T> struct ScatterHelper;

    template<typename T> struct IndexTypeHelper;
    template<> struct IndexTypeHelper<         char > { typedef unsigned char  Type; };
    template<> struct IndexTypeHelper<unsigned char > { typedef unsigned char  Type; };
    template<> struct IndexTypeHelper<         short> { typedef unsigned short Type; };
    template<> struct IndexTypeHelper<unsigned short> { typedef unsigned short Type; };
    template<> struct IndexTypeHelper<         int  > { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<unsigned int  > { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<         float> { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<        double> { typedef unsigned int   Type; }; // _M128I based int32 would be nice

    template<typename T> struct VectorTypeHelper;
    template<> struct VectorTypeHelper<         char > { typedef _M128I Type; };
    template<> struct VectorTypeHelper<unsigned char > { typedef _M128I Type; };
    template<> struct VectorTypeHelper<         short> { typedef _M128I Type; };
    template<> struct VectorTypeHelper<unsigned short> { typedef _M128I Type; };
    template<> struct VectorTypeHelper<         int  > { typedef _M256I Type; };
    template<> struct VectorTypeHelper<unsigned int  > { typedef _M256I Type; };
    template<> struct VectorTypeHelper<         float> { typedef _M256  Type; };
    template<> struct VectorTypeHelper<        double> { typedef _M256D Type; };

    template<typename VectorType, typename EntryType> class VectorMemoryUnion
    {
        public:
            typedef EntryType AliasingEntryType MAY_ALIAS;
            inline VectorMemoryUnion() {}
            inline VectorMemoryUnion(const VectorType &x) : data(x) {}

            VectorType &v() { return data; }
            const VectorType &v() const { return data; }

            AliasingEntryType &m(int index) {
                return reinterpret_cast<AliasingEntryType *>(&data)[index];
            }

            EntryType m(int index) const {
                return reinterpret_cast<const AliasingEntryType *>(&data)[index];
            }

        private:
            VectorType data;
    };

    template<typename T> struct VectorHelperSize;

    namespace VectorSpecialInitializerZero { enum ZEnum { Zero = 0 }; }
    namespace VectorSpecialInitializerOne { enum OEnum { One = 1 }; }
    namespace VectorSpecialInitializerRandom { enum REnum { Random }; }
    namespace VectorSpecialInitializerIndexesFromZero { enum IEnum { IndexesFromZero }; }

    class VectorAlignedBase
    {
        public:
            FREE_STORE_OPERATORS_ALIGNED(32)
    } ALIGN(32);

} // namespace AVX
} // namespace Vc

#endif // AVX_TYPES_H
