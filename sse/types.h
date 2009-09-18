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

namespace Vc
{
namespace SSE
{
    template<typename T> class Vector;

    class Float8Mask;
    class Float8GatherMask;
    template<unsigned int VectorSize> class Mask;

    /*
     * Hack to create a vector object with 8 floats
     */
    class float8 {};

    class M256 {
        public:
            inline M256() {}
            inline M256(_M128 a, _M128 b) { d[0] = a; d[1] = b; }
            inline _M128 &operator[](int i) { return d[i]; }
            inline const _M128 &operator[](int i) const { return d[i]; }
        private:
            _M128 d[2];
    };

    template<typename T> struct VectorHelper {};
    template<typename T> struct GatherHelper;
    template<typename T> struct ScatterHelper;

    template<unsigned int Size> struct IndexTypeHelper;
    template<> struct IndexTypeHelper<2u> { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<4u> { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<8u> { typedef unsigned short Type; };
    template<> struct IndexTypeHelper<16u>{ typedef unsigned char  Type; };

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
            FREE_STORE_OPERATORS_ALIGNED(16)
    } ALIGN(16);

} // namespace SSE
} // namespace Vc

#endif // SSE_TYPES_H
