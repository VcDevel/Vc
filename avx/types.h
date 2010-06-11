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

    class Float16Mask;
    class Float16GatherMask;
    template<unsigned int VectorSize> class Mask;

    /*
     * Hack to create a vector object with 8 floats
     */
    class float16 {};

    class M512 {
        public:
            //inline M512() {}
            //inline M512(_M256 a, _M256 b) { d[0] = a; d[1] = b; }
            static inline M512 create(_M256 a, _M256 b) { M512 r; r.d[0] = a; r.d[1] = b; return r; }
            inline _M256 &operator[](int i) { return d[i]; }
            inline const _M256 &operator[](int i) const { return d[i]; }
        private:
            _M256 d[2];
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

} // namespace AVX
} // namespace Vc

#endif // AVX_TYPES_H
