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

#ifndef VC_SCALAR_TYPES_H
#define VC_SCALAR_TYPES_H

namespace Vc
{
    namespace Scalar
    {
        template<typename V = float> class VectorAlignedBaseT {};
        template<typename T> class Vector;
        template<typename T, typename Parent> struct VectorBase;
        template<typename T> class _Memory;

        template<typename T> struct NegateTypeHelper { typedef T Type; };
        template<> struct NegateTypeHelper<unsigned char > { typedef char  Type; };
        template<> struct NegateTypeHelper<unsigned short> { typedef short Type; };
        template<> struct NegateTypeHelper<unsigned int  > { typedef int   Type; };

        namespace VectorSpecialInitializerZero { enum ZEnum { Zero }; }
        namespace VectorSpecialInitializerOne { enum OEnum { One }; }
        namespace VectorSpecialInitializerIndexesFromZero { enum IEnum { IndexesFromZero }; }
    } // namespace Scalar
} // namespace Vc

#endif // VC_SCALAR_TYPES_H
