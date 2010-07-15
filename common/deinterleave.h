/*  This file is part of the Vc library.

    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_DEINTERLEAVE_H
#define VC_COMMON_DEINTERLEAVE_H

namespace Vc
{

/**
 * supports the following type combinations:
 *   V \  M | float | double | ushort | short | uint | int
 * ---------+----------------------------------------------
 *  float_v |   X   |        |    X   |   X   |      |
 * sfloat_v |   X   |        |    X   |   X   |      |
 * double_v |       |    X   |        |       |      |
 *    int_v |       |        |        |   X   |      |  X
 *   uint_v |       |        |    X   |       |   X  |
 *  short_v |       |        |        |   X   |      |
 * ushort_v |       |        |    X   |       |      |
 */
template<typename V, typename M, typename A> inline void deinterleave(V *a, V *b,
        const M *memory, A align)
{
    Internal::Helper::deinterleave(*a, *b, memory, align);
}

template<typename V, typename M> inline void deinterleave(V *a, V *b,
        const M *memory)
{
    Internal::Helper::deinterleave(*a, *b, memory, Aligned);
}

} // namespace Vc
#endif // VC_COMMON_DEINTERLEAVE_H
