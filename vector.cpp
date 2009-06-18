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

#ifndef ALIGN
# ifdef __GNUC__
#  define V_ALIGN(n) __attribute__((aligned(n)))
# else
#  define V_ALIGN(n) __declspec(align(n))
# endif
#endif

namespace SSE
{
    namespace Internal
    {
        V_ALIGN(16) extern const unsigned int   _IndexesFromZero4[4] = { 0, 1, 2, 3 };
        V_ALIGN(16) extern const unsigned short _IndexesFromZero8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    } // namespace Internal
} // namespace SSE

namespace Larrabee
{
    namespace Internal
    {
        V_ALIGN(16) extern const char _IndexesFromZero[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    } // namespace Internal
} // namespace Larrabee

#undef V_ALIGN
