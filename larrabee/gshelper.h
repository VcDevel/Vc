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

#ifndef VC_LARRABEE_GATHERHELPER_H
#define VC_LARRABEE_GATHERHELPER_H

namespace Vc
{
namespace LRBni
{

/*
 * Gather-Scatter Helper
 */
template<unsigned int Size> struct GSHelper
{
    public:
        template<typename VectorType, typename MemoryType>
        static void gather(VectorType &v, const MemoryType *m, _M512I i);

        template<typename VectorType, typename MemoryType>
        static void scatter(const VectorType &v, MemoryType *m, _M512I i);
};

} // namespace LRBni
} // namespace Vc

#include "gshelper.tcc"

#endif // VC_LARRABEE_GATHERHELPER_H
