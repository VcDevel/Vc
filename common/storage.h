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

#ifndef VC_COMMON_STORAGE_H
#define VC_COMMON_STORAGE_H

#include "macros.h"

namespace Vc
{
namespace Common
{

template<typename VectorType, typename EntryType> class VectorMemoryUnion
{
    public:
        typedef EntryType AliasingEntryType MAY_ALIAS;
        inline VectorMemoryUnion() {}
        inline VectorMemoryUnion(VectorType x) : data(x) {}
        inline VectorMemoryUnion &operator=(VectorType x) {
            data = x; return *this;
        }

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

} // namespace Common
} // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_STORAGE_H
