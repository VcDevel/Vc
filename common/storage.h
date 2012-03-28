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
#if defined VC_ICC || defined VC_MSVC
        inline VectorMemoryUnion(const VectorType &x) { data.v = x; }
        inline VectorMemoryUnion &operator=(const VectorType &x) {
            data.v = x; return *this;
        }

        VectorType &v() { return data.v; }
        const VectorType &v() const { return data.v; }

        EntryType &m(int index) {
            return data.m[index];
        }

        EntryType m(int index) const {
            return data.m[index];
        }

    private:
        union {
            VectorType v;
            EntryType m[sizeof(VectorType)/sizeof(EntryType)];
        } data;
#else
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
#endif
};

#if VC_GCC == 0x40700 || (VC_GCC >= 0x40600 && VC_GCC <= 0x40603)
// workaround bug 52736 in GCC
template<typename T, typename V> static inline T &vectorMemoryUnionAliasedMember(V *data, int index) {
    if (__builtin_constant_p(index) && index == 0) {
        T *ret;
        asm("mov %1,%0" : "=r"(ret) : "r"(data));
        return *ret;
    } else {
        return reinterpret_cast<T *>(data)[index];
    }
}
template<> inline VectorMemoryUnion<__m128d, double>::AliasingEntryType &VectorMemoryUnion<__m128d, double>::m(int index) {
    return vectorMemoryUnionAliasedMember<AliasingEntryType>(&data, index);
}
template<> inline VectorMemoryUnion<__m128i, long long>::AliasingEntryType &VectorMemoryUnion<__m128i, long long>::m(int index) {
    return vectorMemoryUnionAliasedMember<AliasingEntryType>(&data, index);
}
template<> inline VectorMemoryUnion<__m128i, unsigned long long>::AliasingEntryType &VectorMemoryUnion<__m128i, unsigned long long>::m(int index) {
    return vectorMemoryUnionAliasedMember<AliasingEntryType>(&data, index);
}
#endif

} // namespace Common
} // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_STORAGE_H
