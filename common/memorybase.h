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

#ifndef VC_COMMON_MEMORYBASE_H
#define VC_COMMON_MEMORYBASE_H

namespace Vc
{
template<typename V, typename Parent> class MemoryBase
{
    private:
        Parent *p() { return static_cast<Parent *>(this); }
        const Parent *p() const { return static_cast<const Parent *>(this); }
    public:
        typedef typename V::EntryType EntryType;

        inline unsigned int entriesCount() const { return p()->entriesCount; }
        inline unsigned int vectorsCount() const { return p()->vectorsCount; }

        inline       EntryType *entries()       { return p()->entries(); }
        inline const EntryType *entries() const { return p()->entries(); }

        inline EntryType &operator[](int i)       { return entries()[i]; }
        inline EntryType  operator[](int i) const { return entries()[i]; }

        inline operator       EntryType*()       { return entries(); }
        inline operator const EntryType*() const { return entries(); }

        inline       EntryType *vector(int i)       { return &entries()[i * V::Size]; }
        inline const EntryType *vector(int i) const { return &entries()[i * V::Size]; }
};
} // namespace Vc

#endif // VC_COMMON_MEMORYBASE_H
