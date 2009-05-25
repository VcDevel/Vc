/*  This file is part of the KDE project
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) version 3.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public License
    along with this library; see the file COPYING.LIB.  If not, write to
    the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
    Boston, MA 02110-1301, USA.

*/

#ifndef VECTORMEMORYHELPER_H
#define VECTORMEMORYHELPER_H

#include "vector.h"

template<typename Vec>
class VectorMemoryHelper
{
    char *const mem;
    char *const aligned;
    public:
        VectorMemoryHelper(int count)
            : mem(new char[count * sizeof(Vec) + Vc::VectorAlignment]),
            aligned(mem + (Vc::VectorAlignment - (reinterpret_cast<unsigned long>( mem ) & ( Vc::VectorAlignment - 1 ))))
        {
        }
        ~VectorMemoryHelper() { delete[] mem; }

        operator typename Vec::EntryType *() { return reinterpret_cast<typename Vec::EntryType *>(aligned); }
};
#endif // VECTORMEMORYHELPER_H
