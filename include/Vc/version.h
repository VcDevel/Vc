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

#ifndef VC_VERSION_H
#define VC_VERSION_H

#define VC_VERSION_STRING "0.6.1"
#define VC_VERSION_NUMBER 0x000602
#define VC_VERSION_CHECK(major, minor, patch) ((major << 16) | (minor << 8) | (patch << 1))

namespace Vc
{
    static inline const char *versionString() {
        return VC_VERSION_STRING;
    }

    static inline unsigned int versionNumber() {
        return VC_VERSION_NUMBER;
    }
} // namespace Vc

#endif // VC_VERSION_H
