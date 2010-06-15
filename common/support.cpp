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

#include "support.h"
#include "../cpuid.h"

namespace Vc
{

bool isImplementationSupported(Implementation impl)
{
    // for AVX we need to check for OSXSAVE and AVX

    switch (impl) {
    case Scalar:
        return true;
    case SSE2:
        return CpuId::hasSse2();
    case SSE3:
        return CpuId::hasSse3();
    case SSSE3:
        return CpuId::hasSsse3();
    case SSE41:
        return CpuId::hasSse41();
    case SSE42:
        return CpuId::hasSse42();
    case SSE4a:
        return CpuId::hasSse4a();
    case AVX:
        if (CpuId::hasOsxsave() && CpuId::hasAvx()) {
            unsigned int eax;
            asm("xgetbv" : "=a"(eax) :: "edx");
            return (eax & 0x06) == 0x06;
        }
        return false;
    case LRBni:
        // TODO
        return false;
    }
    return false;
}

} // namespace Vc

// vim: sw=4 sts=4 et tw=100
