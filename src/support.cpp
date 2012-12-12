/*  This file is part of the Vc library.

    Copyright (C) 2010-2012 Matthias Kretz <kretz@kde.org>

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

#include <Vc/global.h>
#include <Vc/cpuid.h>
#include "common/support.h"

#ifdef VC_MSVC
#include <intrin.h>
#endif

namespace Vc
{

#ifdef VC_GCC
    __attribute__((target("no-sse2,no-avx")))
#endif
bool isImplementationSupported(Implementation impl)
{
    CpuId::init();

    switch (impl) {
    case ScalarImpl:
        return true;
    case SSE2Impl:
        return CpuId::hasSse2();
    case SSE3Impl:
        return CpuId::hasSse3();
    case SSSE3Impl:
        return CpuId::hasSsse3();
    case SSE41Impl:
        return CpuId::hasSse41();
    case SSE42Impl:
        return CpuId::hasSse42();
    case SSE4aImpl:
        return CpuId::hasSse4a();
    case XopImpl:
        return isImplementationSupported(Vc::AVXImpl) && CpuId::hasXop();
    case Fma4Impl:
        return isImplementationSupported(Vc::AVXImpl) && CpuId::hasFma4();
    case AVXImpl:
        if (CpuId::hasOsxsave() && CpuId::hasAvx()) {
#if defined(VC_MSVC)
#if VC_MSVC >= 160040219 // MSVC 2010 SP1 introduced _xgetbv
            unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
            return (xcrFeatureMask & 0x6) != 0;
#else
            // can't check, but if OSXSAVE is true let's assume it'll work
            return true;
#endif
#elif !defined(VC_NO_XGETBV)
            unsigned int eax;
            asm("xgetbv" : "=a"(eax) : "c"(0) : "edx");
            return (eax & 0x06) == 0x06;
#endif
        }
        return false;
    }
    return false;
}

} // namespace Vc

// vim: sw=4 sts=4 et tw=100
