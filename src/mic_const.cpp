/*  This file is part of the Vc library. {{{

    Copyright (C) 2009-2013 Matthias Kretz <kretz@kde.org>

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

}}}*/

#include "mic/const_data.h"
#include <Vc/version.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/macros.h"

Vc_NAMESPACE_BEGIN(MIC)
    ALIGN(16) extern const char _IndexesFromZero[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    ALIGN(8) const unsigned       int c_general::absMaskFloat[2] = { 0xffffffffu, 0x7fffffffu };
    ALIGN(8) const unsigned       int c_general::signMaskFloat[2] = { 0x0u, 0x80000000u };
    const              float c_general::oneFloat = 1.f;
    const unsigned       int c_general::highMaskFloat = 0xfffff000u;
    ALIGN(4) const unsigned     short c_general::minShort[2] = { 0x8000u, 0x8000u };
    ALIGN(4) const unsigned     short c_general::one16[2] = { 1, 1 };
    const              float c_general::_2power31 = 1u << 31;

    const             double c_general::oneDouble = 1.;
    const unsigned long long c_general::frexpMask = 0xbfefffffffffffffull;
    const unsigned long long c_general::highMaskDouble = 0xfffffffff8000000ull;

    ALIGN(16) const unsigned char c_general::frexpAndMask[16] = {
        1, 0, 2, 0, 4, 0, 8, 0, 16, 0, 32, 0, 64, 0, 128, 0
    };
Vc_NAMESPACE_END

Vc_NAMESPACE_BEGIN(Common)
ALIGN(64) unsigned int RandomState[16] = {
    0x5a383a4fu, 0xc68bd45eu, 0x691d6d86u, 0xb367e14fu,
    0xd689dbaau, 0xfde442aau, 0x3d265423u, 0x1a77885cu,
    0x36ed2684u, 0xfb1f049du, 0x19e52f31u, 0x821e4dd7u,
    0x23996d25u, 0x5962725au, 0x6aced4ceu, 0xd4c610f3u
};

const char LIBRARY_VERSION[] = VC_VERSION_STRING;
const unsigned int LIBRARY_VERSION_NUMBER = VC_VERSION_NUMBER;
const unsigned int LIBRARY_ABI_VERSION = VC_LIBRARY_ABI_VERSION;

void checkLibraryAbi(unsigned int compileTimeAbi, unsigned int versionNumber, const char *compileTimeVersion) {
    if (LIBRARY_ABI_VERSION != compileTimeAbi || LIBRARY_VERSION_NUMBER < versionNumber) {
        printf("The versions of libVc.a (%s) and Vc/version.h (%s) are incompatible. Aborting.\n", LIBRARY_VERSION, compileTimeVersion);
        abort();
    }
}
Vc_NAMESPACE_END
