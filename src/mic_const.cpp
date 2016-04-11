/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#include "mic/const_data.h"
#include <Vc/version.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace MIC
{
    alignas(16) extern const char _IndexesFromZero[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    alignas(8) const unsigned       int c_general::absMaskFloat[2] = { 0xffffffffu, 0x7fffffffu };
    alignas(8) const unsigned       int c_general::signMaskFloat[2] = { 0x0u, 0x80000000u };
    const              float c_general::oneFloat = 1.f;
    const unsigned       int c_general::highMaskFloat = 0xfffff000u;
    alignas(4) const unsigned     short c_general::minShort[2] = { 0x8000u, 0x8000u };
    alignas(4) const unsigned     short c_general::one16[2] = { 1, 1 };
    const              float c_general::_2power31 = 1u << 31;

    const             double c_general::oneDouble = 1.;
    const unsigned long long c_general::frexpMask = 0xbfefffffffffffffull;
    const unsigned long long c_general::highMaskDouble = 0xfffffffff8000000ull;

    alignas(16) const unsigned char c_general::frexpAndMask[16] = {
        1, 0, 2, 0, 4, 0, 8, 0, 16, 0, 32, 0, 64, 0, 128, 0
    };
}
}

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{
alignas(64) unsigned int RandomState[32] = {
    0x5a383a4fu, 0xc68bd45eu, 0x691d6d86u, 0xb367e14fu,
    0xd689dbaau, 0xfde442aau, 0x3d265423u, 0x1a77885cu,
    0x36ed2684u, 0xfb1f049du, 0x19e52f31u, 0x821e4dd7u,
    0x23996d25u, 0x5962725au, 0x6aced4ceu, 0xd4c610f3u,

    0x6ac4c828u, 0x34fcb8a2u, 0x34fe32a9u, 0xdd6fba5du,
    0x112df788u, 0xa8241de1u, 0x0e1d1b1du, 0x813c9552u,
    0xb0f88feeu, 0x1e4364fbu, 0xdb759fb3u, 0xcc01a0f3u,
    0xa94dc0a0u, 0xf6fef349u, 0xcaee8edbu, 0x74af8a26u
};

const char LIBRARY_VERSION[] = Vc_VERSION_STRING;
const unsigned int LIBRARY_VERSION_NUMBER = Vc_VERSION_NUMBER;
const unsigned int LIBRARY_ABI_VERSION = Vc_LIBRARY_ABI_VERSION;

void checkLibraryAbi(unsigned int compileTimeAbi, unsigned int versionNumber, const char *compileTimeVersion) {
    if (LIBRARY_ABI_VERSION != compileTimeAbi || LIBRARY_VERSION_NUMBER < versionNumber) {
        printf("The versions of libVc.a (%s) and Vc/version.h (%s) are incompatible. Aborting.\n", LIBRARY_VERSION, compileTimeVersion);
        abort();
    }
}
}
}
