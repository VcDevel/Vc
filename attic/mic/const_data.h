/*  This file is part of the Vc library. {{{
Copyright © 2013-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_CONST_DATA_H_
#define VC_MIC_CONST_DATA_H_

#include "../common/data.h"
#include "macros.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace MIC
{

alignas(16) extern const char  _IndexesFromZero8[16];

struct alignas(64) c_general
{
    static const float oneFloat;
    static const unsigned int absMaskFloat[2];
    static const unsigned int signMaskFloat[2];
    static const unsigned int highMaskFloat;
    static const unsigned short minShort[2];
    static const unsigned short one16[2];
    static const float _2power31;
    static const double oneDouble;
    static const unsigned long long frexpMask;
    static const unsigned long long highMaskDouble;
    static const unsigned char frexpAndMask[16];
};

}  // namespace MIC
Vc_VERSIONED_NAMESPACE_END

#endif // VC_MIC_CONST_DATA_H_
