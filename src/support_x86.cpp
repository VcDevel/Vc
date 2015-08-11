/*  This file is part of the Vc library. {{{
Copyright © 2010-2013 Matthias Kretz <kretz@kde.org>
All rights reserved.

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

#include <Vc/global.h>
#include <Vc/cpuid.h>
#include <Vc/support.h>

#ifdef VC_MSVC
#include <intrin.h>
#endif

#if defined(VC_GCC) && VC_GCC >= 0x40400
#define VC_TARGET_NO_SIMD __attribute__((target("no-sse2,no-avx")))
#else
#define VC_TARGET_NO_SIMD
#endif

namespace Vc_VERSIONED_NAMESPACE
{

VC_TARGET_NO_SIMD
static inline bool xgetbvCheck(unsigned int bits)
{
#if defined(VC_MSVC) && VC_MSVC >= 160040219 // MSVC 2010 SP1 introduced _xgetbv
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & bits) == bits;
#elif defined(VC_GNU_ASM) && !defined(VC_NO_XGETBV)
    unsigned int eax;
    asm("xgetbv" : "=a"(eax) : "c"(0) : "edx");
    return (eax & bits) == bits;
#else
    // can't check, but if OSXSAVE is true let's assume it'll work
    return bits > 0; // ignore 'warning: unused parameter'
#endif
}

VC_TARGET_NO_SIMD
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
    case AVXImpl:
        return CpuId::hasOsxsave() && CpuId::hasAvx() && xgetbvCheck(0x6);
    case AVX2Impl:
        return false;
    case MICImpl:
        return CpuId::processorFamily() == 0xB && CpuId::processorModel() == 0x1
            && CpuId::isIntel();
    case CUDAImpl:
#ifdef VC_CUDA_TARGET // this is passed as a compiler flag to gcc as libVc_CUDA is compiled entirely with gcc
        return true;
#else
        return false;
#endif
    case ImplementationMask:
        return false;
    }
    return false;
}

VC_TARGET_NO_SIMD
Vc::Implementation bestImplementationSupported()
{
#ifdef VC_CUDA_TARGET
    return Vc::CUDAImpl;
#endif
    CpuId::init();

    if (CpuId::processorFamily() == 0xB && CpuId::processorModel() == 0x1
            && CpuId::isIntel()) {
        return Vc::MICImpl;
    }
    if (!CpuId::hasSse2 ()) return Vc::ScalarImpl;
    if (!CpuId::hasSse3 ()) return Vc::SSE2Impl;
    if (!CpuId::hasSsse3()) return Vc::SSE3Impl;
    if (!CpuId::hasSse41()) return Vc::SSSE3Impl;
    if (!CpuId::hasSse42()) return Vc::SSE41Impl;
    if (CpuId::hasAvx() && CpuId::hasOsxsave() && xgetbvCheck(0x6)) {
        return Vc::AVXImpl;
    }
    return Vc::SSE42Impl;
}

VC_TARGET_NO_SIMD
unsigned int extraInstructionsSupported()
{
    unsigned int flags = 0;
    if (CpuId::hasF16c()) flags |= Vc::Float16cInstructions;
    if (CpuId::hasFma4()) flags |= Vc::Fma4Instructions;
    if (CpuId::hasXop ()) flags |= Vc::XopInstructions;
    if (CpuId::hasPopcnt()) flags |= Vc::PopcntInstructions;
    if (CpuId::hasSse4a()) flags |= Vc::Sse4aInstructions;
    if (CpuId::hasFma ()) flags |= Vc::FmaInstructions;
    //if (CpuId::hasPclmulqdq()) flags |= Vc::PclmulqdqInstructions;
    //if (CpuId::hasAes()) flags |= Vc::AesInstructions;
    //if (CpuId::hasRdrand()) flags |= Vc::RdrandInstructions;
    return flags;
}

}

#undef VC_TARGET_NO_SIMD

// vim: sw=4 sts=4 et tw=100
