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

#include "unittest.h"

TEST(testCompiledImplementation) { VERIFY(Vc::currentImplementationSupported()); }

TEST(testIsSupported)
{
    using Vc::CpuId;
    VERIFY(Vc::isImplementationSupported(Vc::ScalarImpl));
    COMPARE(Vc::isImplementationSupported(Vc::SSE2Impl ), CpuId::hasSse2());
    COMPARE(Vc::isImplementationSupported(Vc::SSE3Impl ), CpuId::hasSse3());
    COMPARE(Vc::isImplementationSupported(Vc::SSSE3Impl), CpuId::hasSsse3());
    COMPARE(Vc::isImplementationSupported(Vc::SSE41Impl), CpuId::hasSse41());
    COMPARE(Vc::isImplementationSupported(Vc::SSE42Impl), CpuId::hasSse42());
    COMPARE(Vc::isImplementationSupported(Vc::AVXImpl  ), CpuId::hasOsxsave() && CpuId::hasAvx());
    COMPARE(Vc::isImplementationSupported(Vc::AVX2Impl ), CpuId::hasOsxsave() && CpuId::hasAvx2());
}

TEST(testBestImplementation)
{
    // when building with a recent and fully featured compiler the following should pass
    // but - old GCC versions have to fall back to Scalar, even though SSE is supported by the CPU
    //     - ICC/MSVC can't use XOP/FMA4
    COMPARE(Vc::bestImplementationSupported(), Vc::CurrentImplementation::current());
}

TEST(testExtraInstructions)
{
    using Vc::CpuId;
    unsigned int extra = Vc::extraInstructionsSupported();
    COMPARE(!(extra & Vc::Float16cInstructions), !CpuId::hasF16c());
    COMPARE(!(extra & Vc::XopInstructions), !CpuId::hasXop());
    COMPARE(!(extra & Vc::Fma4Instructions), !CpuId::hasFma4());
    COMPARE(!(extra & Vc::PopcntInstructions), !CpuId::hasPopcnt());
    COMPARE(!(extra & Vc::Sse4aInstructions), !CpuId::hasSse4a());
    COMPARE(!(extra & Vc::FmaInstructions), !CpuId::hasFma());
    COMPARE(!(extra & Vc::Bmi2Instructions), !CpuId::hasBmi2());
}

// vim: foldmethod=marker
