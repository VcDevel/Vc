/*  This file is part of the Vc library. {{{
Copyright © 2015 Matthias Kretz <kretz@kde.org>

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

#include <Vc/cpuid.h>
#include <iostream>

int Vc_CDECL main()
{
    using Vc::CpuId;
    std::cout << "        cacheLineSize: " << CpuId::cacheLineSize() << '\n';
    std::cout << "        processorType: " << CpuId::processorType() << '\n';
    std::cout << "      processorFamily: " << CpuId::processorFamily() << '\n';
    std::cout << "       processorModel: " << CpuId::processorModel() << '\n';
    std::cout << "    logicalProcessors: " << CpuId::logicalProcessors() << '\n';
    std::cout << "                isAmd: " << CpuId::isAmd   () << '\n';
    std::cout << "              isIntel: " << CpuId::isIntel () << '\n';
    std::cout << "              hasSse3: " << CpuId::hasSse3 () << '\n';
    std::cout << "         hasPclmulqdq: " << CpuId::hasPclmulqdq() << '\n';
    std::cout << "           hasMonitor: " << CpuId::hasMonitor() << '\n';
    std::cout << "               hasVmx: " << CpuId::hasVmx  () << '\n';
    std::cout << "               hasSmx: " << CpuId::hasSmx  () << '\n';
    std::cout << "              hasEist: " << CpuId::hasEist () << '\n';
    std::cout << "               hasTm2: " << CpuId::hasTm2  () << '\n';
    std::cout << "             hasSsse3: " << CpuId::hasSsse3() << '\n';
    std::cout << "               hasFma: " << CpuId::hasFma  () << '\n';
    std::cout << "        hasCmpXchg16b: " << CpuId::hasCmpXchg16b() << '\n';
    std::cout << "              hasPdcm: " << CpuId::hasPdcm () << '\n';
    std::cout << "               hasDca: " << CpuId::hasDca()   << '\n';
    std::cout << "             hasSse41: " << CpuId::hasSse41() << '\n';
    std::cout << "             hasSse42: " << CpuId::hasSse42() << '\n';
    std::cout << "             hasMovbe: " << CpuId::hasMovbe() << '\n';
    std::cout << "            hasPopcnt: " << CpuId::hasPopcnt()<< '\n';
    std::cout << "               hasAes: " << CpuId::hasAes  () << '\n';
    std::cout << "           hasOsxsave: " << CpuId::hasOsxsave() << '\n';
    std::cout << "               hasAvx: " << CpuId::hasAvx  () << '\n';
    std::cout << "              hasBmi1: " << CpuId::hasBmi1 () << '\n';
    std::cout << "               hasHle: " << CpuId::hasHle  () << '\n';
    std::cout << "              hasAvx2: " << CpuId::hasAvx2 () << '\n';
    std::cout << "              hasBmi2: " << CpuId::hasBmi2 () << '\n';
    std::cout << "               hasRtm: " << CpuId::hasRtm  () << '\n';
    std::cout << "           hasAvx512f: " << CpuId::hasAvx512f   () << '\n';
    std::cout << "          hasAvx512dq: " << CpuId::hasAvx512dq  () << '\n';
    std::cout << "        hasAvx512ifma: " << CpuId::hasAvx512ifma() << '\n';
    std::cout << "          hasAvx512pf: " << CpuId::hasAvx512pf  () << '\n';
    std::cout << "          hasAvx512er: " << CpuId::hasAvx512er  () << '\n';
    std::cout << "          hasAvx512cd: " << CpuId::hasAvx512cd  () << '\n';
    std::cout << "          hasAvx512bw: " << CpuId::hasAvx512bw  () << '\n';
    std::cout << "          hasAvx512vl: " << CpuId::hasAvx512vl  () << '\n';
    std::cout << "        hasAvx512vbmi: " << CpuId::hasAvx512vbmi() << '\n';
    std::cout << "              hasF16c: " << CpuId::hasF16c () << '\n';
    std::cout << "            hasRdrand: " << CpuId::hasRdrand()<< '\n';
    std::cout << "               hasFpu: " << CpuId::hasFpu  () << '\n';
    std::cout << "               hasVme: " << CpuId::hasVme  () << '\n';
    std::cout << "                hasDe: " << CpuId::hasDe   () << '\n';
    std::cout << "               hasPse: " << CpuId::hasPse  () << '\n';
    std::cout << "               hasTsc: " << CpuId::hasTsc  () << '\n';
    std::cout << "               hasMsr: " << CpuId::hasMsr  () << '\n';
    std::cout << "               hasPae: " << CpuId::hasPae  () << '\n';
    std::cout << "               hasCx8: " << CpuId::hasCx8  () << '\n';
    std::cout << "              hasMtrr: " << CpuId::hasMtrr () << '\n';
    std::cout << "              hasCmov: " << CpuId::hasCmov () << '\n';
    std::cout << "             hasClfsh: " << CpuId::hasClfsh() << '\n';
    std::cout << "              hasAcpi: " << CpuId::hasAcpi () << '\n';
    std::cout << "               hasMmx: " << CpuId::hasMmx  () << '\n';
    std::cout << "               hasSse: " << CpuId::hasSse  () << '\n';
    std::cout << "              hasSse2: " << CpuId::hasSse2 () << '\n';
    std::cout << "               hasHtt: " << CpuId::hasHtt  () << '\n';
    std::cout << "             hasSse4a: " << CpuId::hasSse4a() << '\n';
    std::cout << "       hasMisAlignSse: " << CpuId::hasMisAlignSse() << '\n';
    std::cout << "       hasAmdPrefetch: " << CpuId::hasAmdPrefetch() << '\n';
    std::cout << "               hasXop: " << CpuId::hasXop ()        << '\n';
    std::cout << "              hasFma4: " << CpuId::hasFma4 ()       << '\n';
    std::cout << "            hasRdtscp: " << CpuId::hasRdtscp()      << '\n';
    std::cout << "             has3DNow: " << CpuId::has3DNow()       << '\n';
    std::cout << "          has3DNowExt: " << CpuId::has3DNowExt()    << '\n';
    std::cout << "        L1Instruction: " << CpuId::L1Instruction() << '\n';
    std::cout << "               L1Data: " << CpuId::L1Data() << '\n';
    std::cout << "               L2Data: " << CpuId::L2Data() << '\n';
    std::cout << "               L3Data: " << CpuId::L3Data() << '\n';
    std::cout << "L1InstructionLineSize: " << CpuId::L1InstructionLineSize() << '\n';
    std::cout << "       L1DataLineSize: " << CpuId::L1DataLineSize() << '\n';
    std::cout << "       L2DataLineSize: " << CpuId::L2DataLineSize() << '\n';
    std::cout << "       L3DataLineSize: " << CpuId::L3DataLineSize() << '\n';
    std::cout << "      L1Associativity: " << CpuId::L1Associativity() << '\n';
    std::cout << "      L2Associativity: " << CpuId::L2Associativity() << '\n';
    std::cout << "      L3Associativity: " << CpuId::L3Associativity() << '\n';
    std::cout << "             prefetch: " << CpuId::prefetch() << '\n';
    return 0;
}

// vim: foldmethod=marker
