/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

#ifndef CPUID_H
#define CPUID_H

#include <iostream>

class CpuId
{
    typedef unsigned char uchar;
    typedef unsigned short ushort;
    typedef unsigned int uint;

    public:
        enum ProcessorType {
            OriginalOemProcessor = 0,
            IntelOverDriveProcessor = 1,
            DualProcessor = 2,
            IntelReserved = 3
        };

        static void init();
        static ushort cacheLineSize() { return static_cast<ushort>(s_cacheLineSize) * 8u; }
        static ProcessorType processorType() { return s_processorType; }
        static uchar processorFamily() { return s_processorFamily; }
        static uchar processorModel() { return s_processorModel; }
        static bool hasSse3 () { return s_processorFeaturesC & (1 << 0); }
        static bool hasVmx  () { return s_processorFeaturesC & (1 << 5); }
        static bool hasSmx  () { return s_processorFeaturesC & (1 << 6); }
        static bool hasEst  () { return s_processorFeaturesC & (1 << 7); }
        static bool hasTm2  () { return s_processorFeaturesC & (1 << 8); }
        static bool hasSsse3() { return s_processorFeaturesC & (1 << 9); }
        static bool hasPdcm () { return s_processorFeaturesC & (1 << 15); }
        static bool hasSse41() { return s_processorFeaturesC & (1 << 19); }
        static bool hasSse42() { return s_processorFeaturesC & (1 << 20); }
        static bool hasAes  () { return s_processorFeaturesC & (1 << 25); }
        static bool hasFpu  () { return s_processorFeaturesD & (1 << 0); }
        static bool hasVme  () { return s_processorFeaturesD & (1 << 1); }
        static bool hasDe   () { return s_processorFeaturesD & (1 << 2); }
        static bool hasPse  () { return s_processorFeaturesD & (1 << 3); }
        static bool hasTsc  () { return s_processorFeaturesD & (1 << 4); }
        static bool hasMsr  () { return s_processorFeaturesD & (1 << 5); }
        static bool hasPae  () { return s_processorFeaturesD & (1 << 6); }
        static bool hasMtrr () { return s_processorFeaturesD & (1 << 12); }
        static bool hasCmov () { return s_processorFeaturesD & (1 << 15); }
        static bool hasClfsh() { return s_processorFeaturesD & (1 << 19); }
        static bool hasAcpi () { return s_processorFeaturesD & (1 << 22); }
        static bool hasMmx  () { return s_processorFeaturesD & (1 << 23); }
        static bool hasSse  () { return s_processorFeaturesD & (1 << 25); }
        static bool hasSse2 () { return s_processorFeaturesD & (1 << 26); }
        static bool hasHtt  () { return s_processorFeaturesD & (1 << 28); }
        static uint   L1Instruction() { return s_L1Instruction; }
        static uint   L1Data() { return s_L1Data; }
        static uint   L2Data() { return s_L2Data; }
        static uint   L3Data() { return s_L3Data; }
        static ushort L1InstructionLineSize() { return s_L1InstructionLineSize; }
        static ushort L1DataLineSize() { return s_L1DataLineSize; }
        static ushort L2DataLineSize() { return s_L2DataLineSize; }
        static ushort L3DataLineSize() { return s_L3DataLineSize; }
        static ushort prefetch() { return s_prefetch; }

    private:
        static void interpret(uchar byte);

        static uint   s_logicalProcessors;
        static uint   s_processorFeaturesC;
        static uint   s_processorFeaturesD;
        static uint   s_L1Instruction;
        static uint   s_L1Data;
        static uint   s_L2Data;
        static uint   s_L3Data;
        static ushort s_L1InstructionLineSize;
        static ushort s_L1DataLineSize;
        static ushort s_L2DataLineSize;
        static ushort s_L3DataLineSize;
        static ushort s_prefetch;
        static uchar  s_brandIndex;
        static uchar  s_cacheLineSize;
        static uchar  s_processorModel;
        static uchar  s_processorFamily;
        static ProcessorType s_processorType;
        static bool   s_noL2orL3;
};

CpuId::uint   CpuId::s_logicalProcessors = 0;
CpuId::uint   CpuId::s_processorFeaturesC = 0;
CpuId::uint   CpuId::s_processorFeaturesD = 0;
CpuId::uint   CpuId::s_L1Instruction = 0;
CpuId::uint   CpuId::s_L1Data = 0;
CpuId::uint   CpuId::s_L2Data = 0;
CpuId::uint   CpuId::s_L3Data = 0;
CpuId::ushort CpuId::s_L1InstructionLineSize = 0;
CpuId::ushort CpuId::s_L1DataLineSize = 0;
CpuId::ushort CpuId::s_L2DataLineSize = 0;
CpuId::ushort CpuId::s_L3DataLineSize = 0;
CpuId::ushort CpuId::s_prefetch = 0;
CpuId::uchar  CpuId::s_brandIndex = 0;
CpuId::uchar  CpuId::s_cacheLineSize = 0;
CpuId::uchar  CpuId::s_processorModel = 0;
CpuId::uchar  CpuId::s_processorFamily = 0;
CpuId::ProcessorType CpuId::s_processorType = CpuId::IntelReserved;
bool   CpuId::s_noL2orL3 = false;

#ifdef _MSC_VER
#define CPUID(id) \
	do { \
		uint &a = eax, &b = ebx, &c = ecx, &d = edx; \
		__asm { \
			mov eax, id \
			cpuid \
			mov a, eax \
			mov b, ebx \
			mov c, ecx \
			mov d, edx \
		} \
	} while (false)
#else
#define CPUID(id) \
    __asm__("mov $" #id ",%%eax\n\tcpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx))
#endif
void CpuId::init()
{
    {
        static bool done = false;
        if (done) return;
        done = true;
    }
    uint eax, ebx, ecx, edx;

    CPUID(0);
    const bool isAmd = (ecx == 0x444D4163);

    CPUID(1);
    s_processorFeaturesC = ecx;
    s_processorFeaturesD = edx;
    s_processorModel  = (eax & 0x000000f0) >> 4;
    s_processorFamily = (eax & 0x00000f00) >> 8;
    if (isAmd) {
        if (s_processorFamily >= 0xf) {
            const uchar processorFamilyExt = (eax & 0x0ff00000) >> 20;
            s_processorFamily += processorFamilyExt;
            const uchar processorModelExt = (eax & 0x000f0000) >> 12;
            s_processorModel += processorModelExt;
        }
    } else if (s_processorFamily == 0xf) {
        const uchar processorFamilyExt = (eax & 0x0ff00000) >> 20;
        s_processorFamily += processorFamilyExt;
        const uchar processorModelExt = (eax & 0x000f0000) >> 12;
        s_processorModel += processorModelExt;
    } else if (s_processorFamily == 0x6) {
        const uchar processorModelExt = (eax & 0x000f0000) >> 12;
        s_processorModel += processorModelExt;
    }
    s_processorType = static_cast<ProcessorType>((eax & 0x00003000) >> 12);

    s_brandIndex = ebx & 0xff;
    ebx >>= 8;
    s_cacheLineSize = ebx & 0xff;
    ebx >>= 8;
    s_logicalProcessors = ebx & 0xff;

    if (isAmd) {
        CPUID(0x80000005);
        s_L1DataLineSize = ecx & 0xff;
        s_L1Data = (ecx >> 24) * 1024;
        s_L1InstructionLineSize = edx & 0xff;
        s_L1Instruction = (edx >> 24) * 1024;

        CPUID(0x80000006);
        s_L2DataLineSize = ecx & 0xff;
        s_L2Data = (ecx >> 16) * 1024;
        s_L3DataLineSize = edx & 0xff;
        s_L3Data = (edx >> 18) * 512 * 1024;
        return;
    }

    // Intel only
    int repeat = 0;
    do {
        CPUID(2);
        if (repeat == 0) {
            repeat = eax & 0xff;
        }
        if (0 == (0x80000000u & eax)) {
            for (int i = 0; i < 3; ++i) {
                eax >>= 8;
                interpret(eax & 0xff);
            }
        }
        if (0 == (0x80000000u & ebx)) {
            for (int i = 0; i < 4; ++i) {
                interpret(ebx & 0xff);
                ebx >>= 8;
            }
        }
        if (0 == (0x80000000u & ecx)) {
            for (int i = 0; i < 4; ++i) {
                interpret(ecx & 0xff);
                ecx >>= 8;
            }
        }
        if (0 == (0x80000000u & edx)) {
            for (int i = 0; i < 4; ++i) {
                interpret(edx & 0xff);
                edx >>= 8;
            }
        }
    } while (--repeat > 0);
}

void CpuId::interpret(uchar byte)
{
    switch (byte) {
    case 0x06:
        s_L1Instruction = 8 * 1024;
        s_L1InstructionLineSize = 32;
        break;
    case 0x08:
        s_L1Instruction = 16 * 1024;
        s_L1InstructionLineSize = 32;
        break;
    case 0x09:
        s_L1Instruction = 32 * 1024;
        s_L1InstructionLineSize = 64;
        break;
    case 0x0A:
        s_L1Data = 8 * 1024;
        s_L1DataLineSize = 32;
        break;
    case 0x0C:
        s_L1Data = 16 * 1024;
        s_L1DataLineSize = 32;
        break;
    case 0x0D:
        s_L1Data = 16 * 1024;
        s_L1DataLineSize = 64;
        break;
    case 0x0E:
        s_L1Data = 24 * 1024;
        s_L1DataLineSize = 64;
        break;
    case 0x21:
        s_L2Data = 256 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0x22:
        s_L3Data = 512 * 1024;
        s_L3DataLineSize = 64;
        break;
    case 0x23:
        s_L3Data = 1024 * 1024;
        s_L3DataLineSize = 64;
        break;
    case 0x25:
        s_L3Data = 2 * 1024 * 1024;
        s_L3DataLineSize = 64;
        break;
    case 0x29:
        s_L3Data = 4 * 1024 * 1024;
        s_L3DataLineSize = 64;
        break;
    case 0x2C:
        s_L1Data = 32 * 1024;
        s_L1DataLineSize = 64;
        break;
    case 0x30:
        s_L1Data = 32 * 1024;
        s_L1DataLineSize = 64;
        break;
    case 0x40:
        s_noL2orL3 = true;
        break;
    case 0x41:
        s_L2Data = 128 * 1024;
        s_L2DataLineSize = 32;
        break;
    case 0x42:
        s_L2Data = 256 * 1024;
        s_L2DataLineSize = 32;
        break;
    case 0x43:
        s_L2Data = 512 * 1024;
        s_L2DataLineSize = 32;
        break;
    case 0x44:
        s_L2Data = 1024 * 1024;
        s_L2DataLineSize = 32;
        break;
    case 0x45:
        s_L2Data = 2 * 1024 * 1024;
        s_L2DataLineSize = 32;
        break;
    case 0x46:
        s_L3Data = 4 * 1024 * 1024;
        s_L3DataLineSize = 64;
        break;
    case 0x47:
        s_L3Data = 8 * 1024 * 1024;
        s_L3DataLineSize = 64;
        break;
    case 0x48:
        s_L2Data = 3 * 1024 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0x49:
        if (s_processorFamily == 0xf && s_processorModel == 0x6) {
            s_L3Data = 4 * 1024 * 1024;
            s_L3DataLineSize = 64;
        } else {
            s_L2Data = 4 * 1024 * 1024;
            s_L2DataLineSize = 64;
        }
        break;
    case 0x4A:
        s_L3Data = 6 * 1024 * 1024;
        s_L3DataLineSize = 64;
        break;
    case 0x4B:
        s_L3Data = 8 * 1024 * 1024;
        s_L3DataLineSize = 64;
        break;
    case 0x4C:
        s_L3Data = 12 * 1024 * 1024;
        s_L3DataLineSize = 64;
        break;
    case 0x4D:
        s_L3Data = 16 * 1024 * 1024;
        s_L3DataLineSize = 64;
        break;
    case 0x4E:
        s_L2Data = 6 * 1024 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0x60:
        s_L1Data = 16 * 1024;
        s_L1DataLineSize = 64;
        break;
    case 0x66:
        s_L1Data = 8 * 1024;
        s_L1DataLineSize = 64;
        break;
    case 0x67:
        s_L1Data = 16 * 1024;
        s_L1DataLineSize = 64;
        break;
    case 0x68:
        s_L1Data = 32 * 1024;
        s_L1DataLineSize = 64;
        break;
    case 0x78:
        s_L2Data = 1024 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0x79:
        s_L2Data = 128 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0x7A:
        s_L2Data = 256 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0x7B:
        s_L2Data = 512 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0x7C:
        s_L2Data = 1024 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0x7D:
        s_L2Data = 2 * 1024 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0x7F:
        s_L2Data = 512 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0x80:
        s_L2Data = 512 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0x82:
        s_L2Data = 256 * 1024;
        s_L2DataLineSize = 32;
        break;
    case 0x83:
        s_L2Data = 512 * 1024;
        s_L2DataLineSize = 32;
        break;
    case 0x84:
        s_L2Data = 1024 * 1024;
        s_L2DataLineSize = 32;
        break;
    case 0x85:
        s_L2Data = 2 * 1024 * 1024;
        s_L2DataLineSize = 32;
        break;
    case 0x86:
        s_L2Data = 512 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0x87:
        s_L2Data = 1024 * 1024;
        s_L2DataLineSize = 64;
        break;
    case 0xE4:
        s_L3Data = 8 * 1024 * 1024;
        s_L3DataLineSize = 64;
        break;
    case 0xF0:
        s_prefetch = 64;
        break;
    case 0xF1:
        s_prefetch = 128;
        break;
    default:
        break;
    }
}

static int _Global_CpuId_Initializer() { CpuId::init(); return 0; }
static int _Global_CpuId_Initializer_Data = _Global_CpuId_Initializer();

#endif // CPUID_H
