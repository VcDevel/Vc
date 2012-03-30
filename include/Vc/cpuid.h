/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

namespace Vc
{
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
        static inline ushort cacheLineSize() { return static_cast<ushort>(s_cacheLineSize) * 8u; }
        static inline ProcessorType processorType() { return s_processorType; }
        static inline uint processorFamily() { return s_processorFamily; }
        static inline uint processorModel() { return s_processorModel; }
        static inline bool isAmd   () { return s_ecx0 == 0x444D4163; }
        static inline bool isIntel () { return s_ecx0 == 0x6C65746E; }
        static inline bool hasSse3 () { return s_processorFeaturesC & (1 << 0); }
        static inline bool hasVmx  () { return s_processorFeaturesC & (1 << 5); }
        static inline bool hasSmx  () { return s_processorFeaturesC & (1 << 6); }
        static inline bool hasEst  () { return s_processorFeaturesC & (1 << 7); }
        static inline bool hasTm2  () { return s_processorFeaturesC & (1 << 8); }
        static inline bool hasSsse3() { return s_processorFeaturesC & (1 << 9); }
        static inline bool hasPdcm () { return s_processorFeaturesC & (1 << 15); }
        static inline bool hasSse41() { return s_processorFeaturesC & (1 << 19); }
        static inline bool hasSse42() { return s_processorFeaturesC & (1 << 20); }
        static inline bool hasAes  () { return s_processorFeaturesC & (1 << 25); }
        static inline bool hasOsxsave() { return s_processorFeaturesC & (1 << 27); }
        static inline bool hasAvx  () { return s_processorFeaturesC & (1 << 28); }
        static inline bool hasFpu  () { return s_processorFeaturesD & (1 << 0); }
        static inline bool hasVme  () { return s_processorFeaturesD & (1 << 1); }
        static inline bool hasDe   () { return s_processorFeaturesD & (1 << 2); }
        static inline bool hasPse  () { return s_processorFeaturesD & (1 << 3); }
        static inline bool hasTsc  () { return s_processorFeaturesD & (1 << 4); }
        static inline bool hasMsr  () { return s_processorFeaturesD & (1 << 5); }
        static inline bool hasPae  () { return s_processorFeaturesD & (1 << 6); }
        static inline bool hasMtrr () { return s_processorFeaturesD & (1 << 12); }
        static inline bool hasCmov () { return s_processorFeaturesD & (1 << 15); }
        static inline bool hasClfsh() { return s_processorFeaturesD & (1 << 19); }
        static inline bool hasAcpi () { return s_processorFeaturesD & (1 << 22); }
        static inline bool hasMmx  () { return s_processorFeaturesD & (1 << 23); }
        static inline bool hasSse  () { return s_processorFeaturesD & (1 << 25); }
        static inline bool hasSse2 () { return s_processorFeaturesD & (1 << 26); }
        static inline bool hasHtt  () { return s_processorFeaturesD & (1 << 28); }
        static inline bool hasSse4a() { return s_processorFeatures8C & (1 << 6); }
        static inline bool hasMisAlignSse() { return s_processorFeatures8C & (1 << 7); }
        static inline bool hasAmdPrefetch() { return s_processorFeatures8C & (1 << 8); }
        static inline bool hasXop ()        { return s_processorFeatures8C & (1 << 11); }
        static inline bool hasFma4 ()       { return s_processorFeatures8C & (1 << 16); }
        static inline bool hasRdtscp()      { return s_processorFeatures8D & (1 << 27); }
        static inline bool has3DNow()       { return s_processorFeatures8D & (1u << 31); }
        static inline bool has3DNowExt()    { return s_processorFeatures8D & (1 << 30); }
        static inline uint   L1Instruction() { return s_L1Instruction; }
        static inline uint   L1Data() { return s_L1Data; }
        static inline uint   L2Data() { return s_L2Data; }
        static inline uint   L3Data() { return s_L3Data; }
        static inline ushort L1InstructionLineSize() { return s_L1InstructionLineSize; }
        static inline ushort L1DataLineSize() { return s_L1DataLineSize; }
        static inline ushort L2DataLineSize() { return s_L2DataLineSize; }
        static inline ushort L3DataLineSize() { return s_L3DataLineSize; }
        static inline ushort prefetch() { return s_prefetch; }

    private:
        static void interpret(uchar byte, bool *checkLeaf4);

        static uint   s_ecx0;
        static uint   s_logicalProcessors;
        static uint   s_processorFeaturesC;
        static uint   s_processorFeaturesD;
        static uint   s_processorFeatures8C;
        static uint   s_processorFeatures8D;
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
} // namespace Vc

#endif // CPUID_H
