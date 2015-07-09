/*  This file is part of the Vc library. {{{
Copyright © 2014 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_COMMON_GATHERIMPLEMENTATION_H
#define VC_COMMON_GATHERIMPLEMENTATION_H

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{

enum class GatherScatterImplementation : int {
    SimpleLoop,
    SetIndexZero,
    BitScanLoop,
    PopcntSwitch
};

using SimpleLoopT   = std::integral_constant<GatherScatterImplementation, GatherScatterImplementation::SimpleLoop>;
using SetIndexZeroT = std::integral_constant<GatherScatterImplementation, GatherScatterImplementation::SetIndexZero>;
using BitScanLoopT  = std::integral_constant<GatherScatterImplementation, GatherScatterImplementation::BitScanLoop>;
using PopcntSwitchT = std::integral_constant<GatherScatterImplementation, GatherScatterImplementation::PopcntSwitch>;

template <typename V, typename MT, typename IT>
Vc_ALWAYS_INLINE void executeGather(SetIndexZeroT,
                                    V &v,
                                    const MT *mem,
                                    IT indexes,
                                    typename V::MaskArgument mask)
{
    indexes.setZeroInverted(static_cast<typename IT::Mask>(mask));
    const V tmp(mem, indexes);
    where(mask) | v = tmp;
}

template <typename V, typename MT, typename IT>
Vc_ALWAYS_INLINE void executeGather(SimpleLoopT,
                                    V &v,
                                    const MT *mem,
                                    const IT &indexes,
                                    typename V::MaskArgument mask)
{
    if (VC_IS_UNLIKELY(mask.isEmpty())) {
        return;
    }
    Common::unrolled_loop<std::size_t, 0, V::Size>([&](std::size_t i) {
        if (mask[i])
            v[i] = mem[indexes[i]];
    });
}

template <typename V, typename MT, typename IT>
Vc_ALWAYS_INLINE void executeGather(BitScanLoopT,
                                    V &v,
                                    const MT *mem,
                                    const IT &indexes,
                                    typename V::MaskArgument mask)
{
    size_t bits = mask.toInt();
    while (VC_IS_LIKELY(bits > 0)) {
        size_t i, j;
        asm("bsf %[bits],%[i]\n\t"
            "bsr %[bits],%[j]\n\t"
            "btr %[i],%[bits]\n\t"
            "btr %[j],%[bits]\n\t"
            : [i] "=r"(i), [j] "=r"(j), [bits] "+r"(bits));
        v[i] = mem[indexes[i]];
        v[j] = mem[indexes[j]];
    }

    /* Alternative from Vc::SSE (0.7)
    int bits = mask.toInt();
    while (bits) {
        const int i = _bit_scan_forward(bits);
        bits &= ~(1 << i); // btr?
        d.set(i, ith_value(i));
    }
    */
}

#ifdef VC_IMPL_POPCNT
template <typename V, typename MT, typename IT>
Vc_ALWAYS_INLINE void executeGather(PopcntSwitchT,
                                    V &v,
                                    const MT *mem,
                                    const IT &indexes,
                                    typename V::MaskArgument mask)
{
    unsigned int bits = mask.toInt();
    unsigned int low, high = 0;
    switch (_mm_popcnt_u32(bits)) {
    case 8:
        v.gather(mem, indexes);
        break;
    case 7:
        low = _bit_scan_forward(bits);
        bits ^= 1 << low;
        v[low] = mem[indexes[low]];
    case 6:
        high = _bit_scan_reverse(bits);
        v[high] = mem[indexes[high]];
        high = (1 << high);
    case 5:
        low = _bit_scan_forward(bits);
        bits ^= high | (1 << low);
        v[low] = mem[indexes[low]];
    case 4:
        high = _bit_scan_reverse(bits);
        v[high] = mem[indexes[high]];
        high = (1 << high);
    case 3:
        low = _bit_scan_forward(bits);
        bits ^= high | (1 << low);
        v[low] = mem[indexes[low]];
    case 2:
        high = _bit_scan_reverse(bits);
        v[high] = mem[indexes[high]];
    case 1:
        low = _bit_scan_forward(bits);
        v[low] = mem[indexes[low]];
    case 0:
        break;
    }
}
#endif

}  // namespace Common
}  // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_GATHERIMPLEMENTATION_H
