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
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_NEON_VECTOR_TCC_
#define VC_NEON_VECTOR_TCC_
#include <random>
#include "macros.h"


namespace Vc_VERSIONED_NAMESPACE
{
namespace NEON
{

template <> Vc_INTRINSIC int_v::Vector(VectorSpecialInitializerZero::ZEnum)
{
}

static const int IndexesFromZeroData[4] = {0, 1, 2, 3};
template <>
Vc_INTRINSIC int_v::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : d(vld1q_s32(IndexesFromZeroData))
{
}

template <> Vc_INTRINSIC int_v::Vector(int x) : d(vdupq_n_s32(x))
{
}
template <> Vc_INTRINSIC int_v &int_v::operator+=(const int_v &x)
{
    d.v() = vaddq_s32(d.v(), x.d.v());
    return *this;
}
template <> Vc_INTRINSIC Vc_PURE int_v int_v::operator+(const int_v &x) const
{
    return vaddq_s32(d.v(), x.d.v());
}

template <> Vc_INTRINSIC int_v int_v::Random()
{
    int_v a;
    static std::default_random_engine ram;
    a.d.v() = vsetq_lane_s32(ram(), a.d.v(), 0);
    a.d.v() = vsetq_lane_s32(ram(), a.d.v(), 1);
    a.d.v() = vsetq_lane_s32(ram(), a.d.v(), 2);
    a.d.v() = vsetq_lane_s32(ram(), a.d.v(), 3);
    return a;
}
template <>
template <typename U, typename Flags, typename>
Vc_INTRINSIC void int_v::load(const U *mem, Flags)
{
    d.v() = vld1q_s32(mem);
}
template <>
template <typename U, typename Flags, typename>
Vc_INTRINSIC void int_v::store(U *mem, Flags) const
{
}

template <> Vc_INTRINSIC void int_v::setZero()
{
    d.v() = VectorType();
}

}  // namespace NEON
}  // namespace Vc

#include "undomacros.h"

#endif  // VC_NEON_VECTOR_TCC_

// vim: foldmethod=marker
