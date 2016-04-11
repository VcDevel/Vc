/*  This file is part of the Vc library. {{{
Copyright © 2012-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SSE_INTERLEAVEDMEMORY_TCC_
#define VC_SSE_INTERLEAVEDMEMORY_TCC_

#include "detail.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{
// InterleavedMemoryAccessBase::interleave {{{1
// 2 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1)
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1);
}
// 3 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1, v2);
}
// 4 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3)
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1, v2, v3);
}
// 5 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4)
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1, v2, v3);
    v4.scatter(m_data + 4, m_indexes);
}
// 6 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4, const typename V::AsArg v5)
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1, v2, v3);
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 4, m_indexes, v4, v5);
}
// 7 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4, const typename V::AsArg v5,
    const typename V::AsArg v6)
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 0, m_indexes, v0, v1, v2, v3);
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 4, m_indexes, v4, v5, v6);
}
// 8 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4, const typename V::AsArg v5,
    const typename V::AsArg v6, const typename V::AsArg v7)
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 0, m_indexes, v0, v1, v2, v3);
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 4, m_indexes, v4, v5, v6, v7);
}

// InterleavedMemoryAccessBase::deinterleave {{{1
// 2 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1)
    const
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1);
}
// 3 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1,
                                                                          V &v2) const
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1, v2);
}
// 4 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1,
                                                                          V &v2, V &v3)
    const
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1, v2, v3);
}
// 5 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1,
                                                                          V &v2, V &v3,
                                                                          V &v4) const
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1, v2, v3, v4);
}
// 6 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1,
                                                                          V &v2, V &v3,
                                                                          V &v4, V &v5)
    const
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1, v2, v3, v4, v5);
}
// 7 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1,
                                                                          V &v2, V &v3,
                                                                          V &v4, V &v5,
                                                                          V &v6) const
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1, v2, v3, v4, v5,
                                             v6);
}
// 8 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(
    V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6, V &v7) const
{
    Vc::Detail::InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1, v2, v3, v4, v5,
                                             v6, v7);
}
//}}}1
}
}

#endif  // VC_SSE_INTERLEAVEDMEMORY_TCC_

// vim: foldmethod=marker
