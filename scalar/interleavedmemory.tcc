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

#ifndef VC_SCALAR_INTERLEAVEDMEMORY_TCC_
#define VC_SCALAR_INTERLEAVEDMEMORY_TCC_

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
    m_data[internal_data(m_indexes).data() + 0] = v0.data();
    m_data[internal_data(m_indexes).data() + 1] = v1.data();
}
// 3 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
{
    m_data[internal_data(m_indexes).data() + 0] = v0.data();
    m_data[internal_data(m_indexes).data() + 1] = v1.data();
    m_data[internal_data(m_indexes).data() + 2] = v2.data();
}
// 4 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3)
{
    m_data[internal_data(m_indexes).data() + 0] = v0.data();
    m_data[internal_data(m_indexes).data() + 1] = v1.data();
    m_data[internal_data(m_indexes).data() + 2] = v2.data();
    m_data[internal_data(m_indexes).data() + 3] = v3.data();
}
// 5 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4)
{
    m_data[internal_data(m_indexes).data() + 0] = v0.data();
    m_data[internal_data(m_indexes).data() + 1] = v1.data();
    m_data[internal_data(m_indexes).data() + 2] = v2.data();
    m_data[internal_data(m_indexes).data() + 3] = v3.data();
    m_data[internal_data(m_indexes).data() + 4] = v4.data();
}
// 6 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4, const typename V::AsArg v5)
{
    m_data[internal_data(m_indexes).data() + 0] = v0.data();
    m_data[internal_data(m_indexes).data() + 1] = v1.data();
    m_data[internal_data(m_indexes).data() + 2] = v2.data();
    m_data[internal_data(m_indexes).data() + 3] = v3.data();
    m_data[internal_data(m_indexes).data() + 4] = v4.data();
    m_data[internal_data(m_indexes).data() + 5] = v5.data();
}
// 7 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4, const typename V::AsArg v5,
    const typename V::AsArg v6)
{
    m_data[internal_data(m_indexes).data() + 0] = v0.data();
    m_data[internal_data(m_indexes).data() + 1] = v1.data();
    m_data[internal_data(m_indexes).data() + 2] = v2.data();
    m_data[internal_data(m_indexes).data() + 3] = v3.data();
    m_data[internal_data(m_indexes).data() + 4] = v4.data();
    m_data[internal_data(m_indexes).data() + 5] = v5.data();
    m_data[internal_data(m_indexes).data() + 6] = v6.data();
}
// 8 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4, const typename V::AsArg v5,
    const typename V::AsArg v6, const typename V::AsArg v7)
{
    m_data[internal_data(m_indexes).data() + 0] = v0.data();
    m_data[internal_data(m_indexes).data() + 1] = v1.data();
    m_data[internal_data(m_indexes).data() + 2] = v2.data();
    m_data[internal_data(m_indexes).data() + 3] = v3.data();
    m_data[internal_data(m_indexes).data() + 4] = v4.data();
    m_data[internal_data(m_indexes).data() + 5] = v5.data();
    m_data[internal_data(m_indexes).data() + 6] = v6.data();
    m_data[internal_data(m_indexes).data() + 7] = v7.data();
}
// InterleavedMemoryAccessBase::deinterleave {{{1
// 2 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0,
                                                                          V &v1) const
{
    v0.data() = m_data[internal_data(m_indexes).data() + 0];
    v1.data() = m_data[internal_data(m_indexes).data() + 1];
}
// 3 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1,
                                                                          V &v2) const
{
    v0.data() = m_data[internal_data(m_indexes).data() + 0];
    v1.data() = m_data[internal_data(m_indexes).data() + 1];
    v2.data() = m_data[internal_data(m_indexes).data() + 2];
}
// 4 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1,
                                                                          V &v2,
                                                                          V &v3) const
{
    v0.data() = m_data[internal_data(m_indexes).data() + 0];
    v1.data() = m_data[internal_data(m_indexes).data() + 1];
    v2.data() = m_data[internal_data(m_indexes).data() + 2];
    v3.data() = m_data[internal_data(m_indexes).data() + 3];
}
// 5 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1,
                                                                          V &v2, V &v3,
                                                                          V &v4) const
{
    v0.data() = m_data[internal_data(m_indexes).data() + 0];
    v1.data() = m_data[internal_data(m_indexes).data() + 1];
    v2.data() = m_data[internal_data(m_indexes).data() + 2];
    v3.data() = m_data[internal_data(m_indexes).data() + 3];
    v4.data() = m_data[internal_data(m_indexes).data() + 4];
}
// 6 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1,
                                                                          V &v2, V &v3,
                                                                          V &v4,
                                                                          V &v5) const
{
    v0.data() = m_data[internal_data(m_indexes).data() + 0];
    v1.data() = m_data[internal_data(m_indexes).data() + 1];
    v2.data() = m_data[internal_data(m_indexes).data() + 2];
    v3.data() = m_data[internal_data(m_indexes).data() + 3];
    v4.data() = m_data[internal_data(m_indexes).data() + 4];
    v5.data() = m_data[internal_data(m_indexes).data() + 5];
}
// 7 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1,
                                                                          V &v2, V &v3,
                                                                          V &v4, V &v5,
                                                                          V &v6) const
{
    v0.data() = m_data[internal_data(m_indexes).data() + 0];
    v1.data() = m_data[internal_data(m_indexes).data() + 1];
    v2.data() = m_data[internal_data(m_indexes).data() + 2];
    v3.data() = m_data[internal_data(m_indexes).data() + 3];
    v4.data() = m_data[internal_data(m_indexes).data() + 4];
    v5.data() = m_data[internal_data(m_indexes).data() + 5];
    v6.data() = m_data[internal_data(m_indexes).data() + 6];
}
// 8 args {{{2
template <typename V, typename I, bool RO>
Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(
    V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6, V &v7) const
{
    v0.data() = m_data[internal_data(m_indexes).data() + 0];
    v1.data() = m_data[internal_data(m_indexes).data() + 1];
    v2.data() = m_data[internal_data(m_indexes).data() + 2];
    v3.data() = m_data[internal_data(m_indexes).data() + 3];
    v4.data() = m_data[internal_data(m_indexes).data() + 4];
    v5.data() = m_data[internal_data(m_indexes).data() + 5];
    v6.data() = m_data[internal_data(m_indexes).data() + 6];
    v7.data() = m_data[internal_data(m_indexes).data() + 7];
}
// }}}1
}  // namespace Common
}  // namespace Vc

#endif // VC_SCALAR_INTERLEAVEDMEMORY_TCC_

// vim: foldmethod=marker
