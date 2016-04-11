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

#ifndef VC_MIC_INTERLEAVEDMEMORY_TCC_
#define VC_MIC_INTERLEAVEDMEMORY_TCC_

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{

namespace
{
using namespace Vc::MicIntrinsics;
typedef std::integral_constant<int, 2> Size2;
typedef std::integral_constant<int, 3> Size3;
typedef std::integral_constant<int, 4> Size4;
typedef std::integral_constant<int, 5> Size5;
typedef std::integral_constant<int, 6> Size6;
typedef std::integral_constant<int, 7> Size7;
typedef std::integral_constant<int, 8> Size8;

template<typename V> struct InterleaveImpl {
    typedef typename V::IndexType IT;
    typedef typename V::EntryType T;
    typedef typename V::VectorEntryType VT;
    typedef MIC::UpDownConversion<VT, T> UpDownC;

    static inline __m512i fixup(const SimdArray<int, 16> &i)
    {
        return internal_data(i).data();
    }
    static inline __m512i fixup(const SimdArray<int, 8> &i)
    {
        return _mm512_mask_loadunpacklo_epi32(_mm512_setzero_epi32(), 0x00ff, &i);
    }
    template <size_t StructSize>
    static inline __m512i fixup(const SuccessiveEntries<StructSize> &i)
    {
        using namespace Detail;
        return (int_v::IndexesFromZero() * int_v(StructSize) + int_v(i.data())).data();
    }

    // deinterleave 2 {{{1
    template <typename I>
    static inline std::tuple<V, V> deinterleave(I &&indexes, const T *const data, Size2)
    {
        const auto i = fixup(std::forward<I>(indexes));
        return std::tuple<V, V>{gather(i, data + 0, UpDownC()),
                                gather(i, data + 1, UpDownC())};
    }

    // deinterleave 3 {{{1
    template <typename I>
    static inline std::tuple<V, V, V> deinterleave(I &&indexes, const T *const data, Size3)
    {
        const auto i = fixup(std::forward<I>(indexes));
        return std::tuple<V, V, V>{gather(i, data + 0, UpDownC()),
                                   gather(i, data + 1, UpDownC()),
                                   gather(i, data + 2, UpDownC())};
    }

    // deinterleave 4 {{{1
    template <typename I>
    static inline std::tuple<V, V, V, V> deinterleave(I &&indexes, const T *const data,
                                                      Size4)
    {
        const auto i = fixup(std::forward<I>(indexes));
        return std::tuple<V, V, V, V>{
            gather(i, data + 0, UpDownC()), gather(i, data + 1, UpDownC()),
            gather(i, data + 2, UpDownC()), gather(i, data + 3, UpDownC())};
    }

    // deinterleave 5 {{{1
    template <typename I>
    static inline std::tuple<V, V, V, V, V> deinterleave(I &&indexes, const T *const data,
                                                         Size5)
    {
        const auto i = fixup(std::forward<I>(indexes));
        return std::tuple<V, V, V, V, V>{
            gather(i, data + 0, UpDownC()), gather(i, data + 1, UpDownC()),
            gather(i, data + 2, UpDownC()), gather(i, data + 3, UpDownC()),
            gather(i, data + 4, UpDownC())};
    }

    // deinterleave 6 {{{1
    template <typename I>
    static inline std::tuple<V, V, V, V, V, V> deinterleave(I &&indexes,
                                                            const T *const data, Size6)
    {
        const auto i = fixup(std::forward<I>(indexes));
        return std::tuple<V, V, V, V, V, V>{
            gather(i, data + 0, UpDownC()), gather(i, data + 1, UpDownC()),
            gather(i, data + 2, UpDownC()), gather(i, data + 3, UpDownC()),
            gather(i, data + 4, UpDownC()), gather(i, data + 5, UpDownC())};
    }

    // deinterleave 7 {{{1
    template <typename I>
    static inline std::tuple<V, V, V, V, V, V, V> deinterleave(I &&indexes,
                                                               const T *const data, Size7)
    {
        const auto i = fixup(std::forward<I>(indexes));
        return std::tuple<V, V, V, V, V, V, V>{
            gather(i, data + 0, UpDownC()), gather(i, data + 1, UpDownC()),
            gather(i, data + 2, UpDownC()), gather(i, data + 3, UpDownC()),
            gather(i, data + 4, UpDownC()), gather(i, data + 5, UpDownC()),
            gather(i, data + 6, UpDownC())};
    }

    // deinterleave 8 {{{1
    template <typename I>
    static inline std::tuple<V, V, V, V, V, V, V, V> deinterleave(I &&indexes,
                                                                  const T *const data,
                                                                  Size8)
    {
        const auto i = fixup(std::forward<I>(indexes));
        return std::tuple<V, V, V, V, V, V, V, V>{
            gather(i, data + 0, UpDownC()), gather(i, data + 1, UpDownC()),
            gather(i, data + 2, UpDownC()), gather(i, data + 3, UpDownC()),
            gather(i, data + 4, UpDownC()), gather(i, data + 5, UpDownC()),
            gather(i, data + 6, UpDownC()), gather(i, data + 7, UpDownC())};
    }

    // interleave 2 {{{1
    template <typename I>
    static inline void interleave(I &&indexes, T *const data, V v0, V v1)
    {
        const int_v i = fixup(indexes);
        v0.scatter(data + 0, i);
        v1.scatter(data + 1, i);
    }

    // interleave 3 {{{1
    template <typename I>
    static inline void interleave(I &&indexes, T *const data, V v0, V v1, V v2)
    {
        const int_v i = fixup(indexes);
        v0.scatter(data + 0, i);
        v1.scatter(data + 1, i);
        v2.scatter(data + 2, i);
    }

    // interleave 4 {{{1
    template <typename I>
    static inline void interleave(I &&indexes, T *const data, V v0, V v1, V v2, V v3)
    {
        const int_v i = fixup(indexes);
        v0.scatter(data + 0, i);
        v1.scatter(data + 1, i);
        v2.scatter(data + 2, i);
        v3.scatter(data + 3, i);
    }

    // interleave 5 {{{1
    template <typename I>
    static inline void interleave(I &&indexes, T *const data, V v0, V v1, V v2, V v3,
                                  V v4)
    {
        const int_v i = fixup(indexes);
        v0.scatter(data + 0, i);
        v1.scatter(data + 1, i);
        v2.scatter(data + 2, i);
        v3.scatter(data + 3, i);
        v4.scatter(data + 4, i);
    }

    // interleave 6 {{{1
    template <typename I>
    static inline void interleave(I &&indexes, T *const data, V v0, V v1, V v2, V v3,
                                  V v4, V v5)
    {
        const int_v i = fixup(indexes);
        v0.scatter(data + 0, i);
        v1.scatter(data + 1, i);
        v2.scatter(data + 2, i);
        v3.scatter(data + 3, i);
        v4.scatter(data + 4, i);
        v5.scatter(data + 5, i);
    }

    // interleave 7 {{{1
    template <typename I>
    static inline void interleave(I &&indexes, T *const data, V v0, V v1, V v2, V v3,
                                  V v4, V v5, V v6)
    {
        const int_v i = fixup(indexes);
        v0.scatter(data + 0, i);
        v1.scatter(data + 1, i);
        v2.scatter(data + 2, i);
        v3.scatter(data + 3, i);
        v4.scatter(data + 4, i);
        v5.scatter(data + 5, i);
        v6.scatter(data + 6, i);
    }

    // interleave 8 {{{1
    template <typename I>
    static inline void interleave(I &&indexes, T *const data, V v0, V v1, V v2, V v3,
                                  V v4, V v5, V v6, V v7)
    {
        const int_v i = fixup(indexes);
        v0.scatter(data + 0, i);
        v1.scatter(data + 1, i);
        v2.scatter(data + 2, i);
        v3.scatter(data + 3, i);
        v4.scatter(data + 4, i);
        v5.scatter(data + 5, i);
        v6.scatter(data + 6, i);
        v7.scatter(data + 7, i);
    }
};
}  // anonymous namespace

// InterleavedMemoryAccessBase::interleave {{{1
// 2 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(const typename V::AsArg v0,
                                                              const typename V::AsArg v1)
{
    InterleaveImpl<V>::interleave(m_indexes, m_data, v0, v1);
}
// 3 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(const typename V::AsArg v0,
                                                              const typename V::AsArg v1,
                                                              const typename V::AsArg v2)
{
    InterleaveImpl<V>::interleave(m_indexes, m_data, v0, v1, v2);
}
// 4 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(const typename V::AsArg v0,
                                                              const typename V::AsArg v1,
                                                              const typename V::AsArg v2,
                                                              const typename V::AsArg v3)
{
    InterleaveImpl<V>::interleave(m_indexes, m_data, v0, v1, v2, v3);
}
// 5 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(const typename V::AsArg v0,
                                                              const typename V::AsArg v1,
                                                              const typename V::AsArg v2,
                                                              const typename V::AsArg v3,
                                                              const typename V::AsArg v4)
{
    InterleaveImpl<V>::interleave(m_indexes, m_data, v0, v1, v2, v3, v4);
}
// 6 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4, const typename V::AsArg v5)
{
    InterleaveImpl<V>::interleave(m_indexes, m_data, v0, v1, v2, v3, v4, v5);
}
// 7 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4, const typename V::AsArg v5,
    const typename V::AsArg v6)
{
    InterleaveImpl<V>::interleave(m_indexes, m_data, v0, v1, v2, v3, v4, v5, v6);
}
// 8 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4, const typename V::AsArg v5,
    const typename V::AsArg v6, const typename V::AsArg v7)
{
    InterleaveImpl<V>::interleave(m_indexes, m_data, v0, v1, v2, v3, v4, v5, v6, v7);
}

// InterleavedMemoryAccessBase::deinterleave {{{1
// 2 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1) const
{
    std::tie(v0, v1) = InterleaveImpl<V>::deinterleave(m_indexes, m_data, Size2());
}
// 3 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1, V &v2) const
{
    std::tie(v0, v1, v2) = InterleaveImpl<V>::deinterleave(m_indexes, m_data, Size3());
}
// 4 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1, V &v2,
                                                                V &v3) const
{
    std::tie(v0, v1, v2, v3) =
        InterleaveImpl<V>::deinterleave(m_indexes, m_data, Size4());
}
// 5 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1, V &v2,
                                                                V &v3, V &v4) const
{
    std::tie(v0, v1, v2, v3, v4) =
        InterleaveImpl<V>::deinterleave(m_indexes, m_data, Size5());
}
// 6 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1, V &v2,
                                                                V &v3, V &v4, V &v5) const
{
    std::tie(v0, v1, v2, v3, v4, v5) =
        InterleaveImpl<V>::deinterleave(m_indexes, m_data, Size6());
}
// 7 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1, V &v2,
                                                                V &v3, V &v4, V &v5,
                                                                V &v6) const
{
    std::tie(v0, v1, v2, v3, v4, v5, v6) =
        InterleaveImpl<V>::deinterleave(m_indexes, m_data, Size7());
}
// 8 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1, V &v2,
                                                                V &v3, V &v4, V &v5,
                                                                V &v6, V &v7) const
{
    std::tie(v0, v1, v2, v3, v4, v5, v6, v7) =
        InterleaveImpl<V>::deinterleave(m_indexes, m_data, Size8());
}
}
}

#endif // VC_MIC_INTERLEAVEDMEMORY_TCC_

// vim: foldmethod=marker
