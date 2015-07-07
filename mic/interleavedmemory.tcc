/*  This file is part of the Vc library. {{{
Copyright © 2012-2015 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_MIC_INTERLEAVEDMEMORY_TCC
#define VC_MIC_INTERLEAVEDMEMORY_TCC

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

    template<typename I> static inline std::tuple<V, V> deinterleave(I indexes, T *const data, Size2)/*{{{*/
    {
        std::tuple<V, V> r;
        indexes *= 2;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<size_t StructSize> static inline std::tuple<V, V> deinterleave(SuccessiveEntries<StructSize> firstIndex, T *const data, Size2)/*{{{*/
    {
        std::tuple<V, V> r;
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 2;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<typename I> static inline std::tuple<V, V, V> deinterleave(I indexes, T *const data, Size3)/*{{{*/
    {
        std::tuple<V, V, V> r;
        indexes *= 3;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        std::get<2>(r).data() = gather(indexes.data(), data + 2, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<size_t StructSize> static inline std::tuple<V, V, V> deinterleave(SuccessiveEntries<StructSize> firstIndex, T *const data, Size3)/*{{{*/
    {
        std::tuple<V, V, V> r;
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 3;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        std::get<2>(r).data() = gather(indexes.data(), data + 2, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<typename I> static inline std::tuple<V, V, V, V> deinterleave(I indexes, T *const data, Size4)/*{{{*/
    {
        std::tuple<V, V, V, V> r;
        indexes *= 4;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        std::get<2>(r).data() = gather(indexes.data(), data + 2, UpDownC());
        std::get<3>(r).data() = gather(indexes.data(), data + 3, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<size_t StructSize> static inline std::tuple<V, V, V, V> deinterleave(SuccessiveEntries<StructSize> firstIndex, T *const data, Size4)/*{{{*/
    {
        std::tuple<V, V, V, V> r;
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 4;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        std::get<2>(r).data() = gather(indexes.data(), data + 2, UpDownC());
        std::get<3>(r).data() = gather(indexes.data(), data + 3, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<typename I> static inline std::tuple<V, V, V, V, V> deinterleave(I indexes, T *const data, Size5)/*{{{*/
    {
        std::tuple<V, V, V, V, V> r;
        indexes *= 5;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        std::get<2>(r).data() = gather(indexes.data(), data + 2, UpDownC());
        std::get<3>(r).data() = gather(indexes.data(), data + 3, UpDownC());
        std::get<4>(r).data() = gather(indexes.data(), data + 4, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<size_t StructSize> static inline std::tuple<V, V, V, V, V> deinterleave(SuccessiveEntries<StructSize> firstIndex, T *const data, Size5)/*{{{*/
    {
        std::tuple<V, V, V, V, V> r;
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 5;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        std::get<2>(r).data() = gather(indexes.data(), data + 2, UpDownC());
        std::get<3>(r).data() = gather(indexes.data(), data + 3, UpDownC());
        std::get<4>(r).data() = gather(indexes.data(), data + 4, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<typename I> static inline std::tuple<V, V, V, V, V, V> deinterleave(I indexes, T *const data, Size6)/*{{{*/
    {
        std::tuple<V, V, V, V, V, V> r;
        indexes *= 6;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        std::get<2>(r).data() = gather(indexes.data(), data + 2, UpDownC());
        std::get<3>(r).data() = gather(indexes.data(), data + 3, UpDownC());
        std::get<4>(r).data() = gather(indexes.data(), data + 4, UpDownC());
        std::get<5>(r).data() = gather(indexes.data(), data + 5, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<size_t StructSize> static inline std::tuple<V, V, V, V, V, V> deinterleave(SuccessiveEntries<StructSize> firstIndex, T *const data, Size6)/*{{{*/
    {
        std::tuple<V, V, V, V, V, V> r;
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 6;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        std::get<2>(r).data() = gather(indexes.data(), data + 2, UpDownC());
        std::get<3>(r).data() = gather(indexes.data(), data + 3, UpDownC());
        std::get<4>(r).data() = gather(indexes.data(), data + 4, UpDownC());
        std::get<5>(r).data() = gather(indexes.data(), data + 5, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<typename I> static inline std::tuple<V, V, V, V, V, V, V> deinterleave(I indexes, T *const data, Size7)/*{{{*/
    {
        std::tuple<V, V, V, V, V, V, V> r;
        indexes *= 7;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        std::get<2>(r).data() = gather(indexes.data(), data + 2, UpDownC());
        std::get<3>(r).data() = gather(indexes.data(), data + 3, UpDownC());
        std::get<4>(r).data() = gather(indexes.data(), data + 4, UpDownC());
        std::get<5>(r).data() = gather(indexes.data(), data + 5, UpDownC());
        std::get<6>(r).data() = gather(indexes.data(), data + 6, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<size_t StructSize> static inline std::tuple<V, V, V, V, V, V, V> deinterleave(SuccessiveEntries<StructSize> firstIndex, T *const data, Size7)/*{{{*/
    {
        std::tuple<V, V, V, V, V, V, V> r;
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 7;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        std::get<2>(r).data() = gather(indexes.data(), data + 2, UpDownC());
        std::get<3>(r).data() = gather(indexes.data(), data + 3, UpDownC());
        std::get<4>(r).data() = gather(indexes.data(), data + 4, UpDownC());
        std::get<5>(r).data() = gather(indexes.data(), data + 5, UpDownC());
        std::get<6>(r).data() = gather(indexes.data(), data + 6, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<typename I> static inline std::tuple<V, V, V, V, V, V, V, V> deinterleave(I indexes, T *const data, Size8)/*{{{*/
    {
        std::tuple<V, V, V, V, V, V, V, V> r;
        indexes *= 8;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        std::get<2>(r).data() = gather(indexes.data(), data + 2, UpDownC());
        std::get<3>(r).data() = gather(indexes.data(), data + 3, UpDownC());
        std::get<4>(r).data() = gather(indexes.data(), data + 4, UpDownC());
        std::get<5>(r).data() = gather(indexes.data(), data + 5, UpDownC());
        std::get<6>(r).data() = gather(indexes.data(), data + 6, UpDownC());
        std::get<7>(r).data() = gather(indexes.data(), data + 7, UpDownC());
        return std::move(r);
    }/*}}}*/
    template<size_t StructSize> static inline std::tuple<V, V, V, V, V, V, V, V> deinterleave(SuccessiveEntries<StructSize> firstIndex, T *const data, Size8)/*{{{*/
    {
        std::tuple<V, V, V, V, V, V, V, V> r;
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 8;
        std::get<0>(r).data() = gather(indexes.data(), data + 0, UpDownC());
        std::get<1>(r).data() = gather(indexes.data(), data + 1, UpDownC());
        std::get<2>(r).data() = gather(indexes.data(), data + 2, UpDownC());
        std::get<3>(r).data() = gather(indexes.data(), data + 3, UpDownC());
        std::get<4>(r).data() = gather(indexes.data(), data + 4, UpDownC());
        std::get<5>(r).data() = gather(indexes.data(), data + 5, UpDownC());
        std::get<6>(r).data() = gather(indexes.data(), data + 6, UpDownC());
        std::get<7>(r).data() = gather(indexes.data(), data + 7, UpDownC());
        return std::move(r);
    }/*}}}*/

    template<typename I> static inline void interleave(I indexes, T *const data, V v0, V v1)/*{{{*/
    {
        indexes *= 2;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<size_t StructSize> static inline void interleave(SuccessiveEntries<StructSize> firstIndex, T *const data, V v0, V v1)/*{{{*/
    {
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 2;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<typename I> static inline void interleave(I indexes, T *const data, V v0, V v1, V v2)/*{{{*/
    {
        indexes *= 3;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
        scatter(data + 2, indexes.data(), v2.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<size_t StructSize> static inline void interleave(SuccessiveEntries<StructSize> firstIndex, T *const data, V v0, V v1, V v2)/*{{{*/
    {
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 3;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
        scatter(data + 2, indexes.data(), v2.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<typename I> static inline void interleave(I indexes, T *const data, V v0, V v1, V v2, V v3)/*{{{*/
    {
        indexes *= 4;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
        scatter(data + 2, indexes.data(), v2.data(), UpDownC(), sizeof(T));
        scatter(data + 3, indexes.data(), v3.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<size_t StructSize> static inline void interleave(SuccessiveEntries<StructSize> firstIndex, T *const data, V v0, V v1, V v2, V v3)/*{{{*/
    {
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 4;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
        scatter(data + 2, indexes.data(), v2.data(), UpDownC(), sizeof(T));
        scatter(data + 3, indexes.data(), v3.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<typename I> static inline void interleave(I indexes, T *const data, V v0, V v1, V v2, V v3, V v4)/*{{{*/
    {
        indexes *= 5;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
        scatter(data + 2, indexes.data(), v2.data(), UpDownC(), sizeof(T));
        scatter(data + 3, indexes.data(), v3.data(), UpDownC(), sizeof(T));
        scatter(data + 4, indexes.data(), v4.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<size_t StructSize> static inline void interleave(SuccessiveEntries<StructSize> firstIndex, T *const data, V v0, V v1, V v2, V v3, V v4)/*{{{*/
    {
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 5;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
        scatter(data + 2, indexes.data(), v2.data(), UpDownC(), sizeof(T));
        scatter(data + 3, indexes.data(), v3.data(), UpDownC(), sizeof(T));
        scatter(data + 4, indexes.data(), v4.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<typename I> static inline void interleave(I indexes, T *const data, V v0, V v1, V v2, V v3, V v4, V v5)/*{{{*/
    {
        indexes *= 6;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
        scatter(data + 2, indexes.data(), v2.data(), UpDownC(), sizeof(T));
        scatter(data + 3, indexes.data(), v3.data(), UpDownC(), sizeof(T));
        scatter(data + 4, indexes.data(), v4.data(), UpDownC(), sizeof(T));
        scatter(data + 5, indexes.data(), v5.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<size_t StructSize> static inline void interleave(SuccessiveEntries<StructSize> firstIndex, T *const data, V v0, V v1, V v2, V v3, V v4, V v5)/*{{{*/
    {
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 6;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
        scatter(data + 2, indexes.data(), v2.data(), UpDownC(), sizeof(T));
        scatter(data + 3, indexes.data(), v3.data(), UpDownC(), sizeof(T));
        scatter(data + 4, indexes.data(), v4.data(), UpDownC(), sizeof(T));
        scatter(data + 5, indexes.data(), v5.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<typename I> static inline void interleave(I indexes, T *const data, V v0, V v1, V v2, V v3, V v4, V v5, V v6)/*{{{*/
    {
        indexes *= 7;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
        scatter(data + 2, indexes.data(), v2.data(), UpDownC(), sizeof(T));
        scatter(data + 3, indexes.data(), v3.data(), UpDownC(), sizeof(T));
        scatter(data + 4, indexes.data(), v4.data(), UpDownC(), sizeof(T));
        scatter(data + 5, indexes.data(), v5.data(), UpDownC(), sizeof(T));
        scatter(data + 6, indexes.data(), v6.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<size_t StructSize> static inline void interleave(SuccessiveEntries<StructSize> firstIndex, T *const data, V v0, V v1, V v2, V v3, V v4, V v5, V v6)/*{{{*/
    {
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 7;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
        scatter(data + 2, indexes.data(), v2.data(), UpDownC(), sizeof(T));
        scatter(data + 3, indexes.data(), v3.data(), UpDownC(), sizeof(T));
        scatter(data + 4, indexes.data(), v4.data(), UpDownC(), sizeof(T));
        scatter(data + 5, indexes.data(), v5.data(), UpDownC(), sizeof(T));
        scatter(data + 6, indexes.data(), v6.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<typename I> static inline void interleave(I indexes, T *const data, V v0, V v1, V v2, V v3, V v4, V v5, V v6, V v7)/*{{{*/
    {
        indexes *= 8;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
        scatter(data + 2, indexes.data(), v2.data(), UpDownC(), sizeof(T));
        scatter(data + 3, indexes.data(), v3.data(), UpDownC(), sizeof(T));
        scatter(data + 4, indexes.data(), v4.data(), UpDownC(), sizeof(T));
        scatter(data + 5, indexes.data(), v5.data(), UpDownC(), sizeof(T));
        scatter(data + 6, indexes.data(), v6.data(), UpDownC(), sizeof(T));
        scatter(data + 7, indexes.data(), v7.data(), UpDownC(), sizeof(T));
    }/*}}}*/
    template<size_t StructSize> static inline void interleave(SuccessiveEntries<StructSize> firstIndex, T *const data, V v0, V v1, V v2, V v3, V v4, V v5, V v6, V v7)/*{{{*/
    {
        const IT indexes = (IT::IndexesFromZero() + firstIndex.data()) * 8;
        scatter(data + 0, indexes.data(), v0.data(), UpDownC(), sizeof(T));
        scatter(data + 1, indexes.data(), v1.data(), UpDownC(), sizeof(T));
        scatter(data + 2, indexes.data(), v2.data(), UpDownC(), sizeof(T));
        scatter(data + 3, indexes.data(), v3.data(), UpDownC(), sizeof(T));
        scatter(data + 4, indexes.data(), v4.data(), UpDownC(), sizeof(T));
        scatter(data + 5, indexes.data(), v5.data(), UpDownC(), sizeof(T));
        scatter(data + 6, indexes.data(), v6.data(), UpDownC(), sizeof(T));
        scatter(data + 7, indexes.data(), v7.data(), UpDownC(), sizeof(T));
    }/*}}}*/
};
} // anonymous namespace

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

#include "undomacros.h"

#endif // VC_MIC_INTERLEAVEDMEMORY_TCC

// vim: foldmethod=marker
