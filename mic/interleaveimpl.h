/*  This file is part of the Vc library. {{{
Copyright © 2016 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_INTERLEAVEIMPL_H_
#define VC_MIC_INTERLEAVEIMPL_H_

#include "macros.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace Detail
{
template <typename T, int Wt, size_t Sizeof>
struct InterleaveImpl<Vector<T, VectorAbi::Mic>, Wt, Sizeof> {
    using V = Vector<T, VectorAbi::Mic>;
    using IT = typename V::IndexType;
    using VT = typename V::VectorEntryType;
    using UpDownC = MIC::UpDownConversion<VT, T>;
    // fixup functions{{{1
    static Vc_INTRINSIC __m512i fixup(__m512i i)
    {
        return i;
    }
    static Vc_INTRINSIC __m512i fixup(const SimdArray<int, 16> &i)
    {
        return internal_data(i).data();
    }
    static Vc_INTRINSIC __m512i fixup(const SimdArray<int, 8> &i)
    {
        return _mm512_mask_loadunpacklo_epi32(_mm512_setzero_epi32(), 0x00ff, &i);
    }
    template <size_t StructSize>
    static Vc_INTRINSIC __m512i fixup(const Common::SuccessiveEntries<StructSize> &i)
    {
        using namespace Detail;
        return (MIC::int_v::IndexesFromZero() * MIC::int_v(StructSize) +
                MIC::int_v(i.data()))
            .data();
    }
    template <size_t... IndexSeq, typename... Vs>  // interleave pack expansion{{{1
    static Vc_INTRINSIC void interleave(index_sequence<IndexSeq...>, T *const data,
                                        const MIC::int_v indexes, Vs &&... vs)
    {
        const auto &&tmp = {(vs.scatter(data + IndexSeq, indexes), 0)...};
    }
    template <typename I, typename... Vs>  // interleave interface{{{1
    static inline void interleave(T *const data, I &&indexes, Vs &&... vs)
    {
        interleave(make_index_sequence<sizeof...(Vs)>(), data,
                   fixup(std::forward<I>(indexes)), std::forward<Vs>(vs)...);
    }
    template <size_t... IndexSeq, typename... Vs>  // deinterleave pack expansion{{{1
    static Vc_INTRINSIC void deinterleave(index_sequence<IndexSeq...>,
                                          const T *const data, const MIC::int_v indexes,
                                          Vs &&... vs)
    {
        using Vc::MicIntrinsics::gather;
        const auto &&tmp = {
            (vs = gather(indexes.data(), data + IndexSeq, UpDownC()), 0)...};
    }
    template <typename I, typename... Vs>  // deinterleave interface{{{1
    static inline void deinterleave(const T *const data, I &&indexes, Vs &&... vs)
    {
        deinterleave(make_index_sequence<sizeof...(Vs)>(), data,
                     fixup(std::forward<I>(indexes)), std::forward<Vs>(vs)...);
    }
    //}}}2
};
}  // namespace Detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_MIC_INTERLEAVEIMPL_H_

// vim: foldmethod=marker
