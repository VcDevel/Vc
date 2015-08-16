/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Jan Stephan <jan.stephan.dd@gmail.com>
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

#ifndef VC_CUDA_MASK_H
#define VC_CUDA_MASK_H

#include "types.h"
#include "macros.h"

namespace Vc_UNVERSIONED_NAMESPACE
{
template <typename T> class Mask<T, VectorAbi::Cuda>
{
    friend class Mask<       double, VectorAbi::Cuda>;
    friend class Mask<        float, VectorAbi::Cuda>;
    friend class Mask< std::int32_t, VectorAbi::Cuda>;
    friend class Mask<std::uint32_t, VectorAbi::Cuda>;
    friend class Mask< std::int16_t, VectorAbi::Cuda>;
    friend class Mask<std::uint16_t, VectorAbi::Cuda>;

public:
    using abi = VectorAbi::Cuda;

    static constexpr size_t Size = CUDA_VECTOR_SIZE;
    __device__ static constexpr std::size_t size() { return CUDA_VECTOR_SIZE; }

    /**
     * The \c EntryType of masks is always bool, independent of \c T.
     */
    typedef bool EntryType;

    using EntryReference = bool &;

    /**
     * The \c VectorEntryType, in contrast to \c EntryType, reveals information about the SIMD
     * implementation. This type is useful for the \c sizeof operator in generic functions.
     */
    typedef bool VectorEntryType;

    /**
     * The \c VectorType reveals the implementation-specific internal type used for the SIMD type.
     */
    using VectorType = bool;

    /**
     * The associated Vector<T> type.
     */
    using Vector = Vector<T, VectorAbi::Cuda>;

    __device__ Vc_INTRINSIC Mask() = default;
    
    __device__ Vc_INTRINSIC explicit Mask(bool b)
    {
        int laneId = threadIdx.x & 0x1F;
        bool value;
        if(laneId == 0)
            value = b;
        value = __shfl(value, 0);
        m[Detail::getThreadId()] = value;
    }

    __device__ Vc_INTRINSIC explicit Mask(VectorSpecialInitializerZero::ZEnum)
        : Mask(false)
    {
    }

    __device__ Vc_INTRINSIC explicit Mask(VectorSpecialInitializerOne::OEnum)
        : Mask(true)
    {
    }

    __device__ Vc_INTRINSIC static Mask Zero() { return Mask(false); }
    __device__ Vc_INTRINSIC static Mask One() { return Mask(true); }

    // implicit cast
    template <typename U>
    __device__ Vc_INTRINSIC Mask(U &&rhs,
                                   Common::enable_if_mask_converts_implicitly<T, U> = nullarg)
    {
        m[Detail::getThreadId()] = rhs.m[Detail::getThreadId()];
    }

    // explicit cast
    template <typename U>
    __device__ Vc_INTRINSIC_L explicit Mask(U &&rhs,
                                            Common::enable_if_mask_converts_explicitly<T, U> =
                                            nullarg) Vc_INTRINSIC_R;

    __device__ Vc_ALWAYS_INLINE explicit Mask(const bool *mem)
    {
        if(mem != nullptr)
            m[Detail::getThreadId()] = mem[Detail::getThreadId()];
    }
    template <typename Flags>
    __device__ Vc_ALWAYS_INLINE explicit Mask(const bool *mem, Flags)
    {
        if(mem != nullptr)
            m[Detail::getThreadId()] = mem[Detail::getThreadId()];
    }

    __device__ Vc_ALWAYS_INLINE void load(const bool *mem)
    {
        if(mem != nullptr)
            m[Detail::getThreadId()] = mem[Detail::getThreadId()];
    }
    template <typename Flags>
    __device__ Vc_ALWAYS_INLINE void load(const bool *mem, Flags)
    {
        if(mem != nullptr)
            m[Detail::getThreadId()] = mem[Detail::getThreadId()];
    }

    __device__ Vc_ALWAYS_INLINE void store(bool *mem) const
    {
        if(mem != nullptr)
            mem[Detail::getThreadId()] = m[Detail::getThreadId()];
    }
    template <typename Flags>
    __device__ Vc_ALWAYS_INLINE void store(bool *mem, Flags) const
    {
        if(mem != nullptr)
            mem[Detail::getThreadId()] = m[Detail::getThreadId()];
    }
   
    // assignment operators {{{1
    __device__ Vc_ALWAYS_INLINE Mask &operator=(const Mask &rhs)
    {
        m[Detail::getThreadId()] = rhs.m[Detail::getThreadId()];
        return *this;
    }
    __device__ Vc_ALWAYS_INLINE Mask &operator=(bool rhs)
    {
        m[Detail::getThreadId()] = rhs;
        return *this;
    }

    // compare operators {{{1
    __device__ Vc_ALWAYS_INLINE bool operator==(const Mask &rhs) const
    {
        __shared__ bool isSame;
        if(Detail::getThreadId() == 0)
            isSame = (m[Detail::getThreadId()] == rhs.m[Detail::getThreadId()]);
        isSame = isSame && (m[Detail::getThreadId()] == rhs.m[Detail::getThreadId()]);
        return isSame;
    }
    __device__ Vc_ALWAYS_INLINE bool operator!=(const Mask &rhs) const
    {
        __shared__ bool isSame;
        if(Detail::getThreadId() == 0)
            isSame = (m[Detail::getThreadId()] == rhs.m[Detail::getThreadId()]);
        isSame = isSame && (m[Detail::getThreadId()] == rhs.m[Detail::getThreadId()]);
        return !isSame;
    }

    // logical operators {{{1
    __device__ Vc_ALWAYS_INLINE Mask operator&&(const Mask &rhs) const
    {
    }
    __device__ Vc_ALWAYS_INLINE Mask operator& (const Mask &rhs) const
    {
    }
    __device__ Vc_ALWAYS_INLINE Mask operator||(const Mask &rhs) const
    {
    }
    __device__ Vc_ALWAYS_INLINE Mask operator| (const Mask &rhs) const
    {
    }
    __device__ Vc_ALWAYS_INLINE Mask operator^ (const Mask &rhs) const
    {
    }
    __device__ Vc_ALWAYS_INLINE Mask operator!() const
    {
    }

    // logical assignment operators {{{1
    __device__ Vc_ALWAYS_INLINE Mask &operator&=(const Mask &rhs) const
    {
        return *this;
    }
    __device__ Vc_ALWAYS_INLINE Mask &operator|=(const Mask &rhs) const
    {
        return *this;
    }
    __device__ Vc_ALWAYS_INLINE Mask &operator^=(const Mask &rhs) const
    {
        return *this;
    }

    // query status {{{1
    __device__ Vc_ALWAYS_INLINE bool isFull() const
    {
        __shared__ bool full;
        if(Detail::getThreadId() == 0)
            full = m[Detail::getThreadId()];
        full = (isFull && m[Detail::getThreadId()]);
        return full;
    }

    __device__ Vc_ALWAYS_INLINE bool isNotEmpty() const
    {
        __shared__ bool notEmpty;
        if(Detail::getThreadId() == 0)
            notEmpty = m[Detail::getThreadId()];
        notEmpty = (notEmpty || m[Detail::getThreadId()]);;
        return notEmpty;
    }

    __device__ Vc_ALWAYS_INLINE bool isEmpty() const
    {
        __shared__ bool empty;
        if(Detail::getThreadId() == 0)
            empty = !(m[Detail::getThreadId()]);
        empty = (empty && !(m[Detail::getThreadId()]));
        return empty;
    }

    __device__ Vc_ALWAYS_INLINE bool isMix() const
    {
    }

    ///\internal Called indirectly from operator[]
    __device__ void setEntry(std::size_t i, bool x)
    {
        if(Vc::Detail::getThreadId() == i)
            m[i] = x;
    }
private:
    bool m[CUDA_VECTOR_SIZE];
};
template <typename T> constexpr std::size_t Mask<T, VectorAbi::Cuda>::Size;

} // namespace Vc

#include "undomacros.h"

#endif // VC_CUDA_MASK_H

