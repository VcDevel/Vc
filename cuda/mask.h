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

namespace Vc_VERSIONED_NAMESPACE
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

    __device__ Vc_INTRINSIC static Mask Zero() { return internalInit(false); }
    __device__ Vc_INTRINSIC static Mask One() { return internalInit(true); }

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

        struct Compare2
        {
            Compare2(bool *ptr, bool var1, bool var2) { *ptr &= (var1 && var2); }
        };

        Detail::reduce2<Compare2>(&isSame, m, rhs.m, Detail::getThreadId());
        return isSame;
    }
    
    __device__ Vc_ALWAYS_INLINE bool operator!=(const Mask &rhs) const
    {
        return !operator==(rhs);
    }

    // logical operators {{{1
    __device__ Vc_ALWAYS_INLINE Mask operator&&(const Mask &rhs) const
    {
        return internalInit(m[Detail::getThreadId()] && rhs.m[Detail::getThreadId()]);
    }
    __device__ Vc_ALWAYS_INLINE Mask operator& (const Mask &rhs) const
    {
        return internalInit(m[Detail::getThreadId()] && rhs.m[Detail::getThreadId()]);
    }
    __device__ Vc_ALWAYS_INLINE Mask operator||(const Mask &rhs) const
    {
        return internalInit(m[Detail::getThreadId()] || rhs.m[Detail::getThreadId()]);
    }
    __device__ Vc_ALWAYS_INLINE Mask operator| (const Mask &rhs) const
    {
        return internalInit(m[Detail::getThreadId()] || rhs.m[Detail::getThreadId()]);
    }
    __device__ Vc_ALWAYS_INLINE Mask operator^ (const Mask &rhs) const
    {
        return internalInit(m[Detail::getThreadId()] ^ rhs.m[Detail::getThreadId()]);
    }
    __device__ Vc_ALWAYS_INLINE Mask operator!() const
    {
        return internalInit(!(m[Detail::getThreadId()]));
    }

    // logical assignment operators {{{1
    __device__ Vc_ALWAYS_INLINE Mask &operator&=(const Mask &rhs) const
    {
        m[Detail::getThreadId()] &= rhs.m[Detail::getThreadId()];
        return *this;
    }
    __device__ Vc_ALWAYS_INLINE Mask &operator|=(const Mask &rhs) const
    {
        m[Detail::getThreadId()] |= rhs.m[Detail::getThreadId()];
        return *this;
    }
    __device__ Vc_ALWAYS_INLINE Mask &operator^=(const Mask &rhs) const
    {
        m[Detail::getThreadId()] ^= rhs.m[Detail::getThreadId()];
        return *this;
    }

    // query status {{{1
    __device__ Vc_ALWAYS_INLINE bool isFull() const
    {
        __shared__ bool full;
        if(Detail::getThreadId() == 0)
            full = m[Detail::getThreadId()];

        struct And
        {
            And(bool *ptr, bool var) { *ptr &= var; }
        };

        Detail::reduce<And>(&full, m, Detail::getThreadId());
        return full;
    }

    __device__ Vc_ALWAYS_INLINE bool isNotEmpty() const
    {
        return !isEmpty();
    }

    __device__ Vc_ALWAYS_INLINE bool isEmpty() const
    {
        __shared__ bool empty;
        if(Detail::getThreadId() == 0)
            empty = !(m[Detail::getThreadId()]);

        struct NotAnd
        {
            NotAnd(bool *ptr, bool var) { *ptr &= !var; }
        };
        
        Detail::reduce<NotAnd>(&empty, m, Detail::getThreadId());
        return empty;
    }

    __device__ Vc_ALWAYS_INLINE bool isMix() const
    {
        return !isEmpty() && !isFull();
    }

    // utility {{{1
    __device__ Vc_ALWAYS_INLINE Vc_PURE int shiftMask() const
    {
    }

    __device__ Vc_ALWAYS_INLINE Vc_PURE int toInt() const
    {
    }

    __device__ Vc_ALWAYS_INLINE Vc_PURE float  data () const { }
    __device__ Vc_ALWAYS_INLINE Vc_PURE int    dataI() const { }
    __device__ Vc_ALWAYS_INLINE Vc_PURE double dataD() const { }

    __device__ Vc_ALWAYS_INLINE EntryReference operator[](std::size_t index)
    {
        return m[index];
    }

    __device__ Vc_ALWAYS_INLINE Vc_PURE bool operator[](std::size_t index) const
    {
        return m[index];
    }

    __device__ Vc_ALWAYS_INLINE Vc_PURE int count() const
    {
    }

    /**
     * Returns the value of the first one in the mask.
     *
     * The return value is undefined if the mask is empty.
     */
    __device__ Vc_ALWAYS_INLINE Vc_PURE int firstOne() const
    {
        bool first = false;
        std::size_t i;
        for(i = 0; i < CUDA_VECTOR_SIZE; ++i)
        {
            first = m[i];
            if(first)
                break;
        }
        return i;
    }

    ///\internal Called indirectly from operator[]
    __device__ void setEntry(std::size_t i, bool x)
    {
        if(Vc::Detail::getThreadId() == i)
            m[i] = x;
    }
    
    __device__ static Vc_INTRINSIC Mask internalInit(EntryType x)
    {
        __shared__ Mask<EntryType> r;
        r[Detail::getThreadId()] = x;
        __syncthreads();
        return r;
    }

private:
    bool m[CUDA_VECTOR_SIZE];
};
template <typename T> constexpr std::size_t Mask<T, VectorAbi::Cuda>::Size;

} // namespace Vc

#include "undomacros.h"

#endif // VC_CUDA_MASK_H

