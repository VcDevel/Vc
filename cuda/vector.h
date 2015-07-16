/*  This file is part of the Vc library. {{{
Copyright © 2015 Jan Stephan <jan.stephan.dd@gmail.com>
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

#ifndef CUDA_VECTOR_H
#define CUDA_VECTOR_H

#include <initializer_list>
#include <type_traits>

//#include "../common/loadstoreflags.h"
//#include "../traits/type_traits.h"

#include "global.h"
#include "initflags.h"
#include "types.h"

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace CUDA
{
// TODO: make an incomplete type in case someone calls this in CPU code
#define VC_CURRENT_CLASS_NAME Vector
template <typename T>
class Vector
{
    static_assert(std::is_arithmetic<T>::value,
                  "Vector<T> only accepts arithmetic builtin types as template parameter T.");

    public:
        typedef T EntryType;
        typedef T VectorType[CUDA_VECTOR_SIZE];

        typedef EntryType VectorEntryType;
        typedef EntryType value_type;
        typedef VectorType vector_type;

    protected:
        VectorType m_data;

    public:
        __device__ Vc_ALWAYS_INLINE VectorType &data() { return m_data; }
        __device__ Vc_ALWAYS_INLINE const VectorType &data() const { return m_data; }

        static constexpr size_t Size = CUDA_VECTOR_SIZE;
        enum Constants {
            MemoryAlignment = alignof(VectorType)
        };

        typedef const Vector<T> AsArg;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // general interface - we have to redefine it here as the declarations in
        // common/generalinterface.h don't have the __device__ annotations
         
        ///////////////////////////////////////////////////////////////////////////////////////////
        // init to zero
        __device__ Vector() = default;
        
        ///////////////////////////////////////////////////////////////////////////////////////////
        // types

        ///////////////////////////////////////////////////////////////////////////////////////////
        // constants
        static constexpr std::size_t size() { return Size; }
        
        ///////////////////////////////////////////////////////////////////////////////////////////
        // constant Vectors
        // TODO: include these
        
        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        __device__ Vector(EntryType a)
        {
            // see example B.14.5.1 (CUDA C Programming Guide)
            int laneId = threadIdx.x & 0x1F;
            int value;
            if(laneId == 0)
                value = a;
            value = __shfl(value, 0);
            m_data[Internal::getThreadId()] = value;
        }
        
        template<typename U>
        __device__
        Vector(U a, typename std::enable_if<std::is_same<U, int>::value &&
                                                !std::is_same<U, EntryType>::value,
                                                void *>::type = nullptr)
            : Vector(static_cast<EntryType>(a))
        {
        }

        __device__ explicit Vector(std::initializer_list<EntryType>)
        {
            static_assert(std::is_same<EntryType, void>::value,
                            "A SIMD vector object cannot be initialized from an initializer list "
                            "because the number of entries in the vector is target-dependent.");
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // internal init
        __device__ Vc_INTRINSIC Vector(InternalInitTag, EntryType x)
        {
            m_data[Internal::getThreadId()] = x;
        }

        __device__ static Vc_INTRINSIC Vector internalInit(EntryType x)
        {
            __shared__ Vector<EntryType> r;
            r[Internal::getThreadId()] = x;
            __syncthreads();
            return r;
        }
        
        ///////////////////////////////////////////////////////////////////////////////////////////
        // load interface
        __device__ explicit Vc_INTRINSIC Vector(const EntryType *x)
        {
            load(x);
        }

        __device__ Vc_INTRINSIC void load(const EntryType *mem)
        {
            if(mem != nullptr)
                memcpy(m_data, mem, sizeof(float) * CUDA_VECTOR_SIZE);
        }
        
        //////////////////////////////////////////////////////////////////////////////////////////
        // store interface
        template <
            typename U>
        __device__ Vc_INTRINSIC void store(U *mem) const
        {
            if(m_data != nullptr)
                memcpy(mem, m_data, sizeof(float) * CUDA_VECTOR_SIZE);
        }

        __device__ Vc_ALWAYS_INLINE EntryType& operator[](std::size_t index)
        {
            VC_ASSERT(index < CUDA_VECTOR_SIZE);
            return m_data[index];
        }

        __device__ Vc_ALWAYS_INLINE EntryType operator[](std::size_t index) const
        {
            VC_ASSERT(index < CUDA_VECTOR_SIZE);
            return m_data[index];
        }
};

} // namespace CUDA
} // namespace Vc

#include "undomacros.h"

#endif // CUDA_VECTOR_H

