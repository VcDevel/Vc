/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>
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
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace CUDA
{
#define VC_CURRENT_CLASS_NAME Vector
template <typename T>
class Vector
{
    static_assert(std::is_arithmetic<T>::value,
                  "Vector<T> only accepts arithmetic builtin types as template parameter T.");

    public:
        typedef T EntryType;
        typedef EntryType* VectorType;

        typedef EntryType VectorEntryType;
        typedef EntryType value_type;
        typedef VectorType vector_type;

    protected:
        VectorType m_data;

    public:
        CUDA_CALLABLE Vc_ALWAYS_INLINE VectorType &data() { return m_data; }
        CUDA_CALLABLE Vc_ALWAYS_INLINE const VectorType &data() const { return m_data; }

        static constexpr size_t Size = CUDA_VECTOR_SIZE;
        enum Constants {
            MemoryAlignment = alignof(VectorType)
        };

        ///////////////////////////////////////////////////////////////////////////////////////////
        // Destructor
        CUDA_CALLABLE ~Vector()
        {
            if(m_data != nullptr)
            {
#               ifdef __CUDA_ARCH__
                block_free(m_data);
#               else
                cudaFree(m_data);
#               endif
            }
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        CUDA_CALLABLE Vector(EntryType a) : m_data{nullptr}
        {
#           ifdef __CUDA_ARCH__
            m_data = static_cast<float*>(block_malloc(sizeof(float) * CUDA_VECTOR_SIZE));
            m_data[getThreadId()] = a;
#           else
            cudaMalloc(&m_data, sizeof(float) * CUDA_VECTOR_SIZE);
            for(std::size_t i = 0; i < CUDA_VECTOR_SIZE; ++i)
                m_data[i] = a;
#           endif
        }
        
        template<typename U>
        CUDA_CALLABLE
        Vector(U a, typename std::enable_if<std::is_same<U, int>::value &&
                                                !std::is_same<U, EntryType>::value,
                                                void *>::type = nullptr)
            : Vector(static_cast<EntryType>(a))
        {
        }

        CUDA_CALLABLE explicit Vector(std::initializer_list<EntryType>)
        {
            static_assert(std::is_same<EntryType, void>::value,
                            "A SIMD vector object cannot be initialized from an initializer list "
                            "because the number of entries in the vector is target-dependent.");
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load interface
        CUDA_CALLABLE explicit Vc_INTRINSIC Vector(const EntryType *x)
            : m_data{nullptr}
        {
            load(x);
        }

        CUDA_CALLABLE Vc_INTRINSIC void load(const EntryType *mem)
        {
            if(m_data == nullptr)
            {
#               ifdef __CUDA_ARCH__
                m_data = static_cast<float*>(block_malloc(sizeof(float) * CUDA_VECTOR_SIZE));;
#               else
                cudaMalloc(&m_data, sizeof(float) * CUDA_VECTOR_SIZE);
#               endif

            }

#           ifdef __CUDA_ARCH__
            memcpy(m_data, mem, sizeof(float) * CUDA_VECTOR_SIZE);
#           else
            cudaMemcpy(m_data, mem, sizeof(float) * CUDA_VECTOR_SIZE, cudaMemcpyDeviceToDevice);
#           endif
        }
        
        //////////////////////////////////////////////////////////////////////////////////////////
        // store interface
        template <
            typename U>
        CUDA_CALLABLE Vc_INTRINSIC void store(U *mem) const
        {
            if(m_data != nullptr && mem != nullptr)
#               ifdef __CUDA_ARCH__
                memcpy(mem, m_data, sizeof(float) * CUDA_VECTOR_SIZE);
#               else
                cudaMemcpy(mem, m_data, sizeof(float) * CUDA_VECTOR_SIZE, cudaMemcpyDeviceToDevice);
#               endif
        }
};
} // namespace CUDA
} // namespace Vc

#include "undomacros.h"

#endif // CUDA_VECTOR_H

