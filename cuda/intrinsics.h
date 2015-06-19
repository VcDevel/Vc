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

// We don't actually use intrinsics but for the sake of standardisation we name this
// file the same as its counterparts for the other implementations

#ifndef VC_CUDA_INTRINSICS_H
#define VC_CUDA_INTRINSICS_H

#include <cstdlib>

#include "global.h"
#include "macros.h"

#include "../common/loadstoreflags.h"
#include "../traits/type_traits.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace CUDA
{
    template <typename T> struct VectorTypeHelper
    {
        typedef struct Type_t
        {
            T data_[CUDA_VECTOR_SIZE];

            Type_t()
            { 
                for(std::size_t i = 0; i < CUDA_VECTOR_SIZE; ++i)
                    data_[i] = 0;
            }
            
            Type_t(const T a)
            {
                for(std::size_t i = 0; i < CUDA_VECTOR_SIZE; ++i)
                    data_[i] = a;
            }

           Type_t(const T* data)
           {
               for(std::size_t i = 0; i < CUDA_VECTOR_SIZE; ++i)
                   data_[i] = data[i];
           }

            // enable array-like semantics
            T& operator[](std::size_t i) { return data_[i]; }
        } Type;
    };
} // namespace CUDA
} // namespace Vc

#include "undomacros.h"

#endif // VC_CUDA_INTRINSICS_H

