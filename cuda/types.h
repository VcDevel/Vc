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

#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

#include "global.h"
#include "../traits/type_traits.h"
#include "macros.h"

#define VC_DOUBLE_V_SIZE    CUDA_VECTOR_SIZE
#define VC_FLOAT_V_SIZE     CUDA_VECTOR_SIZE
#define VC_INT_V_SIZE       CUDA_VECTOR_SIZE
#define VC_UINT_V_SIZE      CUDA_VECTOR_SIZE
#define VC_SHORT_V_SIZE     CUDA_VECTOR_SIZE
#define VC_USHORT_V_SIZE    CUDA_VECTOR_SIZE

namespace Vc_VERSIONED_NAMESPACE
{
namespace CUDA
{
// shared memory banks are organised in consecutive 32bit words
constexpr std::size_t VectorAlignment = 4;

template <typename T> class Vector;
typedef Vector<double>          double_v;
typedef Vector<float>           float_v;
typedef Vector<int>             int_v;
typedef Vector<unsigned int>    uint_v;
typedef Vector<short>           short_v;
typedef Vector<unsigned short>  ushort_v;

template <typename T> class Mask;
typedef Mask<double>            double_m;
typedef Mask<float>             float_m;
typedef Mask<int>               int_m;
typedef Mask<unsigned int>      uint_m;
typedef Mask<short>             short_m;
typedef Mask<unsigned short>    ushort_m;

template <typename V = Vector<float>>
class alignas(alignof(V)) VectorAlignedBaseT;

} // namespace CUDA
} // namespace Vc

#include "undomacros.h"

#endif // CUDA_TYPES_H

