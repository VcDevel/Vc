/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DATAPAR_AARCH_ARITHMETICS_H_
#define VC_DATAPAR_AARCH_ARITHMETICS_H_

#include "storage.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace aarch
{
// plus{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto plus(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() + b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
//Vc_INTRINSIC float32x4_t Vc_VDECL plus(x_f32 a, x_f32 b) { return vadd_f32(a, b); }
//...
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// minus{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto minus(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() - b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
//Vc_INTRINSIC float32x4_t Vc_VDECL minus(x_f32 a, x_f32 b) { return vsub_f32(a, b); }
//...
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// multiplies{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto multiplies(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() * b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
//Vc_INTRINSIC float32x4_t Vc_VDECL multiplies(x_f32 a, x_f32 b) { return vmul_n_f32(a, b); }
//...
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// divides{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto divides(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() / b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
//Vc_INTRINSIC x_f32 Vc_VDECL divides(x_f32 a, x_f32 b) {}
//...
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

//}}}1
}}  // namespaces 
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_AARCH_ARITHMETICS_H_
