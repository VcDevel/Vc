/*  This file is part of the Vc library. {{{
Copyright © 2010-2014 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_COMMON_UNDOMACROS_H_
#define VC_COMMON_UNDOMACROS_H_
#undef VC_COMMON_MACROS_H_

#undef Vc_INTRINSIC
#undef Vc_INTRINSIC_L
#undef Vc_INTRINSIC_R
#undef Vc_CONST
#undef Vc_CONST_L
#undef Vc_CONST_R
#undef Vc_PURE
#undef Vc_PURE_L
#undef Vc_PURE_R
#undef Vc_MAY_ALIAS
#undef Vc_ALWAYS_INLINE
#undef Vc_ALWAYS_INLINE_L
#undef Vc_ALWAYS_INLINE_R
#undef Vc_IS_UNLIKELY
#undef Vc_IS_LIKELY
#undef Vc_RESTRICT
#undef Vc_DEPRECATED
#undef Vc_WARN_UNUSED_RESULT

#undef ALIGN
#undef STRUCT_ALIGN1
#undef STRUCT_ALIGN2
#undef ALIGNED_TYPEDEF
#undef Vc_CAT_IMPL
#undef Vc_CAT2
#undef unrolled_loop16
#undef for_all_vector_entries
#undef FREE_STORE_OPERATORS_ALIGNED

#ifdef Vc_EXTERNAL_ASSERT
#undef Vc_EXTERNAL_ASSERT
#else
#undef Vc_ASSERT
#endif

#undef Vc_HAS_BUILTIN

#undef Vc_APPLY_IMPL_1_
#undef Vc_APPLY_IMPL_2_
#undef Vc_APPLY_IMPL_3_
#undef Vc_APPLY_IMPL_4_
#undef Vc_APPLY_IMPL_5_

#undef Vc_LIST_FLOAT_VECTOR_TYPES
#undef Vc_LIST_INT_VECTOR_TYPES
#undef Vc_LIST_VECTOR_TYPES
#undef Vc_LIST_COMPARES
#undef Vc_LIST_LOGICAL
#undef Vc_LIST_BINARY
#undef Vc_LIST_SHIFTS
#undef Vc_LIST_ARITHMETICS

#undef Vc_APPLY_0
#undef Vc_APPLY_1
#undef Vc_APPLY_2
#undef Vc_APPLY_3
#undef Vc_APPLY_4

#undef Vc_ALL_COMPARES
#undef Vc_ALL_LOGICAL
#undef Vc_ALL_BINARY
#undef Vc_ALL_SHIFTS
#undef Vc_ALL_ARITHMETICS
#undef Vc_ALL_FLOAT_VECTOR_TYPES
#undef Vc_ALL_VECTOR_TYPES

#undef Vc_EXACT_TYPE
#undef Vc_ALIGNED_PARAMETER
#undef Vc_OFFSETOF
#undef Vc_NOEXCEPT

#if defined(Vc_GCC) && !defined(__OPTIMIZE__)
#pragma GCC diagnostic pop
#endif

#endif // VC_COMMON_UNDOMACROS_H_
