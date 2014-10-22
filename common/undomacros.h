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

#ifndef VC_COMMON_UNDOMACROS_H
#define VC_COMMON_UNDOMACROS_H
#undef VC_COMMON_MACROS_H

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
#undef VC_IS_UNLIKELY
#undef VC_IS_LIKELY
#undef VC_RESTRICT
#undef VC_DEPRECATED
#undef Vc_WARN_UNUSED_RESULT

#undef ALIGN
#undef STRUCT_ALIGN1
#undef STRUCT_ALIGN2
#undef ALIGNED_TYPEDEF
#undef _CAT_IMPL
#undef CAT
#undef unrolled_loop16
#undef for_all_vector_entries
#undef FREE_STORE_OPERATORS_ALIGNED

#undef VC_WARN_INLINE
#undef VC_WARN

#ifdef VC_EXTERNAL_ASSERT
#undef VC_EXTERNAL_ASSERT
#else
#undef VC_ASSERT
#endif

#undef VC_HAS_BUILTIN

#undef _VC_APPLY_IMPL_1
#undef _VC_APPLY_IMPL_2
#undef _VC_APPLY_IMPL_3
#undef _VC_APPLY_IMPL_4
#undef _VC_APPLY_IMPL_5

#undef VC_LIST_FLOAT_VECTOR_TYPES
#undef VC_LIST_INT_VECTOR_TYPES
#undef VC_LIST_VECTOR_TYPES
#undef VC_LIST_COMPARES
#undef VC_LIST_LOGICAL
#undef VC_LIST_BINARY
#undef VC_LIST_SHIFTS
#undef VC_LIST_ARITHMETICS

#undef VC_APPLY_0
#undef VC_APPLY_1
#undef VC_APPLY_2
#undef VC_APPLY_3
#undef VC_APPLY_4

#undef VC_ALL_COMPARES
#undef VC_ALL_LOGICAL
#undef VC_ALL_BINARY
#undef VC_ALL_SHIFTS
#undef VC_ALL_ARITHMETICS
#undef VC_ALL_FLOAT_VECTOR_TYPES
#undef VC_ALL_VECTOR_TYPES

#undef VC_EXACT_TYPE
#undef VC_ALIGNED_PARAMETER
#undef VC_OFFSETOF
#undef Vc_NOEXCEPT

#if defined(VC_GCC) && !defined(__OPTIMIZE__)
#pragma GCC diagnostic pop
#endif

#endif // VC_COMMON_UNDOMACROS_H
