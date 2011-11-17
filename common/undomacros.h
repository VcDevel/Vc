/*  This file is part of the Vc library.

    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef VC_COMMON_UNDOMACROS_H
#define VC_COMMON_UNDOMACROS_H
#undef VC_COMMON_MACROS_H

#undef INTRINSIC
#undef INTRINSIC_L
#undef INTRINSIC_R
#undef CONST
#undef CONST_L
#undef CONST_R
#undef PURE
#undef MAY_ALIAS
#undef ALWAYS_INLINE
#undef ALWAYS_INLINE_L
#undef ALWAYS_INLINE_R

#undef ALIGN
#undef STRUCT_ALIGN1
#undef STRUCT_ALIGN2
#undef ALIGNED_TYPEDEF
#undef CAT
#undef CAT_HELPER
#undef CAT3
#undef CAT3_HELPER
#undef unrolled_loop16
#undef for_all_vector_entries
#undef FREE_STORE_OPERATORS_ALIGNED

#undef VC_WARN_INLINE
#undef VC_WARN

#undef VC_STATIC_ASSERT_NC
#undef VC_STATIC_ASSERT

#endif // VC_COMMON_UNDOMACROS_H
