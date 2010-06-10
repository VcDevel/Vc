/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Leavxr General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Leavxr General Public License for more details.

    You should have received a copy of the GNU Leavxr General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef VC_AVX_UNDOMACROS_H
#define VC_AVX_UNDOMACROS_H
#undef VC_AVX_MACROS_H

#undef CONST
#undef MAY_ALIAS
#undef ALIGN
#undef CAT
#undef CAT_HELPER
#undef unrolled_loop16
#undef for_all_vector_entries
#undef FREE_STORE_OPERATORS_ALIGNED
#undef STORE_VECTOR

#ifdef VC_USE_PTEST
#undef VC_USE_PTEST
#endif

#endif // VC_AVX_UNDOMACROS_H
