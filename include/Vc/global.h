/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_IMPL
# define VC_IMPL

# if defined(__SSE4_1__)
#  define VC_IMPL_SSE 1
#  define VC_IMPL_SSE4_1 1
# endif
# if defined(__SSE3__)
#  define VC_IMPL_SSE 1
#  define VC_IMPL_SSE3 1
# endif
# if defined(__SSSE3__)
#  define VC_IMPL_SSE 1
#  define VC_IMPL_SSSE3 1
# endif
# if defined(__SSE2__)
#  define VC_IMPL_SSE 1
#  define VC_IMPL_SSE2 1
# endif

# if defined(VC_IMPL_SSE)
// nothing
# elif defined(__LRB__)
#  define VC_IMPL_LRBni 1
# else
#  define VC_IMPL_Scalar 1
# endif

#else // VC_IMPL

#define SSE    9875294

# if VC_IMPL == SSE
#  undef VC_IMPL
#  if defined(__SSE4_1__)
#   define VC_IMPL_SSE 1
#   define VC_IMPL_SSE4_1 1
#   define VC_IMPL SSE4_1
#  elif defined(__SSSE3__)
#   define VC_IMPL_SSE 1
#   define VC_IMPL_SSSE3 1
#   define VC_IMPL SSSE3
#  elif defined(__SSE3__)
#   define VC_IMPL_SSE 1
#   define VC_IMPL_SSE3 1
#   define VC_IMPL SSE3
#  elif defined(__SSE2__)
#   define VC_IMPL_SSE 1
#   define VC_IMPL_SSE2 1
#   define VC_IMPL SSE2
#  else
#   define VC_IMPL_SSE 1
#   define VC_IMPL_SSE4_1 1
#   define VC_IMPL SSE4_1
#  endif
# endif

#define SSE2   9875295
#define SSE3   9875296
#define SSSE3  9875297
#define SSE4_1 9875298
#define Scalar 9875299
#define LRBni  9875300

# if VC_IMPL == Scalar
#  define VC_IMPL_Scalar 1
# elif VC_IMPL == LRBni
#  define VC_IMPL_LRBni 1
# elif VC_IMPL != SSE2 && VC_IMPL != SSE3 && VC_IMPL != SSSE3 && VC_IMPL != SSE4_1
#  error "VC_IMPL is not set to a known value!"
# endif

#undef SSE
#undef SSE2
#undef SSE3
#undef SSSE3
#undef SSE4_1
#undef Scalar
#undef LRBni

#endif // VC_IMPL
