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

#ifndef VC_SSE_MACROS_H
#define VC_SSE_MACROS_H
#undef VC_SSE_UNDOMACROS_H

#ifndef ALIGN
# ifdef __GNUC__
#  define ALIGN(n) __attribute__((aligned(n)))
# else
#  define ALIGN(n) __declspec(align(n))
# endif
#endif

#define FREE_STORE_OPERATORS_ALIGNED(alignment) \
        void *operator new(size_t size) { return _mm_malloc(size, alignment); } \
        void *operator new[](size_t size) { return _mm_malloc(size, alignment); } \
        void operator delete(void *ptr, size_t) { _mm_free(ptr); } \
        void operator delete[](void *ptr, size_t) { _mm_free(ptr); }

#ifdef __GNUC__
#define CONST __attribute__((const))
#define MAY_ALIAS __attribute__((__may_alias__))
#define ALWAYS_INLINE __attribute__((always_inline))
#else
#define CONST
#define MAY_ALIAS
#define ALWAYS_INLINE
#endif

#define CAT_HELPER(a, b) a##b
#define CAT(a, b) CAT_HELPER(a, b)

#define unrolled_loop16(_it_, _start_, _end_, _code_) \
if (_start_ +  0 < _end_) { enum { _it_ = (_start_ +  0) < _end_ ? (_start_ +  0) : _start_ }; _code_ } \
if (_start_ +  1 < _end_) { enum { _it_ = (_start_ +  1) < _end_ ? (_start_ +  1) : _start_ }; _code_ } \
if (_start_ +  2 < _end_) { enum { _it_ = (_start_ +  2) < _end_ ? (_start_ +  2) : _start_ }; _code_ } \
if (_start_ +  3 < _end_) { enum { _it_ = (_start_ +  3) < _end_ ? (_start_ +  3) : _start_ }; _code_ } \
if (_start_ +  4 < _end_) { enum { _it_ = (_start_ +  4) < _end_ ? (_start_ +  4) : _start_ }; _code_ } \
if (_start_ +  5 < _end_) { enum { _it_ = (_start_ +  5) < _end_ ? (_start_ +  5) : _start_ }; _code_ } \
if (_start_ +  6 < _end_) { enum { _it_ = (_start_ +  6) < _end_ ? (_start_ +  6) : _start_ }; _code_ } \
if (_start_ +  7 < _end_) { enum { _it_ = (_start_ +  7) < _end_ ? (_start_ +  7) : _start_ }; _code_ } \
if (_start_ +  8 < _end_) { enum { _it_ = (_start_ +  8) < _end_ ? (_start_ +  8) : _start_ }; _code_ } \
if (_start_ +  9 < _end_) { enum { _it_ = (_start_ +  9) < _end_ ? (_start_ +  9) : _start_ }; _code_ } \
if (_start_ + 10 < _end_) { enum { _it_ = (_start_ + 10) < _end_ ? (_start_ + 10) : _start_ }; _code_ } \
if (_start_ + 11 < _end_) { enum { _it_ = (_start_ + 11) < _end_ ? (_start_ + 11) : _start_ }; _code_ } \
if (_start_ + 12 < _end_) { enum { _it_ = (_start_ + 12) < _end_ ? (_start_ + 12) : _start_ }; _code_ } \
if (_start_ + 13 < _end_) { enum { _it_ = (_start_ + 13) < _end_ ? (_start_ + 13) : _start_ }; _code_ } \
if (_start_ + 14 < _end_) { enum { _it_ = (_start_ + 14) < _end_ ? (_start_ + 14) : _start_ }; _code_ } \
if (_start_ + 15 < _end_) { enum { _it_ = (_start_ + 15) < _end_ ? (_start_ + 15) : _start_ }; _code_ } \
do {} while ( false )

#define for_all_vector_entries(_it_, _code_) \
  unrolled_loop16(_it_, 0, Size, _code_)

#ifndef _M128
# define _M128 __m128
#endif

#ifndef _M128I
# define _M128I __m128i
#endif

#ifndef _M128D
# define _M128D __m128d
#endif

#define STORE_VECTOR(type, name, vec) \
    union { __m128i p; type v[16 / sizeof(type)]; } CAT(u, __LINE__); \
    _mm_store_si128(&CAT(u, __LINE__).p, vec); \
    const type *const name = &CAT(u, __LINE__).v[0]

#endif // VC_SSE_MACROS_H
