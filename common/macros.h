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

#ifndef VC_COMMON_MACROS_H
#define VC_COMMON_MACROS_H
#undef VC_COMMON_UNDOMACROS_H

#include <Vc/global.h>

#ifdef VC_MSVC
# define ALIGN(n) __declspec(align(n))
# define STRUCT_ALIGN1(n) ALIGN(n)
# define STRUCT_ALIGN2(n)
# define ALIGNED_TYPEDEF(n, _type_, _newType_) typedef ALIGN(n) _type_ _newType_
#else
# define ALIGN(n) __attribute__((aligned(n)))
# define STRUCT_ALIGN1(n)
# define STRUCT_ALIGN2(n) ALIGN(n)
# define ALIGNED_TYPEDEF(n, _type_, _newType_) typedef _type_ _newType_ ALIGN(n)
#endif

#define FREE_STORE_OPERATORS_ALIGNED(alignment) \
        void *operator new(size_t size) { return _mm_malloc(size, alignment); } \
        void *operator new(size_t, void *p) { return p; } \
        void *operator new[](size_t size) { return _mm_malloc(size, alignment); } \
        void operator delete(void *ptr, size_t) { _mm_free(ptr); } \
        void operator delete[](void *ptr, size_t) { _mm_free(ptr); }

#ifdef __GNUC__
#  define INTRINSIC __attribute__((__flatten__, __always_inline__, __artificial__))
#  define INTRINSIC_L
#  define INTRINSIC_R INTRINSIC
#  define FLATTEN __attribute__((__flatten__))
#  define CONST __attribute__((__const__))
#  define CONST_L
#  define CONST_R CONST
#  define PURE __attribute__((__pure__))
#  define MAY_ALIAS __attribute__((__may_alias__))
#  define ALWAYS_INLINE __attribute__((__always_inline__))
#  define ALWAYS_INLINE_L
#  define ALWAYS_INLINE_R ALWAYS_INLINE
#  define VC_IS_UNLIKELY(x) __builtin_expect(x, 0)
#  define VC_IS_LIKELY(x) __builtin_expect(x, 1)
#  define VC_RESTRICT __restrict__
#else
#  define FLATTEN
#  ifdef PURE
#    undef PURE
#  endif
#  define PURE
#  define MAY_ALIAS
#  ifdef VC_MSVC
#    define ALWAYS_INLINE __forceinline
#    define ALWAYS_INLINE_L ALWAYS_INLINE
#    define ALWAYS_INLINE_R
#    define CONST __declspec(noalias)
#    define CONST_L CONST
#    define CONST_R
#    define INTRINSIC __forceinline
#    define INTRINSIC_L INTRINSIC
#    define INTRINSIC_R
#  else
#    define ALWAYS_INLINE
#    define ALWAYS_INLINE_L
#    define ALWAYS_INLINE_R
#    define CONST
#    define CONST_L
#    define CONST_R
#    define INTRINSIC
#    define INTRINSIC_L
#    define INTRINSIC_R
#  endif
#  define VC_IS_UNLIKELY(x) x
#  define VC_IS_LIKELY(x) x
#  define VC_RESTRICT __restrict
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
# define VC_WARN_INLINE
# define VC_WARN(msg) __attribute__((warning("\n\t" msg)))
#else
# define VC_WARN_INLINE inline
# define VC_WARN(msg)
#endif

#define CAT_HELPER(a, b) a##b
#define CAT(a, b) CAT_HELPER(a, b)

#define CAT3_HELPER(a, b, c) a##b##c
#define CAT3(a, b, c) CAT3_HELPER(a, b, c)

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

#ifndef _VC_STATIC_ASSERT_TYPES_H
#define _VC_STATIC_ASSERT_TYPES_H
namespace Vc {
    namespace {
        template<bool> class STATIC_ASSERT_FAILURE;
        template<> class STATIC_ASSERT_FAILURE<true> {};
}}
#endif // _VC_STATIC_ASSERT_TYPES_H

#define VC_STATIC_ASSERT_NC(cond, msg) \
    typedef STATIC_ASSERT_FAILURE<cond> CAT(_STATIC_ASSERTION_FAILED_##msg, __LINE__); \
    CAT(_STATIC_ASSERTION_FAILED_##msg, __LINE__) CAT3(Error_,__LINE__,msg)
#define VC_STATIC_ASSERT(cond, msg) VC_STATIC_ASSERT_NC(cond, msg); (void) CAT3(Error_,__LINE__,msg)


#endif // VC_COMMON_MACROS_H
