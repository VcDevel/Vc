/*  This file is part of the Vc library.

    Copyright (C) 2010-2012 Matthias Kretz <kretz@kde.org>

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

#if defined(VC_GCC) && !defined(__OPTIMIZE__)
// GCC uses lots of old-style-casts in macros that disguise as intrinsics
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

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

#ifdef VC_CLANG
#  define Vc_INTRINSIC_L inline
#  define Vc_INTRINSIC_R __attribute__((always_inline))
#  define Vc_INTRINSIC Vc_INTRINSIC_L Vc_INTRINSIC_R
#  define Vc_FLATTEN
#  define Vc_CONST __attribute__((const))
#  define Vc_CONST_L
#  define Vc_CONST_R Vc_CONST
#  define Vc_PURE __attribute__((pure))
#  define Vc_PURE_L
#  define Vc_PURE_R Vc_PURE
#  define Vc_MAY_ALIAS __attribute__((may_alias))
#  define Vc_ALWAYS_INLINE_L inline
#  define Vc_ALWAYS_INLINE_R __attribute__((always_inline))
#  define Vc_ALWAYS_INLINE Vc_ALWAYS_INLINE_L Vc_ALWAYS_INLINE_R
#  define VC_IS_UNLIKELY(x) __builtin_expect(x, 0)
#  define VC_IS_LIKELY(x) __builtin_expect(x, 1)
#  define VC_RESTRICT __restrict__
#  define VC_DEPRECATED(msg)
#  define Vc_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))
#elif defined(__GNUC__)
#  define Vc_MAY_ALIAS __attribute__((__may_alias__))
#  define Vc_INTRINSIC_R __attribute__((__flatten__, __always_inline__, __artificial__))
#  define Vc_INTRINSIC_L inline
#  define Vc_INTRINSIC Vc_INTRINSIC_L Vc_INTRINSIC_R
#  define Vc_FLATTEN __attribute__((__flatten__))
#  define Vc_ALWAYS_INLINE_L inline
#  define Vc_ALWAYS_INLINE_R __attribute__((__always_inline__))
#  define Vc_ALWAYS_INLINE Vc_ALWAYS_INLINE_L Vc_ALWAYS_INLINE_R
#  ifdef VC_ICC
     // ICC miscompiles if there are functions marked as pure or const
#    define Vc_PURE
#    define Vc_CONST
#  else
#    define Vc_PURE __attribute__((__pure__))
#    define Vc_CONST __attribute__((__const__))
#  endif
#  define Vc_CONST_L
#  define Vc_CONST_R Vc_CONST
#  define Vc_PURE_L
#  define Vc_PURE_R Vc_PURE
#  define VC_IS_UNLIKELY(x) __builtin_expect(x, 0)
#  define VC_IS_LIKELY(x) __builtin_expect(x, 1)
#  define VC_RESTRICT __restrict__
#  define VC_DEPRECATED(msg) __attribute__((__deprecated__(msg)))
#  define Vc_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))
#else
#  define Vc_FLATTEN
#  ifdef Vc_PURE
#    undef Vc_PURE
#  endif
#  define Vc_MAY_ALIAS
#  ifdef VC_MSVC
#    define Vc_ALWAYS_INLINE inline __forceinline
#    define Vc_ALWAYS_INLINE_L Vc_ALWAYS_INLINE
#    define Vc_ALWAYS_INLINE_R
#    define Vc_CONST __declspec(noalias)
#    define Vc_CONST_L Vc_CONST
#    define Vc_CONST_R
#    define Vc_PURE /*Vc_CONST*/
#    define Vc_PURE_L Vc_PURE
#    define Vc_PURE_R
#    define Vc_INTRINSIC inline __forceinline
#    define Vc_INTRINSIC_L Vc_INTRINSIC
#    define Vc_INTRINSIC_R
#  else
#    define Vc_ALWAYS_INLINE
#    define Vc_ALWAYS_INLINE_L
#    define Vc_ALWAYS_INLINE_R
#    define Vc_CONST
#    define Vc_CONST_L
#    define Vc_CONST_R
#    define Vc_PURE
#    define Vc_PURE_L
#    define Vc_PURE_R
#    define Vc_INTRINSIC
#    define Vc_INTRINSIC_L
#    define Vc_INTRINSIC_R
#  endif
#  define VC_IS_UNLIKELY(x) x
#  define VC_IS_LIKELY(x) x
#  define VC_RESTRICT __restrict
#  define VC_DEPRECATED(msg) __declspec(deprecated(msg))
#  define Vc_WARN_UNUSED_RESULT
#endif

#define FREE_STORE_OPERATORS_ALIGNED(alignment) \
        Vc_ALWAYS_INLINE void *operator new(size_t size) { return _mm_malloc(size, alignment); } \
        Vc_ALWAYS_INLINE void *operator new(size_t, void *p) { return p; } \
        Vc_ALWAYS_INLINE void *operator new[](size_t size) { return _mm_malloc(size, alignment); } \
        Vc_ALWAYS_INLINE void *operator new[](size_t , void *p) { return p; } \
        Vc_ALWAYS_INLINE void operator delete(void *ptr, size_t) { _mm_free(ptr); } \
        Vc_ALWAYS_INLINE void operator delete(void *, void *) {} \
        Vc_ALWAYS_INLINE void operator delete[](void *ptr, size_t) { _mm_free(ptr); } \
        Vc_ALWAYS_INLINE void operator delete[](void *, void *) {}

#ifdef VC_GCC
# define VC_WARN_INLINE
# define VC_WARN(msg) __attribute__((warning("\n\t" msg)))
#else
# define VC_WARN_INLINE inline
# define VC_WARN(msg)
#endif

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

#ifdef VC_ASSERT
#define VC_EXTERNAL_ASSERT 1
#else
#ifdef NDEBUG
#define VC_ASSERT(x)
#else
#include <assert.h>
#define VC_ASSERT(x) assert(x);
#endif
#endif

#ifdef VC_CLANG
#define VC_HAS_BUILTIN(x) __has_builtin(x)
#else
#define VC_HAS_BUILTIN(x) 0
#endif

#ifndef VC_COMMON_MACROS_H_ONCE
#define VC_COMMON_MACROS_H_ONCE

#define _VC_CAT_HELPER(a, b, c, d) a##b##c##d
#define _VC_CAT(a, b, c, d) _VC_CAT_HELPER(a, b, c, d)

    template<int e, int center> struct exponentToMultiplier { enum {
        X = exponentToMultiplier<e - 1, center>::X * ((e - center < 31) ? 2 : 1),
        Value = (X == 0 ? 1 : X)
    }; };
    template<int center> struct exponentToMultiplier<center,center> { enum { X = 1, Value = X }; };
    template<int center> struct exponentToMultiplier<   -1, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToMultiplier< -128, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToMultiplier< -256, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToMultiplier< -384, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToMultiplier< -512, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToMultiplier< -640, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToMultiplier< -768, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToMultiplier< -896, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToMultiplier<-1024, center> { enum { X = 0, Value = 1 }; };

    template<int e, int center> struct exponentToDivisor { enum {
        X = exponentToDivisor<e + 1, center>::X * ((center - e < 31) ? 2 : 1),
        Value = (X == 0 ? 1 : X)
    }; };
    template<int center> struct exponentToDivisor<center, center> { enum { X = 1, Value = X }; };
    template<int center> struct exponentToDivisor<     1, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToDivisor<   128, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToDivisor<   256, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToDivisor<   384, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToDivisor<   512, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToDivisor<   640, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToDivisor<   768, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToDivisor<   896, center> { enum { X = 0, Value = 1 }; };
    template<int center> struct exponentToDivisor<  1024, center> { enum { X = 0, Value = 1 }; };
#endif // VC_COMMON_MACROS_H_ONCE

#define _CAT_IMPL(a, b) a##b
#define CAT(a, b) _CAT_IMPL(a, b)

#define _VC_APPLY_IMPL_1(macro, a, b, c, d, e) macro(a)
#define _VC_APPLY_IMPL_2(macro, a, b, c, d, e) macro(a, b)
#define _VC_APPLY_IMPL_3(macro, a, b, c, d, e) macro(a, b, c)
#define _VC_APPLY_IMPL_4(macro, a, b, c, d, e) macro(a, b, c, d)
#define _VC_APPLY_IMPL_5(macro, a, b, c, d, e) macro(a, b, c, d, e)

#define VC_LIST_FLOAT_VECTOR_TYPES(size, macro, a, b, c, d) \
    size(macro, double_v, a, b, c, d) \
    size(macro,  float_v, a, b, c, d)
#define VC_LIST_INT_VECTOR_TYPES(size, macro, a, b, c, d) \
    size(macro,    int_v, a, b, c, d) \
    size(macro,   uint_v, a, b, c, d) \
    size(macro,  short_v, a, b, c, d) \
    size(macro, ushort_v, a, b, c, d)
#define VC_LIST_VECTOR_TYPES(size, macro, a, b, c, d) \
    VC_LIST_FLOAT_VECTOR_TYPES(size, macro, a, b, c, d) \
    VC_LIST_INT_VECTOR_TYPES(size, macro, a, b, c, d)
#define VC_LIST_COMPARES(size, macro, a, b, c, d) \
    size(macro, ==, a, b, c, d) \
    size(macro, !=, a, b, c, d) \
    size(macro, <=, a, b, c, d) \
    size(macro, >=, a, b, c, d) \
    size(macro, < , a, b, c, d) \
    size(macro, > , a, b, c, d)
#define VC_LIST_LOGICAL(size, macro, a, b, c, d) \
    size(macro, &&, a, b, c, d) \
    size(macro, ||, a, b, c, d)
#define VC_LIST_BINARY(size, macro, a, b, c, d) \
    size(macro, |, a, b, c, d) \
    size(macro, &, a, b, c, d) \
    size(macro, ^, a, b, c, d)
#define VC_LIST_SHIFTS(size, macro, a, b, c, d) \
    size(macro, <<, a, b, c, d) \
    size(macro, >>, a, b, c, d)
#define VC_LIST_ARITHMETICS(size, macro, a, b, c, d) \
    size(macro, +, a, b, c, d) \
    size(macro, -, a, b, c, d) \
    size(macro, *, a, b, c, d) \
    size(macro, /, a, b, c, d) \
    size(macro, %, a, b, c, d)

#define VC_APPLY_0(_list, macro)             _list(_VC_APPLY_IMPL_1, macro, 0, 0, 0, 0)
#define VC_APPLY_1(_list, macro, a)          _list(_VC_APPLY_IMPL_2, macro, a, 0, 0, 0)
#define VC_APPLY_2(_list, macro, a, b)       _list(_VC_APPLY_IMPL_3, macro, a, b, 0, 0)
#define VC_APPLY_3(_list, macro, a, b, c)    _list(_VC_APPLY_IMPL_4, macro, a, b, c, 0)
#define VC_APPLY_4(_list, macro, a, b, c, d) _list(_VC_APPLY_IMPL_5, macro, a, b, c, d)

#define VC_ALL_COMPARES(macro)     VC_APPLY_0(VC_LIST_COMPARES, macro)
#define VC_ALL_LOGICAL(macro)      VC_APPLY_0(VC_LIST_LOGICAL, macro)
#define VC_ALL_BINARY(macro)       VC_APPLY_0(VC_LIST_BINARY, macro)
#define VC_ALL_SHIFTS(macro)       VC_APPLY_0(VC_LIST_SHIFTS, macro)
#define VC_ALL_ARITHMETICS(macro)  VC_APPLY_0(VC_LIST_ARITHMETICS, macro)
#define VC_ALL_FLOAT_VECTOR_TYPES(macro) VC_APPLY_0(VC_LIST_FLOAT_VECTOR_TYPES, macro)
#define VC_ALL_VECTOR_TYPES(macro) VC_APPLY_0(VC_LIST_VECTOR_TYPES, macro)

#define VC_EXACT_TYPE(_test, _reference, _type) \
    typename std::enable_if<std::is_same<_test, _reference>::value, _type>::type

#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
#define VC_ALIGNED_PARAMETER(_Type) const _Type &
#else
#define VC_ALIGNED_PARAMETER(_Type) const _Type
#endif

#ifndef Vc__make_unique
#define Vc__make_unique(name) _VC_CAT(Vc__,name,_,__LINE__)
#endif

#if defined(VC_ICC) || defined(VC_CLANG)
#define VC_OFFSETOF(Type, member) (reinterpret_cast<const char *>(&reinterpret_cast<const Type *>(0)->member) - reinterpret_cast<const char *>(0))
#elif defined(VC_GCC) && VC_GCC < 0x40500
#define VC_OFFSETOF(Type, member) (reinterpret_cast<const char *>(&reinterpret_cast<const Type *>(0x1000)->member) - reinterpret_cast<const char *>(0x1000))
#else
#define VC_OFFSETOF(Type, member) offsetof(Type, member)
#endif

#if defined(Vc__NO_NOEXCEPT)
#define Vc_NOEXCEPT throw()
#else
#define Vc_NOEXCEPT noexcept
#endif

// ref-ref:
#ifdef VC_NO_MOVE_CTOR
#define VC_RR_ &
#define VC_FORWARD_(T)
#else
#define VC_RR_ &&
#define VC_FORWARD_(T) std::forward<T>
#endif

#ifdef VC_NO_ALWAYS_INLINE
#undef Vc_ALWAYS_INLINE
#undef Vc_ALWAYS_INLINE_L
#undef Vc_ALWAYS_INLINE_R
#define Vc_ALWAYS_INLINE inline
#define Vc_ALWAYS_INLINE_L inline
#define Vc_ALWAYS_INLINE_R
#undef Vc_INTRINSIC
#undef Vc_INTRINSIC_L
#undef Vc_INTRINSIC_R
#define Vc_INTRINSIC inline
#define Vc_INTRINSIC_L inline
#define Vc_INTRINSIC_R
#endif

#endif // VC_COMMON_MACROS_H
