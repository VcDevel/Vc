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

#ifdef VC_CLANG
#  define INTRINSIC __attribute__((always_inline))
#  define INTRINSIC_L
#  define INTRINSIC_R INTRINSIC
#  define FLATTEN
#  define CONST __attribute__((const))
#  define CONST_L
#  define CONST_R CONST
#  define PURE __attribute__((pure))
#  define PURE_L
#  define PURE_R PURE
#  define MAY_ALIAS __attribute__((may_alias))
#  define ALWAYS_INLINE __attribute__((always_inline))
#  define ALWAYS_INLINE_L
#  define ALWAYS_INLINE_R ALWAYS_INLINE
#  define VC_IS_UNLIKELY(x) __builtin_expect(x, 0)
#  define VC_IS_LIKELY(x) __builtin_expect(x, 1)
#  define VC_RESTRICT __restrict__
#elif defined(__GNUC__)
#  if VC_GCC < 0x40300 || defined(VC_OPEN64)
// GCC 4.1 and 4.2 ICE on may_alias. Since Open64 uses the GCC 4.2 frontend it has the same problem.
#    define MAY_ALIAS
#  else
#    define MAY_ALIAS __attribute__((__may_alias__))
#  endif
#  if VC_GCC < 0x40200
// GCC 4.1 fails with "sorry unimplemented: inlining failed"
#    define INTRINSIC __attribute__((__flatten__))
#  elif VC_GCC < 0x40300 || defined(VC_OPEN64)
// the GCC 4.2 frontend doesn't know the __artificial__ attribute
#    define INTRINSIC __attribute__((__flatten__, __always_inline__))
#  else
#    define INTRINSIC __attribute__((__flatten__, __always_inline__, __artificial__))
#  endif
#  define INTRINSIC_L
#  define INTRINSIC_R INTRINSIC
#  define FLATTEN __attribute__((__flatten__))
#  define CONST __attribute__((__const__))
#  define CONST_L
#  define CONST_R CONST
#  define PURE __attribute__((__pure__))
#  define PURE_L
#  define PURE_R PURE
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
#  define MAY_ALIAS
#  ifdef VC_MSVC
#    define ALWAYS_INLINE __forceinline
#    define ALWAYS_INLINE_L ALWAYS_INLINE
#    define ALWAYS_INLINE_R
#    define CONST __declspec(noalias)
#    define CONST_L CONST
#    define CONST_R
#    define PURE /*CONST*/
#    define PURE_L PURE
#    define PURE_R
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
#    define PURE
#    define PURE_L
#    define PURE_R
#    define INTRINSIC
#    define INTRINSIC_L
#    define INTRINSIC_R
#  endif
#  define VC_IS_UNLIKELY(x) x
#  define VC_IS_LIKELY(x) x
#  define VC_RESTRICT __restrict
#endif

#if __cplusplus >= 201103 /*C++11*/
#define _VC_CONSTEXPR constexpr
#define _VC_CONSTEXPR_L _VC_CONSTEXPR
#define _VC_CONSTEXPR_R
#else
#define _VC_CONSTEXPR inline INTRINSIC CONST
#define _VC_CONSTEXPR_L inline INTRINSIC_L CONST_R
#define _VC_CONSTEXPR_R INTRINSIC_R CONST_R
#endif

#if (defined(__GXX_EXPERIMENTAL_CXX0X__) && VC_GCC >= 0x40600) || __cplusplus >= 201103
# define _VC_NOEXCEPT noexcept
#else
# define _VC_NOEXCEPT throw()
#endif


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

#ifdef NDEBUG
#define VC_ASSERT(x)
#else
#include <assert.h>
#define VC_ASSERT(x) assert(x);
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

#if __cplusplus >= 201103 /*C++11*/ || VC_MSVC >= 160000000
#define VC_STATIC_ASSERT_NC(cond, msg) \
    static_assert(cond, #msg)
#define VC_STATIC_ASSERT(cond, msg) VC_STATIC_ASSERT_NC(cond, msg)
#else // C++98
namespace Vc {
    namespace {
        template<bool> struct STATIC_ASSERT_FAILURE;
        template<> struct STATIC_ASSERT_FAILURE<true> {};
}}

#define VC_STATIC_ASSERT_NC(cond, msg) \
    typedef STATIC_ASSERT_FAILURE<cond> _VC_CAT(static_assert_failed_on_line_,__LINE__,_,msg); \
    enum { \
        _VC_CAT(static_assert_failed__on_line_,__LINE__,_,msg) = sizeof(_VC_CAT(static_assert_failed_on_line_,__LINE__,_,msg)) \
    }
#define VC_STATIC_ASSERT(cond, msg) VC_STATIC_ASSERT_NC(cond, msg)
#endif // C++11/98

    template<int e, int center> struct exponentToMultiplier { enum {
        X = exponentToMultiplier<e - 1, center>::X * ((e - center < 31) ? 2 : 1),
        Value = (X == 0 ? 1 : X)
    }; };
    template<int center> struct exponentToMultiplier<center,center> { enum { X = 1, Value = X }; };
    template<int center> struct exponentToMultiplier<   -1, center> { enum { X = 0 }; };
    template<int center> struct exponentToMultiplier< -256, center> { enum { X = 0 }; };
    template<int center> struct exponentToMultiplier< -512, center> { enum { X = 0 }; };
    template<int center> struct exponentToMultiplier<-1024, center> { enum { X = 0 }; };

    template<int e> struct exponentToDivisor { enum {
        X = exponentToDivisor<e + 1>::X * 2,
      Value = (X == 0 ? 1 : X)
    }; };
    template<> struct exponentToDivisor<0> { enum { X = 1, Value = X }; };
    template<> struct exponentToDivisor<256> { enum { X = 0 }; };
    template<> struct exponentToDivisor<512> { enum { X = 0 }; };
    template<> struct exponentToDivisor<1024> { enum { X = 0 }; };
#endif // VC_COMMON_MACROS_H_ONCE

#define _CAT_IMPL(a, b) a##b
#define CAT(a, b) _CAT_IMPL(a, b)

#define Vc_buildDouble(sign, mantissa, exponent) \
    ((static_cast<double>((mantissa & 0x000fffffffffffffull) | 0x0010000000000000ull) / 0x0010000000000000ull) \
    * exponentToMultiplier<exponent, 0>::Value \
    * exponentToMultiplier<exponent, 30>::Value \
    * exponentToMultiplier<exponent, 60>::Value \
    * exponentToMultiplier<exponent, 90>::Value \
    / exponentToDivisor<exponent>::Value \
    * static_cast<double>(sign))
#define Vc_buildFloat(sign, mantissa, exponent) \
    ((static_cast<float>((mantissa & 0x007fffffu) | 0x00800000) / 0x00800000) \
    * exponentToMultiplier<exponent, 0>::Value \
    * exponentToMultiplier<exponent, 30>::Value \
    * exponentToMultiplier<exponent, 60>::Value \
    * exponentToMultiplier<exponent, 90>::Value \
    / exponentToDivisor<exponent>::Value \
    * static_cast<float>(sign))

#define _VC_APPLY_IMPL_1(macro, a, b, c, d, e) macro(a)
#define _VC_APPLY_IMPL_2(macro, a, b, c, d, e) macro(a, b)
#define _VC_APPLY_IMPL_3(macro, a, b, c, d, e) macro(a, b, c)
#define _VC_APPLY_IMPL_4(macro, a, b, c, d, e) macro(a, b, c, d)
#define _VC_APPLY_IMPL_5(macro, a, b, c, d, e) macro(a, b, c, d, e)

#define VC_LIST_FLOAT_VECTOR_TYPES(size, macro, a, b, c, d) \
    size(macro, double_v, a, b, c, d) \
    size(macro,  float_v, a, b, c, d) \
    size(macro, sfloat_v, a, b, c, d)
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
    typename EnableIf<IsEqualType<_test, _reference>::Value, _type>::Value

#endif // VC_COMMON_MACROS_H
