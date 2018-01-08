/*  This file is part of the Vc library. {{{
Copyright © 2010-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SIMD_MACROS_H_
#define VC_SIMD_MACROS_H_

#include "../version.h"

// warning macro {{{
#define Vc_PRAGMA_(x_) _Pragma(#x_)
#ifdef __GNUC__
#define Vc_CPP_WARNING(msg) Vc_PRAGMA_(GCC warning msg)
#else
#define Vc_CPP_WARNING(msg) Vc_PRAGMA_(message "warning: " msg)
#endif
// }}}

// warning for Vc 1.x code {{{
#if defined VC_IMPL || defined Vc_IMPL
Vc_CPP_WARNING("The Vc_IMPL macro was removed for Vc 2.0. "
               "Instructions are restricted solely via compiler flags. "
               "The vector ABI is chosen via code. "
               "The default vector ABI is currently not selectable.")
#endif
// }}}

// not-yet-optimized warning hack {{{
#ifndef Vc_NO_OPTIMIZATION_WARNINGS
#if defined Vc_GCC || defined Vc_CLANG
// hack GCC's formatting to overwrite "deprecated: " with our own message:
#define Vc_NOT_OPTIMIZED [[deprecated("\x8\x8\x8\x8\x8\x8\x8\x8\x8\x8\x8\x8not optimized yet, if you care about speed please help out! [-DVc_NO_OPTIMIZATION_WARNINGS]")]]
#else
#define Vc_NOT_OPTIMIZED [[deprecated("NOT DEPRECATED, just a note: this function is not optimized yet, if you care about speed please help out! [-DVc_NO_OPTIMIZATION_WARNINGS]")]]
#endif
#else
#define Vc_NOT_OPTIMIZED
#endif
// }}}

// Starting with compiler identification. This is a prerequisite for getting the following
// macro definitions right.
// {{{

#ifdef DOXYGEN
/**
 * \name Compiler Identification Macros
 * \ingroup Utilities
 */
//@{
/**
 * \ingroup Utilities
 * This macro is defined to a number identifying the ICC version if the current
 * translation unit is compiled with the Intel compiler.
 *
 * For any other compiler this macro is not defined.
 */
#define Vc_ICC __INTEL_COMPILER_BUILD_DATE
#undef Vc_ICC
/**
 * \ingroup Utilities
 * This macro is defined to a number identifying the Clang version if the current
 * translation unit is compiled with the Clang compiler.
 *
 * For any other compiler this macro is not defined.
 */
#define Vc_CLANG (__clang_major__ * 0x10000 + __clang_minor__ * 0x100 + __clang_patchlevel__)
#undef Vc_CLANG
/**
 * \ingroup Utilities
 * This macro is defined to a number identifying the Apple Clang version if the current
 * translation unit is compiled with the Apple Clang compiler.
 *
 * For any other compiler this macro is not defined.
 */
#define Vc_APPLECLANG (__clang_major__ * 0x10000 + __clang_minor__ * 0x100 + __clang_patchlevel__)
#undef Vc_APPLECLANG
/**
 * \ingroup Utilities
 * This macro is defined to a number identifying the GCC version if the current
 * translation unit is compiled with the GCC compiler.
 *
 * For any other compiler this macro is not defined.
 */
#define Vc_GCC (__GNUC__ * 0x10000 + __GNUC_MINOR__ * 0x100 + __GNUC_PATCHLEVEL__)
/**
 * \ingroup Utilities
 * This macro is defined to a number identifying the Microsoft Visual C++ version if
 * the current translation unit is compiled with the Visual C++ (MSVC) compiler.
 *
 * For any other compiler this macro is not defined.
 */
#define Vc_MSVC _MSC_FULL_VER
#undef Vc_MSVC
//@}
#endif  // DOXYGEN

#ifdef __INTEL_COMPILER
#  define Vc_ICC __INTEL_COMPILER_BUILD_DATE
#elif defined __clang__ && defined __apple_build_version__
#  define Vc_APPLECLANG (__clang_major__ * 0x10000 + __clang_minor__ * 0x100 + __clang_patchlevel__)
#elif defined(__clang__)
#  define Vc_CLANG (__clang_major__ * 0x10000 + __clang_minor__ * 0x100 + __clang_patchlevel__)
#elif defined(__GNUC__)
#  define Vc_GCC (__GNUC__ * 0x10000 + __GNUC_MINOR__ * 0x100 + __GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#  define Vc_MSVC _MSC_FULL_VER
#else
#  define Vc_UNSUPPORTED_COMPILER 1
#endif
//}}}

// Ensure C++14 feature that are required. Define Vc_CXX17 if C++17 is available.{{{
#if !(defined Vc_ICC || (defined Vc_MSVC && Vc_MSVC >= 191025017) || __cplusplus >= 201402L)
#error "Vc requires support for C++14."
#endif

#define Vc_CXX14 1
#if __cplusplus > 201700L
#  define Vc_CXX17 1
#endif
// }}}

// C++ does not allow attaching overalignment to an existing type via an alias. In
// general, that seems to be the right thing to do. However some workarounds require
// special measures, so here's a macro for doing it with a compilier specific extension.
// {{{
#ifdef Vc_MSVC
#define Vc_ALIGNED_TYPEDEF(n_, type_, new_type_)                                      \
    typedef __declspec(align(n_)) type_ new_type_
#elif __GNUC__
#define Vc_ALIGNED_TYPEDEF(n_, type_, new_type_)                                      \
    typedef type_ new_type_[[gnu::aligned(n_)]]
#else  // the following is actually ill-formed according to C++1[14]
#define Vc_ALIGNED_TYPEDEF(n_, type_, new_type_)                                      \
    using new_type_ alignas(sizeof(n_)) = type_
#endif
// }}}

// On Windows (WIN32) we might see macros called min and max. Just undefine them and hope
// noone (re)defines them (defining NOMINMAX should help).
// {{{
#ifdef WIN32
#define NOMINMAX 1
#if defined min
#undef min
#endif
#if defined max
#undef max
#endif
#endif  // WIN32
// }}}

// ISA extension detection. The following defines all the Vc_HAVE_XXX macros
// ARM{{{
#ifdef __aarch64__
#define Vc_IS_AARCH64 1
#endif  // __aarch64__

#ifdef __ARM_NEON
#define Vc_HAVE_NEON
#define Vc_HAVE_NEON_ABI 1
#define Vc_HAVE_FULL_NEON_ABI 1
#endif  // Vc_HAVE_NEON
//}}}
// x86{{{
#if defined __x86_64__ || defined __amd64__ || defined __amd64 || defined __x86_64 ||    \
    defined _M_AMD64
#define Vc_IS_AMD64 1
#endif

#ifdef __MMX__
#define Vc_HAVE_MMX 1
#endif
#if defined __SSE__ || defined Vc_IS_AMD64 || (defined _M_IX86_FP && _M_IX86_FP >= 1)
#define Vc_HAVE_SSE 1
#endif
#if defined __SSE2__ || defined Vc_IS_AMD64 || (defined _M_IX86_FP && _M_IX86_FP >= 2)
#define Vc_HAVE_SSE2 1
#endif
#ifdef __SSE3__
#define Vc_HAVE_SSE3 1
#endif
#ifdef __SSSE3__
#define Vc_HAVE_SSSE3 1
#endif
#ifdef __SSE4_1__
#define Vc_HAVE_SSE4_1 1
#endif
#ifdef __SSE4_2__
#define Vc_HAVE_SSE4_2 1
#endif
#ifdef __XOP__
#define Vc_HAVE_XOP 1
#endif
#ifdef __AVX__
#define Vc_HAVE_AVX 1
#endif
#ifdef __AVX2__
#define Vc_HAVE_AVX2 1
#define Vc_HAVE_BMI1 1
#define Vc_HAVE_BMI2 1
#define Vc_HAVE_LZCNT 1
#if !defined Vc_ICC && !defined Vc_MSVC
#ifndef __BMI__
#error "expected AVX2 to imply the availability of BMI1"
#endif
#ifndef __BMI2__
#error "expected AVX2 to imply the availability of BMI2"
#endif
#ifndef __LZCNT__
#error "expected AVX2 to imply the availability of LZCNT"
#endif
#endif // !ICC && !MSVC
#endif // __AVX2__
#ifdef __SSE4A__
#  define Vc_HAVE_SSE4A 1
#endif
#ifdef __FMA__
#  define Vc_HAVE_FMA 1
#endif
#ifdef __FMA4__
#  define Vc_HAVE_FMA4 1
#endif
#ifdef __F16C__
#  define Vc_HAVE_F16C 1
#endif
#if defined __POPCNT__ ||                                                                \
    (defined Vc_ICC && (defined Vc_HAVE_SSE4_2 || defined Vc_HAVE_SSE4A))
#  define Vc_HAVE_POPCNT 1
#endif
#ifdef __AVX512F__
#define Vc_HAVE_AVX512F 1
#endif
#ifdef __AVX512DQ__
#define Vc_HAVE_AVX512DQ 1
#endif
#ifdef __AVX512VL__
#define Vc_HAVE_AVX512VL 1
#endif
#ifdef __AVX512BW__
#define Vc_HAVE_AVX512BW 1
#endif
#ifdef __MIC__
#define Vc_HAVE_KNC 1
#endif

#ifdef Vc_HAVE_KNC
//#define Vc_HAVE_KNC_ABI 1
//#define Vc_HAVE_FULL_KNC_ABI 1
#endif

#if defined Vc_HAVE_SSE
#define Vc_HAVE_SSE_ABI 1
#ifdef Vc_HAVE_SSE2
#define Vc_HAVE_FULL_SSE_ABI 1
#endif
#endif

#if defined Vc_HAVE_AVX
#define Vc_HAVE_AVX_ABI 1
#if defined Vc_HAVE_AVX2
#define Vc_HAVE_FULL_AVX_ABI 1
#endif
#endif

#ifdef Vc_HAVE_AVX512F
#define Vc_HAVE_AVX512_ABI 1
#ifdef Vc_HAVE_AVX512BW
#define Vc_HAVE_FULL_AVX512_ABI 1
#endif
#endif
//}}}

// Vc_TEMPLATES_DROP_ATTRIBUTES: GCC 6 drops all attributes on types passed as template
// arguments. This is important if a may_alias gets lost and therefore needs to be readded
// in the implementation of the class template.
// {{{
#if defined Vc_GCC && Vc_GCC >= 0x60000
#define Vc_TEMPLATES_DROP_ATTRIBUTES 1
#endif
// }}}

// Vc_GNU_ASM: defined if GCC compatible inline asm is possible and Vc_NO_INLINE_ASM is
// not defined
// {{{
#if defined(__GNUC__) && !defined(Vc_NO_INLINE_ASM)
#define Vc_GNU_ASM 1
#endif
// }}}

// Vc_USE_BUILTIN_VECTOR_TYPES: defined for GCC and Clang
// TODO: rename to Vc_HAVE_BUILTIN_VECTOR_TYPES
// {{{
#if defined(Vc_GCC) || defined(Vc_CLANG) || defined Vc_APPLECLANG
#define Vc_USE_BUILTIN_VECTOR_TYPES 1
#endif
// }}}

// __cdecl and __vectorcall Windows calling convention macros. Every function with a by
// value simd/simd_mask object needs to be declared with Vc_VDECL.
// {{{
#ifdef Vc_MSVC
#  define Vc_CDECL __cdecl
#  define Vc_VDECL __vectorcall
#else
#  define Vc_CDECL
#  define Vc_VDECL
#endif
// }}}

// Vc_CONCAT{{{
#define Vc_CONCAT_IMPL(a_, b_, c_) a_##b_##c_
#define Vc_CONCAT(a_, b_, c_) Vc_CONCAT_IMPL(a_, b_, c_)
// }}}

#if defined Vc_CLANG || defined Vc_APPLECLANG
#  define Vc_UNREACHABLE __builtin_unreachable
#  define Vc_NEVER_INLINE [[gnu::noinline]]
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
#  define Vc_IS_UNLIKELY(x) __builtin_expect(x, 0)
#  define Vc_IS_LIKELY(x) __builtin_expect(x, 1)
#  define Vc_RESTRICT __restrict__
#  define Vc_DEPRECATED_ALIAS(msg)
#  define Vc_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))
#elif defined(__GNUC__)
#  define Vc_UNREACHABLE __builtin_unreachable
#  if defined Vc_GCC && !defined __OPTIMIZE__
#    define Vc_MAY_ALIAS
#  else
#    define Vc_MAY_ALIAS __attribute__((__may_alias__))
#  endif
#  define Vc_INTRINSIC_R __attribute__((__always_inline__, __artificial__))
#  define Vc_INTRINSIC_L inline
#  define Vc_INTRINSIC Vc_INTRINSIC_L Vc_INTRINSIC_R
#  define Vc_FLATTEN __attribute__((__flatten__))
#  define Vc_ALWAYS_INLINE_L inline
#  define Vc_ALWAYS_INLINE_R __attribute__((__always_inline__))
#  define Vc_ALWAYS_INLINE Vc_ALWAYS_INLINE_L Vc_ALWAYS_INLINE_R
#  ifdef Vc_ICC
// ICC miscompiles if there are functions marked as pure or const
#    define Vc_PURE
#    define Vc_CONST
#    define Vc_NEVER_INLINE
#  else
#    define Vc_NEVER_INLINE [[gnu::noinline]]
#    define Vc_PURE __attribute__((__pure__))
#    define Vc_CONST __attribute__((__const__))
#  endif
#  define Vc_CONST_L
#  define Vc_CONST_R Vc_CONST
#  define Vc_PURE_L
#  define Vc_PURE_R Vc_PURE
#  define Vc_IS_UNLIKELY(x) __builtin_expect(x, 0)
#  define Vc_IS_LIKELY(x) __builtin_expect(x, 1)
#  define Vc_RESTRICT __restrict__
#  ifdef Vc_ICC
#    define Vc_DEPRECATED_ALIAS(msg)
#  else
#    define Vc_DEPRECATED_ALIAS(msg) __attribute__((__deprecated__(msg)))
#  endif
#  define Vc_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))
#else
#  define Vc_NEVER_INLINE
#  define Vc_FLATTEN
#  ifdef Vc_PURE
#    undef Vc_PURE
#  endif
#  define Vc_MAY_ALIAS
#  ifdef Vc_MSVC
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
#    define Vc_UNREACHABLE [] { __assume(0); }
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
#    define Vc_UNREACHABLE std::abort
#  endif
#  define Vc_IS_UNLIKELY(x) x
#  define Vc_IS_LIKELY(x) x
#  define Vc_RESTRICT __restrict
#  define Vc_DEPRECATED_ALIAS(msg)
#  define Vc_WARN_UNUSED_RESULT
#endif

#ifdef Vc_CXX17
#  define Vc_NODISCARD [[nodiscard]]
#elif defined __GNUC__
#  define Vc_NODISCARD [[gnu::warn_unused_result]]
#else
#  define Vc_NODISCARD
#endif

#define Vc_NOTHING_EXPECTING_SEMICOLON static_assert(true, "")

#ifdef Vc_CXX17
// C++17 has solved the issue: operator new will allocate with correct overalignment
#define Vc_FREE_STORE_OPERATORS_ALIGNED(align_)
#else  // Vc_CXX17
#define Vc_FREE_STORE_OPERATORS_ALIGNED(align_)                                          \
    /**\name new/delete overloads for correct alignment */                               \
    /**@{*/                                                                              \
    /*!\brief Allocates correctly aligned memory */                                      \
    Vc_ALWAYS_INLINE void *operator new(size_t size)                                     \
    {                                                                                    \
        return Vc::Common::aligned_malloc<align_>(size);                                 \
    }                                                                                    \
    /*!\brief Returns \p p. */                                                           \
    Vc_ALWAYS_INLINE void *operator new(size_t, void *p) { return p; }                   \
    /*!\brief Allocates correctly aligned memory */                                      \
    Vc_ALWAYS_INLINE void *operator new[](size_t size)                                   \
    {                                                                                    \
        return Vc::Common::aligned_malloc<align_>(size);                                 \
    }                                                                                    \
    /*!\brief Returns \p p. */                                                           \
    Vc_ALWAYS_INLINE void *operator new[](size_t, void *p) { return p; }                 \
    /*!\brief Frees aligned memory. */                                                   \
    Vc_ALWAYS_INLINE void operator delete(void *ptr, size_t) { Vc::Common::free(ptr); }  \
    /*!\brief Does nothing. */                                                           \
    Vc_ALWAYS_INLINE void operator delete(void *, void *) {}                             \
    /*!\brief Frees aligned memory. */                                                   \
    Vc_ALWAYS_INLINE void operator delete[](void *ptr, size_t)                           \
    {                                                                                    \
        Vc::Common::free(ptr);                                                           \
    }                                                                                    \
    /*!\brief Does nothing. */                                                           \
    Vc_ALWAYS_INLINE void operator delete[](void *, void *) {}                           \
    /**@}*/                                                                              \
    Vc_NOTHING_EXPECTING_SEMICOLON
#endif  // Vc_CXX17

#ifdef Vc_ASSERT
#define Vc_EXTERNAL_ASSERT 1
#else
#ifdef NDEBUG
#define Vc_ASSERT(x) Vc::detail::dummy_assert{} << ' '
#else
#define Vc_ASSERT(x) Vc::detail::real_assert(x, #x, __FILE__, __LINE__)
#endif
#endif

#ifdef COMPILE_FOR_UNIT_TESTS
#define Vc_NOEXCEPT_OR_IN_TEST
#else
#define Vc_NOEXCEPT_OR_IN_TEST noexcept
#endif

#if defined Vc_CLANG || defined Vc_APPLECLANG
#define Vc_HAS_BUILTIN(x) __has_builtin(x)
#else
#define Vc_HAS_BUILTIN(x) 0
#endif

#define Vc_APPLY_IMPL_1_(macro, a, b, c, d, e) macro(a)
#define Vc_APPLY_IMPL_2_(macro, a, b, c, d, e) macro(a, b)
#define Vc_APPLY_IMPL_3_(macro, a, b, c, d, e) macro(a, b, c)
#define Vc_APPLY_IMPL_4_(macro, a, b, c, d, e) macro(a, b, c, d)
#define Vc_APPLY_IMPL_5_(macro, a, b, c, d, e) macro(a, b, c, d, e)

#define Vc_LIST_FLOAT_VECTOR_TYPES(size, macro, a, b, c, d) \
    size(macro, double_v, a, b, c, d) \
    size(macro,  float_v, a, b, c, d)
#define Vc_LIST_INT_VECTOR_TYPES(size, macro, a, b, c, d) \
    size(macro,    int_v, a, b, c, d) \
    size(macro,   uint_v, a, b, c, d) \
    size(macro,  short_v, a, b, c, d) \
    size(macro, ushort_v, a, b, c, d)
#define Vc_LIST_VECTOR_TYPES(size, macro, a, b, c, d) \
    Vc_LIST_FLOAT_VECTOR_TYPES(size, macro, a, b, c, d) \
    Vc_LIST_INT_VECTOR_TYPES(size, macro, a, b, c, d)
#define Vc_LIST_COMPARES(size, macro, a, b, c, d) \
    size(macro, ==, a, b, c, d) \
    size(macro, !=, a, b, c, d) \
    size(macro, <=, a, b, c, d) \
    size(macro, >=, a, b, c, d) \
    size(macro, < , a, b, c, d) \
    size(macro, > , a, b, c, d)
#define Vc_LIST_LOGICAL(size, macro, a, b, c, d) \
    size(macro, &&, a, b, c, d) \
    size(macro, ||, a, b, c, d)
#define Vc_LIST_BINARY(size, macro, a, b, c, d) \
    size(macro, |, a, b, c, d) \
    size(macro, &, a, b, c, d) \
    size(macro, ^, a, b, c, d)
#define Vc_LIST_SHIFTS(size, macro, a, b, c, d) \
    size(macro, <<, a, b, c, d) \
    size(macro, >>, a, b, c, d)
#define Vc_LIST_ARITHMETICS(size, macro, a, b, c, d) \
    size(macro, +, a, b, c, d) \
    size(macro, -, a, b, c, d) \
    size(macro, *, a, b, c, d) \
    size(macro, /, a, b, c, d) \
    size(macro, %, a, b, c, d)

#define Vc_APPLY_0(_list, macro)             _list(Vc_APPLY_IMPL_1_, macro, 0, 0, 0, 0) Vc_NOTHING_EXPECTING_SEMICOLON
#define Vc_APPLY_1(_list, macro, a)          _list(Vc_APPLY_IMPL_2_, macro, a, 0, 0, 0) Vc_NOTHING_EXPECTING_SEMICOLON
#define Vc_APPLY_2(_list, macro, a, b)       _list(Vc_APPLY_IMPL_3_, macro, a, b, 0, 0) Vc_NOTHING_EXPECTING_SEMICOLON
#define Vc_APPLY_3(_list, macro, a, b, c)    _list(Vc_APPLY_IMPL_4_, macro, a, b, c, 0) Vc_NOTHING_EXPECTING_SEMICOLON
#define Vc_APPLY_4(_list, macro, a, b, c, d) _list(Vc_APPLY_IMPL_5_, macro, a, b, c, d) Vc_NOTHING_EXPECTING_SEMICOLON

#define Vc_ALL_COMPARES(macro)     Vc_APPLY_0(Vc_LIST_COMPARES, macro)
#define Vc_ALL_LOGICAL(macro)      Vc_APPLY_0(Vc_LIST_LOGICAL, macro)
#define Vc_ALL_BINARY(macro)       Vc_APPLY_0(Vc_LIST_BINARY, macro)
#define Vc_ALL_SHIFTS(macro)       Vc_APPLY_0(Vc_LIST_SHIFTS, macro)
#define Vc_ALL_ARITHMETICS(macro)  Vc_APPLY_0(Vc_LIST_ARITHMETICS, macro)
#define Vc_ALL_FLOAT_VECTOR_TYPES(macro) Vc_APPLY_0(Vc_LIST_FLOAT_VECTOR_TYPES, macro)
#define Vc_ALL_VECTOR_TYPES(macro) Vc_APPLY_0(Vc_LIST_VECTOR_TYPES, macro)

#ifdef Vc_NO_ALWAYS_INLINE
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

#endif  // VC_SIMD_MACROS_H_
