#ifndef _GLIBCXX_EXPERIMENTAL_SIMD_DETAIL_H_
#define _GLIBCXX_EXPERIMENTAL_SIMD_DETAIL_H_

#pragma GCC system_header

#if __cplusplus >= 201703L

#include <cstddef>
#include <cstdint>

// workaround macros {{{
// vector conversions not optimized:
#define _GLIBCXX_SIMD_WORKAROUND_PR85048 1

// zero extension from xmm to zmm not optimized:
//#define _GLIBCXX_SIMD_WORKAROUND_PR85480 1

// incorrect use of k0 register for _kortestc_mask64_u8 and _kortestc_mask32_u8:
#define _GLIBCXX_SIMD_WORKAROUND_PR85538 1

// missed optimization for abs(__vector_type_t<__llong, 2>):
#define _GLIBCXX_SIMD_WORKAROUND_PR85572 1

// very bad codegen for extraction and concatenation of 128/256 "subregisters" with
// sizeof(element type) < 8: https://godbolt.org/g/mqUsgM
#define _GLIBCXX_SIMD_WORKAROUND_XXX_1 1

// bad codegen for 8 Byte memcpy to __vector_type_t<char, 16>
#define _GLIBCXX_SIMD_WORKAROUND_XXX_2 1

// bad codegen for zero-extend using simple concat(x, 0)
#define _GLIBCXX_SIMD_WORKAROUND_XXX_3 1

// bad codegen for integer division
#define _GLIBCXX_SIMD_WORKAROUND_XXX_4 1

// https://github.com/cplusplus/parallelism-ts/issues/65 (incorrect return type of
// static_simd_cast)
#define _GLIBCXX_SIMD_FIX_P2TS_ISSUE65 1

// https://github.com/cplusplus/parallelism-ts/issues/66 (incorrect SFINAE constraint on
// (static)_simd_cast)
#define _GLIBCXX_SIMD_FIX_P2TS_ISSUE66 1
// }}}

#define _GLIBCXX_SIMD_BEGIN_NAMESPACE                                                    \
    namespace std _GLIBCXX_VISIBILITY(default)                                           \
    {                                                                                    \
    _GLIBCXX_BEGIN_NAMESPACE_VERSION namespace experimental                              \
    {                                                                                    \
    inline namespace parallelism_v2                                                      \
    {
#define _GLIBCXX_SIMD_END_NAMESPACE                                                      \
    }                                                                                    \
    }                                                                                    \
    _GLIBCXX_END_NAMESPACE_VERSION                                                       \
    }

_GLIBCXX_SIMD_BEGIN_NAMESPACE
namespace simd_abi  // {{{
{
// implementation details:
struct __scalar_abi;
template <int N> struct __fixed_abi;

template <int Bytes = 16> struct __sse_abi;
template <int Bytes = 32> struct __avx_abi;
template <int Bytes = 64> struct __avx512_abi;
template <int Bytes = 16> struct __neon_abi;

template <int N, class Abi> struct __combine;

// implementation-defined:
template <int NRegisters> using __sse_x = __combine<NRegisters, __sse_abi<>>;
template <int NRegisters> using __avx_x = __combine<NRegisters, __avx_abi<>>;
template <int NRegisters> using __avx512_x = __combine<NRegisters, __avx512_abi<>>;
template <int NRegisters> using __neon_x = __combine<NRegisters, __neon_abi<>>;

template <class T, int N> using __sse_n = __sse_abi<sizeof(T) * N>;
template <class T, int N> using __avx_n = __avx_abi<sizeof(T) * N>;
template <class T, int N> using __avx512_n = __avx512_abi<sizeof(T) * N>;
template <class T, int N> using __neon_n = __neon_abi<sizeof(T) * N>;

using __sse = __sse_abi<>;
using __avx = __avx_abi<>;
using __avx512 = __avx512_abi<>;
using __neon = __neon_abi<>;

using __neon128 = __neon_abi<16>;
using __neon64 = __neon_abi<8>;

// standard:
template <class T, size_t N, class... > struct deduce;
template <int N> using fixed_size = __fixed_abi<N>;
using scalar = __scalar_abi;
}  // namespace simd_abi }}}
// forward declarations is_simd(_mask), simd(_mask), simd_size {{{
template <class T> struct is_simd;
template <class T> struct is_simd_mask;
template <class T, class Abi> class simd;
template <class T, class Abi> class simd_mask;
template <class T, class Abi> struct simd_size;
// }}}
_GLIBCXX_SIMD_END_NAMESPACE

// not-yet-optimized warning hack {{{
#ifndef _GLIBCXX_SIMD_NO_OPTIMIZATION_WARNINGS
#if defined _GLIBCXX_SIMD_GCC || defined _GLIBCXX_SIMD_CLANG
// hack GCC's formatting to overwrite "deprecated: " with our own message:
#define _GLIBCXX_SIMD_NOT_OPTIMIZED [[deprecated("\x8\x8\x8\x8\x8\x8\x8\x8\x8\x8\x8\x8not optimized yet, if you care about speed please help out! [-D_GLIBCXX_SIMD_NO_OPTIMIZATION_WARNINGS]")]]
#else
#define _GLIBCXX_SIMD_NOT_OPTIMIZED [[deprecated("NOT DEPRECATED, just a note: this function is not optimized yet, if you care about speed please help out! [-D_GLIBCXX_SIMD_NO_OPTIMIZATION_WARNINGS]")]]
#endif
#else
#define _GLIBCXX_SIMD_NOT_OPTIMIZED
#endif
// }}}

// Starting with compiler identification. This is a prerequisite for getting the following
// macro definitions right.
// {{{
#if defined(__clang__)
#  define _GLIBCXX_SIMD_CLANG (__clang_major__ * 0x10000 + __clang_minor__ * 0x100 + __clang_patchlevel__)
#elif defined(__GNUC__)
#  define _GLIBCXX_SIMD_GCC (__GNUC__ * 0x10000 + __GNUC_MINOR__ * 0x100 + __GNUC_PATCHLEVEL__)
#endif
//}}}

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

// ISA extension detection. The following defines all the _GLIBCXX_SIMD_HAVE_XXX macros
// ARM{{{
#ifdef __aarch64__
#define _GLIBCXX_SIMD_IS_AARCH64 1
#endif  // __aarch64__

#ifdef __ARM_NEON
#define _GLIBCXX_SIMD_HAVE_NEON
#define _GLIBCXX_SIMD_HAVE_NEON_ABI 1
#define _GLIBCXX_SIMD_HAVE_FULL_NEON_ABI 1
#endif  // _GLIBCXX_SIMD_HAVE_NEON
//}}}
// x86{{{
#ifdef __MMX__
#define _GLIBCXX_SIMD_HAVE_MMX 1
#else
#define _GLIBCXX_SIMD_HAVE_MMX 0
#endif
#if defined __SSE__ || defined __x86_64__
#define _GLIBCXX_SIMD_HAVE_SSE 1
#endif
#if defined __SSE2__ || defined __x86_64__
#define _GLIBCXX_SIMD_HAVE_SSE2 1
#endif
#ifdef __SSE3__
#define _GLIBCXX_SIMD_HAVE_SSE3 1
#endif
#ifdef __SSSE3__
#define _GLIBCXX_SIMD_HAVE_SSSE3 1
#endif
#ifdef __SSE4_1__
#define _GLIBCXX_SIMD_HAVE_SSE4_1 1
#endif
#ifdef __SSE4_2__
#define _GLIBCXX_SIMD_HAVE_SSE4_2 1
#endif
#ifdef __XOP__
#define _GLIBCXX_SIMD_HAVE_XOP 1
#endif
#ifdef __AVX__
#define _GLIBCXX_SIMD_HAVE_AVX 1
#endif
#ifdef __AVX2__
#define _GLIBCXX_SIMD_HAVE_AVX2 1
#endif
#ifdef __BMI__
#define _GLIBCXX_SIMD_HAVE_BMI1 1
#endif
#ifdef __BMI2__
#define _GLIBCXX_SIMD_HAVE_BMI2 1
#endif
#ifdef __LZCNT__
#define _GLIBCXX_SIMD_HAVE_LZCNT 1
#endif
#ifdef __SSE4A__
#define _GLIBCXX_SIMD_HAVE_SSE4A 1
#endif
#ifdef __FMA__
#define _GLIBCXX_SIMD_HAVE_FMA 1
#endif
#ifdef __FMA4__
#define _GLIBCXX_SIMD_HAVE_FMA4 1
#endif
#ifdef __F16C__
#define _GLIBCXX_SIMD_HAVE_F16C 1
#endif
#ifdef __POPCNT__
#define _GLIBCXX_SIMD_HAVE_POPCNT 1
#endif
#ifdef __AVX512F__
#define _GLIBCXX_SIMD_HAVE_AVX512F 1
#endif
#ifdef __AVX512DQ__
#define _GLIBCXX_SIMD_HAVE_AVX512DQ 1
#endif
#ifdef __AVX512VL__
#define _GLIBCXX_SIMD_HAVE_AVX512VL 1
#endif
#ifdef __AVX512BW__
#define _GLIBCXX_SIMD_HAVE_AVX512BW 1
#endif

#if defined _GLIBCXX_SIMD_HAVE_SSE
#define _GLIBCXX_SIMD_HAVE_SSE_ABI 1
#ifdef _GLIBCXX_SIMD_HAVE_SSE2
#define _GLIBCXX_SIMD_HAVE_FULL_SSE_ABI 1
#endif
#endif

#if defined _GLIBCXX_SIMD_HAVE_AVX
#define _GLIBCXX_SIMD_HAVE_AVX_ABI 1
#if defined _GLIBCXX_SIMD_HAVE_AVX2
#define _GLIBCXX_SIMD_HAVE_FULL_AVX_ABI 1
#endif
#endif

#ifdef _GLIBCXX_SIMD_HAVE_AVX512F
#define _GLIBCXX_SIMD_HAVE_AVX512_ABI 1
#ifdef _GLIBCXX_SIMD_HAVE_AVX512BW
#define _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI 1
#endif
#endif

#if defined __x86_64__ && !defined _GLIBCXX_SIMD_HAVE_SSE2
#error "Use of SSE2 is required on AMD64"
#endif
//}}}

// _GLIBCXX_SIMD_USE_BUILTIN_VECTOR_TYPES
// TODO: rename to _GLIBCXX_SIMD_HAVE_BUILTIN_VECTOR_TYPES
// {{{
#define _GLIBCXX_SIMD_USE_BUILTIN_VECTOR_TYPES 1
// }}}

// _GLIBCXX_SIMD_CONCAT{{{
#define _GLIBCXX_SIMD_CONCAT_IMPL(a_, b_, c_) a_##b_##c_
#define _GLIBCXX_SIMD_CONCAT(a_, b_, c_) _GLIBCXX_SIMD_CONCAT_IMPL(a_, b_, c_)
// }}}

#define _GLIBCXX_SIMD_NORMAL_MATH [[gnu::__optimize__("finite-math-only,no-signed-zeros")]]
#define _GLIBCXX_SIMD_UNREACHABLE __builtin_unreachable
#define _GLIBCXX_SIMD_NEVER_INLINE [[gnu::__noinline__]]
#if defined _GLIBCXX_SIMD_CLANG
#  define _GLIBCXX_SIMD_INTRINSIC [[gnu::__always_inline__]] inline
#else
#  define _GLIBCXX_SIMD_INTRINSIC [[gnu::__always_inline__, gnu::__artificial__]] inline
#endif
#define _GLIBCXX_SIMD_CONST __attribute__((__const__))
#define _GLIBCXX_SIMD_PURE __attribute__((__pure__))
#define _GLIBCXX_SIMD_ALWAYS_INLINE [[gnu::__always_inline__]] inline
#define _GLIBCXX_SIMD_IS_UNLIKELY(x) __builtin_expect(x, 0)
#define _GLIBCXX_SIMD_IS_LIKELY(x) __builtin_expect(x, 1)

#ifdef _GLIBCXX_SIMD_CXX17
#  define _GLIBCXX_SIMD_NODISCARD [[nodiscard]]
#elif defined __GNUC__
#  define _GLIBCXX_SIMD_NODISCARD [[gnu::__warn_unused_result__]]
#else
#  define _GLIBCXX_SIMD_NODISCARD
#endif

#define _GLIBCXX_SIMD_NOTHING_EXPECTING_SEMICOLON static_assert(true, "")

#ifdef _GLIBCXX_SIMD_ASSERT
#define _GLIBCXX_SIMD_EXTERNAL_ASSERT 1
#else
#ifdef NDEBUG
#define _GLIBCXX_SIMD_ASSERT(x) std::experimental::__dummy_assert{} << ' '
#else
#define _GLIBCXX_SIMD_ASSERT(x) std::experimental::__real_assert(x, #x, __FILE__, __LINE__)
#endif
#endif

#ifdef COMPILE_FOR_UNIT_TESTS
#define _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
#else
#define _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST noexcept
#endif

#define _GLIBCXX_SIMD_LIST_BINARY(macro) macro(|) macro(&) macro(^)
#define _GLIBCXX_SIMD_LIST_SHIFTS(macro) macro(<<) macro(>>)
#define _GLIBCXX_SIMD_LIST_ARITHMETICS(macro) macro(+) macro(-) macro(*) macro(/) macro(%)

#define _GLIBCXX_SIMD_ALL_BINARY(macro) _GLIBCXX_SIMD_LIST_BINARY(macro) _GLIBCXX_SIMD_NOTHING_EXPECTING_SEMICOLON
#define _GLIBCXX_SIMD_ALL_SHIFTS(macro) _GLIBCXX_SIMD_LIST_SHIFTS(macro) _GLIBCXX_SIMD_NOTHING_EXPECTING_SEMICOLON
#define _GLIBCXX_SIMD_ALL_ARITHMETICS(macro) _GLIBCXX_SIMD_LIST_ARITHMETICS(macro) _GLIBCXX_SIMD_NOTHING_EXPECTING_SEMICOLON

#ifdef _GLIBCXX_SIMD_NO_ALWAYS_INLINE
#undef _GLIBCXX_SIMD_ALWAYS_INLINE
#define _GLIBCXX_SIMD_ALWAYS_INLINE inline
#undef _GLIBCXX_SIMD_INTRINSIC
#define _GLIBCXX_SIMD_INTRINSIC inline
#endif

#endif  // __cplusplus >= 201703L
#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_DETAIL_H_
// vim: foldmethod=marker
