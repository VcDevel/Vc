/*  This file is part of the Vc library. {{{
Copyright © 2012-2015 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_COMMON_TYPES_H_
#define VC_COMMON_TYPES_H_

#ifdef Vc_CHECK_ALIGNMENT
#include <cstdlib>
#include <cstdio>
#endif

#include <Vc/global.h>
#include "../traits/type_traits.h"
#include "permutation.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
using std::size_t;

using llong = long long;
using ullong = unsigned long long;
using ulong = unsigned long;
using uint = unsigned int;
using ushort = unsigned short;
using uchar = unsigned char;
using schar = signed char;

namespace VectorAbi
{
struct Scalar {};
struct Sse {};
struct Avx {};
struct Mic {};
template <typename T>
using Avx1Abi = typename std::conditional<std::is_integral<T>::value, VectorAbi::Sse,
                                          VectorAbi::Avx>::type;
template <typename T>
using Best = typename std::conditional<
    CurrentImplementation::is(ScalarImpl), Scalar,
    typename std::conditional<
        CurrentImplementation::is_between(SSE2Impl, SSE42Impl), Sse,
        typename std::conditional<
            CurrentImplementation::is(AVXImpl), Avx1Abi<T>,
            typename std::conditional<
                CurrentImplementation::is(AVX2Impl), Avx,
                typename std::conditional<CurrentImplementation::is(MICImpl), Mic,
                                          void>::type>::type>::type>::type>::type;
#ifdef Vc_IMPL_AVX2
static_assert(std::is_same<Best<float>, Avx>::value, "");
static_assert(std::is_same<Best<int>, Avx>::value, "");
#elif defined Vc_IMPL_AVX
static_assert(std::is_same<Best<float>, Avx>::value, "");
static_assert(std::is_same<Best<int>, Sse>::value, "");
#elif defined Vc_IMPL_SSE
static_assert(CurrentImplementation::is_between(SSE2Impl, SSE42Impl), "");
static_assert(std::is_same<Best<float>, Sse>::value, "");
static_assert(std::is_same<Best<int>, Sse>::value, "");
#elif defined Vc_IMPL_MIC
static_assert(std::is_same<Best<float>, Mic>::value, "");
static_assert(std::is_same<Best<int>, Mic>::value, "");
#elif defined Vc_IMPL_Scalar
static_assert(std::is_same<Best<float>, Scalar>::value, "");
static_assert(std::is_same<Best<int>, Scalar>::value, "");
#endif
}  // namespace VectorAbi

// TODO: declare the Vector and Mask interfaces here and add the API documentation.
// Specializations should only have internal documentation (if needed).
template<typename T, typename Abi = VectorAbi::Best<T>> class Vector;
template<typename T, typename Abi = VectorAbi::Best<T>> class Mask;

namespace Detail
{
template<typename T> struct MayAliasImpl { typedef T type Vc_MAY_ALIAS; };
//template<size_t Bytes> struct MayAlias<MaskBool<Bytes>> { typedef MaskBool<Bytes> type; };
/**\internal
 * Helper MayAlias<T> that turns T into the type to be used for an aliasing pointer. This
 * adds the may_alias attribute to T (with compilers that support it). But for MaskBool this
 * attribute is already part of the type and applying it a second times leads to warnings/errors,
 * therefore MaskBool is simply forwarded as is.
 */
}  // namespace Detail
#ifdef Vc_ICC
template <typename T> using MayAlias [[gnu::may_alias]] = T;
#else
template <typename T> using MayAlias = typename Detail::MayAliasImpl<T>::type;
#endif

enum class Operator : char {
    Assign,
    Multiply,
    MultiplyAssign,
    Divide,
    DivideAssign,
    Remainder,
    RemainderAssign,
    Plus,
    PlusAssign,
    Minus,
    MinusAssign,
    RightShift,
    RightShiftAssign,
    LeftShift,
    LeftShiftAssign,
    And,
    AndAssign,
    Xor,
    XorAssign,
    Or,
    OrAssign,
    PreIncrement,
    PostIncrement,
    PreDecrement,
    PostDecrement,
    LogicalAnd,
    LogicalOr,
    Comma,
    UnaryPlus,
    UnaryMinus,
    UnaryNot,
    UnaryOnesComplement,
    CompareEqual,
    CompareNotEqual,
    CompareLess,
    CompareGreater,
    CompareLessEqual,
    CompareGreaterEqual
};

template <typename T, std::size_t N> struct array;

/* TODO: add type for half-float, something along these lines:
class half_float
{
    uint16_t data;
public:
    constexpr half_float() : data(0) {}
    constexpr half_float(const half_float &) = default;
    constexpr half_float(half_float &&) = default;
    constexpr half_float &operator=(const half_float &) = default;

    constexpr explicit half_float(float);
    constexpr explicit half_float(double);
    constexpr explicit half_float(int);
    constexpr explicit half_float(unsigned int);

    explicit operator float       () const;
    explicit operator double      () const;
    explicit operator int         () const;
    explicit operator unsigned int() const;

    bool operator==(half_float rhs) const;
    bool operator!=(half_float rhs) const;
    bool operator>=(half_float rhs) const;
    bool operator<=(half_float rhs) const;
    bool operator> (half_float rhs) const;
    bool operator< (half_float rhs) const;

    half_float operator+(half_float rhs) const;
    half_float operator-(half_float rhs) const;
    half_float operator*(half_float rhs) const;
    half_float operator/(half_float rhs) const;
};
*/

/**\addtogroup Utilities
 * @{
 */
/**
 * The special object \p Vc::Zero can be used to construct Vector and Mask objects
 * initialized to zero/false.
 */
static constexpr struct VectorSpecialInitializerZero {} Zero = {};
/**
 * The special object \p Vc::One can be used to construct Vector and Mask objects
 * initialized to one/true.
 */
static constexpr struct VectorSpecialInitializerOne {} One = {};
/**
 * The special object \p Vc::IndexesFromZero can be used to construct Vector objects
 * initialized to values 0, 1, 2, 3, 4, ...
 */
static constexpr struct VectorSpecialInitializerIndexesFromZero {} IndexesFromZero = {};
///@}

// TODO: the following doesn't really belong into the toplevel Vc namespace.
#ifndef Vc_CHECK_ALIGNMENT
template<typename _T> static Vc_ALWAYS_INLINE void assertCorrectAlignment(const _T *){}
#else
template<typename _T> static Vc_ALWAYS_INLINE void assertCorrectAlignment(const _T *ptr)
{
    const size_t s = alignof(_T);
    if((reinterpret_cast<size_t>(ptr) & ((s ^ (s & (s - 1))) - 1)) != 0) {
        fprintf(stderr, "A vector with incorrect alignment has just been created. Look at the stacktrace to find the guilty object.\n");
        abort();
    }
}
#endif

namespace Common
{
/**
 * \internal
 *
 * Helper interface to make m_indexes in InterleavedMemoryAccessBase behave like an integer vector.
 * Only that the entries are successive entries from the given start index.
 */
template<size_t StructSize> class SuccessiveEntries
{
    std::size_t m_first;
public:
    typedef SuccessiveEntries AsArg;
    constexpr SuccessiveEntries(size_t first) : m_first(first) {}
    constexpr Vc_PURE size_t operator[](size_t offset) const { return m_first + offset * StructSize; }
    constexpr Vc_PURE size_t data() const { return m_first; }
    constexpr Vc_PURE SuccessiveEntries operator+(const SuccessiveEntries &rhs) const { return SuccessiveEntries(m_first + rhs.m_first); }
    constexpr Vc_PURE SuccessiveEntries operator*(const SuccessiveEntries &rhs) const { return SuccessiveEntries(m_first * rhs.m_first); }
    constexpr Vc_PURE SuccessiveEntries operator<<(std::size_t x) const { return {m_first << x}; }

    friend SuccessiveEntries &internal_data(SuccessiveEntries &x) { return x; }
    friend const SuccessiveEntries &internal_data(const SuccessiveEntries &x)
    {
        return x;
    }
};

template <std::size_t alignment>
Vc_INTRINSIC_L void *aligned_malloc(std::size_t n) Vc_INTRINSIC_R;
Vc_ALWAYS_INLINE_L void free(void *p) Vc_ALWAYS_INLINE_R;

template <typename T, typename U>
using enable_if_mask_converts_implicitly =
    enable_if<(Traits::is_simd_mask<U>::value && !Traits::isSimdMaskArray<U>::value &&
               Traits::is_implicit_cast_allowed_mask<
                   Traits::entry_type_of<typename Traits::decay<U>::Vector>, T>::value)>;
template <typename T, typename U>
using enable_if_mask_converts_explicitly = enable_if<(
    Traits::isSimdMaskArray<U>::value ||
    (Traits::is_simd_mask<U>::value &&
     !Traits::is_implicit_cast_allowed_mask<
         Traits::entry_type_of<typename Traits::decay<U>::Vector>, T>::value))>;

template <typename T> using WidthT = std::integral_constant<std::size_t, sizeof(T)>;

template<std::size_t Bytes> class MaskBool;

template <typename T, typename IndexVector, typename Scale, bool> class SubscriptOperation;

/**
 * \internal
 * Helper type to pass along the two arguments for a gather operation.
 *
 * \tparam IndexVector  Normally an integer SIMD vector, but an array or std::vector also works
 *                      (though often not as efficient).
 */
template <typename T, typename IndexVector> struct GatherArguments
{
    const IndexVector indexes;
    const T *const address;
};

/**
 * \internal
 * Helper type to pass along the two arguments for a scatter operation.
 *
 * \tparam IndexVector  Normally an integer SIMD vector, but an array or std::vector also works
 *                      (though often not as efficient).
 */
template <typename T, typename IndexVector> struct ScatterArguments
{
    const IndexVector indexes;
    T *const address;
};

/**\internal
 * Break the recursion of the function below.
 */
template <typename I, I Begin, I End, typename F>
Vc_INTRINSIC enable_if<(Begin >= End), void> unrolled_loop(F &&)
{
}

/**\internal
 * Force the code in the lambda \p f to be called with indexes starting from \p Begin up
 * to (excluding) \p End to be called without compare and jump instructions (i.e. an
 * unrolled loop).
 */
template <typename I, I Begin, I End, typename F>
Vc_INTRINSIC Vc_FLATTEN enable_if<(Begin < End), void> unrolled_loop(F &&f)
{
    f(Begin);
    unrolled_loop<I, Begin + 1, End>(f);
}

/**\internal
 * Small simplification of the unrolled_loop call for ranges from 0 to \p Size using
 * std::size_t as the index type.
 */
template <std::size_t Size, typename F> Vc_INTRINSIC void for_all_vector_entries(F &&f)
{
    unrolled_loop<std::size_t, 0u, Size>(std::forward<F>(f));
}

}  // namespace Common
}  // namespace Vc

#include "memoryfwd.h"

#endif // VC_COMMON_TYPES_H_

// vim: foldmethod=marker
