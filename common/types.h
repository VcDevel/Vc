/*  This file is part of the Vc library. {{{
Copyright © 2012-2014 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_COMMON_TYPES_H
#define VC_COMMON_TYPES_H

#ifdef VC_CHECK_ALIGNMENT
#include <cstdlib>
#include <cstdio>
#endif

#include <Vc/global.h>
#include "../traits/type_traits.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
using std::size_t;

namespace Vc_IMPL_NAMESPACE
{
    template<typename T> class Vector;
    template<typename T> class Mask;
}  // namespace Vc_IMPL_NAMESPACE

namespace Detail
{
template<typename T> struct MayAlias { typedef T type Vc_MAY_ALIAS; };
//template<size_t Bytes> struct MayAlias<MaskBool<Bytes>> { typedef MaskBool<Bytes> type; };
/**\internal
 * Helper MayAlias<T> that turns T into the type to be used for an aliasing pointer. This
 * adds the may_alias attribute to T (with compilers that support it). But for MaskBool this
 * attribute is already part of the type and applying it a second times leads to warnings/errors,
 * therefore MaskBool is simply forwarded as is.
 */
}  // namespace Detail
template <typename T> using MayAlias = typename Detail::MayAlias<T>::type;

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

// TODO: all of the following doesn't really belong into the toplevel Vc namespace. An anonymous
// namespace might be enough:

template<typename T> struct DetermineEntryType { typedef T Type; };

// TODO: convert to enum classes
namespace VectorSpecialInitializerZero { enum ZEnum { Zero = 0 }; }
namespace VectorSpecialInitializerOne { enum OEnum { One = 1 }; }
namespace VectorSpecialInitializerIndexesFromZero { enum IEnum { IndexesFromZero }; }

#ifndef VC_ICC
// ICC ICEs if the traits below are in an unnamed namespace
namespace
{
#endif
    enum TestEnum {};

    static_assert(std::is_convertible<TestEnum, short>          ::value ==  true, "HasImplicitCast0_is_broken");
    static_assert(std::is_convertible<int *, void *>            ::value ==  true, "HasImplicitCast1_is_broken");
    static_assert(std::is_convertible<int *, const void *>      ::value ==  true, "HasImplicitCast2_is_broken");
    static_assert(std::is_convertible<const int *, const void *>::value ==  true, "HasImplicitCast3_is_broken");
    static_assert(std::is_convertible<const int *, int *>       ::value == false, "HasImplicitCast4_is_broken");

    template<typename From, typename To> struct is_implicit_cast_allowed : public std::false_type {};
    template<typename T> struct is_implicit_cast_allowed<T, T> : public std::true_type {};
    template<> struct is_implicit_cast_allowed< int32_t, uint32_t> : public std::true_type {};
    template<> struct is_implicit_cast_allowed<uint32_t,  int32_t> : public std::true_type {};
    template<> struct is_implicit_cast_allowed< int16_t, uint16_t> : public std::true_type {};
    template<> struct is_implicit_cast_allowed<uint16_t,  int16_t> : public std::true_type {};

    template<typename From, typename To> struct is_implicit_cast_allowed_mask : public is_implicit_cast_allowed<From, To> {};

    // TODO: issue #605 (make int_v <-> float_v conversion explicit only and drop the int_v::size()
    // == float_v::size() guarantee)
    template<> struct is_implicit_cast_allowed< int32_t,    float> : public std::true_type {};
    template<> struct is_implicit_cast_allowed<uint32_t,    float> : public std::true_type {};
    template<> struct is_implicit_cast_allowed_mask< float,  int32_t> : public std::true_type {};
    template<> struct is_implicit_cast_allowed_mask< float, uint32_t> : public std::true_type {};
#ifndef VC_ICC
} // anonymous namespace
#endif

#ifndef VC_CHECK_ALIGNMENT
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
template <std::size_t alignment>
Vc_INTRINSIC_L void *aligned_malloc(std::size_t n) Vc_INTRINSIC_R;
Vc_ALWAYS_INLINE_L void free(void *p) Vc_ALWAYS_INLINE_R;

template <typename T, typename U>
using enable_if_mask_converts_implicitly =
    enable_if<(Traits::is_simd_mask<U>::value && !Traits::is_simd_mask_array<U>::value &&
               is_implicit_cast_allowed_mask<
                   Traits::entry_type_of<typename Traits::decay<U>::Vector>, T>::value)>;
template <typename T, typename U>
using enable_if_mask_converts_explicitly = enable_if<
    (Traits::is_simd_mask_array<U>::value ||
     (Traits::is_simd_mask<U>::value &&
      !is_implicit_cast_allowed_mask<
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

template <typename I, I Begin, I End, typename F>
Vc_INTRINSIC enable_if<(Begin >= End), void> unrolled_loop(F &&)
{
}

template <typename I, I Begin, I End, typename F>
Vc_INTRINSIC Vc_FLATTEN enable_if<(Begin < End), void> unrolled_loop(F &&f)
{
    f(Begin);
    unrolled_loop<I, Begin + 1, End>(f);
}

}  // namespace Common
}  // namespace Vc

#include "memoryfwd.h"
#include "undomacros.h"

#endif // VC_COMMON_TYPES_H

// vim: foldmethod=marker
