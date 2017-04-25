/*  This file is part of the Vc library. {{{
Copyright © 2016-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DETAIL_SCALAR_H_
#define VC_DETAIL_SCALAR_H_

#include "datapar.h"
#include "detail.h"
#include <bitset>

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
template <class T> using scalar_mask = mask<T, datapar_abi::scalar>;
template <class T> using scalar_datapar = datapar<T, datapar_abi::scalar>;

// datapar impl {{{1
struct scalar_datapar_impl {
    // member types {{{2
    using mask_member_type = bool;
    template <class T> using datapar_member_type = T;
    template <class T> using datapar = Vc::datapar<T, datapar_abi::scalar>;
    template <class T> using mask = Vc::mask<T, datapar_abi::scalar>;
    using size_tag = std::integral_constant<size_t, 1>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    template <class T> static Vc_INTRINSIC T broadcast(T x, size_tag) noexcept
    {
        return x;
    }

    // generator {{{2
    template <class F, class T>
    static Vc_INTRINSIC T generator(F &&gen, type_tag<T>, size_tag)
    {
        return gen(std::integral_constant<size_t, 0>());
    }

    // load {{{2
    template <class T, class U, class F>
    static inline T load(const U *mem, F, type_tag<T>) noexcept
    {
        return static_cast<T>(mem[0]);
    }

    // masked load {{{2
    template <class T, class A, class U, class F>
    static inline void masked_load(datapar<T> &merge, const Vc::mask<T, A> &k,
                                   const U *mem, F) noexcept
    {
        if (k.d) {
            merge.d = static_cast<T>(mem[0]);
        }
    }

    // store {{{2
    template <class T, class U, class F>
    static inline void store(T v, U *mem, F, type_tag<T>) noexcept
    {
        mem[0] = static_cast<T>(v);
    }

    // masked store {{{2
    template <class T, class A, class U, class F>
    static inline void masked_store(const datapar<T> &v, U *mem, F,
                                    const Vc::mask<T, A> &k) noexcept
    {
        if (k[0]) {
            mem[0] = v.d;
        }
    }

    // negation {{{2
    template <class T> static inline mask<T> negate(const datapar<T> &x) noexcept
    {
        return {private_init, !x.d};
    }

    // reductions {{{2
    template <class T, class BinaryOperation>
    static inline T reduce(size_tag, const datapar<T> &x, BinaryOperation &)
    {
        return x.d;
    }

    // min, max, clamp {{{2
    template <class T>
    static inline datapar<T> min(const datapar<T> &a, const datapar<T> &b)
    {
        return {private_init, std::min(a.d, b.d)};
    }

    template <class T>
    static inline datapar<T> max(const datapar<T> &a, const datapar<T> &b)
    {
        return {private_init, std::max(a.d, b.d)};
    }

    // complement {{{2
    template <class T> static inline datapar<T> complement(const datapar<T> &x) noexcept
    {
        return {private_init, static_cast<T>(~x.d)};
    }

    // unary minus {{{2
    template <class T> static inline datapar<T> unary_minus(const datapar<T> &x) noexcept
    {
        return {private_init, static_cast<T>(-x.d)};
    }

    // arithmetic operators {{{2
    template <class T>
    static inline datapar<T> plus(const datapar<T> &x, const datapar<T> &y)
    {
        return {private_init, static_cast<T>(x.d + y.d)};
    }

    template <class T>
    static inline datapar<T> minus(const datapar<T> &x, const datapar<T> &y)
    {
        return {private_init, static_cast<T>(x.d - y.d)};
    }

    template <class T>
    static inline datapar<T> multiplies(const datapar<T> &x, const datapar<T> &y)
    {
        return {private_init, static_cast<T>(x.d * y.d)};
    }

    template <class T>
    static inline datapar<T> divides(const datapar<T> &x, const datapar<T> &y)
    {
        return {private_init, static_cast<T>(x.d / y.d)};
    }

    template <class T>
    static inline datapar<T> modulus(const datapar<T> &x, const datapar<T> &y)
    {
        return {private_init, static_cast<T>(x.d % y.d)};
    }

    template <class T>
    static inline datapar<T> bit_and(const datapar<T> &x, const datapar<T> &y)
    {
        return {private_init, static_cast<T>(x.d & y.d)};
    }

    template <class T>
    static inline datapar<T> bit_or(const datapar<T> &x, const datapar<T> &y)
    {
        return {private_init, static_cast<T>(x.d | y.d)};
    }

    template <class T>
    static inline datapar<T> bit_xor(const datapar<T> &x, const datapar<T> &y)
    {
        return {private_init, static_cast<T>(x.d ^ y.d)};
    }

    template <class T>
    static inline datapar<T> bit_shift_left(const datapar<T> &x, const datapar<T> &y)
    {
        return {private_init, static_cast<T>(x.d << y.d)};
    }

    template <class T>
    static inline datapar<T> bit_shift_right(const datapar<T> &x, const datapar<T> &y)
    {
        return {private_init, static_cast<T>(x.d >> y.d)};
    }

    // increment & decrement{{{2
    template <class T> static inline void increment(T &x) { ++x; }
    template <class T> static inline void decrement(T &x) { --x; }

    // compares {{{2
#define Vc_CMP_OPERATIONS(cmp_)                                                          \
    template <class V> static inline typename V::mask_type cmp_(const V &x, const V &y)  \
    {                                                                                    \
        return {private_init, std::cmp_<typename V::value_type>()(x.d, y.d)};            \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
    Vc_CMP_OPERATIONS(equal_to);
    Vc_CMP_OPERATIONS(not_equal_to);
    Vc_CMP_OPERATIONS(less);
    Vc_CMP_OPERATIONS(greater);
    Vc_CMP_OPERATIONS(less_equal);
    Vc_CMP_OPERATIONS(greater_equal);
#undef Vc_CMP_OPERATIONS

    // smart_reference access {{{2
    template <class T> static T get(const datapar<T> &v, int i) noexcept
    {
        Vc_ASSERT(i == 0);
        unused(i);
        return v.d;
    }
    template <class T, class U> static void set(datapar<T> &v, int i, U &&x) noexcept
    {
        Vc_ASSERT(i == 0);
        unused(i);
        v.d = std::forward<U>(x);
    }
    // }}}2
};

// mask impl {{{1
struct scalar_mask_impl {
    // member types {{{2
    template <class T> using mask = Vc::mask<T, datapar_abi::scalar>;
    using size_tag = std::integral_constant<size_t, 1>;
    template <class T> using type_tag = T *;

    // to_bitset {{{2
    static Vc_INTRINSIC std::bitset<1> to_bitset(bool x) noexcept { return unsigned(x); }

    // from_bitset {{{2
    template <class T>
    static Vc_INTRINSIC bool from_bitset(std::bitset<1> bs, type_tag<T>) noexcept
    {
        return bs[0];
    }

    // broadcast {{{2
    template <class T> static Vc_INTRINSIC bool broadcast(bool x, type_tag<T>) noexcept
    {
        return x;
    }

    // load {{{2
    template <class F>
    static Vc_INTRINSIC bool load(const bool *mem, F, size_tag) noexcept
    {
        return mem[0];
    }

    // masked load {{{2
    template <class F>
    static inline void masked_load(bool &merge, bool mask,
                                   const bool *mem, F, size_tag) noexcept
    {
        if (mask) {
            merge = mem[0];
        }
    }

    // store {{{2
    template <class F> static inline void store(bool v, bool *mem, F, size_tag) noexcept
    {
        mem[0] = v;
    }

    // masked store {{{2
    template <class F>
    static inline void masked_store(bool v, bool *mem, F, bool k, size_tag) noexcept
    {
        if (k) {
            mem[0] = v;
        }
    }

    // negation {{{2
    static inline bool negate(bool x, size_tag) noexcept { return !x; }

    // logical and bitwise operators {{{2
    template <class T>
    static inline mask<T> logical_and(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, x.d && y.d};
    }

    template <class T>
    static inline mask<T> logical_or(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, x.d || y.d};
    }

    template <class T> static inline mask<T> bit_and(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, x.d && y.d};
    }

    template <class T> static inline mask<T> bit_or(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, x.d || y.d};
    }

    template <class T> static inline mask<T> bit_xor(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, x.d != y.d};
    }

    // smart_reference access {{{2
    template <class T> static bool get(const mask<T> &k, int i) noexcept
    {
        Vc_ASSERT(i == 0);
        detail::unused(i);
        return k.d;
    }
    template <class T> static void set(mask<T> &k, int i, bool x) noexcept
    {
        Vc_ASSERT(i == 0);
        detail::unused(i);
        k.d = x;
    }
    // }}}2
};

// traits {{{1
template <class T> struct scalar_traits {
    static constexpr size_t size() noexcept { return 1; }

    using datapar_impl_type = scalar_datapar_impl;
    using datapar_member_type = T;
    static constexpr size_t datapar_member_alignment = alignof(T);
    using datapar_cast_type = const std::array<T, 1>;
    struct datapar_base {};

    using mask_impl_type = scalar_mask_impl;
    using mask_member_type = bool;
    static constexpr size_t mask_member_alignment = alignof(mask_member_type);
    using mask_cast_type = const std::bitset<1>;
    struct mask_base {};
};
template <> struct traits<long double, datapar_abi::scalar> : public scalar_traits<long double> {};
template <> struct traits<double, datapar_abi::scalar> : public scalar_traits<double> {};
template <> struct traits< float, datapar_abi::scalar> : public scalar_traits< float> {};
template <> struct traits<ullong, datapar_abi::scalar> : public scalar_traits<ullong> {};
template <> struct traits< llong, datapar_abi::scalar> : public scalar_traits< llong> {};
template <> struct traits< ulong, datapar_abi::scalar> : public scalar_traits< ulong> {};
template <> struct traits<  long, datapar_abi::scalar> : public scalar_traits<  long> {};
template <> struct traits<  uint, datapar_abi::scalar> : public scalar_traits<  uint> {};
template <> struct traits<   int, datapar_abi::scalar> : public scalar_traits<   int> {};
template <> struct traits<ushort, datapar_abi::scalar> : public scalar_traits<ushort> {};
template <> struct traits< short, datapar_abi::scalar> : public scalar_traits< short> {};
template <> struct traits< uchar, datapar_abi::scalar> : public scalar_traits< uchar> {};
template <> struct traits< schar, datapar_abi::scalar> : public scalar_traits< schar> {};
template <> struct traits<  char, datapar_abi::scalar> : public scalar_traits<  char> {};

// }}}1
}  // namespace detail

// where implementation {{{1
template <typename T>
static Vc_INTRINSIC void masked_assign(
    const mask<T, datapar_abi::scalar> &k, datapar<T, datapar_abi::scalar> &lhs,
    const detail::id<datapar<T, datapar_abi::scalar>> &rhs)
{
    if (k.d) {
        lhs.d = rhs.d;
    }
}

template <typename T>
static Vc_INTRINSIC void masked_assign(
    const mask<T, datapar_abi::scalar> &k, mask<T, datapar_abi::scalar> &lhs,
    const detail::id<mask<T, datapar_abi::scalar>> &rhs)
{
    if (k.d) {
        lhs.d = rhs.d;
    }
}

// Optimization for the case where the RHS is a scalar. No need to broadcast the scalar to a datapar
// first.
template <class T, class U>
static Vc_INTRINSIC
    enable_if<std::is_convertible<U, datapar<T, datapar_abi::scalar>>::value &&
                  std::is_arithmetic<U>::value,
              void>
    masked_assign(const mask<T, datapar_abi::scalar> &k,
                  datapar<T, datapar_abi::scalar> &lhs, const U &rhs)
{
    if (k.d) {
        lhs.d = rhs;
    }
}

template <template <typename> class Op, typename T>
inline void masked_cassign(const detail::scalar_mask<T> &k,
                           detail::scalar_datapar<T> &lhs,
                           const detail::scalar_datapar<T> &rhs)
{
    if (k.d) {
        lhs.d = Op<T>{}(lhs.d, rhs.d);
    }
}

// Optimization for the case where the RHS is a scalar. No need to broadcast the scalar to a datapar
// first.
template <template <typename> class Op, typename T, class U>
inline enable_if<std::is_convertible<U, detail::scalar_datapar<T>>::value &&
                     std::is_arithmetic<U>::value,
                 void>
masked_cassign(const detail::scalar_mask<T> &k, detail::scalar_datapar<T> &lhs,
               const U &rhs)
{
    if (k.d) {
        lhs.d = Op<T>{}(lhs.d, rhs);
    }
}

template <template <typename> class Op, typename T>
inline detail::scalar_datapar<T> masked_unary(const detail::scalar_mask<T> &k,
                                              const detail::scalar_datapar<T> &v)
{
    return static_cast<T>(k.d ? Op<T>{}(v.d) : v.d);
}

// [mask.reductions] {{{1
template <class T> inline bool all_of(const detail::scalar_mask<T> &k) { return k[0]; }
template <class T> inline bool any_of(const detail::scalar_mask<T> &k) { return k[0]; }
template <class T> inline bool none_of(const detail::scalar_mask<T> &k) { return !k[0]; }
template <class T> inline bool some_of(const detail::scalar_mask<T> &) { return false; }
template <class T> inline int popcount(const detail::scalar_mask<T> &k) { return k[0]; }
template <class T> inline int find_first_set(const detail::scalar_mask<T> &) { return 0; }
template <class T> inline int find_last_set(const detail::scalar_mask<T> &) { return 0; }
// }}}1
Vc_VERSIONED_NAMESPACE_END

namespace std
{
// mask operators {{{1
template <class T> struct equal_to<Vc::mask<T, Vc::datapar_abi::scalar>> {
private:
    using M = Vc::mask<T, Vc::datapar_abi::scalar>;

public:
    bool operator()(const M &x, const M &y) const { return x[0] == y[0]; }
};
// }}}1
}  // namespace std

#endif  // VC_DETAIL_SCALAR_H_
// vim: foldmethod=marker
