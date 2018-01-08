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

#ifndef VC_SIMD_FIXED_SIZE_H_
#define VC_SIMD_FIXED_SIZE_H_

#include "simd.h"
#include "detail.h"
#include "simd_tuple.h"
#include "debug.h"
#include <array>

/**
 * The fixed_size ABI gives the following guarantees:
 *  - simd objects are passed via the stack
 *  - memory layout of `simd<T, N>` is equivalent to `std::array<T, N>`
 *  - alignment of `simd<T, N>` is `N * sizeof(T)` if N is a power-of-2 value,
 *    otherwise `next_power_of_2(N * sizeof(T))` (Note: if the alignment were to
 *    exceed the system/compiler maximum, it is bounded to that maximum)
 *  - simd_mask objects are passed like std::bitset<N>
 *  - memory layout of `simd_mask<T, N>` is equivalent to `std::bitset<N>`
 *  - alignment of `simd_mask<T, N>` is equal to the alignment of `std::bitset<N>`
 */

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
#define Vc_FIXED_SIZE_FWD_(name_)                                                        \
    struct name_##_fwd {                                                                 \
        template <class Impl, class Arg0, class... Args>                                 \
        Vc_INTRINSIC_L auto operator()(Impl impl, Arg0 &&arg0,                           \
                                       Args &&... args) noexcept Vc_INTRINSIC_R;         \
    }
Vc_FIXED_SIZE_FWD_(frexp);
#undef Vc_FIXED_SIZE_FWD_

// select_best_vector_type_t<T, N>{{{1
/**
 * \internal
 * Selects the best SIMD type out of a typelist to store N scalar values.
 */
struct dummy : public size_constant<~size_t()> {
};

template <class T, int N, class A, class... More>
struct select_best_vector_type {
    using V = std::conditional_t<std::is_destructible<simd<T, A>>::value,
                                 simd_size<T, A>, dummy>;
    using type =
        std::conditional_t<(N >= V::value), simd<T, A>,
                           typename select_best_vector_type<T, N, More...>::type>;
};
template <class T, int N, class A> struct select_best_vector_type<T, N, A> {
    using type = simd<T, A>;
};
template <class T, int N>
using select_best_vector_type_t = typename select_best_vector_type<T, N,
      simd_abi::avx512,
      simd_abi::avx,
      simd_abi::neon,
      simd_abi::sse,
      simd_abi::scalar
      >::type;

// fixed_size_storage<T, N>{{{1
template <class T, int N, class Tuple, class Next = select_best_vector_type_t<T, N>,
          int Remain = N - int(Next::size())>
struct fixed_size_storage_builder;

template <class T, int N>
struct fixed_size_storage_builder_wrapper
    : public fixed_size_storage_builder<T, N, simd_tuple<T>> {
};

template <class T, int N, class... As, class Next>
struct fixed_size_storage_builder<T, N, simd_tuple<T, As...>, Next, 0> {
    using type = simd_tuple<T, As..., typename Next::abi_type>;
};

template <class T, int N, class... As, class Next, int Remain>
struct fixed_size_storage_builder<T, N, simd_tuple<T, As...>, Next, Remain> {
    using type = typename fixed_size_storage_builder<
        T, Remain, simd_tuple<T, As..., typename Next::abi_type>>::type;
};

namespace tests {
using simd_abi::scalar;
using simd_abi::sse;
using simd_abi::avx;
using simd_abi::avx512;
static_assert(
    std::is_same<fixed_size_storage<float, 1>, simd_tuple<float, scalar>>::value,
    "fixed_size_storage failure");
static_assert(std::is_same<fixed_size_storage<float, 2>,
                           simd_tuple<float, scalar, scalar>>::value,
              "fixed_size_storage failure");
static_assert(std::is_same<fixed_size_storage<float, 3>,
                           simd_tuple<float, scalar, scalar, scalar>>::value,
              "fixed_size_storage failure");
static_assert(
    std::is_same<fixed_size_storage<float, 4>, simd_tuple<float, sse>>::value,
    "fixed_size_storage failure");
static_assert(
    std::is_same<fixed_size_storage<float, 5>, simd_tuple<float, sse, scalar>>::value,
    "fixed_size_storage failure");
#ifdef Vc_HAVE_AVX_ABI
static_assert(
    std::is_same<fixed_size_storage<float, 8>, simd_tuple<float, avx>>::value,
    "fixed_size_storage failure");
static_assert(
    std::is_same<fixed_size_storage<float, 12>, simd_tuple<float, avx, sse>>::value,
    "fixed_size_storage failure");
static_assert(std::is_same<fixed_size_storage<float, 13>,
                           simd_tuple<float, avx, sse, scalar>>::value,
              "fixed_size_storage failure");
#endif
}  // namespace tests

// n_abis_in_tuple {{{1
template <class T> struct seq_op;
template <size_t I0, size_t... Is> struct seq_op<std::index_sequence<I0, Is...>> {
    using first_plus_one = std::index_sequence<I0 + 1, Is...>;
    using notfirst_plus_one = std::index_sequence<I0, (Is + 1)...>;
    template <size_t First, size_t Add>
    using prepend = std::index_sequence<First, I0 + Add, (Is + Add)...>;
};

template <class T> struct n_abis_in_tuple;
template <class T> struct n_abis_in_tuple<simd_tuple<T>> {
    using counts = std::index_sequence<0>;
    using begins = std::index_sequence<0>;
};
template <class T, class A> struct n_abis_in_tuple<simd_tuple<T, A>> {
    using counts = std::index_sequence<1>;
    using begins = std::index_sequence<0>;
};
template <class T, class A0, class... As>
struct n_abis_in_tuple<simd_tuple<T, A0, A0, As...>> {
    using counts = typename seq_op<
        typename n_abis_in_tuple<simd_tuple<T, A0, As...>>::counts>::first_plus_one;
    using begins = typename seq_op<typename n_abis_in_tuple<
        simd_tuple<T, A0, As...>>::begins>::notfirst_plus_one;
};
template <class T, class A0, class A1, class... As>
struct n_abis_in_tuple<simd_tuple<T, A0, A1, As...>> {
    using counts = typename seq_op<typename n_abis_in_tuple<
        simd_tuple<T, A1, As...>>::counts>::template prepend<1, 0>;
    using begins = typename seq_op<typename n_abis_in_tuple<
        simd_tuple<T, A1, As...>>::begins>::template prepend<0, 1>;
};

namespace tests
{
static_assert(
    std::is_same<n_abis_in_tuple<simd_tuple<int, simd_abi::sse, simd_abi::sse,
                                                simd_abi::scalar, simd_abi::scalar,
                                                simd_abi::scalar>>::counts,
                 std::index_sequence<2, 3>>::value,
    "");
static_assert(
    std::is_same<n_abis_in_tuple<simd_tuple<int, simd_abi::sse, simd_abi::sse,
                                                simd_abi::scalar, simd_abi::scalar,
                                                simd_abi::scalar>>::begins,
                 std::index_sequence<0, 2>>::value,
    "");
}  // namespace tests

// tree_reduction {{{1
template <size_t Count, size_t Begin> struct tree_reduction {
    static_assert(Count > 0,
                  "tree_reduction requires at least one simd object to work with");
    template <class T, class... As, class BinaryOperation>
    auto operator()(const simd_tuple<T, As...> &tup,
                    const BinaryOperation &binary_op) const noexcept
    {
        constexpr size_t left = next_power_of_2(Count) / 2;
        constexpr size_t right = Count - left;
        return binary_op(tree_reduction<left, Begin>()(tup, binary_op),
                         tree_reduction<right, Begin + left>()(tup, binary_op));
    }
};
template <size_t Begin> struct tree_reduction<1, Begin> {
    template <class T, class... As, class BinaryOperation>
    auto operator()(const simd_tuple<T, As...> &tup, const BinaryOperation &) const
        noexcept
    {
        return detail::get_simd<Begin>(tup);
    }
};
template <size_t Begin> struct tree_reduction<2, Begin> {
    template <class T, class... As, class BinaryOperation>
    auto operator()(const simd_tuple<T, As...> &tup,
                    const BinaryOperation &binary_op) const noexcept
    {
        return binary_op(detail::get_simd<Begin>(tup),
                         detail::get_simd<Begin + 1>(tup));
    }
};

// vec_to_scalar_reduction {{{1
// This helper function implements the second step in a generic fixed_size reduction.
// -  Input: a tuple of native simd (or scalar) objects of decreasing size.
// - Output: a scalar (the reduction).
// - Approach:
//   1. reduce the first two tuple elements
//      a) If the number of elements differs by a factor of 2, split the first object into
//         two objects of the second type and reduce all three to one object of second
//         type.
//      b) If the number of elements differs by a factor of 4, split the first object into
//         two equally sized objects, reduce, and split to two objects of the second type.
//         Finally, reduce all three remaining objects to one object of second type.
//      c) Otherwise use Vc::reduce to reduce both inputs to a scalar, and binary_op to
//         reduce to a single scalar.
//
//      (This optimizes all native cases on x86, e.g. <AVX512, SSE, Scalar>.)
//
//   2. Concate the result of (1) with the remaining tuple elements to recurse into
//      vec_to_scalar_reduction.
//
//   3. If vec_to_scalar_reduction is called with a one-element tuple, call Vc::reduce to
//      reduce to a scalar and return.
template <class T, class A0, class A1, class BinaryOperation>
Vc_INTRINSIC simd<T, A1> vec_to_scalar_reduction_first_pair(
    const simd<T, A0> left, const simd<T, A1> right, const BinaryOperation &binary_op,
    size_constant<2>) noexcept
{
    const std::array<simd<T, A1>, 2> splitted = split<simd<T, A1>>(left);
    return binary_op(binary_op(splitted[0], right), splitted[1]);
}

template <class T, class A0, class A1, class BinaryOperation>
Vc_INTRINSIC simd<T, A1> vec_to_scalar_reduction_first_pair(
    const simd<T, A0> left, const simd<T, A1> right, const BinaryOperation &binary_op,
    size_constant<4>) noexcept
{
    constexpr auto N0 = simd_size_v<T, A0> / 2;
    const auto left2 = split<simd<T, abi_for_size_t<T, N0>>>(left);
    const std::array<simd<T, A1>, 2> splitted =
        split<simd<T, A1>>(binary_op(left2[0], left2[1]));
    return binary_op(binary_op(splitted[0], right), splitted[1]);
}

template <class T, class A0, class A1, class BinaryOperation, size_t Factor>
Vc_INTRINSIC simd<T, simd_abi::scalar> vec_to_scalar_reduction_first_pair(
    const simd<T, A0> left, const simd<T, A1> right, const BinaryOperation &binary_op,
    size_constant<Factor>) noexcept
{
    return binary_op(Vc::reduce(left, binary_op), Vc::reduce(right, binary_op));
}

template <class T, class A0, class BinaryOperation>
Vc_INTRINSIC T vec_to_scalar_reduction(const simd_tuple<T, A0> &tup,
                                       const BinaryOperation &binary_op) noexcept
{
    return Vc::reduce(simd<T, A0>(detail::private_init, tup.first), binary_op);
}

template <class T, class A0, class A1, class... As, class BinaryOperation>
Vc_INTRINSIC T vec_to_scalar_reduction(const simd_tuple<T, A0, A1, As...> &tup,
                                       const BinaryOperation &binary_op) noexcept
{
    return vec_to_scalar_reduction(
        detail::tuple_concat(
            detail::make_tuple(
                vec_to_scalar_reduction_first_pair<T, A0, A1, BinaryOperation>(
                    {private_init, tup.first}, {private_init, tup.second.first},
                    binary_op, size_constant<simd_size_v<T, A0> / simd_size_v<T, A1>>())),
            tup.second.second),
        binary_op);
}

// partial_bitset_to_member_type {{{1
template <class V, size_t N>
Vc_INTRINSIC auto partial_bitset_to_member_type(std::bitset<N> shifted_bits)
{
    static_assert(V::size() <= N, "");
    using M = typename V::mask_type;
    using T = typename V::value_type;
    constexpr T *type_tag = nullptr;
    return detail::get_impl_t<M>::from_bitset(
        std::bitset<V::size()>(shifted_bits.to_ullong()), type_tag);
}

// simd impl {{{1
template <int N> struct fixed_size_simd_impl {
    // member types {{{2
    using mask_member_type = std::bitset<N>;
    template <class T> using simd_member_type = fixed_size_storage<T, N>;
    template <class T>
    static constexpr std::size_t tuple_size = simd_member_type<T>::tuple_size;
    template <class T>
    static constexpr std::make_index_sequence<tuple_size<T>> index_seq = {};
    template <class T> using simd = Vc::simd<T, simd_abi::fixed_size<N>>;
    template <class T> using simd_mask = Vc::simd_mask<T, simd_abi::fixed_size<N>>;
    using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    template <class T>
    static inline simd_member_type<T> broadcast(T x, size_tag) noexcept
    {
        return simd_member_type<T>::generate(
            [&](auto meta) { return meta.broadcast(x, size_constant<meta.size()>()); });
    }

    // generator {{{2
    template <class F, class T>
    static Vc_INTRINSIC simd_member_type<T> generator(F &&gen, type_tag<T>, size_tag)
    {
        return simd_member_type<T>::generate([&gen](auto meta) {
            return meta.generator(
                [&](auto i_) {
                    return gen(size_constant<meta.offset + decltype(i_)::value>());
                },
                type_tag<T>(), size_constant<meta.size()>());
        });
    }

    // load {{{2
    template <class T, class U, class F>
    static inline simd_member_type<T> load(const U *mem, F f,
                                              type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        return simd_member_type<T>::generate(
            [&](auto meta) { return meta.load(&mem[meta.offset], f, type_tag<T>()); });
    }

    // masked load {{{2
    template <class T, class... As, class U, class F>
    static inline void masked_load(simd_tuple<T, As...> &merge,
                                   const mask_member_type bits, const U *mem,
                                   F f) Vc_NOEXCEPT_OR_IN_TEST
    {
        detail::for_each(merge, [&](auto meta, auto &native) {
            meta.masked_load(native, meta.make_mask(bits), &mem[meta.offset], f);
        });
    }

    // store {{{2
    template <class T, class U, class F>
    static inline void store(const simd_member_type<T> v, U *mem, F f,
                             type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        detail::for_each(v, [&](auto meta, auto native) {
            meta.store(native, &mem[meta.offset], f, type_tag<T>());
        });
    }

    // masked store {{{2
    template <class T, class... As, class U, class F>
    static inline void masked_store(const simd_tuple<T, As...> v, U *mem, F f,
                                    const mask_member_type bits) Vc_NOEXCEPT_OR_IN_TEST
    {
        detail::for_each(v, [&](auto meta, auto native) {
            meta.masked_store(native, &mem[meta.offset], f, meta.make_mask(bits));
        });
    }

    // negation {{{2
    template <class T, class... As>
    static inline mask_member_type negate(simd_tuple<T, As...> x) noexcept
    {
        mask_member_type bits = 0;
        for_each(x, [&bits](auto meta, auto native) {
            bits |= meta.mask_to_shifted_ullong(meta.negate(native));
        });
        return bits;
    }

    // reductions {{{2
private:
    template <class T, class... As, class BinaryOperation, size_t... Counts,
              size_t... Begins>
    static inline T reduce(const simd_tuple<T, As...> &tup,
                           const BinaryOperation &binary_op,
                           std::index_sequence<Counts...>, std::index_sequence<Begins...>)
    {
        // 1. reduce all tuple elements with equal ABI to a single element in the output
        // tuple
        const auto reduced_vec = detail::make_tuple(detail::tree_reduction<Counts, Begins>()(tup, binary_op)...);
        // 2. split and reduce until a scalar results
        return detail::vec_to_scalar_reduction(reduced_vec, binary_op);
    }

public:
    template <class T, class BinaryOperation>
    static inline T reduce(size_tag, const simd<T> &x, const BinaryOperation &binary_op)
    {
        using ranges = n_abis_in_tuple<simd_member_type<T>>;
        return fixed_size_simd_impl::reduce(x.d, binary_op, typename ranges::counts(),
                                               typename ranges::begins());
    }

    // min, max, clamp {{{2
    template <class T, class... As>
    static inline simd_tuple<T, As...> min(const simd_tuple<T, As...> a,
                                              const simd_tuple<T, As...> b)
    {
        return apply([](auto impl, auto aa, auto bb) { return impl.min(aa, bb); }, a, b);
    }

    template <class T, class... As>
    static inline simd_tuple<T, As...> max(const simd_tuple<T, As...> a,
                                              const simd_tuple<T, As...> b)
    {
        return apply([](auto impl, auto aa, auto bb) { return impl.max(aa, bb); }, a, b);
    }

    // complement {{{2
    template <class T, class... As>
    static inline simd_tuple<T, As...> complement(simd_tuple<T, As...> x) noexcept
    {
        return apply([](auto impl, auto xx) { return impl.complement(xx); }, x);
    }

    // unary minus {{{2
    template <class T, class... As>
    static inline simd_tuple<T, As...> unary_minus(simd_tuple<T, As...> x) noexcept
    {
        return apply([](auto impl, auto xx) { return impl.unary_minus(xx); }, x);
    }

    // arithmetic operators {{{2

#define Vc_FIXED_OP(name_, op_)                                                          \
    template <class T, class... As>                                                      \
    static inline simd_tuple<T, As...> name_(simd_tuple<T, As...> x,               \
                                                simd_tuple<T, As...> y)               \
    {                                                                                    \
        return apply([](auto impl, auto xx, auto yy) { return impl.name_(xx, yy); }, x,  \
                     y);                                                                 \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON

        Vc_FIXED_OP(plus, +);
        Vc_FIXED_OP(minus, -);
        Vc_FIXED_OP(multiplies, *);
        Vc_FIXED_OP(divides, /);
        Vc_FIXED_OP(modulus, %);
        Vc_FIXED_OP(bit_and, &);
        Vc_FIXED_OP(bit_or, |);
        Vc_FIXED_OP(bit_xor, ^);
        Vc_FIXED_OP(bit_shift_left, <<);
        Vc_FIXED_OP(bit_shift_right, >>);
#undef Vc_FIXED_OP

    template <class T, class... As>
    static inline simd_tuple<T, As...> bit_shift_left(simd_tuple<T, As...> x, int y)
    {
        return apply([y](auto impl, auto xx) { return impl.bit_shift_left(xx, y); }, x);
    }

    template <class T, class... As>
    static inline simd_tuple<T, As...> bit_shift_right(simd_tuple<T, As...> x,
                                                          int y)
    {
        return apply([y](auto impl, auto xx) { return impl.bit_shift_right(xx, y); }, x);
    }

    // math {{{2
#define Vc_APPLY_ON_TUPLE_(name_)                                                        \
    template <class T, class... As>                                                      \
    static inline simd_tuple<T, As...> name_(simd_tuple<T, As...> x) noexcept            \
    {                                                                                    \
        return apply([](auto impl, auto xx) { return impl.name_(xx); }, x);              \
    }
    Vc_APPLY_ON_TUPLE_(sqrt)
    Vc_APPLY_ON_TUPLE_(abs)
    Vc_APPLY_ON_TUPLE_(logb)
    Vc_APPLY_ON_TUPLE_(trunc)
    Vc_APPLY_ON_TUPLE_(floor)
    Vc_APPLY_ON_TUPLE_(ceil)
#undef Vc_APPLY_ON_TUPLE_

    template <class T, class... As>
    static inline simd_tuple<T, As...> frexp(const simd_tuple<T, As...> &x,
                                             fixed_size_storage<int, N> &exp) noexcept
    {
        //Vc_DEBUG_DEFERRED("exp = ", exp);
        return apply(frexp_fwd(), x, exp);
    }

    template <class T, class... As>
    static inline fixed_size_storage<int, N> fpclassify(simd_tuple<T, As...> x) noexcept
    {
        return detail::optimize_tuple(x.template apply_r<int>(
            [](auto impl, auto xx) { return impl.fpclassify(xx); }));
    }

#define Vc_TEST_ON_TUPLE_(name_)                                                         \
    template <class T, class... As>                                                      \
    static inline mask_member_type name_(simd_tuple<T, As...> x) noexcept                \
    {                                                                                    \
        return test([](auto impl, auto xx) { return impl.name_(xx); }, x);               \
    }
    Vc_TEST_ON_TUPLE_(isinf)
    Vc_TEST_ON_TUPLE_(isfinite)
    Vc_TEST_ON_TUPLE_(isnan)
    Vc_TEST_ON_TUPLE_(isnormal)
    Vc_TEST_ON_TUPLE_(signbit)
#undef Vc_TEST_ON_TUPLE_

    // increment & decrement{{{2
    template <class... Ts> static inline void increment(simd_tuple<Ts...> &x)
    {
        for_each(x, [](auto meta, auto &native) { meta.increment(native); });
    }

    template <class... Ts> static inline void decrement(simd_tuple<Ts...> &x)
    {
        for_each(x, [](auto meta, auto &native) { meta.decrement(native); });
    }

    // compares {{{2
#define Vc_CMP_OPERATIONS(cmp_)                                                          \
    template <class T, class... As>                                                      \
    static inline mask_member_type cmp_(simd_tuple<T, As...> x,                       \
                                        simd_tuple<T, As...> y)                       \
    {                                                                                    \
        mask_member_type bits = 0;                                                       \
        detail::for_each(x, y, [&bits](auto meta, auto native_x, auto native_y) {        \
            bits |= meta.mask_to_shifted_ullong(meta.cmp_(native_x, native_y));          \
        });                                                                              \
        return bits;                                                                     \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
        Vc_CMP_OPERATIONS(equal_to);
        Vc_CMP_OPERATIONS(not_equal_to);
        Vc_CMP_OPERATIONS(less);
        Vc_CMP_OPERATIONS(greater);
        Vc_CMP_OPERATIONS(less_equal);
        Vc_CMP_OPERATIONS(greater_equal);
        Vc_CMP_OPERATIONS(isunordered);
#undef Vc_CMP_OPERATIONS

    // smart_reference access {{{2
    template <class T, class... As>
    static Vc_INTRINSIC T get(const simd_tuple<T, As...> &v, int i) noexcept
    {
        return v[i];
    }
    template <class T, class... As, class U>
    static Vc_INTRINSIC void set(simd_tuple<T, As...> &v, int i, U &&x) noexcept
    {
        v.set(i, std::forward<U>(x));
    }

    // masked_assign {{{2
    template <typename T, class... As>
    static Vc_INTRINSIC void masked_assign(
        const mask_member_type bits, detail::simd_tuple<T, As...> &lhs,
        const detail::id<detail::simd_tuple<T, As...>> rhs)
    {
        detail::for_each(lhs, rhs, [&](auto meta, auto &native_lhs, auto native_rhs) {
            meta.masked_assign(meta.make_mask(bits), native_lhs, native_rhs);
        });
    }

    // Optimization for the case where the RHS is a scalar. No need to broadcast the
    // scalar to a simd first.
    template <typename T, class... As>
    static Vc_INTRINSIC void masked_assign(const mask_member_type bits,
                                           detail::simd_tuple<T, As...> &lhs,
                                           const detail::id<T> rhs)
    {
        detail::for_each(lhs, [&](auto meta, auto &native_lhs) {
            meta.masked_assign(meta.make_mask(bits), native_lhs, rhs);
        });
    }

    // masked_cassign {{{2
    template <template <typename> class Op, typename T, class... As>
    static inline void masked_cassign(const mask_member_type bits,
                                      detail::simd_tuple<T, As...> &lhs,
                                      const detail::simd_tuple<T, As...> rhs)
    {
        detail::for_each(lhs, rhs, [&](auto meta, auto &native_lhs, auto native_rhs) {
            meta.template masked_cassign<Op>(meta.make_mask(bits), native_lhs,
                                             native_rhs);
        });
    }

    // Optimization for the case where the RHS is a scalar. No need to broadcast the
    // scalar to a simd
    // first.
    template <template <typename> class Op, typename T, class... As>
    static inline void masked_cassign(const mask_member_type bits,
                                      detail::simd_tuple<T, As...> &lhs, const T rhs)
    {
        detail::for_each(lhs, [&](auto meta, auto &native_lhs) {
            meta.template masked_cassign<Op>(meta.make_mask(bits), native_lhs, rhs);
        });
    }

    // masked_unary {{{2
    template <template <typename> class Op, class T, class... As>
    static inline detail::simd_tuple<T, As...> masked_unary(
        const mask_member_type bits, const detail::simd_tuple<T, As...> v)
    {
        return v.apply_wrapped([&bits](auto meta, auto native) {
            return meta.template masked_unary<Op>(meta.make_mask(bits), native);
        });
    }

    // }}}2
};

// simd_mask impl {{{1
template <int N> struct fixed_size_mask_impl {
    static_assert(sizeof(ullong) * CHAR_BIT >= N,
                  "The fixed_size implementation relies on one "
                  "ullong being able to store all boolean "
                  "elements.");  // required in load & store

    // member types {{{2
    static constexpr std::make_index_sequence<N> index_seq = {};
    using mask_member_type = std::bitset<N>;
    template <class T> using simd_mask = Vc::simd_mask<T, simd_abi::fixed_size<N>>;
    using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;

    // to_bitset {{{2
    static Vc_INTRINSIC mask_member_type to_bitset(const mask_member_type &bs) noexcept
    {
        return bs;
    }

    // from_bitset {{{2
    template <class T>
    static Vc_INTRINSIC mask_member_type from_bitset(const mask_member_type &bs,
                                                     type_tag<T>) noexcept
    {
        return bs;
    }

    // broadcast {{{2
    template <class T>
    static Vc_INTRINSIC mask_member_type broadcast(bool x, type_tag<T>) noexcept
    {
        return ullong(x) * ((1llu << N) - 1llu);
    }

    // load {{{2
    template <class F>
    static inline mask_member_type load(const bool *mem, F f, size_tag) noexcept
    {
        // TODO: uchar is not necessarily the best type to use here. For smaller N ushort,
        // uint, ullong, float, and double can be more efficient.
        ullong r = 0;
        using Vs = fixed_size_storage<uchar, N>;
        detail::for_each(Vs{}, [&](auto meta, auto) {
            r |= meta.mask_to_shifted_ullong(
                meta.simd_mask.load(&mem[meta.offset], f, size_constant<meta.size()>()));
        });
        return r;
    }

    // masked load {{{2
    template <class F>
    static inline void masked_load(mask_member_type &merge, mask_member_type mask,
                                   const bool *mem, F) noexcept
    {
        execute_n_times<N>([&](auto i) {
            if (mask[i]) {
                merge[i] = mem[i];
            }
        });
    }

    // store {{{2
    template <class F>
    static inline void store(mask_member_type bs, bool *mem, F f, size_tag) noexcept
    {
#ifdef Vc_HAVE_AVX512BW
        const __m512i bool64 =
            and_(_mm512_movm_epi8(bs.to_ullong()), x86::one64(uchar()));
        detail::x86::store_n_bytes(size_constant<N>(), bool64, mem, f);
#elif defined Vc_HAVE_BMI2
#ifdef Vc_IS_AMD64
        unused(f);
        execute_n_times<N / 8>([&](auto i) {
            constexpr size_t offset = i * 8;
            const ullong bool8 =
                _pdep_u64(bs.to_ullong() >> offset, 0x0101010101010101ULL);
            std::memcpy(&mem[offset], &bool8, 8);
        });
        if (N % 8 > 0) {
            constexpr size_t offset = (N / 8) * 8;
            const ullong bool8 =
                _pdep_u64(bs.to_ullong() >> offset, 0x0101010101010101ULL);
            std::memcpy(&mem[offset], &bool8, N % 8);
        }
#else   // Vc_IS_AMD64
        unused(f);
        execute_n_times<N / 4>([&](auto i) {
            constexpr size_t offset = i * 4;
            const ullong bool4 =
                _pdep_u32(bs.to_ullong() >> offset, 0x01010101U);
            std::memcpy(&mem[offset], &bool4, 4);
        });
        if (N % 4 > 0) {
            constexpr size_t offset = (N / 4) * 4;
            const ullong bool4 =
                _pdep_u32(bs.to_ullong() >> offset, 0x01010101U);
            std::memcpy(&mem[offset], &bool4, N % 4);
        }
#endif  // Vc_IS_AMD64
#elif defined Vc_HAVE_SSE2   // !AVX512BW && !BMI2
        using V = simd<uchar, simd_abi::sse>;
        ullong bits = bs.to_ullong();
        execute_n_times<(N + 15) / 16>([&](auto i) {
            constexpr size_t offset = i * 16;
            constexpr size_t remaining = N - offset;
            if (remaining == 1) {
                mem[offset] = static_cast<bool>(bits >> offset);
            } else if (remaining <= 4) {
                const uint bool4 = ((bits >> offset) * 0x00204081U) & 0x01010101U;
                std::memcpy(&mem[offset], &bool4, remaining);
            } else if (remaining <= 7) {
                const ullong bool8 =
                    ((bits >> offset) * 0x40810204081ULL) & 0x0101010101010101ULL;
                std::memcpy(&mem[offset], &bool8, remaining);
            } else {
                auto tmp = _mm_cvtsi32_si128(bits >> offset);
                tmp = _mm_unpacklo_epi8(tmp, tmp);
                tmp = _mm_unpacklo_epi16(tmp, tmp);
                tmp = _mm_unpacklo_epi32(tmp, tmp);
                V tmp2(tmp);
                tmp2 &= V([](auto j) {
                    return static_cast<uchar>(1 << (j % CHAR_BIT));
                });  // mask bit index
                const __m128i bool16 =
                    _mm_add_epi8(data(tmp2 == V()),
                                 x86::one16(uchar()));  // 0xff -> 0x00 | 0x00 -> 0x01
                if (remaining >= 16) {
                    x86::store16(bool16, &mem[offset], f);
                } else if (remaining & 3) {
                    constexpr int to_shift = 16 - int(remaining);
                    _mm_maskmoveu_si128(bool16,
                                        _mm_srli_si128(allone<__m128i>(), to_shift),
                                        reinterpret_cast<char *>(&mem[offset]));
                } else  // at this point: 8 < remaining < 16
                    if (remaining >= 8) {
                    x86::store8(bool16, &mem[offset], f);
                    if (remaining == 12) {
                        x86::store4(_mm_unpackhi_epi64(bool16, bool16), &mem[offset + 8],
                                    f);
                    }
                }
            }
        });
#else
        // TODO: uchar is not necessarily the best type to use here. For smaller N ushort,
        // uint, ullong, float, and double can be more efficient.
        using Vs = fixed_size_storage<uchar, N>;
        detail::for_each(Vs{}, [&](auto meta, auto) {
            meta.store(meta.make_mask(bs), &mem[meta.offset], f, meta.size_tag);
        });
//#else
        //execute_n_times<N>([&](auto i) { mem[i] = bs[i]; });
#endif  // Vc_HAVE_BMI2
    }

    // masked store {{{2
    template <class F>
    static inline void masked_store(const mask_member_type v, bool *mem, F,
                                    const mask_member_type k) noexcept
    {
        execute_n_times<N>([&](auto i) {
            if (k[i]) {
                mem[i] = v[i];
            }
        });
    }

    // negation {{{2
    static Vc_INTRINSIC mask_member_type negate(const mask_member_type &x,
                                                size_tag) noexcept
    {
        return ~x;
    }

    // logical and bitwise operators {{{2
    template <class T>
    static Vc_INTRINSIC simd_mask<T> logical_and(const simd_mask<T> &x, const simd_mask<T> &y) noexcept
    {
        return {bitset_init, x.d & y.d};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> logical_or(const simd_mask<T> &x, const simd_mask<T> &y) noexcept
    {
        return {bitset_init, x.d | y.d};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> bit_and(const simd_mask<T> &x, const simd_mask<T> &y) noexcept
    {
        return {bitset_init, x.d & y.d};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> bit_or(const simd_mask<T> &x, const simd_mask<T> &y) noexcept
    {
        return {bitset_init, x.d | y.d};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> bit_xor(const simd_mask<T> &x, const simd_mask<T> &y) noexcept
    {
        return {bitset_init, x.d ^ y.d};
    }

    // smart_reference access {{{2
    static Vc_INTRINSIC bool get(const mask_member_type k, int i) noexcept
    {
        return k[i];
    }
    static Vc_INTRINSIC void set(mask_member_type &k, int i, bool x) noexcept
    {
        k.set(i, x);
    }

    // masked_assign {{{2
    static Vc_INTRINSIC void masked_assign(const mask_member_type k,
                                           mask_member_type &lhs,
                                           const mask_member_type rhs)
    {
        lhs = (lhs & ~k) | (rhs & k);
    }

    // Optimization for the case where the RHS is a scalar.
    static Vc_INTRINSIC void masked_assign(const mask_member_type k,
                                           mask_member_type &lhs, const bool rhs)
    {
        if (rhs) {
            lhs |= k;
        } else {
            lhs &= ~k;
        }
    }

    // }}}2
};

// simd_converter scalar -> fixed_size<1> {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::scalar, To, simd_abi::fixed_size<1>> {
    simd_tuple<To, simd_abi::scalar> operator()(From x) { return {static_cast<To>(x)}; }
};

// simd_converter fixed_size<1> -> scalar {{{1
template <class From, class To>
struct simd_converter<From, simd_abi::fixed_size<1>, To, simd_abi::scalar> {
    To operator()(simd_tuple<To, simd_abi::scalar> x)
    {
        return {static_cast<To>(x.first)};
    }
};

// simd_converter fixed_size<N> -> fixed_size<N> {{{1
template <class T, int N>
struct simd_converter<T, simd_abi::fixed_size<N>, T, simd_abi::fixed_size<N>> {
    using arg = fixed_size_storage<T, N>;
    Vc_INTRINSIC const arg &operator()(const arg &x) { return x; }
};

template <size_t ChunkSize, class T> struct determine_required_input_chunks;

template <class T, class... Abis>
struct determine_required_input_chunks<0, simd_tuple<T, Abis...>>
    : public std::integral_constant<size_t, 0> {
};

template <size_t ChunkSize, class T, class Abi0, class... Abis>
struct determine_required_input_chunks<ChunkSize, simd_tuple<T, Abi0, Abis...>>
    : public std::integral_constant<
          size_t, determine_required_input_chunks<ChunkSize - simd_size_v<T, Abi0>,
                                                  simd_tuple<T, Abis...>>::value> {
};

template <class From, class To> struct fixed_size_converter {
    struct OneToMultipleChunks {
    };
    template <int N> struct MultipleToOneChunk {
    };
    struct EqualChunks {
    };
    template <class FromAbi, class ToAbi, size_t ToSize = simd_size_v<To, ToAbi>,
              size_t FromSize = simd_size_v<From, FromAbi>>
    using ChunkRelation = std::conditional_t<
        (ToSize < FromSize), OneToMultipleChunks,
        std::conditional_t<(ToSize == FromSize), EqualChunks,
                           MultipleToOneChunk<int(ToSize / FromSize)>>>;

    template <class... Abis>
    using return_type = fixed_size_storage<To, simd_tuple<From, Abis...>::size_v>;


protected:
    // OneToMultipleChunks {{{2
    template <class A0>
    Vc_INTRINSIC return_type<A0> impl(OneToMultipleChunks, const simd_tuple<From, A0> &x)
    {
        using R = return_type<A0>;
        simd_converter<From, A0, To, typename R::first_abi> native_cvt;
        auto &&multiple_return_chunks = native_cvt(x.first);
        return detail::to_tuple<To, typename R::first_abi>(multiple_return_chunks);
    }

    template <class... Abis>
    Vc_INTRINSIC return_type<Abis...> impl(OneToMultipleChunks,
                                           const simd_tuple<From, Abis...> &x)
    {
        using R = return_type<Abis...>;
        using arg = simd_tuple<From, Abis...>;
        constexpr size_t first_chunk = simd_size_v<From, typename arg::first_abi>;
        simd_converter<From, typename arg::first_abi, To, typename R::first_abi>
            native_cvt;
        auto &&multiple_return_chunks = native_cvt(x.first);
        constexpr size_t n_output_chunks =
            first_chunk / simd_size_v<To, typename R::first_abi>;
        return detail::tuple_concat(
            detail::to_tuple<To, typename R::first_abi>(multiple_return_chunks),
            impl(ChunkRelation<typename arg::second_type::first_abi,
                               typename tuple_element<n_output_chunks, R>::type::abi_type>(),
                 x.second));
    }

    // MultipleToOneChunk {{{2
    template <int N, class A0, class... Abis>
    Vc_INTRINSIC return_type<A0, Abis...> impl(MultipleToOneChunk<N>,
                                               const simd_tuple<From, A0, Abis...> &x)
    {
        return impl_mto(std::integral_constant<bool, sizeof...(Abis) + 1 == N>(),
                        std::make_index_sequence<N>(), x);
    }

    template <size_t... Indexes, class A0, class... Abis>
    Vc_INTRINSIC return_type<A0, Abis...> impl_mto(std::true_type,
                                                   std::index_sequence<Indexes...>,
                                                   const simd_tuple<From, A0, Abis...> &x)
    {
        using R = return_type<A0, Abis...>;
        simd_converter<From, A0, To, typename R::first_abi> native_cvt;
        return {native_cvt(detail::get<Indexes>(x)...)};
    }

    template <size_t... Indexes, class A0, class... Abis>
    Vc_INTRINSIC return_type<A0, Abis...> impl_mto(std::false_type,
                                                   std::index_sequence<Indexes...>,
                                                   const simd_tuple<From, A0, Abis...> &x)
    {
        using R = return_type<A0, Abis...>;
        simd_converter<From, A0, To, typename R::first_abi> native_cvt;
        return {
            native_cvt(detail::get<Indexes>(x)...),
            impl(
                ChunkRelation<
                    typename tuple_element<sizeof...(Indexes),
                                           simd_tuple<From, A0, Abis...>>::type::abi_type,
                    typename R::second_type::first_abi>(),
                tuple_pop_front(size_constant<sizeof...(Indexes)>(), x))};
    }

    // EqualChunks {{{2
    template <class A0>
    Vc_INTRINSIC return_type<A0> impl(EqualChunks, const simd_tuple<From, A0> &x)
    {
        simd_converter<From, A0, To, typename return_type<A0>::first_abi> native_cvt;
        return {native_cvt(x.first)};
    }

    template <class A0, class A1, class... Abis>
    Vc_INTRINSIC return_type<A0, A1, Abis...> impl(
        EqualChunks, const simd_tuple<From, A0, A1, Abis...> &x)
    {
        using R = return_type<A0, A1, Abis...>;
        using Rem = typename R::second_type;
        simd_converter<From, A0, To, typename R::first_abi> native_cvt;
        return {native_cvt(x.first),
                impl(ChunkRelation<A1, typename Rem::first_abi>(), x.second)};
    }

    //}}}2
};

template <class From, class To, int N>
struct simd_converter<From, simd_abi::fixed_size<N>, To, simd_abi::fixed_size<N>>
    : public fixed_size_converter<From, To> {
    using base = fixed_size_converter<From, To>;
    using return_type = fixed_size_storage<To, N>;
    using arg = fixed_size_storage<From, N>;

    Vc_INTRINSIC return_type operator()(const arg &x)
    {
        using CR = typename base::template ChunkRelation<typename arg::first_abi,
                                                         typename return_type::first_abi>;
        return base::impl(CR(), x);
    }
};

// simd_converter "native" -> fixed_size<N> {{{1
// i.e. 1 register to ? registers
template <class From, class A, class To, int N>
struct simd_converter<From, A, To, simd_abi::fixed_size<N>> {
    using traits = detail::traits<From, A>;
    using arg = typename traits::simd_member_type;
    using return_type = fixed_size_storage<To, N>;
    static_assert(N == simd_size_v<From, A>,
                  "simd_converter to fixed_size only works for equal element counts");

    Vc_INTRINSIC return_type operator()(arg x)
    {
        return impl(std::make_index_sequence<return_type::tuple_size>(), x);
    }

private:
    return_type impl(std::index_sequence<0>, arg x)
    {
        simd_converter<From, A, To, typename return_type::first_abi> native_cvt;
        return {native_cvt(x)};
    }
    template <size_t... Indexes> return_type impl(std::index_sequence<Indexes...>, arg x)
    {
        simd_converter<From, A, To, typename return_type::first_abi> native_cvt;
        const auto &tmp = native_cvt(x);
        return {tmp[Indexes]...};
    }
};

// simd_converter fixed_size<N> -> "native" {{{1
// i.e. ? register to 1 registers
template <class From, int N, class To, class A>
struct simd_converter<From, simd_abi::fixed_size<N>, To, A> {
    using traits = detail::traits<To, A>;
    using return_type = typename traits::simd_member_type;
    using arg = fixed_size_storage<From, N>;
    static_assert(N == simd_size_v<To, A>,
                  "simd_converter to fixed_size only works for equal element counts");

    Vc_INTRINSIC return_type operator()(arg x)
    {
        return impl(std::make_index_sequence<arg::tuple_size>(), x);
    }

private:
    return_type impl(std::index_sequence<0>, arg x)
    {
        simd_converter<From, typename arg::first_abi, To, A> native_cvt;
        return {native_cvt(x.first)};
    }
    template <size_t... Indexes> return_type impl(std::index_sequence<Indexes...>, arg x)
    {
        simd_converter<From, typename arg::first_abi, To, A> native_cvt;
        return native_cvt(detail::get<Indexes>(x)...);
    }
};

// split_to_tuple {{{1
template <class T, class A0, class... As, int N>
struct split_to_tuple<std::tuple<simd<T, A0>, simd<T, As>...>, simd_abi::fixed_size<N>> {
    template <class A> using V = simd<T, A>;
    std::tuple<V<A0>, V<As>...> operator()(const simd<T, simd_abi::fixed_size<N>> &x)
    {
        using STup = fixed_size_storage<T, N>;
        return impl(A0(), detail::data(x), std::make_index_sequence<1 + sizeof...(As)>(),
                    std::make_index_sequence<STup::tuple_size>());
    }

private:
    template <int N0> using Stor = fixed_size_storage<T, N0>;
    template <std::size_t I, int N0>
    using tuple_abi = typename tuple_element<I, Stor<N0>>::abi_type;

    template <int N0, class... Bs, size_t... Indexes>
    std::tuple<V<A0>, V<As>...> impl2(
        const detail::simd_tuple<T, tuple_abi<Indexes, N0>..., Bs...> &x,
        std::index_sequence<Indexes...>)
    {
        return std::tuple_cat(
            std::tuple<V<A0>>(
                {private_init, detail::make_tuple(detail::get_simd<Indexes>(x)...)}),
            split_to_tuple<std::tuple<V<As>...>, simd_abi::fixed_size<N - N0>>()(
                {private_init, tuple_pop_front(size_constant<sizeof...(Indexes)>(), x)}));
    }

    template <int N0, size_t... Indexes0, size_t... Indexes1>
    std::tuple<V<A0>, V<As>...> impl(simd_abi::fixed_size<N0>,
                                     const detail::simd_tuple<T, A0, As...> &x,
                                     std::index_sequence<Indexes0...>,
                                     std::index_sequence<Indexes1...>)
    {
        return impl2<N0>(x, std::make_index_sequence<Stor<N0>::tuple_size>());
    }

    template <class NotFixedAbi, size_t... Indexes>
    std::tuple<V<A0>, V<As>...> impl(NotFixedAbi,
                                     const detail::simd_tuple<T, A0, As...> &x,
                                     std::index_sequence<Indexes...>,
                                     std::index_sequence<Indexes...>)
    {
        return {detail::get_simd<Indexes>(x)...};
    }

    template <class NotFixedAbi, class... Bs, size_t... Indexes0, size_t... Indexes1>
    std::tuple<V<A0>, V<As>...> impl(NotFixedAbi, const detail::simd_tuple<T, Bs...> &x,
                                     std::index_sequence<Indexes0...>,
                                     std::index_sequence<Indexes1...>)
    {
        std::size_t offset = V<A0>::size();
        std::size_t tmp;
        return {V<A0>(reinterpret_cast<const detail::may_alias<T> *>(&x),
                      flags::vector_aligned),
                V<As>(reinterpret_cast<const detail::may_alias<T> *>(&x) +
                          (tmp = offset, offset += V<As>::size(), tmp),
                      flags::element_aligned)...};
    }

    template <class NotFixedAbi, class... Bs, size_t... Indexes0, size_t... Indexes1>
    std::tuple<V<A0>, V<As>...> impl(NotFixedAbi,
                                     const detail::simd_tuple<T, A0, Bs...> &x,
                                     std::index_sequence<Indexes0...>,
                                     std::index_sequence<Indexes1...>)
    {
        return std::tuple_cat(std::tuple<V<A0>>({private_init, x.first}),
                              split_to_tuple<std::tuple<V<As>...>,
                                             simd_abi::fixed_size<N - V<A0>::size()>>()(
                                  {private_init, x.second}));
    }
};

// }}}1
// traits {{{1
template <class T, int N, bool = ((N <= 32 && N >= 0) || N == 64)>
struct fixed_size_traits {
    static constexpr size_t size() noexcept { return N; }

    using simd_impl_type = fixed_size_simd_impl<N>;
    using simd_member_type = fixed_size_storage<T, N>;
    static constexpr size_t simd_member_alignment =
#ifdef Vc_GCC
        std::min(size_t(
#ifdef __AVX__
                     256
#else
                     128
#endif
                     ),
#else
        (
#endif
                 next_power_of_2(N * sizeof(T)));
    struct simd_cast_type {
        simd_cast_type(const std::array<T, N> &);
        simd_cast_type(simd_member_type dd) : d(dd) {}
        explicit operator simd_member_type() const { return d; }

    private:
        simd_member_type d;
    };
    struct simd_base {
        simd_base() = default;
        Vc_INTRINSIC simd_base(const simd_base &) {}

        explicit operator const simd_member_type &() const
        {
            return data(*static_cast<const fixed_size_simd<T, N> *>(this));
        }
        explicit operator std::array<T, N>() const
        {
            std::array<T, N> r;
            // simd_member_type can be larger because of higher alignment
            static_assert(sizeof(r) <= sizeof(simd_member_type), "");
            std::memcpy(r.data(), &static_cast<const simd_member_type &>(*this),
                        sizeof(r));
            return r;
        }
    };

    using mask_impl_type = fixed_size_mask_impl<N>;
    using mask_member_type = std::bitset<N>;
    static constexpr size_t mask_member_alignment = alignof(mask_member_type);
    class mask_cast_type
    {
        mask_cast_type() = delete;
    };
    struct mask_base {
        //explicit operator std::bitset<size()>() const { return impl::to_bitset(d); }
        // empty. The std::bitset interface suffices
    };

};
template <class T, int N>
struct fixed_size_traits<T, N, false> : public traits<void, void> {
};
template <int N> struct traits<long double, simd_abi::fixed_size<N>> : public fixed_size_traits<long double, N> {};
template <int N> struct traits<double, simd_abi::fixed_size<N>> : public fixed_size_traits<double, N> {};
template <int N> struct traits< float, simd_abi::fixed_size<N>> : public fixed_size_traits< float, N> {};
template <int N> struct traits<ullong, simd_abi::fixed_size<N>> : public fixed_size_traits<ullong, N> {};
template <int N> struct traits< llong, simd_abi::fixed_size<N>> : public fixed_size_traits< llong, N> {};
template <int N> struct traits< ulong, simd_abi::fixed_size<N>> : public fixed_size_traits< ulong, N> {};
template <int N> struct traits<  long, simd_abi::fixed_size<N>> : public fixed_size_traits<  long, N> {};
template <int N> struct traits<  uint, simd_abi::fixed_size<N>> : public fixed_size_traits<  uint, N> {};
template <int N> struct traits<   int, simd_abi::fixed_size<N>> : public fixed_size_traits<   int, N> {};
template <int N> struct traits<ushort, simd_abi::fixed_size<N>> : public fixed_size_traits<ushort, N> {};
template <int N> struct traits< short, simd_abi::fixed_size<N>> : public fixed_size_traits< short, N> {};
template <int N> struct traits< uchar, simd_abi::fixed_size<N>> : public fixed_size_traits< uchar, N> {};
template <int N> struct traits< schar, simd_abi::fixed_size<N>> : public fixed_size_traits< schar, N> {};
template <int N> struct traits<  char, simd_abi::fixed_size<N>> : public fixed_size_traits<  char, N> {};

// }}}1
}  // namespace detail

// [simd_mask.reductions] {{{1
template <class T, int N> inline bool all_of(const fixed_size_mask<T, N> &k)
{
    return data(k).all();
}

template <class T, int N> inline bool any_of(const fixed_size_mask<T, N> &k)
{
    return data(k).any();
}

template <class T, int N> inline bool none_of(const fixed_size_mask<T, N> &k)
{
    return data(k).none();
}

template <class T, int N> inline bool some_of(const fixed_size_mask<T, N> &k)
{
    for (int i = 1; i < N; ++i) {
        if (k[i] != k[i - 1]) {
            return true;
        }
    }
    return false;
}

template <class T, int N> inline int popcount(const fixed_size_mask<T, N> &k)
{
    return data(k).count();
}

template <class T, int N> inline int find_first_set(const fixed_size_mask<T, N> &k)
{
    for (int i = 0; i < N; ++i) {
        if (k[i]) {
            return i;
        }
    }
    return -1;
}

template <class T, int N> inline int find_last_set(const fixed_size_mask<T, N> &k)
{
    for (int i = N - 1; i >= 0; --i) {
        if (k[i]) {
            return i;
        }
    }
    return -1;
}

// }}}1
Vc_VERSIONED_NAMESPACE_END

namespace std
{
// simd_mask operators {{{1
template <class T, int N>
struct equal_to<Vc::simd_mask<T, Vc::simd_abi::fixed_size<N>>> {
private:
    using M = Vc::simd_mask<T, Vc::simd_abi::fixed_size<N>>;

public:
    bool operator()(const M &x, const M &y) const
    {
        bool r = x[0] == y[0];
        for (int i = 1; i < N; ++i) {
            r = r && x[i] == y[i];
        }
        return r;
    }
};
// }}}1
}  // namespace std

#endif  // VC_SIMD_FIXED_SIZE_H_

// vim: foldmethod=marker
