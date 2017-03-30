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

#ifndef VC_DATAPAR_FIXED_SIZE_H_
#define VC_DATAPAR_FIXED_SIZE_H_

#include "datapar.h"
#include "detail.h"
#include <array>

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
// datapar impl {{{1
template <int N> struct fixed_size_datapar_impl {
    // member types {{{2
    static constexpr std::make_index_sequence<N> index_seq = {};
    using mask_member_type = std::array<bool, N>;
    template <class T> using datapar_member_type = std::array<T, N>;
    template <class T> using datapar = Vc::datapar<T, datapar_abi::fixed_size<N>>;
    template <class T> using mask = Vc::mask<T, datapar_abi::fixed_size<N>>;
    using size_tag = std::integral_constant<size_t, N>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    template <class T, size_t... I>
    static Vc_INTRINSIC datapar_member_type<T> broadcast_impl(
        T x, std::index_sequence<I...>) noexcept
    {
        return {((void)I, x)...};
    }
    template <class T>
    static inline datapar_member_type<T> broadcast(T x, size_tag) noexcept
    {
        return broadcast_impl(x, index_seq);
    }

    // load {{{2
    template <class T, class U, size_t... I>
    static Vc_INTRINSIC datapar_member_type<T> load_impl(
        const U *mem, std::index_sequence<I...>) noexcept
    {
        return {static_cast<T>(mem[I])...};
    }
    template <class T, class U, class F>
    static inline datapar_member_type<T> load(const U *mem, F, type_tag<T>) noexcept
    {
        return load_impl<T>(mem, index_seq);
    }

    // masked load {{{2
    template <class T, class U, size_t... I>
    static Vc_INTRINSIC void masked_load_impl(datapar_member_type<T> &merge,
                                              const mask_member_type &mask, const U *mem,
                                              std::index_sequence<I...>) noexcept
    {
        auto &&x = {(merge[I] = mask[I] ? static_cast<T>(mem[I]) : merge[I])...};
        unused(x);
    }
    template <class T, class A, class U, class F>
    static inline void masked_load(datapar<T> &merge, const Vc::mask<T, A> &k,
                                   const U *mem, F) noexcept
    {
        masked_load_impl(merge.d, k.d, mem, index_seq);
    }

    // store {{{2
    template <class T, class U, size_t... I>
    static Vc_INTRINSIC void store_impl(const datapar_member_type<T> &v, U *mem,
                                        std::index_sequence<I...>) noexcept
    {
        auto &&x = {(mem[I] = static_cast<U>(v[I]))...};
        unused(x);
    }
    template <class T, class U, class F>
    static inline void store(const datapar_member_type<T> &v, U *mem, F,
                             type_tag<T>) noexcept
    {
        return store_impl(v, mem, index_seq);
    }

    // masked store {{{2
    template <class T, class U, size_t... I>
    static Vc_INTRINSIC void masked_store_impl(const datapar_member_type<T> &v, U *mem,
                                               std::index_sequence<I...>,
                                               const mask_member_type &k) noexcept
    {
        auto &&x = {(k[I] ? mem[I] = static_cast<U>(v[I]) : false)...};
        unused(x);
    }
    template <class T, class A, class U, class F>
    static inline void masked_store(const datapar<T> &v, U *mem, F,
                                    const Vc::mask<T, A> &k) noexcept
    {
        return masked_store_impl(v.d, mem, index_seq, k.d);
    }

    // negation {{{2
    template <class T, size_t... I>
    static Vc_INTRINSIC mask_member_type negate_impl(const datapar_member_type<T> &x,
                                                     std::index_sequence<I...>) noexcept
    {
        return {!x[I]...};
    }
    template <class T, class A>
    static inline Vc::mask<T, A> negate(const Vc::datapar<T, A> &x) noexcept
    {
        return {private_init, negate_impl(x.d, index_seq)};
    }

    // reductions {{{2
    template <class T, class BinaryOperation>
    static inline T reduce(size_tag, const datapar<T> &x, BinaryOperation &binary_op)
    {
        T r = x[0];
        execute_n_times<N - 1>([&](auto i) { r = binary_op(r, x[i + 1]); });
        return r;
    }

    // min, max, clamp {{{2
    template <class T>
    static inline datapar<T> min(const datapar<T> &a, const datapar<T> &b)
    {
        auto &&x = data(a);
        auto &&y = data(b);
        return {private_init, generate_from_n_evaluations<N, datapar_member_type<T>>(
                                  [&](auto i) { return std::min(x[i], y[i]); })};
    }

    template <class T>
    static inline datapar<T> max(const datapar<T> &a, const datapar<T> &b)
    {
        auto &&x = data(a);
        auto &&y = data(b);
        return {private_init, generate_from_n_evaluations<N, datapar_member_type<T>>(
                                  [&](auto i) { return std::max(x[i], y[i]); })};
    }

    // complement {{{2
    template <class T, class A>
    static inline Vc::datapar<T, A> complement(const Vc::datapar<T, A> &x) noexcept
    {
        return {private_init, generate_from_n_evaluations<N, datapar_member_type<T>>(
                                  [&](auto i) { return static_cast<T>(~x.d[i]); })};
    }

    // unary minus {{{2
    template <class T, class A>
    static inline Vc::datapar<T, A> unary_minus(const Vc::datapar<T, A> &x) noexcept {
        return {private_init, generate_from_n_evaluations<N, datapar_member_type<T>>(
                                  [&](auto i) { return static_cast<T>(-x.d[i]); })};
    }

    // arithmetic operators {{{2

    template <class T, class A>
    static inline Vc::datapar<T, A> plus(const Vc::datapar<T, A> &x,
                                         const Vc::datapar<T, A> &y)
    {
        return {private_init, generate_from_n_evaluations<N, datapar_member_type<T>>([&](
                                  auto i) { return static_cast<T>(x.d[i] + y.d[i]); })};
    }

    template <class T, class A>
    static inline Vc::datapar<T, A> minus(const Vc::datapar<T, A> &x,
                                          const Vc::datapar<T, A> &y)
    {
        return {
            private_init,
            generate_from_n_evaluations<N, datapar_member_type<T>>([&](auto i) {
                return static_cast<T>(Vc::detail::promote_preserving_unsigned(x.d[i]) -
                                      Vc::detail::promote_preserving_unsigned(y.d[i]));
            })};
    }

    template <class T, class A>
    static inline Vc::datapar<T, A> multiplies(const Vc::datapar<T, A> &x,
                                               const Vc::datapar<T, A> &y)
    {
        return {
            private_init,
            generate_from_n_evaluations<N, datapar_member_type<T>>([&](auto i) {
                return static_cast<T>(Vc::detail::promote_preserving_unsigned(x.d[i]) *
                                      Vc::detail::promote_preserving_unsigned(y.d[i]));
            })};
    }

    template <class T, class A>
    static inline Vc::datapar<T, A> divides(const Vc::datapar<T, A> &x,
                                            const Vc::datapar<T, A> &y)
    {
        return {
            private_init,
            generate_from_n_evaluations<N, datapar_member_type<T>>([&](auto i) {
                return static_cast<T>(Vc::detail::promote_preserving_unsigned(x.d[i]) /
                                      Vc::detail::promote_preserving_unsigned(y.d[i]));
            })};
    }

    template <class T, class A>
    static inline Vc::datapar<T, A> modulus(const Vc::datapar<T, A> &x,
                                            const Vc::datapar<T, A> &y)
    {
        return {
            private_init,
            generate_from_n_evaluations<N, datapar_member_type<T>>([&](auto i) {
                return static_cast<T>(Vc::detail::promote_preserving_unsigned(x.d[i]) %
                                      Vc::detail::promote_preserving_unsigned(y.d[i]));
            })};
    }

    template <class T, class A>
    static inline Vc::datapar<T, A> bit_and(const Vc::datapar<T, A> &x,
                                            const Vc::datapar<T, A> &y)
    {
        return {
            private_init,
            generate_from_n_evaluations<N, datapar_member_type<T>>([&](auto i) {
                return static_cast<T>(Vc::detail::promote_preserving_unsigned(x.d[i]) &
                                      Vc::detail::promote_preserving_unsigned(y.d[i]));
            })};
    }

    template <class T, class A>
    static inline Vc::datapar<T, A> bit_or(const Vc::datapar<T, A> &x,
                                           const Vc::datapar<T, A> &y)
    {
        return {
            private_init,
            generate_from_n_evaluations<N, datapar_member_type<T>>([&](auto i) {
                return static_cast<T>(Vc::detail::promote_preserving_unsigned(x.d[i]) |
                                      Vc::detail::promote_preserving_unsigned(y.d[i]));
            })};
    }

    template <class T, class A>
    static inline Vc::datapar<T, A> bit_xor(const Vc::datapar<T, A> &x,
                                            const Vc::datapar<T, A> &y)
    {
        return {
            private_init,
            generate_from_n_evaluations<N, datapar_member_type<T>>([&](auto i) {
                return static_cast<T>(Vc::detail::promote_preserving_unsigned(x.d[i]) ^
                                      Vc::detail::promote_preserving_unsigned(y.d[i]));
            })};
    }

    template <class T, class A>
    static inline Vc::datapar<T, A> bit_shift_left(const Vc::datapar<T, A> &x,
                                                   const Vc::datapar<T, A> &y)
    {
        return {private_init,
                generate_from_n_evaluations<N, datapar_member_type<T>>([&](auto i) {
                    return static_cast<T>(Vc::detail::promote_preserving_unsigned(x.d[i])
                                          << y.d[i]);
                })};
    }

    template <class T, class A>
    static inline Vc::datapar<T, A> bit_shift_right(const Vc::datapar<T, A> &x,
                                                    const Vc::datapar<T, A> &y)
    {
        return {private_init,
                generate_from_n_evaluations<N, datapar_member_type<T>>([&](auto i) {
                    return static_cast<T>(
                        Vc::detail::promote_preserving_unsigned(x.d[i]) >> y.d[i]);
                })};
    }

    // increment & decrement{{{2
    template <class T> static inline void increment(datapar_member_type<T> &x)
    {
        execute_n_times<N>([&](auto i) { ++x[i]; });
    }

    template <class T> static inline void decrement(datapar_member_type<T> &x)
    {
        execute_n_times<N>([&](auto i) { --x[i]; });
    }

    // compares {{{2
    template <template <typename> class Cmp, class T, size_t... I>
    static Vc_INTRINSIC mask_member_type cmp_impl(const datapar_member_type<T> &x,
                                                  const datapar_member_type<T> &y,
                                                  std::index_sequence<I...>)
    {
        Cmp<T> cmp;
        return {cmp(x[I], y[I])...};
    }
#define Vc_CMP_OPERATIONS(cmp_)                                                          \
    template <class V> static inline typename V::mask_type cmp_(const V &x, const V &y)  \
    {                                                                                    \
        return {private_init, cmp_impl<std::cmp_>(x.d, y.d, index_seq)};                 \
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
    template <class T, class A>
    static T get(const Vc::datapar<T, A> &v, int i) noexcept
    {
        return v.d[i];
    }
    template <class T, class A, class U>
    static void set(Vc::datapar<T, A> &v, int i, U &&x) noexcept
    {
        v.d[i] = std::forward<U>(x);
    }
    // }}}2
};

// mask impl {{{1
template <int N> struct fixed_size_mask_impl {
    // member types {{{2
    static constexpr std::make_index_sequence<N> index_seq = {};
    using mask_member_type = std::array<bool, N>;
    template <class T> using mask = Vc::mask<T, datapar_abi::fixed_size<N>>;
    using size_tag = std::integral_constant<size_t, N>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    template <size_t... I>
    static Vc_INTRINSIC mask_member_type
    broadcast_impl(bool x, std::index_sequence<I...>) noexcept
    {
        return {((void)I, x)...};
    }
    template <class T>
    static inline mask_member_type broadcast(bool x, type_tag<T>) noexcept
    {
        return broadcast_impl(x, index_seq);
    }

    // load {{{2
    template <size_t... I>
    static Vc_INTRINSIC mask_member_type load_impl(const bool *mem,
                                                   std::index_sequence<I...>) noexcept
    {
        return {mem[I]...};
    }
    template <class F>
    static inline mask_member_type load(const bool *mem, F, size_tag) noexcept
    {
        return load_impl(mem, index_seq);
    }

    // masked load {{{2
    template <size_t... I>
    static Vc_INTRINSIC void masked_load_impl(mask_member_type &merge,
                                              const mask_member_type &mask,
                                              const bool *mem,
                                              std::index_sequence<I...>) noexcept
    {
        auto &&x = {(merge[I] = mask[I] ? mem[I] : merge[I])...};
        unused(x);
    }
    template <class F>
    static inline void masked_load(mask_member_type &merge, const mask_member_type &mask,
                                   const bool *mem, F, size_tag) noexcept
    {
        masked_load_impl(merge, mask, mem, index_seq);
    }

    // store {{{2
    template <size_t... I>
    static Vc_INTRINSIC void store_impl(const mask_member_type &v, bool *mem,
                                        std::index_sequence<I...>) noexcept
    {
        auto &&x = {(mem[I] = v[I])...};
        unused(x);
    }
    template <class F>
    static inline void store(const mask_member_type &v, bool *mem, F, size_tag) noexcept
    {
        store_impl(v, mem, index_seq);
    }

    // masked store {{{2
    template <class F>
    static inline void masked_store(const mask_member_type &v, bool *mem, F,
                                    const mask_member_type &k, size_tag) noexcept
    {
        execute_n_times<N>([&](auto i) {
            if (k[i]) {
                mem[i] = v[i];
            }
        });
    }

    // negation {{{2
    template <size_t... I>
    static Vc_INTRINSIC mask_member_type negate_impl(const mask_member_type &x,
                                                     std::index_sequence<I...>) noexcept
    {
        return {!x[I]...};
    }
    static inline mask_member_type negate(const mask_member_type &x, size_tag) noexcept
    {
        return negate_impl(x, index_seq);
    }

    // logical and bitwise operators {{{2
    template <class T>
    static inline mask<T> logical_and(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, generate_from_n_evaluations<N, mask_member_type>(
                                  [&](auto i) { return x.d[i] && y.d[i]; })};
    }

    template <class T>
    static inline mask<T> logical_or(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, generate_from_n_evaluations<N, mask_member_type>(
                                  [&](auto i) { return x.d[i] || y.d[i]; })};
    }

    template <class T> static inline mask<T> bit_and(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, generate_from_n_evaluations<N, mask_member_type>(
                                  [&](auto i) { return bool(x.d[i] & y.d[i]); })};
    }

    template <class T> static inline mask<T> bit_or(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, generate_from_n_evaluations<N, mask_member_type>(
                                  [&](auto i) { return bool(x.d[i] | y.d[i]); })};
    }

    template <class T> static inline mask<T> bit_xor(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, generate_from_n_evaluations<N, mask_member_type>(
                                  [&](auto i) { return bool(x.d[i] ^ y.d[i]); })};
    }

    // smart_reference access {{{2
    template <class T, class A> static bool get(const Vc::mask<T, A> &k, int i) noexcept
    {
        return k.d[i];
    }
    template <class T, class A> static void set(Vc::mask<T, A> &k, int i, bool x) noexcept
    {
        k.d[i] = x;
    }
    // }}}2
};

// traits {{{1
template <class T, int N, bool = ((N <= 32 && N >= 0) || N == 64)>
struct fixed_size_traits {
    static constexpr size_t size() noexcept { return N; }

    using datapar_impl_type = fixed_size_datapar_impl<N>;
    using datapar_member_type = std::array<T, N>;
    static constexpr size_t datapar_member_alignment =
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
    using datapar_cast_type = const datapar_member_type &;
    struct datapar_base {};

    using mask_impl_type = fixed_size_mask_impl<N>;
    using mask_member_type = typename mask_impl_type::mask_member_type;
    static constexpr size_t mask_member_alignment = next_power_of_2(N);
    using mask_cast_type = const mask_member_type &;
    struct mask_base {};
};
template <class T, int N>
struct fixed_size_traits<T, N, false> : public traits<void, void> {
};
template <int N> struct traits<long double, datapar_abi::fixed_size<N>> : public fixed_size_traits<long double, N> {};
template <int N> struct traits<double, datapar_abi::fixed_size<N>> : public fixed_size_traits<double, N> {};
template <int N> struct traits< float, datapar_abi::fixed_size<N>> : public fixed_size_traits< float, N> {};
template <int N> struct traits<ullong, datapar_abi::fixed_size<N>> : public fixed_size_traits<ullong, N> {};
template <int N> struct traits< llong, datapar_abi::fixed_size<N>> : public fixed_size_traits< llong, N> {};
template <int N> struct traits< ulong, datapar_abi::fixed_size<N>> : public fixed_size_traits< ulong, N> {};
template <int N> struct traits<  long, datapar_abi::fixed_size<N>> : public fixed_size_traits<  long, N> {};
template <int N> struct traits<  uint, datapar_abi::fixed_size<N>> : public fixed_size_traits<  uint, N> {};
template <int N> struct traits<   int, datapar_abi::fixed_size<N>> : public fixed_size_traits<   int, N> {};
template <int N> struct traits<ushort, datapar_abi::fixed_size<N>> : public fixed_size_traits<ushort, N> {};
template <int N> struct traits< short, datapar_abi::fixed_size<N>> : public fixed_size_traits< short, N> {};
template <int N> struct traits< uchar, datapar_abi::fixed_size<N>> : public fixed_size_traits< uchar, N> {};
template <int N> struct traits< schar, datapar_abi::fixed_size<N>> : public fixed_size_traits< schar, N> {};
template <int N> struct traits<  char, datapar_abi::fixed_size<N>> : public fixed_size_traits<  char, N> {};

// }}}1
}  // namespace detail

// where implementation {{{1
template <typename T, int N>
static Vc_INTRINSIC void masked_assign(
    const mask<T, datapar_abi::fixed_size<N>> &k,
    datapar<T, datapar_abi::fixed_size<N>> &lhs,
    const detail::id<datapar<T, datapar_abi::fixed_size<N>>> &rhs)
{
    detail::execute_n_times<N>([&](auto i) {
        if (k[i]) {
            lhs[i] = rhs[i];
        }
    });
}

template <typename T, int N>
static Vc_INTRINSIC void masked_assign(
    const mask<T, datapar_abi::fixed_size<N>> &k,
    mask<T, datapar_abi::fixed_size<N>> &lhs,
    const detail::id<mask<T, datapar_abi::fixed_size<N>>> &rhs)
{
    detail::execute_n_times<N>([&](auto i) {
        if (k[i]) {
            lhs[i] = rhs[i];
        }
    });
}

// Optimization for the case where the RHS is a scalar. No need to broadcast the scalar to a datapar
// first.
template <class T, int N, class U>
static Vc_INTRINSIC
    enable_if<std::is_convertible<U, datapar<T, datapar_abi::fixed_size<N>>>::value &&
                  std::is_arithmetic<U>::value,
              void>
    masked_assign(const mask<T, datapar_abi::fixed_size<N>> &k,
                  datapar<T, datapar_abi::fixed_size<N>> &lhs, const U &rhs)
{
    detail::execute_n_times<N>([&](auto i) {
        if (k[i]) {
            lhs[i] = rhs;
        }
    });
}

template <template <typename> class Op, typename T, int N>
inline void masked_cassign(const fixed_size_mask<T, N> &k, fixed_size_datapar<T, N> &lhs,
                           const fixed_size_datapar<T, N> &rhs)
{
    detail::execute_n_times<N>([&](auto i) {
        if (k[i]) {
            lhs[i] = Op<T>{}(lhs[i], rhs[i]);
        }
    });
}

// Optimization for the case where the RHS is a scalar. No need to broadcast the scalar to a datapar
// first.
template <template <typename> class Op, typename T, int N, class U>
inline enable_if<std::is_convertible<U, fixed_size_datapar<T, N>>::value &&
                     std::is_arithmetic<U>::value,
                 void>
masked_cassign(const fixed_size_mask<T, N> &k, fixed_size_datapar<T, N> &lhs,
               const U &rhs)
{
    detail::execute_n_times<N>([&](auto i) {
        if (k[i]) {
            lhs[i] = Op<T>{}(lhs[i], rhs);
        }
    });
}

template <template <typename> class Op, typename T, int N>
inline fixed_size_datapar<T, N> masked_unary(const fixed_size_mask<T, N> &k,
                                             const fixed_size_datapar<T, N> &v)
{
    return static_cast<fixed_size_datapar<T, N>>(
        detail::generate_from_n_evaluations<N, std::array<T, N>>([&](auto i) {
            using Vc_VERSIONED_NAMESPACE::detail::data;
            return data(k)[i] ? Op<T>{}(data(v)[i]) : data(v)[i];
        }));
}

// [mask.reductions] {{{1
template <class T, int N> inline bool all_of(const fixed_size_mask<T, N> &k)
{
    for (int i = 0; i < N; ++i) {
        if (!k[i]) {
            return false;
        }
    }
    return true;
}

template <class T, int N> inline bool any_of(const fixed_size_mask<T, N> &k)
{
    for (int i = 0; i < N; ++i) {
        if (k[i]) {
            return true;
        }
    }
    return false;
}

template <class T, int N> inline bool none_of(const fixed_size_mask<T, N> &k)
{
    for (int i = 0; i < N; ++i) {
        if (k[i]) {
            return false;
        }
    }
    return true;
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
    int n = k[0];
    for (int i = 1; i < N; ++i) {
        n += k[i];
    }
    return n;
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
// mask operators {{{1
template <class T, int N>
struct equal_to<Vc::mask<T, Vc::datapar_abi::fixed_size<N>>> {
private:
    using M = Vc::mask<T, Vc::datapar_abi::fixed_size<N>>;

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

#endif  // VC_DATAPAR_FIXED_SIZE_H_

// vim: foldmethod=marker
