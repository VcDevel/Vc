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
#include <tuple>

/**
 * The fixed_size ABI gives the following guarantees:
 *  - datapar objects are passed via the stack
 *  - memory layout of `datapar<T, N>` is equivalent to `std::array<T, N>`
 *  - alignment of `datapar<T, N>` is `N * sizeof(T)` if N is a power-of-2 value,
 *    otherwise `next_power_of_2(N * sizeof(T))` (Note: if the alignment were to
 *    exceed the system/compiler maximum, it is bounded to that maximum)
 *  - mask objects are passed like std::bitset<N>
 *  - memory layout of `mask<T, N>` is equivalent to `std::bitset<N>`
 *  - alignment of `mask<T, N>` is equal to the alignment of `std::bitset<N>`
 */

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
// select_best_vector_type_t<T, N>{{{1
/**
 * \internal
 * Selects the best SIMD type out of a typelist to store N scalar values.
 */
struct dummy {
    static constexpr size_t size() { return ~size_t(); }
};

template <class T, int N, class A, class... More>
struct select_best_vector_type {
    using V = std::conditional_t<std::is_destructible<datapar<T, A>>::value,
                                 datapar<T, A>, dummy>;
    using type =
        std::conditional_t<(N >= V::size()), V,
                           typename select_best_vector_type<T, N, More...>::type>;
};
template <class T, int N, class A> struct select_best_vector_type<T, N, A> {
    using type = datapar<T, A>;
};
template <class T, int N>
using select_best_vector_type_t = typename select_best_vector_type<T, N,
      datapar_abi::avx512,
      datapar_abi::avx,
      datapar_abi::neon,
      datapar_abi::sse,
      datapar_abi::scalar
      >::type;

// fixed_size_storage<T, N>{{{1
template <class T, int N, class Tuple, class Next = select_best_vector_type_t<T, N>,
          int Remain = N - int(Next::size())>
struct fixed_size_storage_builder;

template <class T, int N, class... Ts, class Next>
struct fixed_size_storage_builder<T, N, std::tuple<Ts...>, Next, 0> {
    using type = std::tuple<Ts..., Next>;
};

template <class T, int N, class... Ts, class Next, int Remain>
struct fixed_size_storage_builder<T, N, std::tuple<Ts...>, Next, Remain> {
    using type =
        typename fixed_size_storage_builder<T, Remain, std::tuple<Ts..., Next>>::type;
};

template <class T, int N>
using fixed_size_storage = typename fixed_size_storage_builder<T, N, std::tuple<>>::type;

namespace tests {
using float1 = datapar<float, datapar_abi::scalar>;
using float4 = datapar<float, datapar_abi::sse>;
using float8 = datapar<float, datapar_abi::avx>;
using float16 = datapar<float, datapar_abi::avx512>;
static_assert(std::is_same<fixed_size_storage<float, 1>, std::tuple<float1>>::value, "fixed_size_storage failure");
static_assert(std::is_same<fixed_size_storage<float, 2>, std::tuple<float1, float1>>::value, "fixed_size_storage failure");
static_assert(std::is_same<fixed_size_storage<float, 3>, std::tuple<float1, float1, float1>>::value, "fixed_size_storage failure");
static_assert(std::is_same<fixed_size_storage<float, 4>, std::tuple<float4>>::value, "fixed_size_storage failure");
static_assert(std::is_same<fixed_size_storage<float, 5>, std::tuple<float4, float1>>::value, "fixed_size_storage failure");
#ifdef Vc_HAVE_AVX_ABI
static_assert(std::is_same<fixed_size_storage<float, 8>, std::tuple<float8>>::value, "fixed_size_storage failure");
static_assert(std::is_same<fixed_size_storage<float, 12>, std::tuple<float8, float4>>::value, "fixed_size_storage failure");
static_assert(std::is_same<fixed_size_storage<float, 13>, std::tuple<float8, float4, float1>>::value, "fixed_size_storage failure");
#endif
}  // namespace tests

// datapar impl {{{1
template <int N> struct fixed_size_datapar_impl {
    // member types {{{2
    using mask_member_type = std::bitset<N>;
    template <class T> using datapar_member_type = fixed_size_storage<T, N>;
    template <class T>
    static constexpr std::size_t tuple_size =
        std::tuple_size<datapar_member_type<T>>::value;
    template <class T>
    static constexpr std::make_index_sequence<tuple_size<T>> index_seq = {};
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
    static_assert(sizeof(ullong) * CHAR_BIT >= N,
                  "The fixed_size implementation relies on one "
                  "ullong being able to store all boolean "
                  "elements.");  // required in load & store

    // member types {{{2
    static constexpr std::make_index_sequence<N> index_seq = {};
    using mask_member_type = std::bitset<N>;
    template <class T> using mask = Vc::mask<T, datapar_abi::fixed_size<N>>;
    using size_tag = std::integral_constant<size_t, N>;
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
        detail::for_each(Vs{}, [&](auto v, auto i) {
            typename decltype(v)::mask_type k(&mem[i], f);
            r |= k.to_bitset().to_ullong() << i;
        });
        return r;
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
        // TODO: optimize with maskload intrinsics
        masked_load_impl(merge, mask, mem, std::make_index_sequence<N>());
    }

    // store {{{2
    template <class F>
    static inline void store(mask_member_type bs, bool *mem, F f, size_tag) noexcept
    {
#ifdef Vc_HAVE_AVX512BW
        unused(f);
        const __m512i bool64 =
            and_(_mm512_movm_epi8(bs.to_ullong()), x86::one64(uchar()));
        std::memcpy(mem, &bool64, N);
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
        using V = datapar<uchar, datapar_abi::sse>;
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
                    _mm_maskmoveu_si128(bool16,
                                        _mm_srli_si128(allone<__m128i>(), 16 - remaining),
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
        detail::for_each(Vs{}, [&](auto v, auto i) {
            using M = typename decltype(v)::mask_type;
            M::from_bitset(bs.to_ullong() >> i).memstore(&mem[i], f);
        });
//#else
        //execute_n_times<N>([&](auto i) { mem[i] = bs[i]; });
#endif  // Vc_HAVE_BMI2
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
    static Vc_INTRINSIC mask_member_type negate(const mask_member_type &x,
                                                size_tag) noexcept
    {
        return ~x;
    }

    // logical and bitwise operators {{{2
    template <class T>
    static Vc_INTRINSIC mask<T> logical_and(const mask<T> &x, const mask<T> &y) noexcept
    {
        return {bitset_init, x.d & y.d};
    }

    template <class T>
    static Vc_INTRINSIC mask<T> logical_or(const mask<T> &x, const mask<T> &y) noexcept
    {
        return {bitset_init, x.d | y.d};
    }

    template <class T>
    static Vc_INTRINSIC mask<T> bit_and(const mask<T> &x, const mask<T> &y) noexcept
    {
        return {bitset_init, x.d & y.d};
    }

    template <class T>
    static Vc_INTRINSIC mask<T> bit_or(const mask<T> &x, const mask<T> &y) noexcept
    {
        return {bitset_init, x.d | y.d};
    }

    template <class T>
    static Vc_INTRINSIC mask<T> bit_xor(const mask<T> &x, const mask<T> &y) noexcept
    {
        return {bitset_init, x.d ^ y.d};
    }

    // smart_reference access {{{2
    template <class T, class A>
    static Vc_INTRINSIC bool get(const Vc::mask<T, A> &k, int i) noexcept
    {
        return k.d[i];
    }
    template <class T, class A>
    static Vc_INTRINSIC void set(Vc::mask<T, A> &k, int i, bool x) noexcept
    {
        k.d.set(i, x);
    }
    // }}}2
};

// traits {{{1
template <class T, int N, bool = ((N <= 32 && N >= 0) || N == 64)>
struct fixed_size_traits {
    static constexpr size_t size() noexcept { return N; }

    using datapar_impl_type = fixed_size_datapar_impl<N>;
    using datapar_member_type = fixed_size_storage<T, N>;
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
    using datapar_cast_type = const std::array<T, N> &;
    struct datapar_base {};

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
