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

#ifndef VC_SIMD_GENERICIMPL_H_
#define VC_SIMD_GENERICIMPL_H_

#include "detail.h"
#include "convert_mask.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
// ISA & type detection {{{1
template <class T, size_t N> constexpr bool is_sse_ps()
{
    return have_sse && std::is_same_v<T, float> && N == 4;
}
template <class T, size_t N> constexpr bool is_sse_pd()
{
    return have_sse2 && std::is_same_v<T, double> && N == 2;
}
template <class T, size_t N> constexpr bool is_avx_ps()
{
    return have_avx && std::is_same_v<T, float> && N == 8;
}
template <class T, size_t N> constexpr bool is_avx_pd()
{
    return have_avx && std::is_same_v<T, double> && N == 4;
}
template <class T, size_t N> constexpr bool is_avx512_ps()
{
    return have_avx512f && std::is_same_v<T, float> && N == 16;
}
template <class T, size_t N> constexpr bool is_avx512_pd()
{
    return have_avx512f && std::is_same_v<T, double> && N == 8;
}

template <class T, size_t N> constexpr bool is_neon_ps()
{
    return have_neon && std::is_same_v<T, float> && N == 4;
}
template <class T, size_t N> constexpr bool is_neon_pd()
{
    return have_neon && std::is_same_v<T, double> && N == 2;
}

// simd impl {{{1
template <class Abi> struct generic_simd_impl {
    // member types {{{2
    template <class T> using type_tag = T *;
    template <class T>
    using simd_member_type = typename Abi::template traits<T>::simd_member_type;
    template <class T>
    using mask_member_type = typename Abi::template traits<T>::mask_member_type;
    template <class T> static constexpr size_t full_size = simd_member_type<T>::width;

    // make_simd(Storage) {{{2
    template <class T, size_t N>
    static Vc_INTRINSIC simd<T, Abi> make_simd(Storage<T, N> x)
    {
        return {detail::private_init, x};
    }

    // broadcast {{{2
    template <class T>
    static constexpr Vc_INTRINSIC simd_member_type<T> broadcast(T x) noexcept
    {
        return simd_member_type<T>::broadcast(x);
    }

    // generator {{{2
    template <class F, class T>
    static Vc_INTRINSIC simd_member_type<T> generator(F &&gen, type_tag<T>)
    {
        return detail::generate_storage<T, full_size<T>>(std::forward<F>(gen));
    }

    // load {{{2
    template <class T, class U, class F>
    static Vc_INTRINSIC simd_member_type<T> load(const U *mem, F,
                                                 type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        constexpr size_t N = simd_member_type<T>::width;
        constexpr size_t max_load_size =
            (sizeof(U) >= 4 && have_avx512f) || have_avx512bw
                ? 64
                : (std::is_floating_point_v<U> && have_avx) || have_avx2 ? 32 : 16;
        if constexpr (sizeof(U) > 8) {
            return detail::generate_storage<T, N>(
                [&](auto i) { return static_cast<T>(mem[i]); });
        } else if constexpr (std::is_same_v<U, T>) {
            return detail::builtin_load<U, N>(mem, F());
        } else if constexpr (sizeof(U) * N < 16) {
            return x86::convert<simd_member_type<T>>(
                detail::builtin_load16<U, sizeof(U) * N>(mem, F()));
        } else if constexpr (sizeof(U) * N <= max_load_size) {
            return x86::convert<simd_member_type<T>>(detail::builtin_load<U, N>(mem, F()));
        } else if constexpr (sizeof(U) * N == 2 * max_load_size) {
            return x86::convert<simd_member_type<T>>(
                detail::builtin_load<U, N / 2>(mem, F()),
                detail::builtin_load<U, N / 2>(mem + N / 2, F()));
        } else if constexpr (sizeof(U) * N == 4 * max_load_size) {
            return x86::convert<simd_member_type<T>>(
                detail::builtin_load<U, N / 4>(mem, F()),
                detail::builtin_load<U, N / 4>(mem + 1 * N / 4, F()),
                detail::builtin_load<U, N / 4>(mem + 2 * N / 4, F()),
                detail::builtin_load<U, N / 4>(mem + 3 * N / 4, F()));
        } else if constexpr (sizeof(U) * N == 8 * max_load_size) {
            return x86::convert<simd_member_type<T>>(
                detail::builtin_load<U, N / 8>(mem, F()),
                detail::builtin_load<U, N / 8>(mem + 1 * N / 8, F()),
                detail::builtin_load<U, N / 8>(mem + 2 * N / 8, F()),
                detail::builtin_load<U, N / 8>(mem + 3 * N / 8, F()),
                detail::builtin_load<U, N / 8>(mem + 4 * N / 8, F()),
                detail::builtin_load<U, N / 8>(mem + 5 * N / 8, F()),
                detail::builtin_load<U, N / 8>(mem + 6 * N / 8, F()),
                detail::builtin_load<U, N / 8>(mem + 7 * N / 8, F()));
        } else {
            assert_unreachable<T>();
        }
    }

    // masked load {{{2
    template <class T, size_t N, class U, class F>
    static inline detail::Storage<T, N> masked_load(detail::Storage<T, N> merge, mask_member_type<T> k,
                                   const U *mem, F) Vc_NOEXCEPT_OR_IN_TEST
    {
        if constexpr (std::is_same_v<T, U> ||  // no conversion
                      (sizeof(T) == sizeof(U) &&
                       std::is_integral_v<T> ==
                           std::is_integral_v<U>)  // conversion via bit reinterpretation
        ) {
            constexpr bool have_avx512bw_vl_or_zmm =
                have_avx512bw_vl || (have_avx512bw && sizeof(merge) == 64);
            if constexpr (have_avx512bw_vl_or_zmm && sizeof(T) == 1) {
                if constexpr (sizeof(merge) == 16) {
                    merge = _mm_mask_loadu_epi8(merge, _mm_movemask_epi8(k), mem);
                } else if constexpr (sizeof(merge) == 32) {
                    merge = _mm256_mask_loadu_epi8(merge, _mm256_movemask_epi8(k), mem);
                } else if constexpr (sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_epi8(merge, k, mem);
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (have_avx512bw_vl_or_zmm && sizeof(T) == 2) {
                if constexpr (sizeof(merge) == 16) {
                    merge = _mm_mask_loadu_epi16(merge, x86::movemask_epi16(k), mem);
                } else if constexpr (sizeof(merge) == 32) {
                    merge = _mm256_mask_loadu_epi16(merge, x86::movemask_epi16(k), mem);
                } else if constexpr (sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_epi16(merge, k, mem);
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (have_avx2 && sizeof(T) == 4 && std::is_integral_v<U>) {
                if constexpr (sizeof(merge) == 16) {
                    merge =
                        (~k.d & merge.d) | builtin_cast<T>(_mm_maskload_epi32(
                                               reinterpret_cast<const int *>(mem), k));
                } else if constexpr (sizeof(merge) == 32) {
                    merge =
                        (~k.d & merge.d) | builtin_cast<T>(_mm256_maskload_epi32(
                                               reinterpret_cast<const int *>(mem), k));
                } else if constexpr (have_avx512f && sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_epi32(merge, k, mem);
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (have_avx && sizeof(T) == 4) {
                if constexpr (sizeof(merge) == 16) {
                    merge = or_(andnot_(k.d, merge.d),
                                builtin_cast<T>(_mm_maskload_ps(
                                    reinterpret_cast<const float *>(mem), to_m128i(k))));
                } else if constexpr (sizeof(merge) == 32) {
                    merge = or_(andnot_(k.d, merge.d),
                                _mm256_maskload_ps(reinterpret_cast<const float *>(mem),
                                                   to_m256i(k)));
                } else if constexpr (have_avx512f && sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_ps(merge, k, mem);
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (have_avx2 && sizeof(T) == 8 && std::is_integral_v<U>) {
                if constexpr (sizeof(merge) == 16) {
                    merge =
                        (~k.d & merge.d) | builtin_cast<T>(_mm_maskload_epi64(
                                               reinterpret_cast<const llong *>(mem), k));
                } else if constexpr (sizeof(merge) == 32) {
                    merge =
                        (~k.d & merge.d) | builtin_cast<T>(_mm256_maskload_epi64(
                                               reinterpret_cast<const llong *>(mem), k));
                } else if constexpr (have_avx512f && sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_epi64(merge, k, mem);
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (have_avx && sizeof(T) == 8) {
                if constexpr (sizeof(merge) == 16) {
                    merge = or_(andnot_(k.d, merge.d),
                                builtin_cast<T>(_mm_maskload_pd(
                                    reinterpret_cast<const double *>(mem), to_m128i(k))));
                } else if constexpr (sizeof(merge) == 32) {
                    merge = or_(andnot_(k.d, merge.d),
                                _mm256_maskload_pd(reinterpret_cast<const double *>(mem),
                                                   to_m256i(k)));
                } else if constexpr (have_avx512f && sizeof(merge) == 64) {
                    merge = _mm512_mask_loadu_pd(merge, k, mem);
                } else {
                    assert_unreachable<T>();
                }
            } else {
                detail::bit_iteration(mask_to_int<N>(k), [&](auto i) {
                    merge.set(i, static_cast<T>(mem[i]));
                });
            }
        } else if constexpr (sizeof(U) <= 8 &&  // no long double
                             !detail::converts_via_decomposition_v<
                                 U, T, sizeof(merge)>  // conversion via decomposition is
                                                       // better handled via the
                                                       // bit_iteration fallback below
        ) {
            // TODO: copy pattern from masked_store, which doesn't resort to fixed_size
            using A = simd_abi::deduce_t<
                U, std::max(N, 16 / sizeof(U))  // N or more, so that at least a 16 Byte
                                                // vector is used instead of a fixed_size
                                                // filled with scalars
                >;
            using ATraits = detail::traits<U, A>;
            using AImpl = typename ATraits::simd_impl_type;
            typename ATraits::simd_member_type uncvted{};
            typename ATraits::mask_member_type kk;
            if constexpr (detail::is_fixed_size_abi_v<A>) {
                kk = detail::to_bitset(k.d);
            } else {
                kk = convert_mask<typename ATraits::mask_member_type>(k);
            }
            uncvted = AImpl::masked_load(uncvted, kk, mem, F());
            detail::simd_converter<U, A, T, Abi> converter;
            masked_assign(k, merge, converter(uncvted));
        } else {
            detail::bit_iteration(mask_to_int<N>(k),
                                  [&](auto i) { merge.set(i, static_cast<T>(mem[i])); });
        }
        return merge;
    }

    // store {{{2
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(simd_member_type<T> v, U *mem, F,
                                   type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        // TODO: converting int -> "smaller int" can be optimized with AVX512
        constexpr size_t N = simd_member_type<T>::width;
        constexpr size_t max_store_size =
            (sizeof(U) >= 4 && have_avx512f) || have_avx512bw
                ? 64
                : (std::is_floating_point_v<U> && have_avx) || have_avx2 ? 32 : 16;
        if constexpr (sizeof(U) > 8) {
            detail::execute_n_times<N>([&](auto i) { mem[i] = v[i]; });
        } else if constexpr (std::is_same_v<U, T>) {
            detail::builtin_store(v.d, mem, F());
        } else if constexpr (sizeof(U) * N < 16) {
            detail::builtin_store<sizeof(U) * N>(x86::convert<builtin_type16_t<U>>(v),
                                                 mem, F());
        } else if constexpr (sizeof(U) * N <= max_store_size) {
            detail::builtin_store(x86::convert<builtin_type_t<U, N>>(v), mem, F());
        } else {
            constexpr size_t VSize = max_store_size / sizeof(U);
            constexpr size_t stores = N / VSize;
            using V = builtin_type_t<U, VSize>;
            const std::array<V, stores> converted = x86::convert_all<V>(v);
            detail::execute_n_times<stores>([&](auto i) {
                detail::builtin_store(converted[i], mem + i * VSize, F());
            });
        }
    }

    // masked store {{{2
    template <class T, size_t N, class U, class F>
    static inline void masked_store(const Storage<T, N> v, U *mem, F,
                                    const mask_member_type<T> k) Vc_NOEXCEPT_OR_IN_TEST
    {
        constexpr size_t max_store_size =
            (sizeof(U) >= 4 && have_avx512f) || have_avx512bw
                ? 64
                : (std::is_floating_point_v<U> && have_avx) || have_avx2 ? 32 : 16;
        if constexpr (std::is_same_v<T, U> ||
                      (std::is_integral_v<T> && std::is_integral_v<U> &&
                       sizeof(T) == sizeof(U))) {
            // bitwise or no conversion, reinterpret:
            const auto kk = [&]() {
                if constexpr (detail::is_bitmask_v<decltype(k)>) {
                    return mask_member_type<U>(k.d);
                } else {
                    return detail::storage_bitcast<U>(k);
                }
            }();
            x86::maskstore(storage_bitcast<U>(v), mem, F(), kk);
        } else if constexpr (std::is_integral_v<T> && std::is_integral_v<U> &&
                             sizeof(T) > sizeof(U) && have_avx512f &&
                             (sizeof(T) >= 4 || have_avx512bw) &&
                             (sizeof(v) == 64 || have_avx512vl)) {  // truncating store
            const auto kk = [&]() {
                if constexpr (detail::is_bitmask_v<decltype(k)>) {
                    return k;
                } else {
                    return convert_mask<Storage<bool, N>>(k);
                }
            }();
            if constexpr (sizeof(T) == 8 && sizeof(U) == 4) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi32(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi32(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi64_storeu_epi32(mem, kk, v);
                }
            } else if constexpr (sizeof(T) == 8 && sizeof(U) == 2) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi16(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi16(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi64_storeu_epi16(mem, kk, v);
                }
            } else if constexpr (sizeof(T) == 8 && sizeof(U) == 1) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi64_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi64_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi64_storeu_epi8(mem, kk, v);
                }
            } else if constexpr (sizeof(T) == 4 && sizeof(U) == 2) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi32_storeu_epi16(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi32_storeu_epi16(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi32_storeu_epi16(mem, kk, v);
                }
            } else if constexpr (sizeof(T) == 4 && sizeof(U) == 1) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi32_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi32_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi32_storeu_epi8(mem, kk, v);
                }
            } else if constexpr (sizeof(T) == 2 && sizeof(U) == 1) {
                if constexpr (sizeof(v) == 64) {
                    _mm512_mask_cvtepi16_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 32) {
                    _mm256_mask_cvtepi16_storeu_epi8(mem, kk, v);
                } else if constexpr (sizeof(v) == 16) {
                    _mm_mask_cvtepi16_storeu_epi8(mem, kk, v);
                }
            } else {
                assert_unreachable<T>();
            }
        } else if constexpr (sizeof(U) <= 8 &&  // no long double
                             !detail::converts_via_decomposition_v<
                                 T, U, max_store_size>  // conversion via decomposition is
                                                        // better handled via the
                                                        // bit_iteration fallback below
        ) {
            using VV = Storage<U, std::clamp(N, 16 / sizeof(U), max_store_size / sizeof(U))>;
            using V = typename VV::register_type;
            constexpr bool prefer_bitmask =
                (have_avx512f && sizeof(U) >= 4) || have_avx512bw;
            using M = Storage<std::conditional_t<prefer_bitmask, bool, U>, VV::width>;
            constexpr size_t VN = builtin_traits<V>::width;

            if constexpr (VN >= N) {
                x86::maskstore(VV(convert<V>(v)), mem,
                               // careful, if V has more elements than the input v (N),
                               // vector_aligned is incorrect:
                               std::conditional_t<(builtin_traits<V>::width > N),
                                                  overaligned_tag<sizeof(U) * N>, F>(),
                               convert_mask<M>(k));
            } else if constexpr (VN * 2 == N) {
                const std::array<V, 2> converted = x86::convert_all<V>(v);
                x86::maskstore(VV(converted[0]), mem, F(), convert_mask<M>(detail::extract_part<0, 2>(k)));
                x86::maskstore(VV(converted[1]), mem + VV::width, F(), convert_mask<M>(detail::extract_part<1, 2>(k)));
            } else if constexpr (VN * 4 == N) {
                const std::array<V, 4> converted = x86::convert_all<V>(v);
                x86::maskstore(VV(converted[0]), mem, F(), convert_mask<M>(detail::extract_part<0, 4>(k)));
                x86::maskstore(VV(converted[1]), mem + 1 * VV::width, F(), convert_mask<M>(detail::extract_part<1, 4>(k)));
                x86::maskstore(VV(converted[2]), mem + 2 * VV::width, F(), convert_mask<M>(detail::extract_part<2, 4>(k)));
                x86::maskstore(VV(converted[3]), mem + 3 * VV::width, F(), convert_mask<M>(detail::extract_part<3, 4>(k)));
            } else if constexpr (VN * 8 == N) {
                const std::array<V, 8> converted = x86::convert_all<V>(v);
                x86::maskstore(VV(converted[0]), mem, F(), convert_mask<M>(detail::extract_part<0, 8>(k)));
                x86::maskstore(VV(converted[1]), mem + 1 * VV::width, F(), convert_mask<M>(detail::extract_part<1, 8>(k)));
                x86::maskstore(VV(converted[2]), mem + 2 * VV::width, F(), convert_mask<M>(detail::extract_part<2, 8>(k)));
                x86::maskstore(VV(converted[3]), mem + 3 * VV::width, F(), convert_mask<M>(detail::extract_part<3, 8>(k)));
                x86::maskstore(VV(converted[4]), mem + 4 * VV::width, F(), convert_mask<M>(detail::extract_part<4, 8>(k)));
                x86::maskstore(VV(converted[5]), mem + 5 * VV::width, F(), convert_mask<M>(detail::extract_part<5, 8>(k)));
                x86::maskstore(VV(converted[6]), mem + 6 * VV::width, F(), convert_mask<M>(detail::extract_part<6, 8>(k)));
                x86::maskstore(VV(converted[7]), mem + 7 * VV::width, F(), convert_mask<M>(detail::extract_part<7, 8>(k)));
            } else {
                assert_unreachable<T>();
            }
        } else {
            detail::bit_iteration(mask_to_int<N>(k),
                                  [&](auto i) { mem[i] = static_cast<U>(v[i]); });
        }
    }

    // complement {{{2
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> complement(Storage<T, N> x) noexcept
    {
        return detail::x86::complement(x);
    }

    // unary minus {{{2
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> unary_minus(Storage<T, N> x) noexcept
    {
        return detail::x86::unary_minus(x);
    }

    // arithmetic operators {{{2
#define Vc_ARITHMETIC_OP_(name_)                                                         \
    template <class T, size_t N>                                                         \
    static Vc_INTRINSIC Storage<T, N> name_(Storage<T, N> x, Storage<T, N> y)            \
    {                                                                                    \
        return detail::x86::name_(x, y);                                                 \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
    Vc_ARITHMETIC_OP_(plus);
    Vc_ARITHMETIC_OP_(minus);
    Vc_ARITHMETIC_OP_(multiplies);
    Vc_ARITHMETIC_OP_(divides);
    Vc_ARITHMETIC_OP_(modulus);
    Vc_ARITHMETIC_OP_(bit_and);
    Vc_ARITHMETIC_OP_(bit_or);
    Vc_ARITHMETIC_OP_(bit_xor);
    Vc_ARITHMETIC_OP_(bit_shift_left);
    Vc_ARITHMETIC_OP_(bit_shift_right);
#undef Vc_ARITHMETIC_OP_

    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> bit_shift_left(Storage<T, N> x,
                                                                        int y)
    {
        return detail::x86::bit_shift_left(x, y);
    }
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> bit_shift_right(Storage<T, N> x,
                                                                         int y)
    {
        return detail::x86::bit_shift_right(x, y);
    }

    // compares {{{2
    // equal_to {{{3
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC mask_member_type<T> equal_to(Storage<T, N> x,
                                                               Storage<T, N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_EQ_OQ);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ);
                } else { assert_unreachable<T>(); }
            } else {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmpeq_epi64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmpeq_epi32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmpeq_epi16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmpeq_epi8_mask(x, y);
                } else { assert_unreachable<T>(); }
            }
        } else {
            return to_storage(x.d == y.d);
        }
    }

    // not_equal_to {{{3
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC mask_member_type<T> not_equal_to(Storage<T, N> x,
                                                                   Storage<T, N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_NEQ_UQ);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_NEQ_UQ);
                } else { assert_unreachable<T>(); }
            } else {
                       if constexpr (sizeof(T) == 8) { return ~_mm512_cmpeq_epi64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return ~_mm512_cmpeq_epi32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return ~_mm512_cmpeq_epi16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return ~_mm512_cmpeq_epi8_mask(x, y);
                } else { assert_unreachable<T>(); }
            }
        } else {
            return to_storage(x.d != y.d);
        }
    }

    // less {{{3
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC mask_member_type<T> less(Storage<T, N> x,
                                                           Storage<T, N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_LT_OS);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_LT_OS);
                } else { assert_unreachable<T>(); }
            } else if constexpr (std::is_signed_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmplt_epi64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmplt_epi32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmplt_epi16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmplt_epi8_mask(x, y);
                } else { assert_unreachable<T>(); }
            } else {
                static_assert(std::is_unsigned_v<T>);
                       if constexpr (sizeof(T) == 8) { return _mm512_cmplt_epu64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmplt_epu32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmplt_epu16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmplt_epu8_mask(x, y);
                } else { assert_unreachable<T>(); }
            }
        } else {
            return to_storage(x.d < y.d);
        }
    }

    // less_equal {{{3
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC mask_member_type<T> less_equal(Storage<T, N> x,
                                                                 Storage<T, N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_LE_OS);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_LE_OS);
                } else { assert_unreachable<T>(); }
            } else if constexpr (std::is_signed_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmple_epi64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmple_epi32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmple_epi16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmple_epi8_mask(x, y);
                } else { assert_unreachable<T>(); }
            } else {
                static_assert(std::is_unsigned_v<T>);
                       if constexpr (sizeof(T) == 8) { return _mm512_cmple_epu64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmple_epu32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmple_epu16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmple_epu8_mask(x, y);
                } else { assert_unreachable<T>(); }
            }
        } else {
            return to_storage(x.d <= y.d);
        }
    }

    // negation {{{2
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC mask_member_type<T> negate(Storage<T, N> x) noexcept
    {
        if constexpr (detail::is_abi<Abi, simd_abi::__avx512_abi>()) {
            return equal_to(x, simd_member_type<T>());
        } else {
            return detail::to_storage(!x.d);
        }
    }

    // min, max, clamp {{{2
    template <class T, size_t N>
    Vc_NORMAL_MATH static constexpr Vc_INTRINSIC Storage<T, N> min(Storage<T, N> a,
                                                                   Storage<T, N> b)
    {
        return a.d < b.d ? a.d : b.d;
    }
    template <class T, size_t N>
    Vc_NORMAL_MATH static constexpr Vc_INTRINSIC Storage<T, N> max(Storage<T, N> a,
                                                                   Storage<T, N> b)
    {
        return a.d > b.d ? a.d : b.d;
    }

    template <class T, size_t N>
    Vc_NORMAL_MATH static constexpr Vc_INTRINSIC std::pair<Storage<T, N>, Storage<T, N>>
    minmax(Storage<T, N> a, Storage<T, N> b)
    {
        return {a.d < b.d ? a.d : b.d, a.d < b.d ? b.d : a.d};
    }

    // reductions {{{2
    template <class T, class BinaryOperation>
    static Vc_INTRINSIC T reduce(simd<T, Abi> x, BinaryOperation &&binary_op)
    {
        constexpr size_t N = simd_size_v<T, Abi>;
        if constexpr (sizeof(x) > 16) {
            using A = simd_abi::deduce_t<T, N / 2>;
            using V = Vc::simd<T, A>;
            return traits<T, A>::simd_impl_type::reduce(
                binary_op(V(detail::private_init, extract<0, 2>(data(x).d)),
                          V(detail::private_init, extract<1, 2>(data(x).d))),
                std::forward<BinaryOperation>(binary_op));
        } else {
            const auto &xx = x.d;
            if constexpr (N == 16) {
                x = binary_op(make_simd<T, N>(_mm_unpacklo_epi8(xx, xx)),
                              make_simd<T, N>(_mm_unpackhi_epi8(xx, xx)));
            }
            if constexpr (N >= 8) {
                x = binary_op(make_simd<T, N>(_mm_unpacklo_epi16(xx, xx)),
                              make_simd<T, N>(_mm_unpackhi_epi16(xx, xx)));
            }
            if constexpr (N >= 4) {
                using U = std::conditional_t<std::is_floating_point_v<T>, float, int>;
                const auto y = builtin_cast<U>(xx.d);
                x = binary_op(x, make_simd<T, N>(to_storage(
                                     builtin_type_t<U, 4>{y[3], y[2], y[1], y[0]})));
            }
            const auto y = builtin_cast<llong>(xx.d);
            return binary_op(
                x, make_simd<T, N>(to_storage(builtin_type_t<llong, 2>{y[1], y[1]})))[0];
        }
    }

    // math {{{2
    // sqrt {{{3
    template <class T, size_t N> static Vc_INTRINSIC Storage<T, N> sqrt(Storage<T, N> x)
    {
               if constexpr (is_sse_ps   <T, N>()) { return _mm_sqrt_ps(x);
        } else if constexpr (is_sse_pd   <T, N>()) { return _mm_sqrt_pd(x);
        } else if constexpr (is_avx_ps   <T, N>()) { return _mm256_sqrt_ps(x);
        } else if constexpr (is_avx_pd   <T, N>()) { return _mm256_sqrt_pd(x);
        } else if constexpr (is_avx512_ps<T, N>()) { return _mm512_sqrt_ps(x);
        } else if constexpr (is_avx512_pd<T, N>()) { return _mm512_sqrt_pd(x);
        } else { assert_unreachable<T>(); }
    }

    // abs {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC Storage<T, N> abs(Storage<T, N> x) noexcept
    {
        return detail::x86::abs(x);
    }

    // trunc {{{3
    template <class T, size_t N> static Vc_INTRINSIC Storage<T, N> trunc(Storage<T, N> x)
    {
        if constexpr (is_avx512_ps<T, N>()) {
            return _mm512_roundscale_round_ps(x, 0x03, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx512_pd<T, N>()) {
            return _mm512_roundscale_round_pd(x, 0x03, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx_ps<T, N>()) {
            return _mm256_round_ps(x, 0x3);
        } else if constexpr (is_avx_pd<T, N>()) {
            return _mm256_round_pd(x, 0x3);
        } else if constexpr (have_sse4_1 && is_sse_ps<T, N>()) {
            return _mm_round_ps(x, 0x3);
        } else if constexpr (have_sse4_1 && is_sse_pd<T, N>()) {
            return _mm_round_pd(x, 0x3);
        } else if constexpr (is_sse_ps<T, N>()) {
            auto truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
            const auto no_fractional_values = builtin_cast<float>(
                builtin_cast<int>(builtin_cast<uint>(x.d) & 0x7f800000u) <
                0x4b000000);  // the exponent is so large that no mantissa bits signify
                              // fractional values (0x3f8 + 23*8 = 0x4b0)
            return x86::blend(no_fractional_values, x, truncated);
        } else if constexpr (is_sse_pd<T, N>()) {
            const auto abs_x = abs(x).d;
            const auto min_no_fractional_bits = builtin_cast<double>(
                builtin_broadcast<2>(0x4330'0000'0000'0000ull));  // 0x3ff + 52 = 0x433
            builtin_type16_t<double> truncated =
                (abs_x + min_no_fractional_bits) - min_no_fractional_bits;
            // due to rounding, the result can be too large. In this case `truncated >
            // abs(x)` holds, so subtract 1 to truncated if `abs(x) < truncated`
            truncated -=
                and_(builtin_cast<double>(abs_x < truncated), builtin_broadcast<2>(1.));
            // finally, fix the sign bit:
            return or_(
                and_(builtin_cast<double>(builtin_broadcast<2>(0x8000'0000'0000'0000ull)),
                     x),
                truncated);
        } else {
            assert_unreachable<T>();
        }
    }

    // floor {{{3
    template <class T, size_t N> static Vc_INTRINSIC Storage<T, N> floor(Storage<T, N> x)
    {
        if constexpr (is_avx512_ps<T, N>()) {
            return _mm512_roundscale_round_ps(x, 0x01, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx512_pd<T, N>()) {
            return _mm512_roundscale_round_pd(x, 0x01, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx_ps<T, N>()) {
            return _mm256_round_ps(x, 0x1);
        } else if constexpr (is_avx_pd<T, N>()) {
            return _mm256_round_pd(x, 0x1);
        } else if constexpr (have_sse4_1 && is_sse_ps<T, N>()) {
            return _mm_floor_ps(x);
        } else if constexpr (have_sse4_1 && is_sse_pd<T, N>()) {
            return _mm_floor_pd(x);
        } else {
            const auto y = trunc(x).d;
            const auto negative_input = builtin_cast<T>(x.d < builtin_broadcast<N, T>(0));
            const auto mask = andnot_(builtin_cast<T>(y == x.d), negative_input);
            return or_(andnot_(mask, y), and_(mask, y - builtin_broadcast<N, T>(1)));
        }
    }

    // ceil {{{3
    template <class T, size_t N> static Vc_INTRINSIC Storage<T, N> ceil(Storage<T, N> x)
    {
        if constexpr (is_avx512_ps<T, N>()) {
            return _mm512_roundscale_round_ps(x, 0x02, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx512_pd<T, N>()) {
            return _mm512_roundscale_round_pd(x, 0x02, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx_ps<T, N>()) {
            return _mm256_round_ps(x, 0x2);
        } else if constexpr (is_avx_pd<T, N>()) {
            return _mm256_round_pd(x, 0x2);
        } else if constexpr (have_sse4_1 && is_sse_ps<T, N>()) {
            return _mm_ceil_ps(x);
        } else if constexpr (have_sse4_1 && is_sse_pd<T, N>()) {
            return _mm_ceil_pd(x);
        } else {
            const auto y = trunc(x).d;
            const auto negative_input = builtin_cast<T>(x.d < builtin_broadcast<N, T>(0));
            const auto inv_mask = or_(builtin_cast<T>(y == x.d), negative_input);
            return or_(and_(inv_mask, y),
                       andnot_(inv_mask, y + builtin_broadcast<N, T>(1)));
        }
    }

    // isnan {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC mask_member_type<T> isnan(Storage<T, N> x)
    {
             if constexpr (is_sse_ps   <T, N>()) { return _mm_cmpunord_ps(x, x); }
        else if constexpr (is_avx_ps   <T, N>()) { return _mm256_cmp_ps(x, x, _CMP_UNORD_Q); }
        else if constexpr (is_avx512_ps<T, N>()) { return _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q); }
        else if constexpr (is_sse_pd   <T, N>()) { return _mm_cmpunord_pd(x, x); }
        else if constexpr (is_avx_pd   <T, N>()) { return _mm256_cmp_pd(x, x, _CMP_UNORD_Q); }
        else if constexpr (is_avx512_pd<T, N>()) { return _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q); }
        else { assert_unreachable<T>(); }
    }

    // isfinite {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC mask_member_type<T> isfinite(Storage<T, N> x)
    {
        return x86::cmpord(x, x.d * T());
    }

    // isunordered {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC mask_member_type<T> isunordered(Storage<T, N> x, Storage<T, N> y)
    {
        return x86::cmpunord(x, y);
    }

    // signbit {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC mask_member_type<T> signbit(Storage<T, N> x)
    {
        using I = int_for_sizeof_t<T>;
        if constexpr (have_avx512dq && is_avx512_ps<T, N>()) {
            return _mm512_movepi32_mask(to_m512i(x));
        } else if constexpr (have_avx512dq && is_avx512_pd<T, N>()) {
            return _mm512_movepi64_mask(to_m512i(x));
        } else if constexpr (sizeof(x) == 64) {
            const auto signmask = builtin_broadcast<N>(std::numeric_limits<I>::min());
            return equal_to(Storage<I, N>(builtin_cast<I>(x.d) & signmask),
                            Storage<I, N>(signmask));
        } else {
            const auto xx = builtin_cast<I>(x.d);
            constexpr I signmask = std::numeric_limits<I>::min();
            if constexpr ((sizeof(T) == 4 && (have_avx2 || sizeof(x) == 16)) ||
                          have_avx512vl) {
                (void)signmask;
                return builtin_cast<T>(xx >> std::numeric_limits<I>::digits);
            } else if constexpr ((have_avx2 || (have_ssse3 && sizeof(x) == 16))) {
                return builtin_cast<T>((xx & signmask) == signmask);
            } else {  // SSE2/3 or AVX (w/o AVX2)
                constexpr auto one = builtin_broadcast<N, T>(1);
                return builtin_cast<T>(
                    builtin_cast<T>((xx & signmask) | builtin_cast<I>(one))  // -1 or 1
                    != one);
            }
        }
    }

    // isnonzerovalue (isnormal | is subnormal == !isinf & !isnan & !is zero) {{{3
    template <class T> static Vc_INTRINSIC auto isnonzerovalue(T x)
    {
        using U = typename builtin_traits<T>::value_type;
        return x86::cmpord(x * std::numeric_limits<U>::infinity(),  // NaN if x == 0
                           x * U()                                  // NaN if x == inf
        );
    }

    template <class T> static Vc_INTRINSIC auto isnonzerovalue_mask(T x)
    {
        using U = typename builtin_traits<T>::value_type;
        constexpr size_t N = builtin_traits<T>::width;
        const auto a = x * std::numeric_limits<U>::infinity();  // NaN if x == 0
        const auto b = x * U();                                 // NaN if x == inf
        if constexpr (have_avx512vl && is_sse_ps<U, N>()) {
            return _mm_cmp_ps_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (have_avx512f && is_sse_ps<U, N>()) {
            return __mmask8(0xf &
                            _mm512_cmp_ps_mask(auto_cast(a), auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (have_avx512vl && is_sse_pd<U, N>()) {
            return _mm_cmp_pd_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (have_avx512f && is_sse_pd<U, N>()) {
            return __mmask8(0x3 &
                            _mm512_cmp_pd_mask(auto_cast(a), auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (have_avx512vl && is_avx_ps<U, N>()) {
            return _mm256_cmp_ps_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (have_avx512f && is_avx_ps<U, N>()) {
            return __mmask8(_mm512_cmp_ps_mask(auto_cast(a), auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (have_avx512vl && is_avx_pd<U, N>()) {
            return _mm256_cmp_pd_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (have_avx512f && is_avx_pd<U, N>()) {
            return __mmask8(0xf &
                            _mm512_cmp_pd_mask(auto_cast(a), auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (is_avx512_ps<U, N>()) {
            return _mm512_cmp_ps_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (is_avx512_pd<U, N>()) {
            return _mm512_cmp_pd_mask(a, b, _CMP_ORD_Q);
        } else {
            assert_unreachable<T>();
        }
    }

    // isinf {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC mask_member_type<T> isinf(Storage<T, N> x)
    {
        if constexpr (is_avx512_pd<T, N>()) {
            if constexpr (have_avx512dq) {
                return _mm512_fpclass_pd_mask(x, 0x08) | _mm512_fpclass_pd_mask(x, 0x10);
            } else {
                return _mm512_cmp_epi64_mask(to_m512i(x86::abs(x)),
                                             builtin_broadcast<N>(0x7ff0000000000000ll),
                                             _CMP_EQ_OQ);
            }
        } else if constexpr (is_avx512_ps<T, N>()) {
            if constexpr (have_avx512dq) {
                return _mm512_fpclass_ps_mask(x, 0x08) | _mm512_fpclass_ps_mask(x, 0x10);
            } else {
                return _mm512_cmp_epi32_mask(to_m512i(x86::abs(x)),
                                             auto_cast(builtin_broadcast<N>(0x7f800000u)),
                                             _CMP_EQ_OQ);
            }
        } else if constexpr (have_avx512dq_vl) {
            if constexpr (is_sse_pd<T, N>()) {
                return builtin_cast<double>(_mm_movm_epi64(_mm_fpclass_pd_mask(x, 0x08) |
                                                           _mm_fpclass_pd_mask(x, 0x10)));
            } else if constexpr (is_avx_pd<T, N>()) {
                return builtin_cast<double>(_mm256_movm_epi64(
                    _mm256_fpclass_pd_mask(x, 0x08) | _mm256_fpclass_pd_mask(x, 0x10)));
            } else if constexpr (is_sse_ps<T, N>()) {
                return builtin_cast<float>(_mm_movm_epi32(_mm_fpclass_ps_mask(x, 0x08) |
                                                          _mm_fpclass_ps_mask(x, 0x10)));
            } else if constexpr (is_avx_ps<T, N>()) {
                return builtin_cast<float>(_mm256_movm_epi32(
                    _mm256_fpclass_ps_mask(x, 0x08) | _mm256_fpclass_ps_mask(x, 0x10)));
            } else {
                assert_unreachable<T>();
            }
        } else {
            // compares to inf using the corresponding integer type
            return builtin_cast<T>(builtin_cast<int_for_sizeof_t<T>>(abs(x).d) ==
                                   builtin_cast<int_for_sizeof_t<T>>(builtin_broadcast<N>(
                                       std::numeric_limits<T>::infinity())));
            // alternative:
            //return builtin_cast<T>(abs(x).d > std::numeric_limits<T>::max());
        }
    }
    // isnormal {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC mask_member_type<T> isnormal(Storage<T, N> x)
    {
        // subnormals -> 0
        // 0 -> 0
        // inf -> inf
        // -inf -> inf
        // nan -> inf
        // normal value -> positive value / not 0
        return isnonzerovalue(
            and_(x.d, builtin_broadcast<N>(std::numeric_limits<T>::infinity())));
    }

    // fpclassify {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC fixed_size_storage<int, N> fpclassify(Storage<T, N> x)
    {
        if constexpr (is_avx512_pd<T, N>()) {
            // AVX512 is special because we want to use an __mmask to blend int vectors
            // (coming from double vectors). GCC doesn't allow this combination on the
            // ternary operator. Thus, resort to intrinsics:
            if constexpr (have_avx512vl) {
                auto &&b = [](int y) { return to_intrin(builtin_broadcast<N>(y)); };
                return {_mm256_mask_mov_epi32(
                    _mm256_mask_mov_epi32(
                        _mm256_mask_mov_epi32(b(FP_NORMAL), isnan(x), b(FP_NAN)),
                        isinf(x), b(FP_INFINITE)),
                    _mm512_cmp_pd_mask(
                        abs(x), builtin_broadcast<N>(std::numeric_limits<double>::min()),
                        _CMP_LT_OS),
                    _mm256_mask_mov_epi32(
                        b(FP_SUBNORMAL),
                        _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                        b(FP_ZERO)))};
            } else {
                auto &&b = [](int y) {
                    return _mm512_castsi256_si512(to_intrin(builtin_broadcast<N>(y)));
                };
                return {lo256(_mm512_mask_mov_epi32(
                    _mm512_mask_mov_epi32(
                        _mm512_mask_mov_epi32(b(FP_NORMAL), isnan(x), b(FP_NAN)),
                        isinf(x), b(FP_INFINITE)),
                    _mm512_cmp_pd_mask(
                        abs(x), builtin_broadcast<N>(std::numeric_limits<double>::min()),
                        _CMP_LT_OS),
                    _mm512_mask_mov_epi32(
                        b(FP_SUBNORMAL),
                        _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                        b(FP_ZERO))))};
            }
        } else {
            constexpr auto fp_normal =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_NORMAL));
            constexpr auto fp_nan =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_NAN));
            constexpr auto fp_infinite =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_INFINITE));
            constexpr auto fp_subnormal =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_SUBNORMAL));
            constexpr auto fp_zero =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_ZERO));

            const auto tmp = builtin_cast<llong>(
                abs(x).d < std::numeric_limits<T>::min()
                    ? (x.d == 0 ? fp_zero : fp_subnormal)
                    : x86::blend(isinf(x).d, x86::blend(isnan(x).d, fp_normal, fp_nan),
                                 fp_infinite));
            if constexpr (std::is_same_v<T, float>) {
                if constexpr (fixed_size_storage<int, N>::tuple_size == 1) {
                    return {tmp};
                } else if constexpr (fixed_size_storage<int, N>::tuple_size == 2) {
                    return {extract<0, 2>(tmp), extract<1, 2>(tmp)};
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (is_sse_pd<T, N>()) {
                static_assert(fixed_size_storage<int, N>::tuple_size == 2);
                return {_mm_cvtsi128_si32(tmp),
                        {_mm_cvtsi128_si32(_mm_unpackhi_epi64(tmp, tmp))}};
            } else if constexpr (is_avx_pd<T, N>()) {
                static_assert(fixed_size_storage<int, N>::tuple_size == 1);
                return {_mm_packs_epi32(lo128(tmp), hi128(tmp))};
            } else {
                assert_unreachable<T>();
            }
        }
    }

    // increment & decrement{{{2
    template <class T, size_t N> static Vc_INTRINSIC void increment(Storage<T, N> &x)
    {
        x = plus(x, Storage<T, N>::broadcast(T(1)));
    }
    template <class T, size_t N> static Vc_INTRINSIC void decrement(Storage<T, N> &x)
    {
        x = minus(x, Storage<T, N>::broadcast(T(1)));
    }

    // smart_reference access {{{2
    template <class T, size_t N, class U>
    static Vc_INTRINSIC void set(Storage<T, N> &v, int i, U &&x) noexcept
    {
        v.set(i, std::forward<U>(x));
    }

    // masked_assign{{{2
    template <class T, class K, size_t N>
    static Vc_INTRINSIC void masked_assign(Storage<K, N> k, Storage<T, N> &lhs,
                                           detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(k.d, lhs.d, rhs.d);
    }

    template <class T, class K, size_t N>
    static Vc_INTRINSIC void masked_assign(Storage<K, N> k, Storage<T, N> &lhs,
                                           detail::id<T> rhs)
    {
        if (__builtin_constant_p(rhs) && rhs == 0 && std::is_same<K, T>::value) {
            if constexpr (!detail::is_bitmask(k)) {
                // the andnot_ optimization only makes sense if k.d is a vector register
                lhs.d = andnot_(k.d, lhs.d);
                return;
            } else {
                // for AVX512/__mmask, a _mm512_maskz_mov is best
                lhs = detail::x86::blend(k, lhs, intrinsic_type_t<T, N>());
                return;
            }
        }
        lhs = detail::x86::blend(k.d, lhs.d, builtin_broadcast<N>(rhs));
    }

    // masked_cassign {{{2
    template <template <typename> class Op, class T, class K, size_t N>
    static Vc_INTRINSIC void masked_cassign(const Storage<K, N> k, Storage<T, N> &lhs,
                                            const detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(
            k.d, lhs.d, detail::data(Op<void>{}(make_simd(lhs), make_simd(rhs))).d);
    }

    template <template <typename> class Op, class T, class K, size_t N>
    static Vc_INTRINSIC void masked_cassign(const Storage<K, N> k, Storage<T, N> &lhs,
                                            const detail::id<T> rhs)
    {
        lhs = detail::x86::blend(
            k.d, lhs.d,
            detail::data(
                Op<void>{}(make_simd(lhs), make_simd<T, N>(builtin_broadcast<N>(rhs))))
                .d);
    }

    // masked_unary {{{2
    template <template <typename> class Op, class T, class K, size_t N>
    static Vc_INTRINSIC Storage<T, N> masked_unary(const Storage<K, N> k,
                                                            const Storage<T, N> v)
    {
        auto vv = make_simd(v);
        Op<decltype(vv)> op;
        return detail::x86::blend(k, v, detail::data(op(vv)));
    }

    //}}}2
};

// simd_mask impl {{{1
template <class Abi> struct generic_mask_impl {
    // member types {{{2
    template <class T> using type_tag = T *;
    template <class T> using simd_mask = Vc::simd_mask<T, Abi>;
    template <class T>
    using simd_member_type = typename Abi::template traits<T>::simd_member_type;
    template <class T>
    using mask_member_type = typename Abi::template traits<T>::mask_member_type;

    // masked load {{{2
    template <class T, size_t N, class F>
    static inline Storage<T, N> masked_load(Storage<T, N> merge, Storage<T, N> mask,
                                            const bool *mem, F) noexcept
    {
        if constexpr (detail::is_abi<Abi, simd_abi::__avx512_abi>()) {
            if constexpr (detail::have_avx512bw_vl) {
                if constexpr (N == 8) {
                    const auto a = _mm_mask_loadu_epi8(zero<__m128i>(), mask, mem);
                    return (merge & ~mask) | _mm_test_epi8_mask(a, a);
                } else if constexpr (N == 16) {
                    const auto a = _mm_mask_loadu_epi8(zero<__m128i>(), mask, mem);
                    return (merge & ~mask) | _mm_test_epi8_mask(a, a);
                } else if constexpr (N == 32) {
                    const auto a = _mm256_mask_loadu_epi8(zero<__m256i>(), mask, mem);
                    return (merge & ~mask) | _mm256_test_epi8_mask(a, a);
                } else if constexpr (N == 64) {
                    const auto a = _mm512_mask_loadu_epi8(zero<__m512i>(), mask, mem);
                    return (merge & ~mask) | _mm512_test_epi8_mask(a, a);
                } else {
                    assert_unreachable<T>();
                }
            } else {
                detail::bit_iteration(mask, [&](auto i) { merge.set(i, mem[i]); });
                return merge;
            }
        } else if constexpr (have_avx512bw_vl && N == 32 && sizeof(T) == 1) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(
                _mm256_mask_sub_epi8(to_m256i(merge), k, __m256i(),
                                     _mm256_mask_loadu_epi8(__m256i(), k, mem)));
        } else if constexpr (have_avx512bw_vl && N == 16 && sizeof(T) == 1) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm_mask_sub_epi8(to_m128i(merge), k, __m128i(),
                                                 _mm_mask_loadu_epi8(__m128i(), k, mem)));
        } else if constexpr (have_avx512bw_vl && N == 16 && sizeof(T) == 2) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm256_mask_sub_epi16(
                to_m256i(merge), k, __m256i(),
                _mm256_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 8 && sizeof(T) == 2) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm_mask_sub_epi16(
                to_m128i(merge), k, __m128i(),
                _mm_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 8 && sizeof(T) == 4) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm256_mask_sub_epi32(
                to_m256i(merge), k, __m256i(),
                _mm256_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 4 && sizeof(T) == 4) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm_mask_sub_epi32(
                to_m128i(merge), k, __m128i(),
                _mm_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 4 && sizeof(T) == 8) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm256_mask_sub_epi64(
                to_m256i(merge), k, __m256i(),
                _mm256_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 2 && sizeof(T) == 8) {
            const auto k = convert_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm_mask_sub_epi64(
                to_m128i(merge), k, __m128i(),
                _mm_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else {
            // AVX(2) has 32/64 bit maskload, but nothing at 8 bit granularity
            auto tmp = storage_bitcast<detail::int_for_sizeof_t<T>>(merge);
            detail::bit_iteration(mask_to_int<N>(mask),
                                  [&](auto i) { tmp.set(i, -mem[i]); });
            merge = storage_bitcast<T>(tmp);
        }
        return merge;
    }

    // store {{{2
    template <class F> static Vc_INTRINSIC void store4(int x, bool *mem, F)
    {
        if constexpr (detail::is_aligned_v<F, 4>) {
            *reinterpret_cast<may_alias<int> *>(mem) = x;
        } else {
            std::memcpy(mem, &x, 4);
        }
    }
    template <class T, size_t N, class F>
    static Vc_INTRINSIC void store(Storage<T, N> v, bool *mem, F) noexcept
    {
        if constexpr (detail::is_abi<Abi, simd_abi::__sse_abi>()) {
            if constexpr (N == 2 && have_sse2) {
                const auto k = builtin_cast<int>(v.d);
                mem[0] = -k[1];
                mem[1] = -k[3];
            } else if constexpr (N == 4 && have_sse2) {
                const unsigned bool4 =
                    builtin_cast<uint>(_mm_packs_epi16(
                        _mm_packs_epi32(to_m128i(v), __m128i()), __m128i()))[0] &
                    0x01010101u;
                std::memcpy(mem, &bool4, 4);
            } else if constexpr (std::is_same_v<T, float> && have_mmx) {
                const __m128 k(v);
                const __m64 kk = _mm_cvtps_pi8(and_(k, _mm_set1_ps(1.f)));
                builtin_store<4>(kk, mem, F());
                _mm_empty();
            } else if constexpr (N == 8 && have_sse2) {
                builtin_store<8>(
                    _mm_packs_epi16(to_intrin(builtin_cast<ushort>(v.d) >> 15),
                                    __m128i()),
                    mem, F());
            } else if constexpr (N == 16 && have_sse2) {
                builtin_store(v.d & 1, mem, F());
            } else {
                assert_unreachable<T>();
            }
        } else if constexpr (detail::is_abi<Abi, simd_abi::__avx_abi>()) {
            if constexpr (N == 4 && have_avx) {
                auto k = to_m256i(v);
                int bool4;
                if constexpr (have_avx2) {
                    bool4 = _mm256_movemask_epi8(k);
                } else {
                    bool4 = (_mm_movemask_epi8(lo128(k)) |
                             (_mm_movemask_epi8(hi128(k)) << 16));
                }
                bool4 &= 0x01010101;
                std::memcpy(mem, &bool4, 4);
            } else if constexpr (N == 8 && have_avx) {
                const auto k = to_m256i(v);
                const auto k2 = _mm_srli_epi16(_mm_packs_epi16(lo128(k), hi128(k)), 15);
                const auto k3 = _mm_packs_epi16(k2, __m128i());
                builtin_store<8>(k3, mem, F());
            } else if constexpr (N == 16 && have_avx2) {
                const auto x = _mm256_srli_epi16(v, 15);
                const auto bools = _mm_packs_epi16(lo128(x), hi128(x));
                builtin_store<16>(bools, mem, F());
            } else if constexpr (N == 16 && have_avx) {
                const auto bools = 1 & builtin_cast<uchar>(_mm_packs_epi16(
                                           lo128(v.intrin()), hi128(v.intrin())));
                builtin_store<16>(bools, mem, F());
            } else if constexpr (N == 32 && have_avx) {
                builtin_store<32>(1 & v.d, mem, F());
            } else {
                assert_unreachable<T>();
            }
        } else if constexpr (detail::is_abi<Abi, simd_abi::__avx512_abi>()) {
            if constexpr (N == 8) {
                builtin_store<8>(
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
                    _mm_maskz_set1_epi8(v, 1),
#elif defined __x86_64__
                    make_storage<ullong>(_pdep_u64(v, 0x0101010101010101ULL), 0ull),
#else
                    make_storage<uint>(_pdep_u32(v, 0x01010101U),
                                       _pdep_u32(v >> 4, 0x01010101U)),
#endif
                    mem, F());
            } else if constexpr (N == 16 && detail::have_avx512bw_vl) {
                builtin_store(_mm_maskz_set1_epi8(v, 1), mem, F());
            } else if constexpr (N == 16 && detail::have_avx512f) {
                _mm512_mask_cvtepi32_storeu_epi8(mem, ~__mmask16(),
                                                 _mm512_maskz_set1_epi32(v, 1));
            } else if constexpr (N == 32 && detail::have_avx512bw_vl) {
                builtin_store(_mm256_maskz_set1_epi8(v, 1), mem, F());
            } else if constexpr (N == 32 && detail::have_avx512bw) {
                builtin_store(lo256(_mm512_maskz_set1_epi8(v, 1)), mem, F());
            } else if constexpr (N == 64 && detail::have_avx512bw) {
                builtin_store(_mm512_maskz_set1_epi8(v, 1), mem, F());
            } else {
                assert_unreachable<T>();
            }
        } else {
            assert_unreachable<T>();
        }
    }

    // masked store {{{2
    template <class T, size_t N, class F>
    static inline void masked_store(const Storage<T, N> v, bool *mem, F,
                                    const Storage<T, N> k) noexcept
    {
        if constexpr (detail::is_abi<Abi, simd_abi::__avx512_abi>()) {
            if constexpr (N == 8 && detail::have_avx512bw_vl) {
                _mm_mask_cvtepi16_storeu_epi8(mem, k, _mm_maskz_set1_epi16(v, 1));
            } else if constexpr (N == 8 && detail::have_avx512vl) {
                _mm256_mask_cvtepi32_storeu_epi8(mem, k, _mm256_maskz_set1_epi32(v, 1));
            } else if constexpr (N == 8) {
                // we rely on k < 0x100:
                _mm512_mask_cvtepi32_storeu_epi8(mem, k, _mm512_maskz_set1_epi32(v, 1));
            } else if constexpr (N == 16 && detail::have_avx512bw_vl) {
                _mm_mask_storeu_epi8(mem, k, _mm_maskz_set1_epi8(v, 1));
            } else if constexpr (N == 16) {
                _mm512_mask_cvtepi32_storeu_epi8(mem, k, _mm512_maskz_set1_epi32(v, 1));
            } else if constexpr (N == 32 && detail::have_avx512bw_vl) {
                _mm256_mask_storeu_epi8(mem, k, _mm256_maskz_set1_epi8(v, 1));
            } else if constexpr (N == 32 && detail::have_avx512bw) {
                _mm256_mask_storeu_epi8(mem, k, lo256(_mm512_maskz_set1_epi8(v, 1)));
            } else if constexpr (N == 64 && detail::have_avx512bw) {
                _mm512_mask_storeu_epi8(mem, k, _mm512_maskz_set1_epi8(v, 1));
            } else {
                assert_unreachable<T>();
            }
        } else {
            detail::bit_iteration(mask_to_int<N>(k), [&](auto i) { mem[i] = v[i]; });
        }
    }

    // from_bitset{{{2
    template <size_t N, class T>
    static Vc_INTRINSIC mask_member_type<T> from_bitset(std::bitset<N> bits, type_tag<T>)
    {
        return convert_mask<typename mask_member_type<T>::register_type>(bits);
    }

    // logical and bitwise operators {{{2
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> logical_and(const Storage<T, N> &x,
                                                            const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d & y.d;
        } else {
            return detail::and_(x.d, y.d);
        }
    }

    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> logical_or(const Storage<T, N> &x,
                                                           const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d | y.d;
        } else {
            return detail::or_(x.d, y.d);
        }
    }

    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> bit_and(const Storage<T, N> &x,
                                                        const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d & y.d;
        } else {
            return detail::and_(x.d, y.d);
        }
    }

    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> bit_or(const Storage<T, N> &x,
                                                       const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d | y.d;
        } else {
            return detail::or_(x.d, y.d);
        }
    }

    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> bit_xor(const Storage<T, N> &x,
                                                        const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d ^ y.d;
        } else {
            return detail::xor_(x.d, y.d);
        }
    }

    // smart_reference access {{{2
    template <class T, size_t N> static void set(Storage<T, N> &k, int i, bool x) noexcept
    {
        if constexpr (std::is_same_v<T, bool>) {
            k.set(i, x);
        } else {
            using int_t = builtin_type_t<int_for_sizeof_t<T>, N>;
            auto tmp = reinterpret_cast<int_t>(k.d);
            tmp[i] = -x;
            k.d = auto_cast(tmp);
        }
    }
    // masked_assign{{{2
    template <class T, size_t N>
    static Vc_INTRINSIC void masked_assign(Storage<T, N> k, Storage<T, N> &lhs,
                                           detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(k.d, lhs.d, rhs.d);
    }

    template <class T, size_t N>
    static Vc_INTRINSIC void masked_assign(Storage<T, N> k, Storage<T, N> &lhs, bool rhs)
    {
        if (__builtin_constant_p(rhs)) {
            if (rhs == false) {
                lhs = andnot_(k.d, lhs.d);
            } else {
                lhs = or_(k.d, lhs.d);
            }
            return;
        }
        lhs = detail::x86::blend(k, lhs, detail::data(simd_mask<T>(rhs)));
    }

    //}}}2
};

//}}}1
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END


#endif  // VC_SIMD_GENERICIMPL_H_
