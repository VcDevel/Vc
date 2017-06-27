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

#ifndef VC_DATAPAR_GENERICIMPL_H_
#define VC_DATAPAR_GENERICIMPL_H_

#include "detail.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
// datapar impl {{{1
template <class Derived> struct generic_datapar_impl {
    // member types {{{2
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;

    template <class T, size_t N>
    static Vc_INTRINSIC auto Vc_VDECL datapar(Storage<T, N> x)
    {
        return Derived::make_datapar(x);
    }

    // adjust_for_long{{{2
    template <size_t Size>
    static Vc_INTRINSIC Storage<equal_int_type_t<long>, Size> Vc_VDECL
    adjust_for_long(Storage<long, Size> x)
    {
        return {x.v()};
    }
    template <size_t Size>
    static Vc_INTRINSIC Storage<equal_int_type_t<ulong>, Size> Vc_VDECL
    adjust_for_long(Storage<ulong, Size> x)
    {
        return {x.v()};
    }
    template <class T, size_t Size>
    static Vc_INTRINSIC const Storage<T, Size> &adjust_for_long(const Storage<T, Size> &x)
    {
        return x;
    }

    // generator {{{2
    template <class F, class T, size_t N>
    static Vc_INTRINSIC Storage<T, N> generator(F &&gen, type_tag<T>, size_tag<N>)
    {
        return detail::generate_from_n_evaluations<N, Storage<T, N>>(
            [&gen](auto element_idx_) { return gen(element_idx_); });
    }

    // complement {{{2
    template <class T, size_t N>
    static Vc_INTRINSIC Storage<T, N> complement(Storage<T, N> x) noexcept
    {
        return static_cast<typename Storage<T, N>::VectorType>(
            detail::x86::complement(adjust_for_long(x)));
    }

    // unary minus {{{2
    template <class T, size_t N>
    static Vc_INTRINSIC Storage<T, N> unary_minus(Storage<T, N> x) noexcept
    {
        using detail::x86::unary_minus;
        return static_cast<typename Storage<T, N>::VectorType>(
            unary_minus(adjust_for_long(x)));
    }

    // arithmetic operators {{{2
#define Vc_ARITHMETIC_OP_(name_)                                                         \
    template <size_t N>                                                                  \
    static Vc_INTRINSIC Storage<long, N> Vc_VDECL name_(Storage<long, N> x,              \
                                                        Storage<long, N> y)              \
    {                                                                                    \
        using Adjusted = detail::Storage<equal_int_type_t<long>, N>;                     \
        return static_cast<typename Adjusted::VectorType>(                               \
            detail::name_(Adjusted(x.v()), Adjusted(y.v())));                            \
    }                                                                                    \
    template <size_t N>                                                                  \
    static Vc_INTRINSIC Storage<unsigned long, N> Vc_VDECL name_(                        \
        Storage<unsigned long, N> x, Storage<unsigned long, N> y)                        \
    {                                                                                    \
        using Adjusted = detail::Storage<equal_int_type_t<unsigned long>, N>;            \
        return static_cast<typename Adjusted::VectorType>(                               \
            detail::name_(Adjusted(x.v()), Adjusted(y.v())));                            \
    }                                                                                    \
    template <class T, size_t N>                                                         \
    static Vc_INTRINSIC Storage<T, N> Vc_VDECL name_(Storage<T, N> x, Storage<T, N> y)   \
    {                                                                                    \
        return detail::name_(x, y);                                                      \
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
    static Vc_INTRINSIC Storage<T, N> Vc_VDECL bit_shift_left(Storage<T, N> x, int y)
    {
        return static_cast<typename Storage<T, N>::VectorType>(
            detail::bit_shift_left(adjust_for_long(x), y));
    }
    template <class T, size_t N>
    static Vc_INTRINSIC Storage<T, N> Vc_VDECL bit_shift_right(Storage<T, N> x, int y)
    {
        return static_cast<typename Storage<T, N>::VectorType>(
            detail::bit_shift_right(adjust_for_long(x), y));
    }

    // sqrt {{{2
    template <class T, size_t N>
    static Vc_INTRINSIC Storage<T, N> sqrt(Storage<T, N> x) noexcept
    {
        using detail::x86::sqrt;
        return sqrt(adjust_for_long(x));
    }

    // abs {{{2
    template <class T, size_t N>
    static Vc_INTRINSIC Storage<T, N> abs(Storage<T, N> x) noexcept
    {
        using detail::x86::abs;
        return abs(adjust_for_long(x));
    }

    // increment & decrement{{{2
    template <class T, size_t N> static Vc_INTRINSIC void increment(Storage<T, N> &x)
    {
        x = plus(x, Storage<T, N>(Derived::broadcast(T(1), size_tag<N>())));
    }
    template <class T, size_t N> static Vc_INTRINSIC void decrement(Storage<T, N> &x)
    {
        x = minus(x, Storage<T, N>(Derived::broadcast(T(1), size_tag<N>())));
    }

    // masked_assign{{{2
    template <class T, class K, size_t N>
    static Vc_INTRINSIC void Vc_VDECL masked_assign(Storage<K, N> k, Storage<T, N> &lhs,
                                             detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(k, lhs, rhs);
    }
    template <class T, class K, size_t N>
    static Vc_INTRINSIC void Vc_VDECL masked_assign(Storage<K, N> k, Storage<T, N> &lhs,
                                                    detail::id<T> rhs)
    {
#ifdef __GNUC__
        if (__builtin_constant_p(rhs) && rhs == 0 && std::is_same<K, T>::value) {
            lhs = x86::andnot_(k, lhs);
            return;
        }
#endif  // __GNUC__
        lhs =
            detail::x86::blend(k, lhs, x86::broadcast(rhs, size_constant<sizeof(lhs)>()));
    }

    // masked_cassign {{{2
    template <template <typename> class Op, class T, class K, size_t N>
    static Vc_INTRINSIC void Vc_VDECL masked_cassign(const Storage<K, N> k,
                                                     Storage<T, N> &lhs,
                                                     const detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(k, lhs,
                                 detail::data(Op<void>{}(datapar(lhs), datapar(rhs))));
    }

    template <template <typename> class Op, class T, class K, size_t N>
    static Vc_INTRINSIC void Vc_VDECL masked_cassign(const Storage<K, N> k,
                                                     Storage<T, N> &lhs,
                                                     const detail::id<T> rhs)
    {
        lhs = detail::x86::blend(
            k, lhs,
            detail::data(Op<void>{}(
                datapar(lhs), datapar<T, N>(Derived::broadcast(rhs, size_tag<N>())))));
    }

    // masked_unary {{{2
    template <template <typename> class Op, class T, class K, size_t N>
    static Vc_INTRINSIC Storage<T, N> Vc_VDECL masked_unary(const Storage<K, N> k,
                                                            const Storage<T, N> v)
    {
        auto vv = datapar(v);
        Op<decltype(vv)> op;
        return detail::x86::blend(k, v, detail::data(op(vv)));
    }

    //}}}2
};

// mask impl {{{1
template <class abi, template <class> class mask_member_type> struct generic_mask_impl {
    // member types {{{2
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;
    template <class T> using mask = Vc::mask<T, abi>;

    // masked load {{{2
    template <class T, size_t N, class F>
    static inline void Vc_VDECL masked_load(Storage<T, N> &merge, Storage<T, N> mask,
                                            const bool *mem, F) noexcept
    {
        detail::execute_n_times<N>([&](auto i) {
            if (mask[i]) {
                merge.set(i, MaskBool<sizeof(T)>{mem[i]});
            }
        });
    }

    // masked store {{{2
    template <class T, size_t N, class F>
    static inline void Vc_VDECL masked_store(const Storage<T, N> v, bool *mem, F,
                                             const Storage<T, N> k) noexcept
    {
        detail::execute_n_times<N>([&](auto i) {
            if (k[i]) {
                mem[i] = v[i];
            }
        });
    }

    // to_bitset {{{2
    template <class T, size_t N>
    static Vc_INTRINSIC std::bitset<N> to_bitset(Storage<T, N> v,
                                                 std::integral_constant<int, 1>) noexcept
    {
        return x86::movemask(v);
    }
    template <class T>
    static Vc_INTRINSIC std::bitset<8> to_bitset(Storage<T, 8> v,
                                                 std::integral_constant<int, 2>) noexcept
    {
        return x86::movemask_epi16(v);
    }

#ifdef Vc_HAVE_AVX2
    template <class T>
    static Vc_INTRINSIC std::bitset<16> to_bitset(Storage<T, 16> v,
                                                 std::integral_constant<int, 2>) noexcept
    {
        return x86::movemask_epi16(v);
    }
#endif  // Vc_HAVE_AVX2

    template <class T, size_t N>
    static Vc_INTRINSIC std::bitset<N> to_bitset(Storage<T, N> v,
                                                 std::integral_constant<int, 4>) noexcept
    {
        return x86::movemask(Storage<float, N>(v.v()));
    }
    template <class T, size_t N>
    static Vc_INTRINSIC std::bitset<N> to_bitset(Storage<T, N> v,
                                                 std::integral_constant<int, 8>) noexcept
    {
        return x86::movemask(Storage<double, N>(v.v()));
    }
    template <class T, size_t N>
    static Vc_INTRINSIC std::bitset<N> to_bitset(Storage<T, N> v) noexcept
    {
        static_assert(N <= sizeof(uint) * CHAR_BIT,
                      "Needs missing 64-bit implementation");
        if (std::is_integral<T>::value && sizeof(T) > 1) {
#if 0 //defined Vc_HAVE_BMI2
            switch (sizeof(T)) {
            case 2: return _pext_u32(x86::movemask(v), 0xaaaaaaaa);
            case 4: return _pext_u32(x86::movemask(v), 0x88888888);
            case 8: return _pext_u32(x86::movemask(v), 0x80808080);
            default: Vc_UNREACHABLE();
            }
#else
            return to_bitset(v, std::integral_constant<int, sizeof(T)>());
#endif
        } else {
            return x86::movemask(v);
        }
    }

    // from_bitset{{{2
    template <size_t N, class T>
    static Vc_INTRINSIC mask_member_type<T> from_bitset(std::bitset<N> bits, type_tag<T>)
    {
#ifdef Vc_HAVE_AVX512BW
        if (sizeof(T) <= 2u) {
            return detail::intrin_cast<detail::intrinsic_type<T, N>>(
                x86::convert_mask<sizeof(T), sizeof(mask_member_type<T>)>(bits));
        }
#endif  // Vc_HAVE_AVX512BW
#ifdef Vc_HAVE_AVX512DQ
        if (sizeof(T) >= 4u) {
            return detail::intrin_cast<detail::intrinsic_type<T, N>>(
                x86::convert_mask<sizeof(T), sizeof(mask_member_type<T>)>(bits));
        }
#endif  // Vc_HAVE_AVX512DQ
        using U = std::conditional_t<sizeof(T) == 8, ullong,
                  std::conditional_t<sizeof(T) == 4, uint,
                  std::conditional_t<sizeof(T) == 2, ushort,
                  std::conditional_t<sizeof(T) == 1, uchar, void>>>>;
        using V = datapar<U, abi>;
        constexpr size_t bits_per_element = sizeof(U) * CHAR_BIT;
        if (bits_per_element >= N) {
            V tmp(static_cast<U>(bits.to_ullong()));                  // broadcast
            tmp &= V([](auto i) { return static_cast<U>(1 << i); });  // mask bit index
            return detail::intrin_cast<detail::intrinsic_type<T, N>>(
                detail::data(tmp != V()));
        } else {
            V tmp([&](auto i) {
                return static_cast<U>(bits.to_ullong() >>
                                      (bits_per_element * (i / bits_per_element)));
            });
            tmp &= V([](auto i) {
                return static_cast<U>(1 << (i % bits_per_element));
            });  // mask bit index
            return detail::intrin_cast<detail::intrinsic_type<T, N>>(
                detail::data(tmp != V()));
        }
    }

    // masked_assign{{{2
    template <class T, size_t N>
    static Vc_INTRINSIC void Vc_VDECL masked_assign(Storage<T, N> k, Storage<T, N> &lhs,
                                                    detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(k, lhs, rhs);
    }
    template <class T, size_t N>
    static Vc_INTRINSIC void Vc_VDECL masked_assign(Storage<T, N> k, Storage<T, N> &lhs,
                                                    bool rhs)
    {
#ifdef __GNUC__
        if (__builtin_constant_p(rhs)) {
            if (rhs == false) {
                lhs = x86::andnot_(k, lhs);
            } else {
                lhs = x86::or_(k, lhs);
            }
            return;
        }
#endif  // __GNUC__
        lhs = detail::x86::blend(k, lhs, detail::data(mask<T>(rhs)));
    }

    //}}}2
};

//}}}1
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END


#endif  // VC_DATAPAR_GENERICIMPL_H_
