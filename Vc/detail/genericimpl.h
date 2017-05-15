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

    template <class T, class A, class U>
    static Vc_INTRINSIC Vc::datapar<T, A> make_datapar(const U &x)
    {
        using traits = typename Vc::datapar<T, A>::traits;
        using V = typename traits::datapar_member_type;
        return {private_init, static_cast<V>(x)};
    }

    // generator {{{2
    template <class F, class T, size_t N>
    static Vc_INTRINSIC Storage<T, N> generator(F &&gen, type_tag<T>, size_tag<N>)
    {
        return detail::generate_from_n_evaluations<N, Storage<T, N>>(
            [&gen](auto element_idx_) { return gen(element_idx_); });
    }

    // complement {{{2
    template <class T, class A>
    static Vc_INTRINSIC Vc::datapar<T, A> complement(const Vc::datapar<T, A> &x) noexcept
    {
        using detail::x86::complement;
        return make_datapar<T, A>(complement(adjust_for_long(detail::data(x))));
    }

    // unary minus {{{2
    template <class T, class A>
    static Vc_INTRINSIC Vc::datapar<T, A> unary_minus(const Vc::datapar<T, A> &x) noexcept
    {
        using detail::x86::unary_minus;
        return make_datapar<T, A>(unary_minus(adjust_for_long(detail::data(x))));
    }

    // arithmetic operators {{{2
#define Vc_ARITHMETIC_OP_(name_)                                                         \
    template <class T, class A>                                                          \
    static Vc_INTRINSIC datapar<T, A> Vc_VDECL name_(datapar<T, A> x, datapar<T, A> y)   \
    {                                                                                    \
        return make_datapar<T, A>(                                                       \
            detail::name_(adjust_for_long(x.d), adjust_for_long(y.d)));                  \
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

    // increment & decrement{{{2
    template <class T, size_t N> static Vc_INTRINSIC void increment(Storage<T, N> &x)
    {
        x = detail::plus(x, Storage<T, N>(Derived::broadcast(T(1), size_tag<N>())));
    }
    template <size_t N> static Vc_INTRINSIC void increment(Storage<long, N> &x)
    {
        x = detail::plus(adjust_for_long(x), Storage<equal_int_type_t<long>, N>(
                                                 Derived::broadcast(1L, size_tag<N>())));
    }
    template <size_t N> static Vc_INTRINSIC void increment(Storage<ulong, N> &x)
    {
        x = detail::plus(adjust_for_long(x), Storage<equal_int_type_t<ulong>, N>(
                                                 Derived::broadcast(1L, size_tag<N>())));
    }

    template <class T, size_t N> static Vc_INTRINSIC void decrement(Storage<T, N> &x)
    {
        x = detail::minus(x, Storage<T, N>(Derived::broadcast(T(1), size_tag<N>())));
    }
    template <size_t N> static Vc_INTRINSIC void decrement(Storage<long, N> &x)
    {
        x = detail::minus(adjust_for_long(x), Storage<equal_int_type_t<long>, N>(
                                                  Derived::broadcast(1L, size_tag<N>())));
    }
    template <size_t N> static Vc_INTRINSIC void decrement(Storage<ulong, N> &x)
    {
        x = detail::minus(adjust_for_long(x), Storage<equal_int_type_t<ulong>, N>(
                                                  Derived::broadcast(1L, size_tag<N>())));
    }
};
// mask impl {{{1
template <class abi, template <class> class mask_member_type> struct generic_mask_impl {
    // member types {{{2
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;
    template <class T> using mask = Vc::mask<T, abi>;

    // masked load {{{2
    template <class T, class F>
    static inline void Vc_VDECL masked_load(mask<T> &merge, mask<T> mask, const bool *mem,
                                            F) noexcept
    {
        constexpr auto N = datapar_size_v<T, abi>;
        detail::execute_n_times<N>([&](auto i) {
            if (detail::data(mask)[i]) {
                detail::data(merge).set(i, MaskBool<sizeof(T)>{mem[i]});
            }
        });
    }

    // masked store {{{2
    template <class T, class F>
    static inline void Vc_VDECL masked_store(mask<T> v, bool *mem, F, mask<T> k) noexcept
    {
        constexpr auto N = datapar_size_v<T, abi>;
        detail::execute_n_times<N>([&](auto i) {
            if (detail::data(k)[i]) {
                mem[i] = detail::data(v)[i];
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
        return x86::movemask(_mm_packs_epi16(v, zero<__m128i>()));
    }
#ifdef Vc_HAVE_AVX2
    template <class T>
    static Vc_INTRINSIC std::bitset<16> to_bitset(Storage<T, 16> v,
                                                 std::integral_constant<int, 2>) noexcept
    {
        return x86::movemask(_mm_packs_epi16(x86::lo128(v), x86::hi128(v)));
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
};
//}}}1
}  // namespace detail

// where implementation {{{1
template <class T, class A>
inline void Vc_VDECL masked_assign(mask<T, A> k, datapar<T, A> &lhs,
                                   const detail::id<datapar<T, A>> &rhs)
{
    detail::data(lhs) =
        detail::x86::blend(detail::data(k), detail::data(lhs), detail::data(rhs));
}

template <class T, class A>
inline void Vc_VDECL masked_assign(mask<T, A> k, mask<T, A> &lhs,
                                   const detail::id<mask<T, A>> &rhs)
{
    detail::data(lhs) =
        detail::x86::blend(detail::data(k), detail::data(lhs), detail::data(rhs));
}

template <template <typename> class Op, typename T, class A,
          int = 1  // the int parameter is used to disambiguate the function template
                   // specialization for the avx512 ABI
          >
inline void Vc_VDECL masked_cassign(mask<T, A> k, datapar<T, A> &lhs,
                                    const datapar<T, A> &rhs)
{
    detail::data(lhs) = detail::x86::blend(detail::data(k), detail::data(lhs),
                                           detail::data(Op<void>{}(lhs, rhs)));
}

template <template <typename> class Op, typename T, class A, class U>
inline enable_if<std::is_convertible<U, datapar<T, A>>::value, void> Vc_VDECL
masked_cassign(mask<T, A> k, datapar<T, A> &lhs, const U &rhs)
{
    masked_cassign<Op>(k, lhs, datapar<T, A>(rhs));
}

template <template <typename> class Op, typename T, class A,
          int = 1  // the int parameter is used to disambiguate the function template
                   // specialization for the avx512 ABI
          >
inline datapar<T, A> Vc_VDECL masked_unary(mask<T, A> k, datapar<T, A> v)
{
    Op<datapar<T, A>> op;
    return static_cast<datapar<T, A>>(
        detail::x86::blend(detail::data(k), detail::data(v), detail::data(op(v))));
}

//}}}1
Vc_VERSIONED_NAMESPACE_END


#endif  // VC_DATAPAR_GENERICIMPL_H_
