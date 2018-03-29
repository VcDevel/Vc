/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SIMD_NEON_H_
#define VC_SIMD_NEON_H_

#include "macros.h"
#ifdef Vc_HAVE_NEON
#include "storage.h"
#include "aarch/intrinsics.h"
#include "aarch/convert.h"
#include "aarch/arithmetics.h"
#include "maskbool.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
struct neon_mask_impl;
struct neon_simd_impl;

template <class T> using neon_simd_member_type = Storage<T, 16 / sizeof(T)>;
template <class T> using neon_mask_member_type = Storage<T, 16 / sizeof(T)>;

template <class T> struct traits<T, simd_abi::neon> {
    static_assert(sizeof(T) <= 8,
                  "NEON can only implement operations on element types with sizeof <= 8");

    using simd_member_type = neon_simd_member_type<T>;
    using simd_impl_type = neon_simd_impl;
    static constexpr size_t simd_member_alignment = alignof(simd_member_type);
    using simd_cast_type = typename simd_member_type::register_type;
    struct simd_base {
        explicit operator simd_cast_type() const
        {
            return data(*static_cast<const simd<T, simd_abi::neon> *>(this));
        }
    };

    using mask_member_type = neon_mask_member_type<T>;
    using mask_impl_type = neon_mask_impl;
    static constexpr size_t mask_member_alignment = alignof(mask_member_type);
    using mask_cast_type = typename mask_member_type::register_type;
    struct mask_base {
        explicit operator typename mask_member_type::register_type() const
        {
            return data(*static_cast<const simd_mask<T, simd_abi::neon> *>(this));
        }
    };
};
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
// simd impl {{{1
template <class Derived> struct generic_simd_impl {
    // member types {{{2
    template <size_t N> using size_tag = std::integral_constant<size_t, N>;

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
    static Vc_INTRINSIC Vc::simd<T, A> make_simd(const U &x)
    {
        using traits = typename Vc::simd<T, A>::traits;
        using V = typename traits::simd_member_type;
        return {private_init, static_cast<V>(x)};
    }

    // complement {{{2
    template <class T, class A>
    static Vc_INTRINSIC Vc::simd<T, A> complement(const Vc::simd<T, A> &x) noexcept
    {
        using detail::aarch::complement;
        return make_simd<T, A>(complement(adjust_for_long(detail::data(x))));
    }

    // unary minus {{{2
    template <class T, class A>
    static Vc_INTRINSIC Vc::simd<T, A> unary_minus(const Vc::simd<T, A> &x) noexcept
    {
        using detail::aarch::unary_minus;
        return make_simd<T, A>(unary_minus(adjust_for_long(detail::data(x))));
    }
    // arithmetic operators {{{2
#define Vc_ARITHMETIC_OP_(name_)                                                         \
    template <class T, class A>                                                          \
    static Vc_INTRINSIC simd<T, A> Vc_VDECL name_(simd<T, A> x, simd<T, A> y)   \
    {                                                                                    \
        return make_simd<T, A>(                                                       \
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
    //  increment & decrement{{{2
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
//}}}1
}  // namespace detail

// where implementation {{{1
template <class T, class A>
inline void Vc_VDECL masked_assign(simd_mask<T, A> k, simd<T, A> &lhs,
                                   const detail::id<simd<T, A>> &rhs)
{
    lhs = static_cast<simd<T, A>>(
        detail::aarch::blend(detail::data(k), detail::data(lhs), detail::data(rhs)));
}

template <class T, class A>
inline void Vc_VDECL masked_assign(simd_mask<T, A> k, simd_mask<T, A> &lhs,
                                   const detail::id<simd_mask<T, A>> &rhs)
{
    lhs = static_cast<simd_mask<T, A>>(
        detail::aarch::blend(detail::data(k), detail::data(lhs), detail::data(rhs)));
}

template <template <typename> class Op, typename T, class A,
          int = 1>
inline void Vc_VDECL masked_cassign(simd_mask<T, A> k, simd<T, A> &lhs,
                                    const simd<T, A> &rhs)
{
    lhs = static_cast<simd<T, A>>(detail::aarch::blend(
        detail::data(k), detail::data(lhs), detail::data(Op<void>{}(lhs, rhs))));
}

template <template <typename> class Op, typename T, class A, class U>
inline enable_if<std::is_convertible<U, simd<T, A>>::value, void> Vc_VDECL
masked_cassign(simd_mask<T, A> k, simd<T, A> &lhs, const U &rhs)
{
    masked_cassign<Op>(k, lhs, simd<T, A>(rhs));
}

template <template <typename> class Op, typename T, class A,
          int = 1>
inline simd<T, A> Vc_VDECL masked_unary(simd_mask<T, A> k, simd<T, A> v)
{
    Op<simd<T, A>> op;
    return static_cast<simd<T, A>>(
        detail::aarch::blend(detail::data(k), detail::data(v), detail::data(op(v))));
}

//}}}1
Vc_VERSIONED_NAMESPACE_END

#ifdef Vc_HAVE_NEON_ABI
Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// simd impl {{{1
struct neon_simd_impl : public generic_simd_impl<neon_simd_impl> {
    // member types {{{2
    using abi = simd_abi::neon;
    template <class T> static constexpr size_t size() { return simd_size_v<T, abi>; }
    template <class T> using simd_member_type = neon_simd_member_type<T>;
    template <class T> using intrinsic_type = typename simd_member_type<T>::register_type;
    template <class T> using mask_member_type = neon_mask_member_type<T>;
    template <class T> using simd = Vc::simd<T, abi>;
    template <class T> using simd_mask = Vc::simd_mask<T, abi>;
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;
    /**
    // broadcast {{{2
    static Vc_INTRINSIC intrinsic_type<float> broadcast(float x, size_tag<4>) noexcept
    {
        return vld1q_dup_f32(x);
    }
#ifdef Vc_HAVE_AARCH64
    static Vc_INTRINSIC intrinsic_type<double> broadcast(double x, size_tag<2>) noexcept
    {
        return vld1q_dub_f64(x);
    }
#endif

    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<2>) noexcept
    {
        return vld1_dub_f64(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<4>) noexcept
    {
        return vld1_dub_f32(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<8>) noexcept
    {
        return vld1_dub_f16(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<16>) noexcept
    {
        return vld1_dub_f8(x);
    }
    */
    //  load {{{2
    //  from long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC simd_member_type<T> load(const long double *mem, F,
                                                    type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        return generate_from_n_evaluations<size<T>(), simd_member_type<T>>(
            [&](auto i) { return static_cast<T>(mem[i]); });
    }
     // load without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(const T *mem, F f, type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        return detail::load16(mem, f);
    }
    // store {{{2
    // store to long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC void store(simd_member_type<T> v, long double *mem, F,
                                   type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        // alignment F doesn't matter
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }
    // store without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC void store(simd_member_type<T> v, T *mem, F f,
                                   type_tag<T>) Vc_NOEXCEPT_OR_IN_TEST
    {
        store16(v, mem, f);
    }
  /**
    // convert and 16-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(simd_member_type<T> v, U *mem, F f, type_tag<T>,
                                   enable_if<sizeof(T) == sizeof(U) * 8> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
        store2(convert<simd_member_type<T>, simd_member_type<U>>(v), mem, f);
    }
    // convert and 32-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(simd_member_type<T> v, U *mem, F f, type_tag<T>,
                                   enable_if<sizeof(T) == sizeof(U) * 4> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_AARCH_ABI
        store4(convert<simd_member_type<T>, simd_member_type<U>>(v), mem, f);
#else
        unused(f);
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
#endif
    }
    // convert and 64-bit  store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(simd_member_type<T> v, U *mem, F f, type_tag<T>,
                                   enable_if<sizeof(T) == sizeof(U) * 2> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_AARCH_ABI
        store8(convert<simd_member_type<T>, simd_member_type<U>>(v), mem, f);
#else
        unused(f);
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
#endif
    }
    // convert and 128-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(simd_member_type<T> v, U *mem, F f, type_tag<T>,
                                   enable_if<sizeof(T) == sizeof(U)> = nullarg) Vc_NOEXCEPT_OR_IN_TEST
    {
#ifdef Vc_HAVE_FULL_AARCH_ABI
        store16(convert<simd_member_type<T>, simd_member_type<U>>(v), mem, f);
#else
        unused(f);
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
#endif
    }
    // masked store {{{2
    template <class T, class F>
    static Vc_INTRINSIC void masked_store(simd<T> v, long double *mem, F,
                                          simd_mask<T> k) Vc_NOEXCEPT_OR_IN_TEST
    {
        // no support for long double?
        execute_n_times<size<T>()>([&](auto i) {
            if (k.d.m(i)) {
                mem[i] = v.d.m(i);
            }
        });
    }
    template <class T, class U, class F>
    static Vc_INTRINSIC void masked_store(simd<T> v, U *mem, F,
                                          simd_mask<T> k) Vc_NOEXCEPT_OR_IN_TEST
    {
        //TODO: detail::masked_store(mem, v.v(), k.d.v(), f);
        execute_n_times<size<T>()>([&](auto i) {
            if (k.d.m(i)) {
                mem[i] = static_cast<T>(v.d.m(i));
            }
        });
    }
      // negation {{{2
    template <class T> static Vc_INTRINSIC simd_mask<T> negate(simd<T> x) noexcept
    {
#if defined Vc_GCC && defined Vc_USE_BUILTIN_VECTOR_TYPES
        return {private_init, !x.d.builtin()};
#else
        return equal_to(x, simd<T>(0));
#endif
    }
      // compares {{{2
#if defined Vc_USE_BUILTIN_VECTOR_TYPES
    template <class T>
    static Vc_INTRINSIC simd_mask<T> equal_to(simd<T> x, simd<T> y)
    {
        return {private_init, x.d.builtin() == y.d.builtin()};
    }
    template <class T>
    static Vc_INTRINSIC simd_mask<T> not_equal_to(simd<T> x, simd<T> y)
    {
        return {private_init, x.d.builtin() != y.d.builtin()};
    }
    template <class T>
    static Vc_INTRINSIC simd_mask<T> less(simd<T> x, simd<T> y)
    {
        return {private_init, x.d.builtin() < y.d.builtin()};
    }
    template <class T>
    static Vc_INTRINSIC simd_mask<T> less_equal(simd<T> x, simd<T> y)
    {
        return {private_init, x.d.builtin() <= y.d.builtin()};
    }
#else
    static Vc_INTRINSIC simd_mask<double> equal_to(simd<double> x, simd<double> y) { return {private_init, vceqq_f64(x.d, y.d)}; }
    static Vc_INTRINSIC simd_mask< float> equal_to(simd< float> x, simd< float> y) { return {private_init, vceqq_f32(x.d, y.d)}; }
    static Vc_INTRINSIC simd_mask< llong> equal_to(simd< llong> x, simd< llong> y) { return {private_init, vceqq_s64(x.d, y.d)}; }
    static Vc_INTRINSIC simd_mask<ullong> equal_to(simd<ullong> x, simd<ullong> y) { return {private_init, vceqq_u64(x.d, y.d)}; }
    static Vc_INTRINSIC simd_mask<  long> equal_to(simd<  long> x, simd<  long> y) { return {private_init, not_(sizeof(long) == 8 ?  vceqq_s64(x.d, y.d) : vceqq_s32(x.d, y.d)); }
    static Vc_INTRINSIC simd_mask< ulong> equal_to(simd< ulong> x, simd< ulong> y) { return {private_init, not_(sizeof(long) == 8 ?  vceqq_u64(x.d, y.d) : vceqq_u32(x.d, y.d)); }
    static Vc_INTRINSIC simd_mask<   int> equal_to(simd<   int> x, simd<   int> y) { return {private_init, vceqq_s32(x.d, y.d)}; }
    static Vc_INTRINSIC simd_mask<  uint> equal_to(simd<  uint> x, simd<  uint> y) { return {private_init, vceqq_u32(x.d, y.d)}; }
    static Vc_INTRINSIC simd_mask< short> equal_to(simd< short> x, simd< short> y) { return {private_init, vceqq_s16(x.d, y.d)}; }
    static Vc_INTRINSIC simd_mask<ushort> equal_to(simd<ushort> x, simd<ushort> y) { return {private_init, vceqq_u16(x.d, y.d)}; }
    static Vc_INTRINSIC simd_mask< schar> equal_to(simd< schar> x, simd< schar> y) { return {private_init, vceqq_s8(x.d, y.d)}; }
    static Vc_INTRINSIC simd_mask< uchar> equal_to(simd< uchar> x, simd< uchar> y) { return {private_init, vceqq_u8(x.d, y.d)}; }

    static Vc_INTRINSIC simd_mask<double> not_equal_to(simd<double> x, simd<double> y) { return {private_init, detail::not_(vceqq_f64(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask< float> not_equal_to(simd< float> x, simd< float> y) { return {private_init, detail::not_(vceqq_f32(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask< llong> not_equal_to(simd< llong> x, simd< llong> y) { return {private_init, detail::not_(vceqq_s64(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask<ullong> not_equal_to(simd<ullong> x, simd<ullong> y) { return {private_init, detail::not_(vceqq_u64(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask<  long> not_equal_to(simd<  long> x, simd<  long> y) { return {private_init, detail::not_(sizeof(long) == 8 ? vceqq_s64(x.d, y.d) : vceqq_s32(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask< ulong> not_equal_to(simd< ulong> x, simd< ulong> y) { return {private_init, detail::not_(sizeof(long) == 8 ? vceqq_u64(x.d, y.d) : vceqq_u32(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask<   int> not_equal_to(simd<   int> x, simd<   int> y) { return {private_init, detail::not_(vceqq_s32(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask<  uint> not_equal_to(simd<  uint> x, simd<  uint> y) { return {private_init, detail::not_(vceqq_u32(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask< short> not_equal_to(simd< short> x, simd< short> y) { return {private_init, detail::not_(vceqq_s16(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask<ushort> not_equal_to(simd<ushort> x, simd<ushort> y) { return {private_init, detail::not_(vceqq_u16(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask< schar> not_equal_to(simd< schar> x, simd< schar> y) { return {private_init, detail::not_(vceqq_s8(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask< uchar> not_equal_to(simd< uchar> x, simd< uchar> y) { return {private_init, detail::not_(vceqq_u8(x.d, y.d))}; }

    static Vc_INTRINSIC simd_mask<double> less(simd<double> x, simd<double> y) { return {private_init, vclt_f64(x.d, y.d)}; }
    static Vc_INTRINSIC simd_mask< float> less(simd< float> x, simd< float> y) { return {private_init, vclt_f32(x.d, y.d)}; }
    static Vc_INTRINSIC simd_mask< llong> less(simd< llong> x, simd< llong> y) { return {private_init, vclt_s64(y.d, x.d)}; }
    static Vc_INTRINSIC simd_mask<ullong> less(simd<ullong> x, simd<ullong> y) { return {private_init, vclt_u64(y.d, x.d)}; }
    static Vc_INTRINSIC simd_mask<  long> less(simd<  long> x, simd<  long> y) { return {private_init, sizeof(long) == 8 ? vclt_s64(y.d, x.d) :  vclt_s32(y.d, x.d)}; }
    static Vc_INTRINSIC simd_mask< ulong> less(simd< ulong> x, simd< ulong> y) { return {private_init, sizeof(long) == 8 ? vclt_u64(y.d, x.d) : vclt_u32(y.d, x.d)}; }
    static Vc_INTRINSIC simd_mask<   int> less(simd<   int> x, simd<   int> y) { return {private_init, vclt_s32(y.d, x.d)}; }
    static Vc_INTRINSIC simd_mask<  uint> less(simd<  uint> x, simd<  uint> y) { return {private_init, vclt_u32(y.d, x.d)}; }
    static Vc_INTRINSIC simd_mask< short> less(simd< short> x, simd< short> y) { return {private_init, vclt_s16(y.d, x.d)}; }
    static Vc_INTRINSIC simd_mask<ushort> less(simd<ushort> x, simd<ushort> y) { return {private_init, vclt_u16(y.d, x.d)}; }
    static Vc_INTRINSIC simd_mask< schar> less(simd< schar> x, simd< schar> y) { return {private_init, vclt_s8(y.d, x.d)}; }
    static Vc_INTRINSIC simd_mask< uchar> less(simd< uchar> x, simd< uchar> y) { return {private_init, vclt_u8(y.d, x.d)}; }

    static Vc_INTRINSIC simd_mask<double> less_equal(simd<double> x, simd<double> y) { return {private_init, detail::not_(vcle_f64(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask< float> less_equal(simd< float> x, simd< float> y) { return {private_init, detail::not_(vcle_f32(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask< llong> less_equal(simd< llong> x, simd< llong> y) { return {private_init, detail::not_(vcle_s64(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask<ullong> less_equal(simd<ullong> x, simd<ullong> y) { return {private_init, detail::not_(vcle_u64(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask<  long> less_equal(simd<  long> x, simd<  long> y) { return {private_init, detail::not_(sizeof(long) == 8 ?  vcle_s64(x.d, y.d) : vcle_s32(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask< ulong> less_equal(simd< ulong> x, simd< ulong> y) { return {private_init, detail::not_(sizeof(long) == 8 ?  vcle_u64(x.d, y.d) :  vcle_u32(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask<   int> less_equal(simd<   int> x, simd<   int> y) { return {private_init, detail::not_( vcle_s32(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask<  uint> less_equal(simd<  uint> x, simd<  uint> y) { return {private_init, detail::not_( vcle_u32(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask< short> less_equal(simd< short> x, simd< short> y) { return {private_init, detail::not_( vcle_s16(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask<ushort> less_equal(simd<ushort> x, simd<ushort> y) { return {private_init, detail::not_( vcle_u16(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask< schar> less_equal(simd< schar> x, simd< schar> y) { return {private_init, detail::not_( vcle_s8(x.d, y.d))}; }
    static Vc_INTRINSIC simd_mask< uchar> less_equal(simd< uchar> x, simd< uchar> y) { return {private_init, detail::not_( vcle_u8(x.d, y.d))}; }
#endif
     // smart_reference access {{{2
    template <class T, class A>
    static Vc_INTRINSIC T get(Vc::simd<T, A> v, int i) noexcept
    {
        return v.d.m(i);
    }
    template <class T, class A, class U>
    static Vc_INTRINSIC void set(Vc::simd<T, A> &v, int i, U &&x) noexcept
    {
        v.d.set(i, std::forward<U>(x));
    }
      // }}}2
*/
};
// simd_mask impl {{{1
struct neon_mask_impl {
     // memb er types {{{2
    using abi = simd_abi::neon;
    template <class T> static constexpr size_t size() { return simd_size_v<T, abi>; }
    template <class T> using mask_member_type = neon_mask_member_type<T>;
    template <class T> using simd_mask = Vc::simd_mask<T, simd_abi::neon>;
    template <class T> using mask_bool = MaskBool<sizeof(T)>;
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;
    // broadcast {{{2
    template <class T> static Vc_INTRINSIC auto broadcast(bool x, type_tag<T>) noexcept
    {
        return detail::aarch::broadcast16(T(mask_bool<T>{x}));
    }

    // load {{{2
    template <class F>
    static Vc_INTRINSIC auto load(const bool *mem, F, size_tag<4>) noexcept
    {
    }

    template <class F>
    static Vc_INTRINSIC auto load(const bool *mem, F, size_tag<2>) noexcept
    {
    }
    template <class F>
    static Vc_INTRINSIC auto load(const bool *mem, F, size_tag<8>) noexcept
    {
    }
    template <class F>
    static Vc_INTRINSIC auto load(const bool *mem, F, size_tag<16>) noexcept
    {
    }

    // masked load {{{2
    template <class T, class F, class SizeTag>
    static Vc_INTRINSIC void masked_load(mask_member_type<T> &merge,
                                         mask_member_type<T> mask, const bool *mem, F,
                                         SizeTag s) noexcept
    {
        for (std::size_t i = 0; i < s; ++i) {
            if (mask.m(i)) {
                merge.set(i, mask_bool<T>{mem[i]});
            }
        }
    }
    // store {{{2
    template <class T, class F>
    static Vc_INTRINSIC void store(mask_member_type<T> v, bool *mem, F,
                                   size_tag<2>) noexcept
    {
    }
    template <class T, class F>
    static Vc_INTRINSIC void store(mask_member_type<T> v, bool *mem, F,
                                   size_tag<4>) noexcept
    {
    }
    template <class T, class F>
    static Vc_INTRINSIC void store(mask_member_type<T> v, bool *mem, F,
                                   size_tag<8>) noexcept
    {
    }
    template <class T, class F>
    static Vc_INTRINSIC void store(mask_member_type<T> v, bool *mem, F,
                                   size_tag<16>) noexcept
    {
    }
    // masked store {{{2
    template <class T, class F, class SizeTag>
    static Vc_INTRINSIC void masked_store(mask_member_type<T> v, bool *mem, F,
                                          mask_member_type<T> k, SizeTag) noexcept
    {
        for (std::size_t i = 0; i < size<T>(); ++i) {
            if (k.m(i)) {
                mem[i] = v.m(i);
            }
        }
    }
	/*
    // negation {{{2
    template <class T, class SizeTag>
    static Vc_INTRINSIC mask_member_type<T> negate(const mask_member_type<T> &x,
                                                   SizeTag) noexcept
    {
#if defined Vc_GCC && defined Vc_USE_BUILTIN_VECTOR_TYPES
        return !x.builtin();
#else
        return detail::not_(x.v());
#endif
    }
 	 */
    // logical and bitwise operator s {{{2
    template <class T>
    static Vc_INTRINSIC simd_mask<T> logical_and(const simd_mask<T> &x, const simd_mask<T> &y)
    {
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> logical_or(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, detail::or_(x.d, y.d)};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> bit_and(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, detail::and_(x.d, y.d)};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> bit_or(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, detail::or_(x.d, y.d)};
    }

    template <class T>
    static Vc_INTRINSIC simd_mask<T> bit_xor(const simd_mask<T> &x, const simd_mask<T> &y)
    {
        return {private_init, detail::xor_(x.d, y.d)};
    }

    // smart_reference access {{{2
    template <class T> static bool get(const simd_mask<T> &k, int i) noexcept
    {
        return k.d.m(i);
    }
    template <class T> static void set(simd_mask<T> &k, int i, bool x) noexcept
    {
        k.d.set(i, mask_bool<T>(x));
    }
    // }}}2
};
// }}}1
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

// [simd_mask.reductions] {{{
Vc_VERSIONED_NAMESPACE_BEGIN

//NEON really doesn't have mask reduction implentation?

/*

Vc_ALWAYS_INLINE bool all_of(simd_mask<float, simd_abi::neon> k)
{
    const float32x4_t d(k);
    return movemask_f32(d) == 0xf;
}

Vc_ALWAYS_INLINE bool any_of(simd_mask<float, simd_abi::neon> k)
{
    const float32x4_t d(k);
    return movemask_f32(d) != 0;
}

Vc_ALWAYS_INLINE bool none_of(simd_mask<float, simd_abi::neon> k)
{
    const float32x4_t d(k);
    return movemask_f32(d) == 0;
}

Vc_ALWAYS_INLINE bool some_of(simd_mask<float, simd_abi::neon> k)
{
    const float32x4_t d(k);
    const int tmp = movemask_f32(d);
    return tmp != 0 && (tmp ^ 0xf) != 0;
}

Vc_ALWAYS_INLINE bool all_of(simd_mask<double, simd_abi::neon> k)
{
    float64x2_t d(k);
    return movemask_f64(d) == 0x3;
}

Vc_ALWAYS_INLINE bool any_of(simd_mask<double, simd_abi::neon> k)
{
    const float64x2_t d(k);
    return movemask_f64(d) != 0;
}

Vc_ALWAYS_INLINE bool none_of(simd_mask<double, simd_abi::neon> k)
{
    const float64x2_t d(k);
    return movemask_f64(d) == 0;
}

Vc_ALWAYS_INLINE bool some_of(simd_mask<double, simd_abi::neon> k)
{
    const float64x2_t d(k);
    const int tmp = movemask_f64(d);
    return tmp == 1 || tmp == 2;
}

template <class T> Vc_ALWAYS_INLINE bool all_of(simd_mask<T, simd_abi::neon> k)
{
    const int32x4_t d(k);
    return movemask_s32(d) == 0xffff;
}

template <class T> Vc_ALWAYS_INLINE bool any_of(simd_mask<T, simd_abi::neon> k)
{
    const int32x4_t d(k);
    return movemask_s32(d) != 0x0000;
}

template <class T> Vc_ALWAYS_INLINE bool none_of(simd_mask<T, simd_abi::neon> k)
{
    const int32x4_t d(k);
    return movemask_s32(d) == 0x0000;
}

template <class T> Vc_ALWAYS_INLINE bool some_of(simd_mask<T, simd_abi::neon> k)
{
    const int32x4_t d(k);
    const int tmp = movemask_s32(d);
    return tmp != 0 && (tmp ^ 0xffff) != 0;
}

template <class T> Vc_ALWAYS_INLINE int popcount(simd_mask<T, simd_abi::neon> k)
{
    const auto d =
        static_cast<typename detail::traits<T, simd_abi::neon>::mask_cast_type>(k);
    return detail::mask_count<k.size()>(d);
}

template <class T> Vc_ALWAYS_INLINE int find_first_set(simd_mask<T, simd_abi::neon> k)
{
    const auto d =
        static_cast<typename detail::traits<T, simd_abi::neon>::mask_cast_type>(k);
    return detail::firstbit(detail::mask_to_int<k.size()>(d));
}

template <class T> Vc_ALWAYS_INLINE int find_last_set(simd_mask<T, simd_abi::neon> k)
{
    const auto d =
        static_cast<typename detail::traits<T, simd_abi::neon>::mask_cast_type>(k);
    return detail::lastbit(detail::mask_to_int<k.size()>(d));
}
*/
Vc_VERSIONED_NAMESPACE_END
// }}}

#endif  // Vc_HAVE_NEON_ABI
#endif  // Vc_HAVE_NEON

#endif  // VC_SIMD_NEON_H_

// vim: foldmethod=marker
