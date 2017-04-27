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

#ifndef VC_DATAPAR_NEON_H_
#define VC_DATAPAR_NEON_H_

#include "fixed_size.h"
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
struct neon_datapar_impl;

template <class T> using neon_datapar_member_type = Storage<T, 16 / sizeof(T)>;
template <class T> using neon_mask_member_type = Storage<T, 16 / sizeof(T)>;

template <class T> struct traits<T, datapar_abi::neon> {
    static_assert(sizeof(T) <= 8,
                  "NEON can only implement operations on element types with sizeof <= 8");
    static constexpr size_t size() noexcept { return 16 / sizeof(T); }

    using datapar_member_type = neon_datapar_member_type<T>;
    using datapar_impl_type = neon_datapar_impl;
    static constexpr size_t datapar_member_alignment = alignof(datapar_member_type);
    using datapar_cast_type = typename datapar_member_type::VectorType;

    using mask_member_type = neon_mask_member_type<T>;
    using mask_impl_type = neon_mask_impl;
    static constexpr size_t mask_member_alignment = alignof(mask_member_type);
    using mask_cast_type = typename mask_member_type::VectorType;
};
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
// datapar impl {{{1
template <class Derived> struct generic_datapar_impl {
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
    static Vc_INTRINSIC Vc::datapar<T, A> make_datapar(const U &x)
    {
        using traits = typename Vc::datapar<T, A>::traits;
        using V = typename traits::datapar_member_type;
        return {private_init, static_cast<V>(x)};
    }
    // complement {{{2
    template <class T, class A>
    static Vc_INTRINSIC Vc::datapar<T, A> complement(const Vc::datapar<T, A> &x) noexcept
    {
        using detail::aarch::complement;
        return make_datapar<T, A>(complement(adjust_for_long(detail::data(x))));
    }
    // unary minus {{{2
    template <class T, class A>
    static Vc_INTRINSIC Vc::datapar<T, A> unary_minus(const Vc::datapar<T, A> &x) noexcept
    {
        using detail::aarch::unary_minus;
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
inline void Vc_VDECL masked_assign(mask<T, A> k, datapar<T, A> &lhs,
                                   const detail::id<datapar<T, A>> &rhs)
{
    lhs = static_cast<datapar<T, A>>(
        detail::aarch::blend(detail::data(k), detail::data(lhs), detail::data(rhs)));
}

template <class T, class A>
inline void Vc_VDECL masked_assign(mask<T, A> k, mask<T, A> &lhs,
                                   const detail::id<mask<T, A>> &rhs)
{
    lhs = static_cast<mask<T, A>>(
        detail::aarch::blend(detail::data(k), detail::data(lhs), detail::data(rhs)));
}

template <template <typename> class Op, typename T, class A,
          int = 1>
inline void Vc_VDECL masked_cassign(mask<T, A> k, datapar<T, A> &lhs,
                                    const datapar<T, A> &rhs)
{
    lhs = static_cast<datapar<T, A>>(detail::aarch::blend(
        detail::data(k), detail::data(lhs), detail::data(Op<void>{}(lhs, rhs))));
}

template <template <typename> class Op, typename T, class A, class U>
inline enable_if<std::is_convertible<U, datapar<T, A>>::value, void> Vc_VDECL
masked_cassign(mask<T, A> k, datapar<T, A> &lhs, const U &rhs)
{
    masked_cassign<Op>(k, lhs, datapar<T, A>(rhs));
}

template <template <typename> class Op, typename T, class A,
          int = 1>
inline datapar<T, A> Vc_VDECL masked_unary(mask<T, A> k, datapar<T, A> v)
{
    Op<datapar<T, A>> op;
    return static_cast<datapar<T, A>>(
        detail::aarch::blend(detail::data(k), detail::data(v), detail::data(op(v))));
}

// }}}1
Vc_VERSIONED_NAMESPACE_END

#ifdef Vc_HAVE_NEON_ABI
Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// datapar impl {{{1
struct neon_datapar_impl : public generic_datapar_impl<neon_datapar_impl> {
    // member types {{{2
    using abi = datapar_abi::neon;
    template <class T> static constexpr size_t size() { return datapar_size_v<T, abi>; }
    template <class T> using datapar_member_type = neon_datapar_member_type<T>;
    template <class T> using intrinsic_type = typename datapar_member_type<T>::VectorType;
    template <class T> using mask_member_type = neon_mask_member_type<T>;
    template <class T> using datapar = Vc::datapar<T, abi>;
    template <class T> using mask = Vc::mask<T, abi>;
    template <size_t N> using size_tag = std::integral_constant<size_t, N>;
    template <class T> using type_tag = T *;
    // broadcast {{{2
/*
    static Vc_INTRINSIC intrinsic_type<float> broadcast(float x, size_tag<4>) noexcept
    {
        return vld1q_dup_f32(x);
    }
    static Vc_INTRINSIC intrinsic_type<double> broadcast(double x, size_tag<2>) noexcept
    {
        return vld1q_dub_f64(x);
    }
*/
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<2>) noexcept
    {
        return vld1_dub_s64(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<4>) noexcept
    {
        return vld1_dub_s32(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<8>) noexcept
    {
        return vld1_dub_s16(x);
    }
    template <class T>
    static Vc_INTRINSIC intrinsic_type<T> broadcast(T x, size_tag<16>) noexcept
    {
        return vld1_dub_s8(x);
    }
    // load {{{2
    // from long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC datapar_member_type<T> load(const long double *mem, F,
                                                    type_tag<T>) noexcept
    {
        return generate_from_n_evaluations<size<T>(), datapar_member_type<T>>(
            [&](auto i) { return static_cast<T>(mem[i]); });
    }
    // load without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC intrinsic_type<T> load(const T *mem, F f, type_tag<T>) noexcept
    {
        return detail::load16(mem, f);
    }
    // store {{{2
    // store to long double has no vector implementation{{{3
    template <class T, class F>
    static Vc_INTRINSIC void store(datapar_member_type<T> v, long double *mem, F,
                                   type_tag<T>) noexcept
    {
        // alignment F doesn't matter
        execute_n_times<size<T>()>([&](auto i) { mem[i] = v.m(i); });
    }
    // store without conversion{{{3
    template <class T, class F>
    static Vc_INTRINSIC void store(datapar_member_type<T> v, T *mem, F f,
                                   type_tag<T>) noexcept
    {
        store16(v, mem, f);
    }
    // convert and 16-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(datapar_member_type<T> v, U *mem, F f, type_tag<T>,
                                   enable_if<sizeof(T) == sizeof(U) * 8> = nullarg) noexcept
    {
        store2(convert<datapar_member_type<T>, datapar_member_type<U>>(v), mem, f);
    }
    // convert and 32-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(datapar_member_type<T> v, U *mem, F f, type_tag<T>,
                                   enable_if<sizeof(T) == sizeof(U) * 4> = nullarg) noexcept
    {
#ifdef Vc_HAVE_FULL_AARCH_ABI
        store4(convert<datapar_member_type<T>, datapar_member_type<U>>(v), mem, f);
#else
        unused(f);
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
#endif
    }
    // convert and 64-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(datapar_member_type<T> v, U *mem, F f, type_tag<T>,
                                   enable_if<sizeof(T) == sizeof(U) * 2> = nullarg) noexcept
    {
#ifdef Vc_HAVE_FULL_AARCH_ABI
        store8(convert<datapar_member_type<T>, datapar_member_type<U>>(v), mem, f);
#else
        unused(f);
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
#endif
    }
    // convert and 128-bit store{{{3
    template <class T, class U, class F>
    static Vc_INTRINSIC void store(datapar_member_type<T> v, U *mem, F f, type_tag<T>,
                                   enable_if<sizeof(T) == sizeof(U)> = nullarg) noexcept
    {
#ifdef Vc_HAVE_FULL_AARCH_ABI
        store16(convert<datapar_member_type<T>, datapar_member_type<U>>(v), mem, f);
#else
        unused(f);
        execute_n_times<size<T>()>([&](auto i) { mem[i] = static_cast<U>(v[i]); });
#endif
    }
    // masked store {{{2
    template <class T, class U, class F>
    static Vc_INTRINSIC void masked_store(datapar<T> v, U *mem, F,
                                          mask<T> k) noexcept
    {
        //TODO: detail::masked_store(mem, v.v(), k.d.v(), f);
        execute_n_times<size<T>()>([&](auto i) {
            if (k.d.m(i)) {
                mem[i] = static_cast<T>(v.d.m(i));
            }
        });
    }
    // negation {{{2
    template <class T> static Vc_INTRINSIC mask<T> negate(datapar<T> x) noexcept
    {
#if defined Vc_GCC && defined Vc_USE_BUILTIN_VECTOR_TYPES
        return {private_init, !x.d.builtin()};
#else
        return equal_to(x, datapar<T>(0));
#endif
    }
    // compares {{{2
#if defined Vc_USE_BUILTIN_VECTOR_TYPES
    template <class T>
    static Vc_INTRINSIC mask<T> equal_to(datapar<T> x, datapar<T> y)
    {
        return {private_init, x.d.builtin() == y.d.builtin()};
    }
    template <class T>
    static Vc_INTRINSIC mask<T> not_equal_to(datapar<T> x, datapar<T> y)
    {
        return {private_init, x.d.builtin() != y.d.builtin()};
    }
    template <class T>
    static Vc_INTRINSIC mask<T> less(datapar<T> x, datapar<T> y)
    {
        return {private_init, x.d.builtin() < y.d.builtin()};
    }
    template <class T>
    static Vc_INTRINSIC mask<T> less_equal(datapar<T> x, datapar<T> y)
    {
        return {private_init, x.d.builtin() <= y.d.builtin()};
    }
#else
    static Vc_INTRINSIC mask<double> equal_to(datapar<double> x, datapar<double> y) { return {private_init, vceqq_f64(x.d, y.d)}; }
    static Vc_INTRINSIC mask< float> equal_to(datapar< float> x, datapar< float> y) { return {private_init, vceqq_f32(x.d, y.d)}; }
    static Vc_INTRINSIC mask< llong> equal_to(datapar< llong> x, datapar< llong> y) { return {private_init, vceqq_s64(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ullong> equal_to(datapar<ullong> x, datapar<ullong> y) { return {private_init, vceqq_u64(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  long> equal_to(datapar<  long> x, datapar<  long> y) { return {private_init, not_(sizeof(long) == 8 ?  vceqq_s64(x.d, y.d) : vceqq_s32(x.d, y.d)); }
    static Vc_INTRINSIC mask< ulong> equal_to(datapar< ulong> x, datapar< ulong> y) { return {private_init, not_(sizeof(long) == 8 ?  vceqq_u64(x.d, y.d) : vceqq_u32(x.d, y.d)); }
    static Vc_INTRINSIC mask<   int> equal_to(datapar<   int> x, datapar<   int> y) { return {private_init, vceqq_s32(x.d, y.d)}; }
    static Vc_INTRINSIC mask<  uint> equal_to(datapar<  uint> x, datapar<  uint> y) { return {private_init, vceqq_u32(x.d, y.d)}; }
    static Vc_INTRINSIC mask< short> equal_to(datapar< short> x, datapar< short> y) { return {private_init, vceqq_s16(x.d, y.d)}; }
    static Vc_INTRINSIC mask<ushort> equal_to(datapar<ushort> x, datapar<ushort> y) { return {private_init, vceqq_u16(x.d, y.d)}; }
    static Vc_INTRINSIC mask< schar> equal_to(datapar< schar> x, datapar< schar> y) { return {private_init, vceqq_s8(x.d, y.d)}; }
    static Vc_INTRINSIC mask< uchar> equal_to(datapar< uchar> x, datapar< uchar> y) { return {private_init, vceqq_u8(x.d, y.d)}; }

    static Vc_INTRINSIC mask<double> not_equal_to(datapar<double> x, datapar<double> y) { return {private_init, detail::not_(vceqq_f64(x.d, y.d))}; }
    static Vc_INTRINSIC mask< float> not_equal_to(datapar< float> x, datapar< float> y) { return {private_init, detail::not_(vceqq_f32(x.d, y.d))}; }
    static Vc_INTRINSIC mask< llong> not_equal_to(datapar< llong> x, datapar< llong> y) { return {private_init, detail::not_(vceqq_s64(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ullong> not_equal_to(datapar<ullong> x, datapar<ullong> y) { return {private_init, detail::not_(vceqq_u64(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  long> not_equal_to(datapar<  long> x, datapar<  long> y) { return {private_init, detail::not_(sizeof(long) == 8 ? vceqq_s64(x.d, y.d) : vceqq_s32(x.d, y.d))}; }
    static Vc_INTRINSIC mask< ulong> not_equal_to(datapar< ulong> x, datapar< ulong> y) { return {private_init, detail::not_(sizeof(long) == 8 ? vceqq_u64(x.d, y.d) : vceqq_u32(x.d, y.d))}; }
    static Vc_INTRINSIC mask<   int> not_equal_to(datapar<   int> x, datapar<   int> y) { return {private_init, detail::not_(vceqq_s32(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  uint> not_equal_to(datapar<  uint> x, datapar<  uint> y) { return {private_init, detail::not_(vceqq_u32(x.d, y.d))}; }
    static Vc_INTRINSIC mask< short> not_equal_to(datapar< short> x, datapar< short> y) { return {private_init, detail::not_(vceqq_s16(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ushort> not_equal_to(datapar<ushort> x, datapar<ushort> y) { return {private_init, detail::not_(vceqq_u16(x.d, y.d))}; }
    static Vc_INTRINSIC mask< schar> not_equal_to(datapar< schar> x, datapar< schar> y) { return {private_init, detail::not_(vceqq_s8(x.d, y.d))}; }
    static Vc_INTRINSIC mask< uchar> not_equal_to(datapar< uchar> x, datapar< uchar> y) { return {private_init, detail::not_(vceqq_u8(x.d, y.d))}; }

    static Vc_INTRINSIC mask<double> less(datapar<double> x, datapar<double> y) { return {private_init, vclt_f64(x.d, y.d)}; }
    static Vc_INTRINSIC mask< float> less(datapar< float> x, datapar< float> y) { return {private_init, vclt_f32(x.d, y.d)}; }
    static Vc_INTRINSIC mask< llong> less(datapar< llong> x, datapar< llong> y) { return {private_init, vclt_s64(y.d, x.d)}; }
    static Vc_INTRINSIC mask<ullong> less(datapar<ullong> x, datapar<ullong> y) { return {private_init, vclt_u64(y.d, x.d)}; }
    static Vc_INTRINSIC mask<  long> less(datapar<  long> x, datapar<  long> y) { return {private_init, sizeof(long) == 8 ? vclt_s64(y.d, x.d) :  vclt_s32(y.d, x.d)}; }
    static Vc_INTRINSIC mask< ulong> less(datapar< ulong> x, datapar< ulong> y) { return {private_init, sizeof(long) == 8 ? vclt_u64(y.d, x.d) : vclt_u32(y.d, x.d)}; }
    static Vc_INTRINSIC mask<   int> less(datapar<   int> x, datapar<   int> y) { return {private_init, vclt_s32(y.d, x.d)}; }
    static Vc_INTRINSIC mask<  uint> less(datapar<  uint> x, datapar<  uint> y) { return {private_init, vclt_u32(y.d, x.d)}; }
    static Vc_INTRINSIC mask< short> less(datapar< short> x, datapar< short> y) { return {private_init, vclt_s16(y.d, x.d)}; }
    static Vc_INTRINSIC mask<ushort> less(datapar<ushort> x, datapar<ushort> y) { return {private_init, vclt_u16(y.d, x.d)}; }
    static Vc_INTRINSIC mask< schar> less(datapar< schar> x, datapar< schar> y) { return {private_init, vclt_s8(y.d, x.d)}; }
    static Vc_INTRINSIC mask< uchar> less(datapar< uchar> x, datapar< uchar> y) { return {private_init, vclt_u8(y.d, x.d)}; }

    static Vc_INTRINSIC mask<double> less_equal(datapar<double> x, datapar<double> y) { return {private_init, detail::not_(vcle_f64(x.d, y.d))}; }
    static Vc_INTRINSIC mask< float> less_equal(datapar< float> x, datapar< float> y) { return {private_init, detail::not_(vcle_f32(x.d, y.d))}; }
    static Vc_INTRINSIC mask< llong> less_equal(datapar< llong> x, datapar< llong> y) { return {private_init, detail::not_(vcle_s64(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ullong> less_equal(datapar<ullong> x, datapar<ullong> y) { return {private_init, detail::not_(vcle_u64(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  long> less_equal(datapar<  long> x, datapar<  long> y) { return {private_init, detail::not_(sizeof(long) == 8 ?  vcle_s64(x.d, y.d) : vcle_s32(x.d, y.d))}; }
    static Vc_INTRINSIC mask< ulong> less_equal(datapar< ulong> x, datapar< ulong> y) { return {private_init, detail::not_(sizeof(long) == 8 ?  vcle_u64(x.d, y.d) :  vcle_u32(x.d, y.d))}; }
    static Vc_INTRINSIC mask<   int> less_equal(datapar<   int> x, datapar<   int> y) { return {private_init, detail::not_( vcle_s32(x.d, y.d))}; }
    static Vc_INTRINSIC mask<  uint> less_equal(datapar<  uint> x, datapar<  uint> y) { return {private_init, detail::not_( vcle_u32(x.d, y.d))}; }
    static Vc_INTRINSIC mask< short> less_equal(datapar< short> x, datapar< short> y) { return {private_init, detail::not_( vcle_s16(x.d, y.d))}; }
    static Vc_INTRINSIC mask<ushort> less_equal(datapar<ushort> x, datapar<ushort> y) { return {private_init, detail::not_( vcle_u16(x.d, y.d))}; }
    static Vc_INTRINSIC mask< schar> less_equal(datapar< schar> x, datapar< schar> y) { return {private_init, detail::not_( vcle_s8(x.d, y.d))}; }
    static Vc_INTRINSIC mask< uchar> less_equal(datapar< uchar> x, datapar< uchar> y) { return {private_init, detail::not_( vcle_u8(x.d, y.d))}; }
#endif
    // smart_reference access {{{2
    template <class T, class A>
    static Vc_INTRINSIC T get(Vc::datapar<T, A> v, int i) noexcept
    {
        return v.d.m(i);
    }
    template <class T, class A, class U>
    static Vc_INTRINSIC void set(Vc::datapar<T, A> &v, int i, U &&x) noexcept
    {
        v.d.set(i, std::forward<U>(x));
    }
    // }}}2
};
// mask impl {{{1
struct neon_mask_impl {
    // member types {{{2
    using abi = datapar_abi::neon;
    template <class T> static constexpr size_t size() { return datapar_size_v<T, abi>; }
    template <class T> using mask_member_type = neon_mask_member_type<T>;
    template <class T> using mask = Vc::mask<T, datapar_abi::neon>;
    template <class T> using mask_bool = MaskBool<sizeof(T)>;
    template <size_t N> using size_tag = std::integral_constant<size_t, N>;
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
    // logical and bitwise operator s {{{2
    template <class T>
    static Vc_INTRINSIC mask<T> logical_and(const mask<T> &x, const mask<T> &y)
    {
    }
    template <class T>
    static Vc_INTRINSIC mask<T> logical_or(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, detail::or_(x.d, y.d)};
    }
    template <class T>
    static Vc_INTRINSIC mask<T> bit_and(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, detail::and_(x.d, y.d)};
    }
    template <class T>
    static Vc_INTRINSIC mask<T> bit_or(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, detail::or_(x.d, y.d)};
    }
    template <class T>
    static Vc_INTRINSIC mask<T> bit_xor(const mask<T> &x, const mask<T> &y)
    {
        return {private_init, detail::xor_(x.d, y.d)};
    }
    // smart_reference access {{{2
    template <class T> static bool get(const mask<T> &k, int i) noexcept
    {
        return k.d.m(i);
    }
    template <class T> static void set(mask<T> &k, int i, bool x) noexcept
    {
        k.d.set(i, mask_bool<T>(x));
    }
    // }}}2
};
// }}}1
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

// [mask.reductions] {{{
Vc_VERSIONED_NAMESPACE_BEGIN
/*
//NEON really doesn't have mask reduction implentation?
Vc_ALWAYS_INLINE bool all_of(mask<float, datapar_abi::neon> k)
{
    const float32x4_t d(k);
    return movemask_f32(d) == 0xf;
}

Vc_ALWAYS_INLINE bool any_of(mask<float, datapar_abi::neon> k)
{
    const float32x4_t d(k);
    return movemask_f32(d) != 0;
}

Vc_ALWAYS_INLINE bool none_of(mask<float, datapar_abi::neon> k)
{
    const float32x4_t d(k);
    return movemask_f32(d) == 0;
}

Vc_ALWAYS_INLINE bool some_of(mask<float, datapar_abi::neon> k)
{
    const float32x4_t d(k);
    const int tmp = movemask_f32(d);
    return tmp != 0 && (tmp ^ 0xf) != 0;
}

Vc_ALWAYS_INLINE bool all_of(mask<double, datapar_abi::neon> k)
{
    float64x2_t d(k);
    return movemask_f64(d) == 0x3;
}

Vc_ALWAYS_INLINE bool any_of(mask<double, datapar_abi::neon> k)
{
    const float64x2_t d(k);
    return movemask_f64(d) != 0;
}

Vc_ALWAYS_INLINE bool none_of(mask<double, datapar_abi::neon> k)
{
    const float64x2_t d(k);
    return movemask_f64(d) == 0;
}

Vc_ALWAYS_INLINE bool some_of(mask<double, datapar_abi::neon> k)
{
    const float64x2_t d(k);
    const int tmp = movemask_f64(d);
    return tmp == 1 || tmp == 2;
}

template <class T> Vc_ALWAYS_INLINE bool all_of(mask<T, datapar_abi::neon> k)
{
    const int32x4_t d(k);
    return movemask_s32(d) == 0xffff;
}

template <class T> Vc_ALWAYS_INLINE bool any_of(mask<T, datapar_abi::neon> k)
{
    const int32x4_t d(k);
    return movemask_s32(d) != 0x0000;
}

template <class T> Vc_ALWAYS_INLINE bool none_of(mask<T, datapar_abi::neon> k)
{
    const int32x4_t d(k);
    return movemask_s32(d) == 0x0000;
}

template <class T> Vc_ALWAYS_INLINE bool some_of(mask<T, datapar_abi::neon> k)
{
    const int32x4_t d(k);
    const int tmp = movemask_s32(d);
    return tmp != 0 && (tmp ^ 0xffff) != 0;
}

template <class T> Vc_ALWAYS_INLINE int popcount(mask<T, datapar_abi::neon> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::neon>::mask_cast_type>(k);
    return detail::mask_count<k.size()>(d);
}

template <class T> Vc_ALWAYS_INLINE int find_first_set(mask<T, datapar_abi::neon> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::neon>::mask_cast_type>(k);
    return detail::firstbit(detail::mask_to_int<k.size()>(d));
}

template <class T> Vc_ALWAYS_INLINE int find_last_set(mask<T, datapar_abi::neon> k)
{
    const auto d =
        static_cast<typename detail::traits<T, datapar_abi::neon>::mask_cast_type>(k);
    return detail::lastbit(detail::mask_to_int<k.size()>(d));
}
*/
Vc_VERSIONED_NAMESPACE_END
// }}}

#endif  // Vc_HAVE_NEON_ABI
#endif  // Vc_HAVE_NEON

#endif  // VC_DATAPAR_NEON_H_

// vim: foldmethod=marker
