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

#ifndef VC_SIMD_X86_STORAGE_H_
#define VC_SIMD_X86_STORAGE_H_

#ifndef VC_SIMD_STORAGE_H_
#error "Do not include detail/x86/storage.h directly. Include detail/storage.h instead."
#endif
#include "intrinsics.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace x86
{
// extract_part {{{1
// identity {{{2
template <class T>
constexpr Vc_INTRINSIC const storage16_t<T>& extract_part_impl(std::true_type,
                                                               size_constant<0>,
                                                               size_constant<1>,
                                                               const storage16_t<T>& x)
{
    return x;
}

// by 2 and by 4 splits {{{2
template <class T, size_t N, size_t Index, size_t Total>
constexpr Vc_INTRINSIC Storage<T, N / Total> extract_part_impl(std::true_type,
                                                               size_constant<Index>,
                                                               size_constant<Total>,
                                                               Storage<T, N> x)
{
    return detail::extract<Index, Total>(x.d);
}

// partial SSE (shifts) {{{2
template <class T, size_t Index, size_t Total, size_t N>
Vc_INTRINSIC Storage<T, 16 / sizeof(T)> extract_part_impl(std::false_type,
                                                                   size_constant<Index>,
                                                                   size_constant<Total>,
                                                                   Storage<T, N> x)
{
    constexpr int split = sizeof(x) / 16;
    constexpr int shift = (sizeof(x) / Total * Index) % 16;
    return x86::shift_right<shift>(
        extract_part_impl<T>(std::true_type(), size_constant<Index * split / Total>(),
                             size_constant<split>(), x));
}

// public interface {{{2
template <size_t Index, size_t Total, class T, size_t N>
Vc_INTRINSIC Vc_CONST Storage<T, std::max(16 / sizeof(T), N / Total)> extract_part(
    Storage<T, N> x)
{
    constexpr size_t NewN = N / Total;
    static_assert(Total > 1, "Total must be greater than 1");
    static_assert(NewN * Total == N, "N must be divisible by Total");
    return extract_part_impl<T>(
        bool_constant<(sizeof(T) * NewN >= 16)>(),  // dispatch on whether the result is a
                                                    // partial SSE register or larger
        size_constant<Index>(), size_constant<Total>(), x);
}

// }}}1
// extract_part(Storage<bool, N>) {{{
template <size_t Offset, size_t SplitBy, size_t N>
constexpr Vc_INTRINSIC Storage<bool, N / SplitBy> extract_part(Storage<bool, N> x)
{
    static_assert(SplitBy >= 2 && Offset < SplitBy && Offset >= 0);
    return x.d >> (Offset * N / SplitBy);
}

// }}}
}  // namespace x86

// to_storage specializations for bitset and __mmask<N> {{{
#ifdef Vc_HAVE_AVX512_ABI
template <size_t N> class to_storage<std::bitset<N>>
{
    std::bitset<N> d;

public:
    [[deprecated("use convert_mask<To>(bitset)")]]
    constexpr to_storage(std::bitset<N> x) : d(x) {}

    // can convert to larger storage for Abi::is_partial == true
    template <class U, size_t M> constexpr operator Storage<U, M>() const
    {
        static_assert(M >= N);
        return convert_mask<Storage<U, M>>(d);
    }
};

#define Vc_TO_STORAGE(type_)                                                             \
    template <> class to_storage<type_>                                                  \
    {                                                                                    \
        type_ d;                                                                         \
                                                                                         \
    public:                                                                              \
        [[deprecated("use convert_mask<To>(bitset)")]] constexpr to_storage(type_ x)     \
            : d(x)                                                                       \
        {                                                                                \
        }                                                                                \
                                                                                         \
        template <class U, size_t N> constexpr operator Storage<U, N>() const            \
        {                                                                                \
            static_assert(N >= sizeof(type_) * CHAR_BIT);                                \
            return reinterpret_cast<builtin_type_t<U, N>>(                               \
                convert_mask<Storage<U, N>>(d));                                         \
        }                                                                                \
                                                                                         \
        template <size_t N> constexpr operator Storage<bool, N>() const                  \
        {                                                                                \
            static_assert(                                                               \
                std::is_same_v<type_, typename bool_storage_member_type<N>::type>);      \
            return d;                                                                    \
        }                                                                                \
    }
Vc_TO_STORAGE(__mmask8);
Vc_TO_STORAGE(__mmask16);
Vc_TO_STORAGE(__mmask32);
Vc_TO_STORAGE(__mmask64);
#undef Vc_TO_STORAGE
#endif  // Vc_HAVE_AVX512_ABI

// }}}
// concat {{{
// These functions are part of the Storage interface => same namespace.
// These functions are only available when AVX or higher is enabled. In the future there
// may be more cases (e.g. half SSE -> full SSE or even MMX -> SSE).
#if 0//def Vc_HAVE_SSE2
template <class T>
Vc_INTRINSIC Vc_CONST Storage<T, 4 / sizeof(T)> Vc_VDECL
    concat(Storage<T, 2 / sizeof(T)> a, Storage<T, 2 / sizeof(T)> b)
{
    static_assert(std::is_integral_v<T>);
    return to_storage_unsafe(_mm_unpacklo_epi16(to_m128i(a), to_m128i(b)));
}

template <class T>
Vc_INTRINSIC Vc_CONST Storage<T, 8 / sizeof(T)> Vc_VDECL
    concat(Storage<T, 4 / sizeof(T)> a, Storage<T, 4 / sizeof(T)> b)
{
    static_assert(std::is_integral_v<T>);
    return to_storage_unsafe(_mm_unpacklo_epi32(to_m128i(a), to_m128i(b)));
}

Vc_INTRINSIC Vc_CONST Storage<float, 4> Vc_VDECL concat(Storage<float, 2> a,
                                                        Storage<float, 2> b)
{
    return to_storage(_mm_unpacklo_pd(to_m128d(a), to_m128d(b)));
}

template <class T>
Vc_INTRINSIC Vc_CONST Storage<T, 16 / sizeof(T)> Vc_VDECL
    concat(Storage<T, 8 / sizeof(T)> a, Storage<T, 8 / sizeof(T)> b)
{
    static_assert(std::is_integral_v<T>);
    return to_storage(_mm_unpacklo_epi64(to_m128d(a), to_m128d(b)));
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <class T>
Vc_INTRINSIC Vc_CONST Storage<T, 32 / sizeof(T)> Vc_VDECL
    concat(Storage<T, 16 / sizeof(T)> a, Storage<T, 16 / sizeof(T)> b)
{
    return concat(a.d, b.d);
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <class T>
Vc_INTRINSIC Vc_CONST Storage<T, 64 / sizeof(T)> Vc_VDECL
    concat(Storage<T, 32 / sizeof(T)> a, Storage<T, 32 / sizeof(T)> b)
{
    return concat(a.d, b.d);
}
#endif  // Vc_HAVE_AVX512F

template <class T, size_t N>
Vc_INTRINSIC Vc_CONST Storage<T, 4 * N> Vc_VDECL concat(Storage<T, N> a, Storage<T, N> b,
                                                        Storage<T, N> c, Storage<T, N> d)
{
    return concat(concat(a, b), concat(c, d));
}

template <class T, size_t N>
Vc_INTRINSIC Vc_CONST Storage<T, 8 * N> Vc_VDECL concat(Storage<T, N> a, Storage<T, N> b,
                                                        Storage<T, N> c, Storage<T, N> d,
                                                        Storage<T, N> e, Storage<T, N> f,
                                                        Storage<T, N> g, Storage<T, N> h)
{
    return concat(concat(concat(a, b), concat(c, d)), concat(concat(e, f), concat(g, h)));
}

//}}}

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_SIMD_X86_STORAGE_H_
