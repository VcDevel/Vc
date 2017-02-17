/*  This file is part of the Vc library. {{{
Copyright © 2010-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DATAPAR_STORAGE_H_
#define VC_DATAPAR_STORAGE_H_

#include <iosfwd>

#include "macros.h"
#include "x86/intrinsics.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// AliasStrategy{{{1
namespace AliasStrategy
{
struct Union {};
struct MayAlias {};
struct VectorBuiltin {};
struct UnionMembers {};
}  // namespace AliasStrategy

using DefaultStrategy =
// manual selection:
#if defined Vc_USE_ALIASSTRATEGY_VECTORBUILTIN
#ifndef Vc_USE_BUILTIN_VECTOR_TYPES
#define Vc_USE_BUILTIN_VECTOR_TYPES
#endif
    AliasStrategy::VectorBuiltin;
#elif defined Vc_USE_ALIASSTRATEGY_UNIONMEMBERS
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
#undef Vc_USE_BUILTIN_VECTOR_TYPES
#endif
    AliasStrategy::UnionMembers;
#elif defined Vc_USE_ALIASSTRATEGY_UNION
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
#undef Vc_USE_BUILTIN_VECTOR_TYPES
#endif
    AliasStrategy::Union;
#elif defined Vc_USE_ALIASSTRATEGY_MAYALIAS
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
#undef Vc_USE_BUILTIN_VECTOR_TYPES
#endif
    AliasStrategy::MayAlias;
// automatic selection:
#elif defined Vc_USE_BUILTIN_VECTOR_TYPES
    AliasStrategy::VectorBuiltin;
#elif defined Vc_MSVC
    AliasStrategy::UnionMembers;
#elif defined Vc_ICC
    AliasStrategy::Union;
#elif defined __GNUC__
    AliasStrategy::MayAlias;
#else
    AliasStrategy::Union;
#endif

// assertCorrectAlignment{{{1
#ifndef Vc_CHECK_ALIGNMENT
template<typename _T> static Vc_ALWAYS_INLINE void assertCorrectAlignment(const _T *){}
#else
template<typename _T> static Vc_ALWAYS_INLINE void assertCorrectAlignment(const _T *ptr)
{
    const size_t s = alignof(_T);
    if((reinterpret_cast<size_t>(ptr) & ((s ^ (s & (s - 1))) - 1)) != 0) {
        std::fprintf(stderr, "A vector with incorrect alignment has just been created. Look at the stacktrace to find the guilty object.\n");
        std::abort();
    }
}
#endif

// Storage decl{{{1
template <typename ValueType, size_t Size, typename Strategy = DefaultStrategy>
class Storage;
//}}}1

#if defined Vc_HAVE_SSE  // need at least one SIMD ISA to make sense
// Storage<bool>{{{1
template <size_t Size> struct bool_storage_member_type;
template <size_t Size> class Storage<bool, Size, DefaultStrategy>
{
public:
    using VectorType = typename bool_storage_member_type<Size>::type;
    using value_type = bool;
    using EntryType = value_type;

    static constexpr size_t size() { return Size; }
    Vc_INTRINSIC Storage() = default;
    template <class... Args, class = enable_if<sizeof...(Args) == Size>>
    Vc_INTRINSIC Storage(Args &&...init)
        : data{static_cast<EntryType>(std::forward<Args>(init))...}
    {
    }

    Vc_INTRINSIC Storage(const VectorType &x) : data(x) {}
    Vc_INTRINSIC Storage(const Storage &) = default;
    Vc_INTRINSIC Storage &operator=(const Storage &) = default;

    Vc_INTRINSIC Vc_PURE operator const VectorType &() const { return v(); }
    Vc_INTRINSIC Vc_PURE VectorType &v() { return data; }
    Vc_INTRINSIC Vc_PURE const VectorType &v() const { return data; }

    Vc_INTRINSIC Vc_PURE EntryType operator[](size_t i) const { return m(i); }
    Vc_INTRINSIC Vc_PURE EntryType m(size_t i) const { return data & (VectorType(1) << i); }
    Vc_INTRINSIC void set(size_t i, EntryType x)
    {
        if (x) {
            data |= (VectorType(1) << i);
        } else {
            data &= ~(VectorType(1) << i);
        }
    }

private:
    VectorType data;
};

// Storage<Union>{{{1
template <typename ValueType, size_t Size>
class Storage<ValueType, Size, AliasStrategy::Union>
{
    static_assert(std::is_fundamental<ValueType>::value &&
                      std::is_arithmetic<ValueType>::value,
                  "Only works for fundamental arithmetic types.");

public:
    using VectorType = intrinsic_type<ValueType, Size>;
    using value_type = ValueType;
    using EntryType = value_type;

    union Alias {
        Vc_INTRINSIC Alias(VectorType vv) : v(vv) {}
        VectorType v;
        EntryType m[Size];
    };

    static constexpr size_t size() { return Size; }

    Vc_INTRINSIC Storage() : data(x86::zero<VectorType>()) { assertCorrectAlignment(&data); }

    template <class... Args, class = enable_if<sizeof...(Args) == Size>>
    Vc_INTRINSIC Storage(Args &&...init)
        : data(x86::set(static_cast<EntryType>(std::forward<Args>(init))...))
    {
        assertCorrectAlignment(&data);
    }

#ifdef Vc_HAVE_AVX512BW
    inline Storage(__mmask64 k)
        : data(intrin_cast<VectorType>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
        assertCorrectAlignment(&data);
    }
    inline Storage(__mmask32 k)
        : data(intrin_cast<VectorType>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
        assertCorrectAlignment(&data);
    }
#endif  // Vc_HAVE_AVX512BW
#if defined Vc_HAVE_AVX512DQ || (defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL)
    inline Storage(__mmask16 k)
        : data(intrin_cast<VectorType>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
        assertCorrectAlignment(&data);
    }
    inline Storage(__mmask8 k)
        : data(intrin_cast<VectorType>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
        assertCorrectAlignment(&data);
    }
#endif  // Vc_HAVE_AVX512BW

    Vc_INTRINSIC Storage(const VectorType &x) : data(x)
    {
        assertCorrectAlignment(&data);
    }

    template <typename U>
    Vc_INTRINSIC explicit Storage(const U &x
#ifndef Vc_MSVC
                                  ,
                                  enable_if<sizeof(U) == sizeof(VectorType)> = nullarg
#endif
                                  )
        : data(reinterpret_cast<const VectorType &>(x))
    {
        static_assert(sizeof(U) == sizeof(VectorType),
                      "invalid call to converting Storage constructor");
        assertCorrectAlignment(&data);
    }

    static const VectorType &adjustVectorType(const VectorType &x) { return x; }
    template <typename T> static VectorType adjustVectorType(const T &x)
    {
        return reinterpret_cast<VectorType>(x);
    }
    template <typename U>
    Vc_INTRINSIC explicit Storage(const Storage<U, Size, AliasStrategy::Union> &x)
        : data(adjustVectorType(x.v()))
    {
        assertCorrectAlignment(&data);
    }

    Vc_INTRINSIC Storage(const Storage &) = default;
    Vc_INTRINSIC Storage &operator=(const Storage &) = default;

    Vc_INTRINSIC operator const VectorType &() const { return data; }
    Vc_INTRINSIC Vc_PURE VectorType &v() { return data; }
    Vc_INTRINSIC Vc_PURE const VectorType &v() const { return data; }

    Vc_INTRINSIC Vc_PURE EntryType operator[](size_t i) const { return m(i); }
    Vc_INTRINSIC Vc_PURE EntryType m(size_t i) const { return Alias(data).m[i]; }
    Vc_INTRINSIC void set(size_t i, EntryType x)
    {
        Alias a(data);
        a.m[i] = x;
        data = a.v;
    }

private:
    VectorType data;
};

// Storage<MayAlias>{{{1
template <typename ValueType, size_t Size>
class Storage<ValueType, Size, AliasStrategy::MayAlias>
{
    static_assert(std::is_fundamental<ValueType>::value &&
                      std::is_arithmetic<ValueType>::value,
                  "Only works for fundamental arithmetic types.");

    struct
#ifndef Vc_MSVC
        [[gnu::may_alias]]
#endif
        aliased_construction
    {
        may_alias<ValueType> d[Size];
    };

public:
    using VectorType = intrinsic_type<ValueType, Size>;
    using value_type = ValueType;
    using EntryType = value_type;

    static constexpr size_t size() { return Size; }

    Vc_INTRINSIC Storage() : data(x86::zero<VectorType>()) { assertCorrectAlignment(&data); }

    template <class... Args, class = enable_if<sizeof...(Args) == Size>>
    Vc_INTRINSIC Storage(Args &&...init)
        : data(x86::set(static_cast<EntryType>(std::forward<Args>(init))...))
    {
        assertCorrectAlignment(&data);
    }

#ifdef Vc_HAVE_AVX512BW
    inline Storage(__mmask64 k)
        : data(intrin_cast<VectorType>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
    }
    inline Storage(__mmask32 k)
        : data(intrin_cast<VectorType>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
    }
#endif  // Vc_HAVE_AVX512BW
#if defined Vc_HAVE_AVX512DQ || (defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL)
    inline Storage(__mmask16 k)
        : data(intrin_cast<VectorType>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
    }
    inline Storage(__mmask8 k)
        : data(intrin_cast<VectorType>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
    }
#endif  // Vc_HAVE_AVX512BW

    Vc_INTRINSIC Storage(const VectorType &x) : data(x)
    {
        assertCorrectAlignment(&data);
    }

    template <typename U>
    Vc_INTRINSIC explicit Storage(const U &x,
                                  enable_if<sizeof(U) == sizeof(VectorType)> = nullarg)
        : data(reinterpret_cast<const VectorType &>(x))
    {
        assertCorrectAlignment(&data);
    }

    template <typename U>
    Vc_INTRINSIC explicit Storage(Storage<U, Size, AliasStrategy::MayAlias> x)
        : data(reinterpret_cast<VectorType>(x.v()))
    {
        assertCorrectAlignment(&data);
    }

    Vc_INTRINSIC Storage(const Storage &) = default;
    Vc_INTRINSIC Storage &operator=(const Storage &) = default;

    Vc_INTRINSIC operator const VectorType &() const { return v(); }
    Vc_INTRINSIC Vc_PURE VectorType &v() { return data; }
    Vc_INTRINSIC Vc_PURE const VectorType &v() const { return data; }

    Vc_INTRINSIC Vc_PURE EntryType operator[](size_t i) const { return m(i); }
    Vc_INTRINSIC Vc_PURE EntryType m(size_t i) const
    {
        return reinterpret_cast<const may_alias<EntryType> *>(&data)[i];
    }
    Vc_INTRINSIC void set(size_t i, EntryType x)
    {
        reinterpret_cast<may_alias<EntryType> *>(&data)[i] = x;
    }

private:
    VectorType data;
};

// Storage<VectorBuiltin>{{{1
template <typename ValueType, size_t Size>
class Storage<ValueType, Size, AliasStrategy::VectorBuiltin>
{
    static_assert(std::is_fundamental<ValueType>::value &&
                      std::is_arithmetic<ValueType>::value,
                  "Only works for fundamental arithmetic types.");

public:
    using Builtin = builtin_type<ValueType, Size>;

    using VectorType =
#ifdef Vc_TEMPLATES_DROP_ATTRIBUTES
        may_alias<intrinsic_type<ValueType, Size>>;
#else
        intrinsic_type<ValueType, Size>;
#endif
    using value_type = ValueType;
    using EntryType = value_type;

    static constexpr size_t size() { return Size; }

    Vc_INTRINSIC Storage() : data{} { assertCorrectAlignment(&data); }

    template <class... Args, class = enable_if<sizeof...(Args) == Size>>
    Vc_INTRINSIC Storage(Args &&... init)
        : data{static_cast<EntryType>(std::forward<Args>(init))...}
    {
    }

#ifdef Vc_HAVE_AVX512BW
    inline Storage(__mmask64 k)
        : data(reinterpret_cast<Builtin>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
    }
    inline Storage(__mmask32 k)
        : data(reinterpret_cast<Builtin>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
    }
#endif  // Vc_HAVE_AVX512BW
#if defined Vc_HAVE_AVX512DQ || (defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL)
    inline Storage(__mmask16 k)
        : data(reinterpret_cast<Builtin>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
    }
    inline Storage(__mmask8 k)
        : data(reinterpret_cast<Builtin>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
    }
#endif  // Vc_HAVE_AVX512BW

    Vc_INTRINSIC Storage(Builtin x) : data(x) { assertCorrectAlignment(&data); }

    template <typename U>
    Vc_INTRINSIC Storage(
        const U &x,
        enable_if<is_builtin_vector_v<U> && sizeof(U) == sizeof(VectorType)> = nullarg)
        : data(reinterpret_cast<Builtin>(x))
    {
        assertCorrectAlignment(&data);
    }

    template <typename U>
    Vc_INTRINSIC explicit Storage(Storage<U, Size, AliasStrategy::VectorBuiltin> x)
        : data(reinterpret_cast<Builtin>(x.v()))
    {
        assertCorrectAlignment(&data);
    }

    Vc_INTRINSIC Storage(const Storage &) = default;
    Vc_INTRINSIC Storage &operator=(const Storage &) = default;

    //Vc_INTRINSIC operator const Builtin &() const { return data; }
    Vc_INTRINSIC operator const VectorType &() const { return v(); }
    Vc_INTRINSIC Vc_PURE VectorType &v() { return reinterpret_cast<VectorType &>(data); }
    Vc_INTRINSIC Vc_PURE const VectorType &v() const { return reinterpret_cast<const VectorType &>(data); }

    Vc_INTRINSIC Vc_PURE EntryType operator[](size_t i) const { return m(i); }
    Vc_INTRINSIC Vc_PURE EntryType m(size_t i) const { return data[i]; }
    Vc_INTRINSIC void set(size_t i, EntryType x) { data[i] = x; }

    Vc_INTRINSIC Builtin &builtin() { return data; }
    Vc_INTRINSIC const Builtin &builtin() const { return data; }

private:
    Builtin data;
};

// Storage<UnionMembers>{{{1
template <typename ValueType, size_t Size>
class Storage<ValueType, Size, AliasStrategy::UnionMembers>
{
    static_assert(std::is_fundamental<ValueType>::value &&
                      std::is_arithmetic<ValueType>::value,
                  "Only works for fundamental arithmetic types.");

public:
    using VectorType = intrinsic_type<ValueType, Size>;
    using value_type = ValueType;
    using EntryType = value_type;

    static constexpr size_t size() { return Size; }

    Vc_INTRINSIC Storage() : data(x86::zero<VectorType>())
    {
        assertCorrectAlignment(&data);
    }

    template <class... Args, class = enable_if<sizeof...(Args) == Size>>
    Vc_INTRINSIC Storage(Args &&...init)
        : data(x86::set(static_cast<EntryType>(std::forward<Args>(init))...))
    {
        assertCorrectAlignment(&data);
    }

#ifdef Vc_HAVE_AVX512BW
    inline Storage(__mmask64 k)
        : data(intrin_cast<VectorType>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
        assertCorrectAlignment(&data);
    }
    inline Storage(__mmask32 k)
        : data(intrin_cast<VectorType>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
        assertCorrectAlignment(&data);
    }
#endif  // Vc_HAVE_AVX512BW
#if defined Vc_HAVE_AVX512DQ || (defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL)
    inline Storage(__mmask16 k)
        : data(intrin_cast<VectorType>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
        assertCorrectAlignment(&data);
    }
    inline Storage(__mmask8 k)
        : data(intrin_cast<VectorType>(
              convert_mask<sizeof(EntryType), sizeof(VectorType)>(k)))
    {
        assertCorrectAlignment(&data);
    }
#endif  // Vc_HAVE_AVX512BW

    Vc_INTRINSIC Storage(const VectorType &x) : data(x)
    {
        assertCorrectAlignment(&data);
    }
    template <typename U>
    Vc_INTRINSIC explicit Storage(const U &x
#ifndef Vc_MSVC
                                  ,
                                  enable_if<sizeof(U) == sizeof(VectorType)> = nullarg
#endif  // Vc_MSVC
                                  )
        : data(reinterpret_cast<const VectorType &>(x))
    {
        static_assert(sizeof(U) == sizeof(VectorType),
                      "invalid call to converting Storage constructor");
        assertCorrectAlignment(&data);
    }

    static const VectorType &adjustVectorType(const VectorType &x) { return x; }
    template <typename T> static VectorType adjustVectorType(const T &x)
    {
        return reinterpret_cast<VectorType>(x);
    }
    template <typename U>
    Vc_INTRINSIC explicit Storage(const Storage<U, Size, AliasStrategy::UnionMembers> &x)
        : data(adjustVectorType(x.v()))
    {
        assertCorrectAlignment(&data);
    }

    Vc_INTRINSIC Storage(const Storage &) = default;
    Vc_INTRINSIC Storage &operator=(const Storage &) = default;

    Vc_INTRINSIC operator const VectorType &() const { return v(); }
    Vc_INTRINSIC Vc_PURE VectorType &v() { return data; }
    Vc_INTRINSIC Vc_PURE const VectorType &v() const { return data; }

    Vc_INTRINSIC Vc_PURE EntryType operator[](size_t i) const { return m(i); }
    Vc_INTRINSIC_L Vc_PURE_L EntryType m(size_t i) const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC void set(size_t i, EntryType x) { ref(i) = x; }

private:
    Vc_INTRINSIC_L Vc_PURE_L typename std::conditional<
        std::is_same<EntryType, signed char>::value, char,
        typename std::conditional<
            std::is_same<EntryType, long>::value, int,
            typename std::conditional<std::is_same<EntryType, ulong>::value, uint,
                                      EntryType>::type>::type>::type &
    ref(size_t i) Vc_INTRINSIC_R Vc_PURE_R;
    VectorType data;
};

#ifdef Vc_MSVC
template <> Vc_INTRINSIC Vc_PURE double Storage<double, 2, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m128d_f64[i]; }
template <> Vc_INTRINSIC Vc_PURE  float Storage< float, 4, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m128_f32[i]; }
template <> Vc_INTRINSIC Vc_PURE  llong Storage< llong, 2, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m128i_i64[i]; }
template <> Vc_INTRINSIC Vc_PURE   long Storage<  long, 4, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m128i_i32[i]; }
template <> Vc_INTRINSIC Vc_PURE    int Storage<   int, 4, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m128i_i32[i]; }
template <> Vc_INTRINSIC Vc_PURE  short Storage< short, 8, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m128i_i16[i]; }
template <> Vc_INTRINSIC Vc_PURE  schar Storage< schar,16, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m128i_i8[i]; }
template <> Vc_INTRINSIC Vc_PURE ullong Storage<ullong, 2, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m128i_u64[i]; }
template <> Vc_INTRINSIC Vc_PURE  ulong Storage< ulong, 4, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m128i_u32[i]; }
template <> Vc_INTRINSIC Vc_PURE   uint Storage<  uint, 4, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m128i_u32[i]; }
template <> Vc_INTRINSIC Vc_PURE ushort Storage<ushort, 8, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m128i_u16[i]; }
template <> Vc_INTRINSIC Vc_PURE  uchar Storage< uchar,16, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m128i_u8[i]; }

template <> Vc_INTRINSIC Vc_PURE double &Storage<double, 2, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m128d_f64[i]; }
template <> Vc_INTRINSIC Vc_PURE  float &Storage< float, 4, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m128_f32[i]; }
template <> Vc_INTRINSIC Vc_PURE  llong &Storage< llong, 2, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m128i_i64[i]; }
template <> Vc_INTRINSIC Vc_PURE    int &Storage<  long, 4, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m128i_i32[i]; }
template <> Vc_INTRINSIC Vc_PURE    int &Storage<   int, 4, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m128i_i32[i]; }
template <> Vc_INTRINSIC Vc_PURE  short &Storage< short, 8, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m128i_i16[i]; }
template <> Vc_INTRINSIC Vc_PURE   char &Storage< schar,16, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m128i_i8[i]; }
template <> Vc_INTRINSIC Vc_PURE ullong &Storage<ullong, 2, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m128i_u64[i]; }
template <> Vc_INTRINSIC Vc_PURE   uint &Storage< ulong, 4, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m128i_u32[i]; }
template <> Vc_INTRINSIC Vc_PURE   uint &Storage<  uint, 4, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m128i_u32[i]; }
template <> Vc_INTRINSIC Vc_PURE ushort &Storage<ushort, 8, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m128i_u16[i]; }
template <> Vc_INTRINSIC Vc_PURE  uchar &Storage< uchar,16, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m128i_u8[i]; }

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC Vc_PURE double Storage<double, 4, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m256d_f64[i]; }
template <> Vc_INTRINSIC Vc_PURE  float Storage< float, 8, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m256_f32[i]; }
template <> Vc_INTRINSIC Vc_PURE  llong Storage< llong, 4, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m256i_i64[i]; }
template <> Vc_INTRINSIC Vc_PURE   long Storage<  long, 8, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m256i_i32[i]; }
template <> Vc_INTRINSIC Vc_PURE    int Storage<   int, 8, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m256i_i32[i]; }
template <> Vc_INTRINSIC Vc_PURE  short Storage< short,16, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m256i_i16[i]; }
template <> Vc_INTRINSIC Vc_PURE  schar Storage< schar,32, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m256i_i8[i]; }
template <> Vc_INTRINSIC Vc_PURE ullong Storage<ullong, 4, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m256i_u64[i]; }
template <> Vc_INTRINSIC Vc_PURE  ulong Storage< ulong, 8, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m256i_u32[i]; }
template <> Vc_INTRINSIC Vc_PURE   uint Storage<  uint, 8, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m256i_u32[i]; }
template <> Vc_INTRINSIC Vc_PURE ushort Storage<ushort,16, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m256i_u16[i]; }
template <> Vc_INTRINSIC Vc_PURE  uchar Storage< uchar,32, AliasStrategy::UnionMembers>::m(size_t i) const { return data.m256i_u8[i]; }

template <> Vc_INTRINSIC Vc_PURE double &Storage<double, 4, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m256d_f64[i]; }
template <> Vc_INTRINSIC Vc_PURE  float &Storage< float, 8, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m256_f32[i]; }
template <> Vc_INTRINSIC Vc_PURE  llong &Storage< llong, 4, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m256i_i64[i]; }
template <> Vc_INTRINSIC Vc_PURE    int &Storage<  long, 8, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m256i_i32[i]; }
template <> Vc_INTRINSIC Vc_PURE    int &Storage<   int, 8, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m256i_i32[i]; }
template <> Vc_INTRINSIC Vc_PURE  short &Storage< short,16, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m256i_i16[i]; }
template <> Vc_INTRINSIC Vc_PURE   char &Storage< schar,32, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m256i_i8[i]; }
template <> Vc_INTRINSIC Vc_PURE ullong &Storage<ullong, 4, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m256i_u64[i]; }
template <> Vc_INTRINSIC Vc_PURE   uint &Storage< ulong, 8, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m256i_u32[i]; }
template <> Vc_INTRINSIC Vc_PURE   uint &Storage<  uint, 8, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m256i_u32[i]; }
template <> Vc_INTRINSIC Vc_PURE ushort &Storage<ushort,16, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m256i_u16[i]; }
template <> Vc_INTRINSIC Vc_PURE  uchar &Storage< uchar,32, AliasStrategy::UnionMembers>::ref(size_t i) { return data.m256i_u8[i]; }
#endif
#endif  // Vc_MSVC

// Storage ostream operators{{{1
template <class CharT, class T, size_t N>
inline std::basic_ostream<CharT> &operator<<(std::basic_ostream<CharT> & s,
                                             const Storage<T, N> &v)
{
    s << '[' << v[0];
    for (size_t i = 1; i < N; ++i) {
        s << ((i % 4) ? " " : " | ") << v[i];
    }
    return s << ']';
}

//}}}1
#endif  // Vc_HAVE_SSE
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_STORAGE_H_

// vim: foldmethod=marker
