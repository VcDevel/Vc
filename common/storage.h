/*  This file is part of the Vc library.

    Copyright (C) 2010-2012 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef VC_COMMON_STORAGE_H
#define VC_COMMON_STORAGE_H

#include "aliasingentryhelper.h"
#include "types.h"
#include <utility>
#include <cstring>

#include "macros.h"

Vc_NAMESPACE_BEGIN(Common)

// accessScalar for MSVC/*{{{*/
#ifdef VC_MSVC
#ifdef VC_IMPL_AVX
template<typename EntryType, typename VectorType> inline EntryType &accessScalar(VectorType &d, size_t i) { return accessScalar<EntryType>(d._d, i); }
template<typename EntryType, typename VectorType> inline EntryType accessScalar(const VectorType &d, size_t i) { return accessScalar<EntryType>(d._d, i); }
#else
template<typename EntryType, typename VectorType> inline EntryType &accessScalar(VectorType &d, size_t i) { return accessScalar<EntryType>(d[i/4], i % 4); }
template<typename EntryType, typename VectorType> inline EntryType accessScalar(const VectorType &d, size_t i) { return accessScalar<EntryType>(d[i/4], i % 4); }
#endif

template<> Vc_ALWAYS_INLINE double &accessScalar<double, __m128d>(__m128d &d, size_t i) { return d.m128d_f64[i]; }
template<> Vc_ALWAYS_INLINE float  &accessScalar<float , __m128 >(__m128  &d, size_t i) { return d.m128_f32[i]; }
template<> Vc_ALWAYS_INLINE short  &accessScalar<short , __m128i>(__m128i &d, size_t i) { return d.m128i_i16[i]; }
template<> Vc_ALWAYS_INLINE unsigned short  &accessScalar<unsigned short , __m128i>(__m128i &d, size_t i) { return d.m128i_u16[i]; }
template<> Vc_ALWAYS_INLINE int  &accessScalar<int , __m128i>(__m128i &d, size_t i) { return d.m128i_i32[i]; }
template<> Vc_ALWAYS_INLINE unsigned int  &accessScalar<unsigned int , __m128i>(__m128i &d, size_t i) { return d.m128i_u32[i]; }
template<> Vc_ALWAYS_INLINE char  &accessScalar<char , __m128i>(__m128i &d, size_t i) { return d.m128i_i8[i]; }
template<> Vc_ALWAYS_INLINE unsigned char  &accessScalar<unsigned char , __m128i>(__m128i &d, size_t i) { return d.m128i_u8[i]; }

template<> Vc_ALWAYS_INLINE double accessScalar<double, __m128d>(const __m128d &d, size_t i) { return d.m128d_f64[i]; }
template<> Vc_ALWAYS_INLINE float  accessScalar<float , __m128 >(const __m128  &d, size_t i) { return d.m128_f32[i]; }
template<> Vc_ALWAYS_INLINE short  accessScalar<short , __m128i>(const __m128i &d, size_t i) { return d.m128i_i16[i]; }
template<> Vc_ALWAYS_INLINE unsigned short  accessScalar<unsigned short , __m128i>(const __m128i &d, size_t i) { return d.m128i_u16[i]; }
template<> Vc_ALWAYS_INLINE int  accessScalar<int , __m128i>(const __m128i &d, size_t i) { return d.m128i_i32[i]; }
template<> Vc_ALWAYS_INLINE unsigned int  accessScalar<unsigned int , __m128i>(const __m128i &d, size_t i) { return d.m128i_u32[i]; }
template<> Vc_ALWAYS_INLINE char  accessScalar<char , __m128i>(const __m128i &d, size_t i) { return d.m128i_i8[i]; }
template<> Vc_ALWAYS_INLINE unsigned char  accessScalar<unsigned char , __m128i>(const __m128i &d, size_t i) { return d.m128i_u8[i]; }

#ifdef VC_IMPL_AVX
template<> Vc_ALWAYS_INLINE double &accessScalar<double, __m256d>(__m256d &d, size_t i) { return d.m256d_f64[i]; }
template<> Vc_ALWAYS_INLINE float  &accessScalar<float , __m256 >(__m256  &d, size_t i) { return d.m256_f32[i]; }
template<> Vc_ALWAYS_INLINE short  &accessScalar<short , __m256i>(__m256i &d, size_t i) { return d.m256i_i16[i]; }
template<> Vc_ALWAYS_INLINE unsigned short  &accessScalar<unsigned short , __m256i>(__m256i &d, size_t i) { return d.m256i_u16[i]; }
template<> Vc_ALWAYS_INLINE int  &accessScalar<int , __m256i>(__m256i &d, size_t i) { return d.m256i_i32[i]; }
template<> Vc_ALWAYS_INLINE unsigned int  &accessScalar<unsigned int , __m256i>(__m256i &d, size_t i) { return d.m256i_u32[i]; }

template<> Vc_ALWAYS_INLINE double accessScalar<double, __m256d>(const __m256d &d, size_t i) { return d.m256d_f64[i]; }
template<> Vc_ALWAYS_INLINE float  accessScalar<float , __m256 >(const __m256  &d, size_t i) { return d.m256_f32[i]; }
template<> Vc_ALWAYS_INLINE short  accessScalar<short , __m256i>(const __m256i &d, size_t i) { return d.m256i_i16[i]; }
template<> Vc_ALWAYS_INLINE unsigned short  accessScalar<unsigned short , __m256i>(const __m256i &d, size_t i) { return d.m256i_u16[i]; }
template<> Vc_ALWAYS_INLINE int  accessScalar<int , __m256i>(const __m256i &d, size_t i) { return d.m256i_i32[i]; }
template<> Vc_ALWAYS_INLINE unsigned int  accessScalar<unsigned int , __m256i>(const __m256i &d, size_t i) { return d.m256i_u32[i]; }
#endif
#endif
/*}}}*/
// GccTypeHelper/*{{{*/
#ifdef VC_USE_BUILTIN_VECTOR_TYPES
template<typename EntryType, typename VectorType> struct GccTypeHelper;
template<> struct GccTypeHelper<double        , __m128d> { typedef  __v2df Type; };
template<> struct GccTypeHelper<float         , __m128 > { typedef  __v4sf Type; };
template<> struct GccTypeHelper<long long     , __m128i> { typedef  __v2di Type; };
template<> struct GccTypeHelper<unsigned long long, __m128i> { typedef  __v2di Type; };
template<> struct GccTypeHelper<int           , __m128i> { typedef  __v4si Type; };
template<> struct GccTypeHelper<unsigned int  , __m128i> { typedef  __v4si Type; };
template<> struct GccTypeHelper<short         , __m128i> { typedef  __v8hi Type; };
template<> struct GccTypeHelper<unsigned short, __m128i> { typedef  __v8hi Type; };
template<> struct GccTypeHelper<char          , __m128i> { typedef __v16qi Type; };
template<> struct GccTypeHelper<unsigned char , __m128i> { typedef __v16qi Type; };
#ifdef VC_IMPL_SSE
template<typename VectorType> struct GccTypeHelper<float, VectorType> { typedef  __v4sf Type; };
#endif
#ifdef VC_IMPL_AVX
template<> struct GccTypeHelper<double        , __m256d> { typedef  __v4df Type; };
template<> struct GccTypeHelper<float         , __m256 > { typedef  __v8sf Type; };
template<> struct GccTypeHelper<long long     , __m256i> { typedef  __v4di Type; };
template<> struct GccTypeHelper<unsigned long long, __m256i> { typedef  __v4di Type; };
template<> struct GccTypeHelper<int           , __m256i> { typedef  __v8si Type; };
template<> struct GccTypeHelper<unsigned int  , __m256i> { typedef  __v8si Type; };
template<> struct GccTypeHelper<short         , __m256i> { typedef __v16hi Type; };
template<> struct GccTypeHelper<unsigned short, __m256i> { typedef __v16hi Type; };
template<> struct GccTypeHelper<char          , __m256i> { typedef __v32qi Type; };
template<> struct GccTypeHelper<unsigned char , __m256i> { typedef __v32qi Type; };
#endif
#endif
/*}}}*/
namespace
{
template<typename T> struct MayAlias { typedef T Type Vc_MAY_ALIAS; };
template<size_t Bytes> struct MayAlias<MaskBool<Bytes>> { typedef MaskBool<Bytes> Type; };
} // anonymous namespace

template <typename VectorType, typename EntryType> class AliasedVectorEntry/*{{{*/
{
    typedef typename std::conditional<std::is_const<VectorType>::value,
            const EntryType *const, EntryType *const>::type PointerType;
    PointerType scalar;

public:
    constexpr AliasedVectorEntry(VectorType &d, size_t i) : scalar(&reinterpret_cast<PointerType>(&d)[i])
    {
    }

    AliasedVectorEntry(const AliasedVectorEntry &rhs) = delete;
    AliasedVectorEntry &operator=(const AliasedVectorEntry &rhs) = delete;
    AliasedVectorEntry(AliasedVectorEntry &&rhs) = delete;
    AliasedVectorEntry &operator=(AliasedVectorEntry &&rhs) = delete;

    template <typename U,
              typename std::enable_if<std::is_convertible<typename std::decay<U>::type, EntryType>::value &&
                                          !std::is_const<VectorType>::value,
                                      int>::type = 0>
    Vc_INTRINSIC AliasedVectorEntry &operator=(U &&x)
    {
        EntryType tmp = std::forward<U>(x);
        // memcpy does '*scalar = x' but in a way that tells the compiler that pointer aliasing
        // occurred
        std::memcpy(reinterpret_cast<VectorType *>(scalar), &tmp, sizeof(EntryType));
        return *this;
    }
    Vc_INTRINSIC operator EntryType() const
    {
        return *scalar;
    }
    Vc_INTRINSIC operator EntryType &()
    {
        return *scalar;
    }

    template <typename U>
    Vc_INTRINSIC decltype(std::declval<EntryType &>()[std::forward<U>(0)]) operator[](U &&i)
    {
        return (*scalar)[std::forward<U>(i)];
    }
    template <typename U>
    Vc_INTRINSIC decltype(std::declval<const EntryType &>()[std::forward<U>(0)]) operator[](
        U &&i) const
    {
        return (*scalar)[std::forward<U>(i)];
    }

    template <typename U,
              typename std::enable_if<
                  std::is_class<EntryType>::value &&std::is_convertible<EntryType, U>::value,
                  int>::type = 0>
    Vc_INTRINSIC operator U() const
    {
        return *scalar;
    }

#define VC_OP__(op__)                                                                              \
    template <typename U> AliasedVectorEntry &operator op__##=(U &&x)                              \
    {                                                                                              \
        EntryType tmp = (*scalar)op__ std::forward<U>(x);                                          \
        return operator=(tmp);                                                                     \
    }
    VC_ALL_BINARY(VC_OP__)
    VC_ALL_SHIFTS(VC_OP__)
    VC_ALL_ARITHMETICS(VC_OP__)
#undef VC_OP__

    AliasedVectorEntry &operator++()
    {
        EntryType tmp = *scalar;
        return operator=(++tmp);
    }
    EntryType operator++(int)
    {
        EntryType tmp = *scalar;
        EntryType r = tmp++;
        operator=(tmp);
        return r;
    }
    AliasedVectorEntry &operator--()
    {
        EntryType tmp = *scalar;
        return operator=(--tmp);
    }
    EntryType operator--(int)
    {
        EntryType tmp = *scalar;
        EntryType r = tmp--;
        operator=(tmp);
        return r;
    }

    // needs SFINAE
    //decltype(std::declval<EntryType &>().operator->()) operator->() { return scalar->operator->(); }
    //decltype(std::declval<const EntryType &>().operator->()) operator->() const { return scalar->operator->(); }
};/*}}}*/

#if 0 //defined VC_ICC
template <typename _VectorType, typename _EntryType> class VectorMemoryUnion/*{{{*/
{
public:
    typedef _VectorType VectorType;
    typedef _EntryType EntryType;
    Vc_ALWAYS_INLINE VectorMemoryUnion()
        : data()
    {
        assertCorrectAlignment(&v());
    }
    Vc_ALWAYS_INLINE VectorMemoryUnion(VectorType vv)
        : data(vv)
    {
        assertCorrectAlignment(&v());
    }

    Vc_ALWAYS_INLINE Vc_PURE VectorType &v()
    {
        return data;
    }
    Vc_ALWAYS_INLINE Vc_PURE const VectorType &v() const
    {
        return data;
    }

    Vc_ALWAYS_INLINE Vc_PURE AliasedVectorEntry<VectorType, EntryType> m(size_t index)
    {
        return AliasedVectorEntry<VectorType, EntryType>(data, index);
    }
    Vc_ALWAYS_INLINE Vc_PURE EntryType m(size_t index) const
    {
        return reinterpret_cast<const EntryType *>(&data)[index];
    }

private:
    VectorType data;
};/*}}}*/
#else
template<typename _VectorType, typename _EntryType, typename _VectorEntryType = _EntryType> class VectorMemoryUnion/*{{{*/
{
    public:
        typedef _VectorType VectorType;
        typedef _EntryType EntryType;
        typedef _VectorEntryType VectorEntryType;

        Vc_ALWAYS_INLINE VectorMemoryUnion() { assertCorrectAlignment(&v()); }
#if defined VC_ICC
        Vc_ALWAYS_INLINE VectorMemoryUnion(VectorType x) : data(x) {
            assertCorrectAlignment(&v());
        }
        Vc_ALWAYS_INLINE VectorMemoryUnion(const VectorMemoryUnion &) = default;
        /*
        Vc_ALWAYS_INLINE VectorMemoryUnion(VectorMemoryUnion &&) = default;
        Vc_ALWAYS_INLINE VectorMemoryUnion(const VectorMemoryUnion &x) : data(x.v()) {
            assertCorrectAlignment(&v());
        }
        Vc_ALWAYS_INLINE VectorMemoryUnion(VectorMemoryUnion &&x) : data(std::move(x.data.v)) {
            assertCorrectAlignment(&v());
        }
        */

        Vc_ALWAYS_INLINE VectorMemoryUnion &operator=(const VectorMemoryUnion &rhs) { data.v = rhs.data.v; return *this; }// = default;
        //Vc_ALWAYS_INLINE VectorMemoryUnion &operator=(VectorMemoryUnion &&) = default;

        Vc_ALWAYS_INLINE Vc_PURE VectorType &v() { return data.v; }
        Vc_ALWAYS_INLINE Vc_PURE const VectorType &v() const { return data.v; }

        Vc_ALWAYS_INLINE Vc_PURE VectorEntryType &m(size_t index) {
            return data.m[index];
        }
        Vc_ALWAYS_INLINE Vc_PURE VectorEntryType m(size_t index) const {
            return data.m[index];
        }

        Vc_ALWAYS_INLINE Vc_PURE EntryType &entry(size_t index) {
            return data.m2[index * (sizeof(VectorEntryType) / sizeof(EntryType))];
        }
        Vc_ALWAYS_INLINE Vc_PURE EntryType entry(size_t index) const {
            return data.m2[index * (sizeof(VectorEntryType) / sizeof(EntryType))];
        }

    private:
        union VectorScalarUnion {
            Vc_INTRINSIC VectorScalarUnion() : v() {}
            Vc_INTRINSIC VectorScalarUnion(VectorType vv) : v(vv) {}
            Vc_INTRINSIC VectorScalarUnion(const VectorScalarUnion &x) = default;
            VectorType v;
            VectorEntryType m[];
            EntryType m2[];
        } data;
#else
        Vc_ALWAYS_INLINE VectorMemoryUnion(VC_ALIGNED_PARAMETER(VectorType) x) : data(x) { assertCorrectAlignment(&data); }
        Vc_ALWAYS_INLINE VectorMemoryUnion &operator=(VC_ALIGNED_PARAMETER(VectorType) x) {
            data = x; return *this;
        }

        Vc_ALWAYS_INLINE Vc_PURE VectorType &v() { return data; }
        Vc_ALWAYS_INLINE Vc_PURE const VectorType &v() const { return data; }

#ifdef VC_MSVC
        Vc_ALWAYS_INLINE EntryType &m(size_t index) {
            return accessScalar<EntryType>(data, index);
        }

        Vc_ALWAYS_INLINE EntryType m(size_t index) const {
            return accessScalar<EntryType>(data, index);
        }
#else
        typedef typename MayAlias<EntryType>::Type AliasingEntryType;
        Vc_ALWAYS_INLINE Vc_PURE AliasingEntryType &m(size_t index) {
            return reinterpret_cast<AliasingEntryType *>(&data)[index];
        }

        Vc_ALWAYS_INLINE Vc_PURE EntryType m(size_t index) const {
            return reinterpret_cast<const AliasingEntryType *>(&data)[index];
        }
#endif
        Vc_ALWAYS_INLINE Vc_PURE decltype(m(0)) entry(size_t index) {
            return m(index);
        }
        Vc_ALWAYS_INLINE Vc_PURE decltype(m(0)) entry(size_t index) const {
            return m(index);
        }

#ifdef VC_USE_BUILTIN_VECTOR_TYPES
        template<typename JustForSfinae = void>
        Vc_ALWAYS_INLINE Vc_PURE
        typename GccTypeHelper<typename std::conditional<true, EntryType, JustForSfinae>::type, VectorType>::Type
        gcc() const { return typename GccTypeHelper<EntryType, VectorType>::Type(data); }
#endif

    private:
#ifdef VC_COMPILE_BENCHMARKS
    public:
#endif
        VectorType data;
#endif
};/*}}}*/
#endif

#if defined(VC_GCC) && (VC_GCC == 0x40700 || (VC_GCC >= 0x40600 && VC_GCC <= 0x40603))
// workaround bug 52736 in GCC/*{{{*/
template<typename T, typename V> static Vc_ALWAYS_INLINE Vc_CONST T &vectorMemoryUnionAliasedMember(V *data, size_t index) {
    if (__builtin_constant_p(index) && index == 0) {
        T *ret;
        asm("mov %1,%0" : "=r"(ret) : "r"(data));
        return *ret;
    } else {
        return reinterpret_cast<T *>(data)[index];
    }
}
template<> Vc_ALWAYS_INLINE Vc_PURE VectorMemoryUnion<__m128d, double>::AliasingEntryType &VectorMemoryUnion<__m128d, double>::m(size_t index) {
    return vectorMemoryUnionAliasedMember<AliasingEntryType>(&data, index);
}
template<> Vc_ALWAYS_INLINE Vc_PURE VectorMemoryUnion<__m128i, long long>::AliasingEntryType &VectorMemoryUnion<__m128i, long long>::m(size_t index) {
    return vectorMemoryUnionAliasedMember<AliasingEntryType>(&data, index);
}
template<> Vc_ALWAYS_INLINE Vc_PURE VectorMemoryUnion<__m128i, unsigned long long>::AliasingEntryType &VectorMemoryUnion<__m128i, unsigned long long>::m(size_t index) {
    return vectorMemoryUnionAliasedMember<AliasingEntryType>(&data, index);
}/*}}}*/
#endif

Vc_NAMESPACE_END

#include "undomacros.h"

#ifdef __SSE2__
#include "maskentry.h"
    static_assert(
        std::is_convertible<Vc::Common::AliasedVectorEntry<__m128, Vc::Common::MaskBool<4>>,
                            bool>::value,
        "std::is_convertible<MaskBool<4>, bool> failed");
#endif

#endif // VC_COMMON_STORAGE_H

// vim: foldmethod=marker
