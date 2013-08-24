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
#include "macros.h"

Vc_NAMESPACE_BEGIN(Common)

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

namespace
{
template<typename T> struct MayAlias { typedef T Type Vc_MAY_ALIAS; };
template<size_t Bytes> struct MayAlias<MaskBool<Bytes>> { typedef MaskBool<Bytes> Type; };
} // anonymous namespace
template<typename _VectorType, typename _EntryType> class VectorMemoryUnion
{
    public:
        typedef _VectorType VectorType;
        typedef _EntryType EntryType;
        Vc_ALWAYS_INLINE VectorMemoryUnion() { assertCorrectAlignment(&v()); }
#if defined VC_ICC
        Vc_ALWAYS_INLINE VectorMemoryUnion(const VectorType &x) { data.v = x; assertCorrectAlignment(&data.v); }
        Vc_ALWAYS_INLINE VectorMemoryUnion &operator=(const VectorType &x) {
            data.v = x; return *this;
        }

        Vc_ALWAYS_INLINE Vc_PURE VectorType &v() { return data.v; }
        Vc_ALWAYS_INLINE Vc_PURE const VectorType &v() const { return data.v; }

        Vc_ALWAYS_INLINE Vc_PURE EntryType &m(size_t index) {
            return data.m[index];
        }
        Vc_ALWAYS_INLINE Vc_PURE EntryType m(size_t index) const {
            return data.m[index];
        }

#ifdef VC_COMPILE_BENCHMARKS
    public:
#endif
    private:
        union VectorScalarUnion {
            Vc_INTRINSIC VectorScalarUnion() {}
            Vc_INTRINSIC VectorScalarUnion(const VectorScalarUnion &rhs) : v(rhs.v) {}
            Vc_INTRINSIC VectorScalarUnion &operator=(const VectorScalarUnion &rhs) { v = rhs.v; return *this; }
            VectorType v;
            EntryType m[sizeof(VectorType)/sizeof(EntryType)];
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
};

#if defined(VC_GCC) && (VC_GCC == 0x40700 || (VC_GCC >= 0x40600 && VC_GCC <= 0x40603))
// workaround bug 52736 in GCC
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
}
#endif

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_STORAGE_H
