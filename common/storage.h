/*  This file is part of the Vc library. {{{
Copyright © 2010-2014 Matthias Kretz <kretz@kde.org>
All rights reserved.

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

#ifndef VC_COMMON_STORAGE_H
#define VC_COMMON_STORAGE_H

#include "aliasingentryhelper.h"
#include "types.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{

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

template<typename EntryType, typename VectorType> struct BuiltinTypeHelper { typedef VectorType Type; };
#ifdef VC_USE_BUILTIN_VECTOR_TYPES
template<> struct BuiltinTypeHelper<double        , __m128d> { typedef         double Type __attribute__((__vector_size__(16))); };
template<> struct BuiltinTypeHelper<float         , __m128 > { typedef          float Type __attribute__((__vector_size__(16))); };
template<> struct BuiltinTypeHelper<long long     , __m128i> { typedef      long long Type __attribute__((__vector_size__(16))); };
template<> struct BuiltinTypeHelper<unsigned long long, __m128i> { typedef  unsigned long long Type __attribute__((__vector_size__(16))); };
template<> struct BuiltinTypeHelper<int           , __m128i> { typedef            int Type __attribute__((__vector_size__(16))); };
template<> struct BuiltinTypeHelper<unsigned int  , __m128i> { typedef   unsigned int Type __attribute__((__vector_size__(16))); };
template<> struct BuiltinTypeHelper<short         , __m128i> { typedef          short Type __attribute__((__vector_size__(16))); };
template<> struct BuiltinTypeHelper<unsigned short, __m128i> { typedef unsigned short Type __attribute__((__vector_size__(16))); };
template<> struct BuiltinTypeHelper<char          , __m128i> { typedef           char Type __attribute__((__vector_size__(16))); };
template<> struct BuiltinTypeHelper<unsigned char , __m128i> { typedef  unsigned char Type __attribute__((__vector_size__(16))); };
template<> struct BuiltinTypeHelper<  signed char , __m128i> { typedef    signed char Type __attribute__((__vector_size__(16))); };
#ifdef VC_IMPL_SSE
template<typename VectorType> struct BuiltinTypeHelper<float, VectorType> { typedef  __v4sf Type; };
#endif
#ifdef VC_IMPL_AVX
template<> struct BuiltinTypeHelper<double        , __m256d> { typedef  __v4df Type; };
template<> struct BuiltinTypeHelper<float         , __m256 > { typedef  __v8sf Type; };
template<> struct BuiltinTypeHelper<long long     , __m256i> { typedef  __v4di Type; };
template<> struct BuiltinTypeHelper<unsigned long long, __m256i> { typedef  __v4di Type; };
template<> struct BuiltinTypeHelper<int           , __m256i> { typedef  __v8si Type; };
template<> struct BuiltinTypeHelper<unsigned int  , __m256i> { typedef  __v8si Type; };
template<> struct BuiltinTypeHelper<short         , __m256i> { typedef __v16hi Type; };
template<> struct BuiltinTypeHelper<unsigned short, __m256i> { typedef __v16hi Type; };
template<> struct BuiltinTypeHelper<char          , __m256i> { typedef __v32qi Type; };
template<> struct BuiltinTypeHelper<unsigned char , __m256i> { typedef __v32qi Type; };
#endif
#endif

template<typename T> struct MayAliasImpl { typedef T Type Vc_MAY_ALIAS; };
template<size_t Bytes> struct MayAliasImpl<MaskBool<Bytes>> { typedef MaskBool<Bytes> Type; };
/**
 * \internal
 * Helper MayAlias<T> that turns T into the type to be used for an aliasing pointer. This
 * adds the may_alias attribute to T (with compilers that support it). But for MaskBool this
 * attribute is already part of the type and applying it a second times leads to warnings/errors,
 * therefore MaskBool is simply forwarded as is.
 */
template<typename T> using MayAlias = typename MayAliasImpl<T>::Type;

/**
 * \internal
 * Helper class that abstracts the hackery needed for aliasing SIMD and fundamental types to the
 * same memory. The C++ standard says that two pointers of different type may be assumed by the
 * compiler to point to different memory. But all supported compilers have some extension that
 * allows to work around this limitation. GCC (and compatible compilers) can use the `may_alias`
 * attribute to specify memory aliasing exactly where it happens. Other compilers provide unions to
 * allow memory aliasing. VectorMemoryUnion hides all this behind one common interface.
 */
template<typename _VectorType, typename _EntryType> class VectorMemoryUnion
{
    public:
        typedef _VectorType VectorType;
        typedef _EntryType EntryType;
        Vc_ALWAYS_INLINE VectorMemoryUnion() : data() { assertCorrectAlignment(&v()); }
#if defined VC_ICC
        Vc_ALWAYS_INLINE VectorMemoryUnion(const VectorType &x) : data(x) { assertCorrectAlignment(&data.v); }
        Vc_ALWAYS_INLINE VectorMemoryUnion &operator=(const VectorType &x) {
            data.v = x; return *this;
        }
        VectorMemoryUnion(const VectorMemoryUnion &) = default;

        Vc_ALWAYS_INLINE Vc_PURE VectorType &v() { return data.v; }
        Vc_ALWAYS_INLINE Vc_PURE const VectorType &v() const { return data.v; }

        Vc_ALWAYS_INLINE Vc_PURE EntryType &ref(size_t index) {
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
            Vc_INTRINSIC VectorScalarUnion() : v() {}
            Vc_INTRINSIC VectorScalarUnion(VectorType vv) : v(vv) {}
            VectorType v;
            EntryType m[];
        } data;
#else
        typedef typename BuiltinTypeHelper<EntryType, VectorType>::Type BuiltinType;
        typedef BuiltinType BuiltinTypeAlias Vc_MAY_ALIAS;

        Vc_ALWAYS_INLINE VectorMemoryUnion(VectorType x)
            // this copies from x to data, so aliasing is covered via the localized
            // may_alias
            : data(reinterpret_cast<const BuiltinTypeAlias &>(x))
        {
            assertCorrectAlignment(&data);
        }

        Vc_ALWAYS_INLINE Vc_PURE VectorType &v() { return reinterpret_cast<VectorType &>(data); }
        Vc_ALWAYS_INLINE Vc_PURE const VectorType &v() const { return reinterpret_cast<const VectorType &>(data); }

#ifdef VC_MSVC
        Vc_ALWAYS_INLINE EntryType &ref(size_t index) {
            return accessScalar<EntryType>(data, index);
        }

        Vc_ALWAYS_INLINE EntryType m(size_t index) const {
            return accessScalar<EntryType>(data, index);
        }

        Vc_INTRINSIC void set(size_t index, EntryType x)
        {
            accessScalar<EntryType>(data, index) = x;
        }
#elif VC_USE_BUILTIN_VECTOR_TYPES
        Vc_INTRINSIC void set(size_t index, EntryType x) { data[index] = x; }

        Vc_ALWAYS_INLINE EntryType m(size_t index) const { return data[index]; }
        Vc_ALWAYS_INLINE EntryType &ref(size_t index) {
            return data[index];
        }
#else
        Vc_ALWAYS_INLINE Vc_PURE MayAlias<EntryType> &ref(size_t index) {
            return reinterpret_cast<MayAlias<EntryType> *>(&data)[index];
        }

        Vc_ALWAYS_INLINE Vc_PURE EntryType m(size_t index) const {
            return reinterpret_cast<const MayAlias<EntryType> *>(&data)[index];
        }

        Vc_INTRINSIC void set(size_t index, EntryType x) { ref(index) = x; }
#endif
#ifdef VC_USE_BUILTIN_VECTOR_TYPES
        Vc_ALWAYS_INLINE BuiltinType &builtin() { return data; }
        Vc_ALWAYS_INLINE const BuiltinType &builtin() const { return data; }
#endif

    private:
#ifdef VC_COMPILE_BENCHMARKS
    public:
#endif
        typename BuiltinTypeHelper<EntryType, VectorType>::Type data;
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
template<> Vc_ALWAYS_INLINE Vc_PURE MayAlias<double> &VectorMemoryUnion<__m128d, double>::ref(size_t index) {
    return vectorMemoryUnionAliasedMember<MayAlias<EntryType>>(&data, index);
}
template<> Vc_ALWAYS_INLINE Vc_PURE MayAlias<long long> &VectorMemoryUnion<__m128i, long long>::ref(size_t index) {
    return vectorMemoryUnionAliasedMember<MayAlias<EntryType>>(&data, index);
}
template<> Vc_ALWAYS_INLINE Vc_PURE MayAlias<unsigned long long> &VectorMemoryUnion<__m128i, unsigned long long>::ref(size_t index) {
    return vectorMemoryUnionAliasedMember<MayAlias<EntryType>>(&data, index);
}
#endif

}  // namespace Common
}  // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_STORAGE_H
