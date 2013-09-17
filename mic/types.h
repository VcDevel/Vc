/*  This file is part of the Vc library.

    Copyright (C) 2010-2011 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_TYPES_H
#define VC_MIC_TYPES_H

#include <cstdlib>
#include "intrinsics.h"
#include "../common/memoryfwd.h"
#include "macros.h"

#define VC_DOUBLE_V_SIZE 8
#define VC_FLOAT_V_SIZE 16
#define VC_SFLOAT_V_SIZE 16
#define VC_INT_V_SIZE 16
#define VC_UINT_V_SIZE 16
#define VC_SHORT_V_SIZE 16
#define VC_USHORT_V_SIZE 16

#include "../common/types.h"

Vc_PUBLIC_NAMESPACE_BEGIN
template<typename T> struct DetermineVectorEntryType { typedef T Type; };
// MIC does not support epi8/epu8 operations, thus we change the EntryType to int/uint
template<> struct DetermineVectorEntryType<char> { typedef int Type; };
template<> struct DetermineVectorEntryType<signed char> { typedef int Type; };
template<> struct DetermineVectorEntryType<unsigned char> { typedef unsigned int Type; };
// MIC does not support epi16/epu16 operations, thus we change the EntryType to int/uint
template<> struct DetermineVectorEntryType<short> { typedef int Type; };
template<> struct DetermineVectorEntryType<unsigned short> { typedef unsigned int Type; };
// MIC does not support epi64/epu64 operations, thus we change the EntryType to int/uint
template<> struct DetermineVectorEntryType<long> { typedef int Type; };
template<> struct DetermineVectorEntryType<unsigned long> { typedef unsigned int Type; };
// MIC does not support epi64/epu64 operations, thus we change the EntryType to int/uint
template<> struct DetermineVectorEntryType<long long> { typedef int Type; };
template<> struct DetermineVectorEntryType<unsigned long long> { typedef unsigned int Type; };
Vc_NAMESPACE_END

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

    template<typename T> struct VectorHelper;
    template<typename T> class VectorMultiplication;
    template<typename T> class Vector;
    template<typename T> struct SwizzledVector;
    template<typename T> class Mask;
    ALIGN(16) extern const char _IndexesFromZero[16];

    template<typename T> struct ConcatTypeHelper { typedef T Type; };
    template<> struct ConcatTypeHelper<         float> { typedef double Type; };
    template<> struct ConcatTypeHelper<           int> { typedef long long Type; };
    template<> struct ConcatTypeHelper<  unsigned int> { typedef unsigned long long Type; };
    template<> struct ConcatTypeHelper<         short> { typedef int Type; };
    template<> struct ConcatTypeHelper<unsigned short> { typedef unsigned int Type; };


    template<typename T> struct VectorTypeHelper;
    template<> struct VectorTypeHelper<         char > { typedef __m512i Type; };
    template<> struct VectorTypeHelper<  signed char > { typedef __m512i Type; };
    template<> struct VectorTypeHelper<unsigned char > { typedef __m512i Type; };
    template<> struct VectorTypeHelper<         short> { typedef __m512i Type; };
    template<> struct VectorTypeHelper<unsigned short> { typedef __m512i Type; };
    template<> struct VectorTypeHelper<         int  > { typedef __m512i Type; };
    template<> struct VectorTypeHelper<unsigned int  > { typedef __m512i Type; };
    template<> struct VectorTypeHelper<         long > { typedef __m512i Type; };
    template<> struct VectorTypeHelper<unsigned long > { typedef __m512i Type; };
    template<> struct VectorTypeHelper<         long long> { typedef __m512i Type; };
    template<> struct VectorTypeHelper<unsigned long long> { typedef __m512i Type; };
    template<> struct VectorTypeHelper<         float> { typedef __m512  Type; };
    template<> struct VectorTypeHelper<        double> { typedef __m512d Type; };
    template<> struct VectorTypeHelper<       __m512i> { typedef __m512i Type; };
    template<> struct VectorTypeHelper<       __m512 > { typedef __m512  Type; };
    template<> struct VectorTypeHelper<       __m512d> { typedef __m512d Type; };

    template<typename T> struct MaskTypeHelper { typedef __mmask16 Type; };
    template<> struct MaskTypeHelper<__m512d>  { typedef __mmask8  Type; };
    template<> struct MaskTypeHelper< double>  { typedef __mmask8  Type; };

    template<typename T> struct ReturnTypeHelper { typedef char Type; };
    template<> struct ReturnTypeHelper<unsigned int> { typedef unsigned char Type; };
    template<> struct ReturnTypeHelper<int> { typedef signed char Type; };
    template<typename T> const typename ReturnTypeHelper<T>::Type *IndexesFromZeroHelper() {
        return reinterpret_cast<const typename ReturnTypeHelper<T>::Type *>(&_IndexesFromZero[0]);
    }

    template<size_t Size> struct IndexScaleHelper;
    template<> struct IndexScaleHelper<8> { static inline _MM_INDEX_SCALE_ENUM value() { return _MM_SCALE_4; } };
    template<> struct IndexScaleHelper<4> { static inline _MM_INDEX_SCALE_ENUM value() { return _MM_SCALE_4; } };
    template<> struct IndexScaleHelper<2> { static inline _MM_INDEX_SCALE_ENUM value() { return _MM_SCALE_2; } };
    template<> struct IndexScaleHelper<1> { static inline _MM_INDEX_SCALE_ENUM value() { return _MM_SCALE_1; } };
    template<typename T> struct IndexScale {
        static inline _MM_INDEX_SCALE_ENUM value() { return IndexScaleHelper<sizeof(T)>::value(); }
    };

    template<typename EntryType, typename MemType> struct UpDownConversion;
    template<> struct UpDownConversion<double, double> {
        constexpr operator _MM_DOWNCONV_PD_ENUM() const { return _MM_DOWNCONV_PD_NONE; }
        constexpr operator _MM_UPCONV_PD_ENUM() const { return _MM_UPCONV_PD_NONE; }
    };
    template<> struct UpDownConversion<float, float> {
        constexpr operator _MM_DOWNCONV_PS_ENUM() const { return _MM_DOWNCONV_PS_NONE; }
        constexpr operator _MM_UPCONV_PS_ENUM() const { return _MM_UPCONV_PS_NONE; }
    };
    /*template<> struct UpDownConversion<float, half_float> {
        constexpr operator _MM_DOWNCONV_PS_ENUM() const { return _MM_DOWNCONV_PS_FLOAT16; }
        constexpr operator _MM_UPCONV_PS_ENUM() const { return _MM_UPCONV_PS_FLOAT16; }
    };*/
    template<> struct UpDownConversion<float, unsigned char> {
        constexpr operator _MM_DOWNCONV_PS_ENUM() const { return _MM_DOWNCONV_PS_UINT8; }
        constexpr operator _MM_UPCONV_PS_ENUM() const { return _MM_UPCONV_PS_UINT8; }
    };
    template<> struct UpDownConversion<float, signed char> {
        constexpr operator _MM_DOWNCONV_PS_ENUM() const { return _MM_DOWNCONV_PS_SINT8; }
        constexpr operator _MM_UPCONV_PS_ENUM() const { return _MM_UPCONV_PS_SINT8; }
    };
    template<> struct UpDownConversion<float, unsigned short> {
        constexpr operator _MM_DOWNCONV_PS_ENUM() const { return _MM_DOWNCONV_PS_UINT16; }
        constexpr operator _MM_UPCONV_PS_ENUM() const { return _MM_UPCONV_PS_UINT16; }
    };
    template<> struct UpDownConversion<float, signed short> {
        constexpr operator _MM_DOWNCONV_PS_ENUM() const { return _MM_DOWNCONV_PS_SINT16; }
        constexpr operator _MM_UPCONV_PS_ENUM() const { return _MM_UPCONV_PS_SINT16; }
    };
    template<> struct UpDownConversion<unsigned int, char> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_SINT8; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_SINT8; }
    };
    template<> struct UpDownConversion<unsigned int, signed char> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_SINT8; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_SINT8; }
    };
    template<> struct UpDownConversion<unsigned int, unsigned char> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_UINT8; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_UINT8; }
    };
    template<> struct UpDownConversion<unsigned int, short> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_SINT16; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_SINT16; }
    };
    template<> struct UpDownConversion<unsigned int, unsigned short> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_UINT16; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_UINT16; }
    };
    template<> struct UpDownConversion<unsigned int, int> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_NONE; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_NONE; }
    };
    template<> struct UpDownConversion<unsigned int, unsigned int> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_NONE; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_NONE; }
    };
    template<> struct UpDownConversion<int, char> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_SINT8; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_SINT8; }
    };
    template<> struct UpDownConversion<int, signed char> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_SINT8; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_SINT8; }
    };
    template<> struct UpDownConversion<int, unsigned char> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_UINT8; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_UINT8; }
    };
    template<> struct UpDownConversion<int, short> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_SINT16; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_SINT16; }
    };
    template<> struct UpDownConversion<int, unsigned short> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_UINT16; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_UINT16; }
    };
    template<> struct UpDownConversion<int, int> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_NONE; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_NONE; }
    };
    template<> struct UpDownConversion<int, unsigned int> {
        constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_NONE; }
        constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_NONE; }
    };

    enum { VectorAlignment = 64 };

    template<typename V = Vector<float> >
    class STRUCT_ALIGN1(sizeof(V)) VectorAlignedBaseT
    {
        public:
            FREE_STORE_OPERATORS_ALIGNED(sizeof(V))
    } STRUCT_ALIGN2(sizeof(V));

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_MIC_TYPES_H
