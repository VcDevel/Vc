/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_AVX_CONST_H_
#define VC_AVX_CONST_H_

#include <cstddef>
#include "types.h"
#include "const_data.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace AVX
{
    template<typename T> struct IndexesFromZeroData;
    template<> struct IndexesFromZeroData<int> {
        static Vc_ALWAYS_INLINE Vc_CONST const int *address() { return reinterpret_cast<const int *>(&_IndexesFromZero32[0]); }
    };
    template<> struct IndexesFromZeroData<unsigned int> {
        static Vc_ALWAYS_INLINE Vc_CONST const unsigned int *address() { return &_IndexesFromZero32[0]; }
    };
    template<> struct IndexesFromZeroData<short> {
        static Vc_ALWAYS_INLINE Vc_CONST const short *address() { return reinterpret_cast<const short *>(&_IndexesFromZero16[0]); }
    };
    template<> struct IndexesFromZeroData<unsigned short> {
        static Vc_ALWAYS_INLINE Vc_CONST const unsigned short *address() { return &_IndexesFromZero16[0]; }
    };
    template<> struct IndexesFromZeroData<signed char> {
        static Vc_ALWAYS_INLINE Vc_CONST const signed char *address() { return reinterpret_cast<const signed char *>(&_IndexesFromZero8[0]); }
    };
    template<> struct IndexesFromZeroData<char> {
        static Vc_ALWAYS_INLINE Vc_CONST const char *address() { return reinterpret_cast<const char *>(&_IndexesFromZero8[0]); }
    };
    template<> struct IndexesFromZeroData<unsigned char> {
        static Vc_ALWAYS_INLINE Vc_CONST const unsigned char *address() { return &_IndexesFromZero8[0]; }
    };

    template<typename _T> struct Const
    {
        typedef Vector<_T> V;
        typedef typename V::EntryType T;
        typedef typename V::Mask M;

        static Vc_ALWAYS_INLINE Vc_CONST V _pi_4()        { return V(c_trig<T>::data[0]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _pi_4_hi()     { return V(c_trig<T>::data[1]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _pi_4_rem1()   { return V(c_trig<T>::data[2]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _pi_4_rem2()   { return V(c_trig<T>::data[3]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _1_16()        { return V(c_trig<T>::data[4]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _16()          { return V(c_trig<T>::data[5]); }

        static Vc_ALWAYS_INLINE Vc_CONST V cosCoeff(int i)   { return V(c_trig<T>::data[( 8 + i)]); }
        static Vc_ALWAYS_INLINE Vc_CONST V sinCoeff(int i)   { return V(c_trig<T>::data[(14 + i)]); }
        static Vc_ALWAYS_INLINE Vc_CONST V atanP(int i)      { return V(c_trig<T>::data[(24 + i)]); }
        static Vc_ALWAYS_INLINE Vc_CONST V atanQ(int i)      { return V(c_trig<T>::data[(29 + i)]); }
        static Vc_ALWAYS_INLINE Vc_CONST V atanThrsHi()      { return V(c_trig<T>::data[34]); }
        static Vc_ALWAYS_INLINE Vc_CONST V atanThrsLo()      { return V(c_trig<T>::data[35]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _pi_2_rem()       { return V(c_trig<T>::data[36]); }
        static Vc_ALWAYS_INLINE Vc_CONST V lossThreshold()   { return V(c_trig<T>::data[20]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _4_pi()           { return V(c_trig<T>::data[21]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _pi_2()           { return V(c_trig<T>::data[22]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _pi()             { return V(c_trig<T>::data[23]); }
        static Vc_ALWAYS_INLINE Vc_CONST V asinCoeff0(int i) { return V(c_trig<T>::data[(40 + i)]); }
        static Vc_ALWAYS_INLINE Vc_CONST V asinCoeff1(int i) { return V(c_trig<T>::data[(45 + i)]); }
        static Vc_ALWAYS_INLINE Vc_CONST V asinCoeff2(int i) { return V(c_trig<T>::data[(49 + i)]); }
        static Vc_ALWAYS_INLINE Vc_CONST V asinCoeff3(int i) { return V(c_trig<T>::data[(55 + i)]); }
        static Vc_ALWAYS_INLINE Vc_CONST V smallAsinInput()  { return V(c_trig<T>::data[37]); }
        static Vc_ALWAYS_INLINE Vc_CONST V largeAsinInput()  { return V(c_trig<T>::data[38]); }

        static Vc_ALWAYS_INLINE Vc_CONST M exponentMask() { return M(V(c_log<T>::d(1)).data()); }
        static Vc_ALWAYS_INLINE Vc_CONST V _1_2()         { return V(c_log<T>::d(18)); }
        static Vc_ALWAYS_INLINE Vc_CONST V _1_sqrt2()     { return V(c_log<T>::d(15)); }
        static Vc_ALWAYS_INLINE Vc_CONST V P(int i)       { return V(c_log<T>::d(2 + i)); }
        static Vc_ALWAYS_INLINE Vc_CONST V Q(int i)       { return V(c_log<T>::d(8 + i)); }
        static Vc_ALWAYS_INLINE Vc_CONST V min()          { return V(c_log<T>::d(14)); }
        static Vc_ALWAYS_INLINE Vc_CONST V ln2_small()    { return V(c_log<T>::d(17)); }
        static Vc_ALWAYS_INLINE Vc_CONST V ln2_large()    { return V(c_log<T>::d(16)); }
        static Vc_ALWAYS_INLINE Vc_CONST V neginf()       { return V(c_log<T>::d(13)); }
        static Vc_ALWAYS_INLINE Vc_CONST V log10_e()      { return V(c_log<T>::d(19)); }
        static Vc_ALWAYS_INLINE Vc_CONST V log2_e()       { return V(c_log<T>::d(20)); }

        static Vc_ALWAYS_INLINE_L Vc_CONST_L V highMask() Vc_ALWAYS_INLINE_R Vc_CONST_R;
    };

    template<> Vc_ALWAYS_INLINE Vc_CONST Vector<float>  Const<float>::highMask() { return _mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::highMaskFloat)); }
    template<> Vc_ALWAYS_INLINE Vc_CONST Vector<double> Const<double>::highMask() { return _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::highMaskDouble)); }
}  // namespace AVX

namespace AVX2
{
using AVX::IndexesFromZeroData;
using AVX::Const;
}  // namespace AVX2
}  // namespace Vc

#endif // VC_AVX_CONST_H_
