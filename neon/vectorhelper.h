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

#ifndef VC_NEON_VECTORHELPER_H_
#define VC_NEON_VECTORHELPER_H_

#include "types.h"
#include "../common/loadstoreflags.h"
#include <limits>
#include "const_data.h"
#include "macros.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace NEON
{
#define Vc_OP0(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name() { return code; }
#define Vc_OP1(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name(const VectorType a) { return code; }
#define Vc_OP2(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name(const VectorType a, const VectorType b) { return code; }
#define Vc_OP3(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name(const VectorType a, const VectorType b, const VectorType c) { return code; }

        template<> struct VectorHelper<_NEON128>
        {
            typedef _NEON128 VectorType;

            template<typename Flags> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const float *x, typename Flags::EnableIfAligned  = nullptr) { return vld1q_f32(x); }
// How to check aligned or not?
            template<typename Flags> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const float *x, typename Flags::EnableIfUnaligned = nullptr) { return vld1q_f32(x); }
            template<typename Flags> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const float *x, typename Flags::EnableIfStreaming = nullptr) { return ; }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VectorType x, typename Flags::EnableIfAligned               = nullptr) {  }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VectorType x, typename Flags::EnableIfUnalignedNotStreaming = nullptr) {  }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VectorType x, typename Flags::EnableIfStreaming             = nullptr) {  }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VectorType x, typename Flags::EnableIfUnalignedAndStreaming = nullptr) { }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VectorType x, VectorType m) { }
        };


        template<> struct VectorHelper<_NEON128D>
        {
            typedef _NEON128D VectorType;
            template<typename Flags> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const double *x, typename Flags::EnableIfAligned   = nullptr) { return _mm_load_pd(x); }
            template<typename Flags> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const double *x, typename Flags::EnableIfUnaligned = nullptr) { return ; }
            template<typename Flags> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const double *x, typename Flags::EnableIfStreaming = nullptr) { return ; }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VectorType x, typename Flags::EnableIfAligned               = nullptr) { ; }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VectorType x, typename Flags::EnableIfUnalignedNotStreaming = nullptr) { ; }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VectorType x, typename Flags::EnableIfStreaming             = nullptr) { ; }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VectorType x, typename Flags::EnableIfUnalignedAndStreaming = nullptr) {  }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VectorType x, VectorType m) { ; }
        };

        template<> struct VectorHelper<_NEON128I>
        {
            typedef _NEON128I VectorType;
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const T *x, typename Flags::EnableIfAligned   = nullptr) { return ; }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const T *x, typename Flags::EnableIfUnaligned = nullptr) { return ; }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const T *x, typename Flags::EnableIfStreaming = nullptr) { return ; }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VectorType x, typename Flags::EnableIfAligned               = nullptr) {  }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VectorType x, typename Flags::EnableIfUnalignedNotStreaming = nullptr) {; }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VectorType x, typename Flags::EnableIfStreaming             = nullptr) { }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VectorType x, typename Flags::EnableIfUnalignedAndStreaming = nullptr) { }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VectorType x, VectorType m) { _mm_maskmoveu_si128(x, m, reinterpret_cast<char *>(mem)); }
        };
        static Vc_ALWAYS_INLINE Vc_CONST VectorType op(const VectorType a) { return ; }
        static Vc_ALWAYS_INLINE Vc_CONST VectorType op(const VectorType a, const VectorType b) { return ; }
        static Vc_ALWAYS_INLINE Vc_CONST VectorType op(const VectorType a, const VectorType b) { return ; }
        static Vc_ALWAYS_INLINE Vc_CONST VectorType op(const VectorType a, const VectorType b) { return ; }
        static Vc_ALWAYS_INLINE Vc_CONST VectorType op(const VectorType a, const VectorType b) { return ; }
        static Vc_ALWAYS_INLINE Vc_CONST VectorType min(VectorType a, VectorType b) { return ; }
        static Vc_ALWAYS_INLINE Vc_CONST VectorType max(VectorType a, VectorType b) { return ; }

        template<> struct VectorHelper<double> {
            typedef _NEON128D VectorType;
            typedef double EntryType;
            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VectorType a, _NEON128 mask) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const double a) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const double a, const double b) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one()  { return ; }// set(1.); }
            static Vc_ALWAYS_INLINE void fma(VectorType &v1, VectorType v2, VectorType v3) {
            }
            static inline void fma(VectorType &v1, VectorType v2, VectorType v3) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType rsqrt(VectorType x) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType reciprocal(VectorType x) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isNaN(VectorType x) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isFinite(VectorType x) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isInfinite(VectorType x) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType abs(const VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VectorType a) {
            }
        };

        template<> struct VectorHelper<float> {
            typedef float EntryType;
            typedef _NEON128 VectorType;
            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VectorType a, _NEON128 mask) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const float a) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const float a, const float b, const float c, const float d) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one()  { return ; }// set(1.f); }
            static Vc_ALWAYS_INLINE Vc_CONST _NEON128 concat(_NEON128D a, _NEON128D b) { return ; }
            static Vc_ALWAYS_INLINE void fma(VectorType &v1, VectorType v2, VectorType v3) {
            }
            static inline void fma(VectorType &v1, VectorType v2, VectorType v3) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isNaN(VectorType x) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isFinite(VectorType x) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isInfinite(VectorType x) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType reciprocal(VectorType x) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType abs(const VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VectorType a) {
            }
        };

        template<> struct VectorHelper<int> {
            typedef int EntryType;
            typedef _NEON128I VectorType;
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VectorType a, _NEON128 mask) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one() { return ; }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const int a) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const int a, const int b, const int c, const int d) { return ; }

            static Vc_ALWAYS_INLINE void fma(VectorType &v1, VectorType v2, VectorType v3) { v1 = add(mul(v1, v2), v3); }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftLeft(VectorType a, int shift) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftRight(VectorType a, int shift) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType abs(const VectorType a) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType min(VectorType a, VectorType b) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType max(VectorType a, VectorType b) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType mul(VectorType a, VectorType b) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VectorType a) {
            }
            static inline Vc_CONST VectorType mul(const VectorType a, const VectorType b) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VectorType a) { return a; }
        };

        template<> struct VectorHelper<unsigned int> {
            typedef unsigned int EntryType;
            typedef _NEON128I VectorType;
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VectorType a, _NEON128 mask) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one() { return ; }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType min(VectorType a, VectorType b) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType max(VectorType a, VectorType b) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VectorType a) {
            }

            static Vc_ALWAYS_INLINE void fma(VectorType &v1, VectorType v2, VectorType v3) { v1 = add(mul(v1, v2), v3); }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType mul(const VectorType a, const VectorType b) {
                return VectorHelper<int>::mul(a, b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftLeft(VectorType a, int shift) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftRight(VectorType a, int shift) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const unsigned int a) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const unsigned int a, const unsigned int b, const unsigned int c, const unsigned int d) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VectorType a) { return a; }
        };

        template<> struct VectorHelper<signed short> {
            typedef _NEON128I VectorType;
            typedef signed short EntryType;

            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VectorType a, _NEON128 mask) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST _NEON128I concat(_NEON128I a, _NEON128I b) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST _NEON128I expand0(_NEON128I x) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST _NEON128I expand1(_NEON128I x) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one() { return ; }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftLeft(VectorType a, int shift) {
                return ;
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftRight(VectorType a, int shift) {
                return ;
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const EntryType a) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const EntryType a, const EntryType b, const EntryType c, const EntryType d,
                    const EntryType e, const EntryType f, const EntryType g, const EntryType h) {
                return ;
            }

            static Vc_ALWAYS_INLINE void fma(VectorType &v1, VectorType v2, VectorType v3) {
                v1 = add(mul(v1, v2), v3); }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType abs(const VectorType a) { return ; }

            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VectorType a) {
            }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VectorType a) { return a; }
        };

        template<> struct VectorHelper<unsigned short> {
            typedef _NEON128I VectorType;
            typedef unsigned short EntryType;
            Vc_OP_CAST_(or_) Vc_OP_CAST_(and_) Vc_OP_CAST_(xor_)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VectorType a, _NEON128 mask) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST _NEON128I concat(_NEON128I a, _NEON128I b) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST _NEON128I concat(_NEON128I a, _NEON128I b) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST _NEON128I expand0(_NEON128I x) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST _NEON128I expand1(_NEON128I x) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one() { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType min(VectorType a, VectorType b) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType max(VectorType a, VectorType b) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftLeft(VectorType a, int shift) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftRight(VectorType a, int shift) {
            }
            static Vc_ALWAYS_INLINE void fma(VectorType &v1, VectorType v2, VectorType v3) { v1 = add(mul(v1, v2), v3); }

            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VectorType a) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const EntryType a) { return ; }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const EntryType a, const EntryType b, const EntryType c,
                    const EntryType d, const EntryType e, const EntryType f,
                    const EntryType g, const EntryType h) {
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VectorType a) { return a; }
        };

}  // namespace NEON
Vc_VERSIONED_NAMESPACE_END

#include "vectorhelper.tcc"

#endif // VC_NEON_VECTORHELPER_H_
