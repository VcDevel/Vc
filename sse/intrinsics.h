/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

#ifndef SSE_INTRINSICS_H
#define SSE_INTRINSICS_H

#if defined(_MSC_VER) && !defined(__midl)
// MSVC sucks. If you include intrin.h you get all SSE and AVX intrinsics
// declared. This is a problem because we need to implement the intrinsics
// that are not supported in hardware ourselves.
// Something always includes intrin.h even if you don't
// do it explicitly. Therefore we try to be the first to include it
// but with __midl defined, in which case it is basically empty.
#ifdef __INTRIN_H_
#error "intrin.h was already included, polluting the namespace. Please fix your code to include the Vc headers before anything that includes intrin.h. (Vc will declare the relevant intrinsics as they are required by some system headers.)"
#endif
#define __midl
#include <intrin.h>
#undef __midl
#include <crtdefs.h>
#include <setjmp.h>
#include <stddef.h>
extern "C" {

#ifdef _WIN64
_CRTIMP double ceil(_In_ double);
__int64 _InterlockedDecrement64(__int64 volatile *);
__int64 _InterlockedExchange64(__int64 volatile *, __int64);
void * _InterlockedExchangePointer(void * volatile *, void *);
__int64 _InterlockedExchangeAdd64(__int64 volatile *, __int64);
void *_InterlockedCompareExchangePointer (void * volatile *, void *, void *);
__int64 _InterlockedIncrement64(__int64 volatile *);
int __cdecl _setjmpex(jmp_buf);
void __faststorefence(void);
__int64 __mulh(__int64,__int64);
unsigned __int64 __umulh(unsigned __int64,unsigned __int64);
unsigned __int64 __readcr0(void);
unsigned __int64 __readcr2(void);
unsigned __int64 __readcr3(void);
unsigned __int64 __readcr4(void);
unsigned __int64 __readcr8(void);
void __writecr0(unsigned __int64);
void __writecr3(unsigned __int64);
void __writecr4(unsigned __int64);
void __writecr8(unsigned __int64);
unsigned __int64 __readdr(unsigned int);
void __writedr(unsigned int, unsigned __int64);
unsigned __int64 __readeflags(void);
void __writeeflags(unsigned __int64);
void __movsq(unsigned long long *, unsigned long long const *, size_t);
unsigned char __readgsbyte(unsigned long Offset);
unsigned short __readgsword(unsigned long Offset);
unsigned long __readgsdword(unsigned long Offset);
unsigned __int64 __readgsqword(unsigned long Offset);
void __writegsbyte(unsigned long Offset, unsigned char Data);
void __writegsword(unsigned long Offset, unsigned short Data);
void __writegsdword(unsigned long Offset, unsigned long Data);
void __writegsqword(unsigned long Offset, unsigned __int64 Data);
void __addgsbyte(unsigned long Offset, unsigned char Data);
void __addgsword(unsigned long Offset, unsigned short Data);
void __addgsdword(unsigned long Offset, unsigned long Data);
void __addgsqword(unsigned long Offset, unsigned __int64 Data);
void __incgsbyte(unsigned long Offset);
void __incgsword(unsigned long Offset);
void __incgsdword(unsigned long Offset);
void __incgsqword(unsigned long Offset);
unsigned char __vmx_vmclear(unsigned __int64*);
unsigned char __vmx_vmlaunch(void);
unsigned char __vmx_vmptrld(unsigned __int64*);
unsigned char __vmx_vmread(size_t, size_t*);
unsigned char __vmx_vmresume(void);
unsigned char __vmx_vmwrite(size_t, size_t);
unsigned char __vmx_on(unsigned __int64*);
void __stosq(unsigned __int64 *,  unsigned __int64, size_t);
unsigned char _interlockedbittestandset64(__int64 volatile *a, __int64 b);
unsigned char _interlockedbittestandreset64(__int64 volatile *a, __int64 b);
short _InterlockedCompareExchange16_np(short volatile *Destination, short Exchange, short Comparand);
long _InterlockedCompareExchange_np (long volatile *, long, long);
__int64 _InterlockedCompareExchange64_np(__int64 volatile *, __int64, __int64);
void *_InterlockedCompareExchangePointer_np (void * volatile *, void *, void *);
unsigned char _InterlockedCompareExchange128(__int64 volatile *, __int64, __int64, __int64 *);
unsigned char _InterlockedCompareExchange128_np(__int64 volatile *, __int64, __int64, __int64 *);
long _InterlockedAnd_np(long volatile *, long);
char _InterlockedAnd8_np(char volatile *, char);
short _InterlockedAnd16_np(short volatile *, short);
__int64 _InterlockedAnd64_np(__int64 volatile *, __int64);
long _InterlockedOr_np(long volatile *, long);
char _InterlockedOr8_np(char volatile *, char);
short _InterlockedOr16_np(short volatile *, short);
__int64 _InterlockedOr64_np(__int64 volatile *, __int64);
long _InterlockedXor_np(long volatile *, long);
char _InterlockedXor8_np(char volatile *, char);
short _InterlockedXor16_np(short volatile *, short);
__int64 _InterlockedXor64_np(__int64 volatile *, __int64);
unsigned __int64 __lzcnt64(unsigned __int64);
unsigned __int64 __popcnt64(unsigned __int64);
__int64 _InterlockedOr64(__int64 volatile *, __int64);
__int64 _InterlockedXor64(__int64 volatile *, __int64);
__int64 _InterlockedAnd64(__int64 volatile *, __int64);
unsigned char _bittest64(__int64 const *a, __int64 b);
unsigned char _bittestandset64(__int64 *a, __int64 b);
unsigned char _bittestandreset64(__int64 *a, __int64 b);
unsigned char _bittestandcomplement64(__int64 *a, __int64 b);
unsigned char _BitScanForward64(unsigned long* Index, unsigned __int64 Mask);
unsigned char _BitScanReverse64(unsigned long* Index, unsigned __int64 Mask);
unsigned __int64 __shiftleft128(unsigned __int64 LowPart, unsigned __int64 HighPart, unsigned char Shift);
unsigned __int64 __shiftright128(unsigned __int64 LowPart, unsigned __int64 HighPart, unsigned char Shift);
unsigned __int64 _umul128(unsigned __int64 multiplier, unsigned __int64 multiplicand, unsigned __int64 *highproduct);
__int64 _mul128(__int64 multiplier, __int64 multiplicand, __int64 *highproduct);
#endif

long _InterlockedOr(long volatile *, long);
char _InterlockedOr8(char volatile *, char);
short _InterlockedOr16(short volatile *, short);
long _InterlockedXor(long volatile *, long);
char _InterlockedXor8(char volatile *, char);
short _InterlockedXor16(short volatile *, short);
long _InterlockedAnd(long volatile *, long);
char _InterlockedAnd8(char volatile *, char);
short _InterlockedAnd16(short volatile *, short);
unsigned char _bittest(long const *a, long b);
unsigned char _bittestandset(long *a, long b);
unsigned char _bittestandreset(long *a, long b);
unsigned char _bittestandcomplement(long *a, long b);
unsigned char _BitScanForward(unsigned long* Index, unsigned long Mask);
unsigned char _BitScanReverse(unsigned long* Index, unsigned long Mask);
_CRTIMP wchar_t * __cdecl wcscat( _Pre_cap_for_(_Source) _Prepost_z_ wchar_t *, _In_z_ const wchar_t * _Source);
_Check_return_ _CRTIMP int __cdecl wcscmp(_In_z_ const wchar_t *,_In_z_  const wchar_t *);
_CRTIMP wchar_t * __cdecl wcscpy(_Pre_cap_for_(_Source) _Post_z_ wchar_t *, _In_z_ const wchar_t * _Source);
_Check_return_ _CRTIMP size_t __cdecl wcslen(_In_z_ const wchar_t *);
#pragma warning(suppress: 4985)
_CRTIMP wchar_t * __cdecl _wcsset(_Inout_z_ wchar_t *, wchar_t);
void _ReadBarrier(void);
unsigned char _rotr8(unsigned char value, unsigned char shift);
unsigned short _rotr16(unsigned short value, unsigned char shift);
unsigned char _rotl8(unsigned char value, unsigned char shift);
unsigned short _rotl16(unsigned short value, unsigned char shift);
short _InterlockedIncrement16(short volatile *Addend);
short _InterlockedDecrement16(short volatile *Addend);
short _InterlockedCompareExchange16(short volatile *Destination, short Exchange, short Comparand);
void __nvreg_save_fence(void);
void __nvreg_restore_fence(void);

#ifdef _M_IX86
unsigned long __readcr0(void);
unsigned long __readcr2(void);
unsigned long __readcr3(void);
unsigned long __readcr4(void);
unsigned long __readcr8(void);
void __writecr0(unsigned);
void __writecr3(unsigned);
void __writecr4(unsigned);
void __writecr8(unsigned);
unsigned __readdr(unsigned int);
void __writedr(unsigned int, unsigned);
unsigned __readeflags(void);
void __writeeflags(unsigned);
void __addfsbyte(unsigned long Offset, unsigned char Data);
void __addfsword(unsigned long Offset, unsigned short Data);
void __addfsdword(unsigned long Offset, unsigned long Data);
void __incfsbyte(unsigned long Offset);
void __incfsword(unsigned long Offset);
void __incfsdword(unsigned long Offset);
unsigned char __readfsbyte(unsigned long Offset);
unsigned short __readfsword(unsigned long Offset);
unsigned long __readfsdword(unsigned long Offset);
unsigned __int64 __readfsqword(unsigned long Offset);
void __writefsbyte(unsigned long Offset, unsigned char Data);
void __writefsword(unsigned long Offset, unsigned short Data);
void __writefsdword(unsigned long Offset, unsigned long Data);
void __writefsqword(unsigned long Offset, unsigned __int64 Data);
long _InterlockedAddLargeStatistic(__int64 volatile *, long);
#endif

_Ret_bytecap_(_Size) void * __cdecl _alloca(size_t _Size);
int __cdecl abs(_In_ int);
_Check_return_ unsigned short __cdecl _byteswap_ushort(_In_ unsigned short value);
_Check_return_ unsigned long __cdecl _byteswap_ulong(_In_ unsigned long value);
_Check_return_ unsigned __int64 __cdecl _byteswap_uint64(_In_ unsigned __int64 value);
void __cdecl __debugbreak(void);
void __cdecl _disable(void);
__int64 __emul(int,int);
unsigned __int64 __emulu(unsigned int,unsigned int);
void __cdecl _enable(void);
long __cdecl _InterlockedDecrement(long volatile *);
long _InterlockedExchange(long volatile *, long);
short _InterlockedExchange16(short volatile *, short);
char _InterlockedExchange8(char volatile *, char);
long _InterlockedExchangeAdd(long volatile *, long);
short _InterlockedExchangeAdd16(short volatile *, short);
char _InterlockedExchangeAdd8(char volatile *, char);
long _InterlockedCompareExchange (long volatile *, long, long);
__int64 _InterlockedCompareExchange64(__int64 volatile *, __int64, __int64);
long __cdecl _InterlockedIncrement(long volatile *);
int __cdecl _inp(unsigned short);
int __cdecl inp(unsigned short);
unsigned long __cdecl _inpd(unsigned short);
unsigned long __cdecl inpd(unsigned short);
unsigned short __cdecl _inpw(unsigned short);
unsigned short __cdecl inpw(unsigned short);
long __cdecl labs(_In_ long);
_Check_return_ unsigned long __cdecl _lrotl(_In_ unsigned long,_In_ int);
_Check_return_ unsigned long __cdecl _lrotr(_In_ unsigned long,_In_ int);
unsigned __int64  __ll_lshift(unsigned __int64,int);
__int64  __ll_rshift(__int64,int);
_Check_return_ int __cdecl memcmp(_In_opt_bytecount_(_Size) const void *,_In_opt_bytecount_(_Size) const void *,_In_ size_t _Size);
void * __cdecl memcpy(_Out_opt_bytecapcount_(_Size) void *,_In_opt_bytecount_(_Size) const void *,_In_ size_t _Size);
void * __cdecl memset(_Out_opt_bytecapcount_(_Size) void *,_In_ int,_In_ size_t _Size);
int __cdecl _outp(unsigned short,int);
int __cdecl outp(unsigned short,int);
unsigned long __cdecl _outpd(unsigned short,unsigned long);
unsigned long __cdecl outpd(unsigned short,unsigned long);
unsigned short __cdecl _outpw(unsigned short,unsigned short);
unsigned short __cdecl outpw(unsigned short,unsigned short);
void * _ReturnAddress(void);
_Check_return_ unsigned int __cdecl _rotl(_In_ unsigned int,_In_ int);
_Check_return_ unsigned int __cdecl _rotr(_In_ unsigned int,_In_ int);
int __cdecl _setjmp(jmp_buf);
_Check_return_ int __cdecl strcmp(_In_z_ const char *,_In_z_ const char *);
_Check_return_ size_t __cdecl strlen(_In_z_ const char *);
char * __cdecl strset(_Inout_z_ char *,_In_ int);
unsigned __int64 __ull_rshift(unsigned __int64,int);
void * _AddressOfReturnAddress(void);

void _WriteBarrier(void);
void _ReadWriteBarrier(void);
void __wbinvd(void);
void __invlpg(void*);
unsigned __int64 __readmsr(unsigned long);
void __writemsr(unsigned long, unsigned __int64);
unsigned __int64 __rdtsc(void);
void __movsb(unsigned char *, unsigned char const *, size_t);
void __movsw(unsigned short *, unsigned short const *, size_t);
void __movsd(unsigned long *, unsigned long const *, size_t);
unsigned char __inbyte(unsigned short Port);
unsigned short __inword(unsigned short Port);
unsigned long __indword(unsigned short Port);
void __outbyte(unsigned short Port, unsigned char Data);
void __outword(unsigned short Port, unsigned short Data);
void __outdword(unsigned short Port, unsigned long Data);
void __inbytestring(unsigned short Port, unsigned char *Buffer, unsigned long Count);
void __inwordstring(unsigned short Port, unsigned short *Buffer, unsigned long Count);
void __indwordstring(unsigned short Port, unsigned long *Buffer, unsigned long Count);
void __outbytestring(unsigned short Port, unsigned char *Buffer, unsigned long Count);
void __outwordstring(unsigned short Port, unsigned short *Buffer, unsigned long Count);
void __outdwordstring(unsigned short Port, unsigned long *Buffer, unsigned long Count);
unsigned int __getcallerseflags();
void __vmx_vmptrst(unsigned __int64 *);
void __vmx_off(void);
void __svm_clgi(void);
void __svm_invlpga(void*, int);
void __svm_skinit(int);
void __svm_stgi(void);
void __svm_vmload(size_t);
void __svm_vmrun(size_t);
void __svm_vmsave(size_t);
void __halt(void);
void __sidt(void*);
void __lidt(void*);
void __ud2(void);
void __nop(void);
void __stosb(unsigned char *, unsigned char, size_t);
void __stosw(unsigned short *,  unsigned short, size_t);
void __stosd(unsigned long *,  unsigned long, size_t);
unsigned char _interlockedbittestandset(long volatile *a, long b);
unsigned char _interlockedbittestandreset(long volatile *a, long b);
void __cpuid(int a[4], int b);
void __cpuidex(int a[4], int b, int c);
unsigned __int64 __readpmc(unsigned long a);
unsigned long __segmentlimit(unsigned long a);
_Check_return_ unsigned __int64 __cdecl _rotl64(_In_ unsigned __int64,_In_ int);
_Check_return_ unsigned __int64 __cdecl _rotr64(_In_ unsigned __int64,_In_ int);
__int64 __cdecl _abs64(__int64);
void __int2c(void);
char _InterlockedCompareExchange8(char volatile *Destination, char Exchange, char Comparand);
unsigned short __lzcnt16(unsigned short);
unsigned int __lzcnt(unsigned int);
unsigned short __popcnt16(unsigned short);
unsigned int __popcnt(unsigned int);
unsigned __int64 __rdtscp(unsigned int*);
}
#endif

// MMX
#include <mmintrin.h>
// SSE
#include <xmmintrin.h>
// SSE2
#include <emmintrin.h>

#if defined(__GNUC__) && !defined(VC_IMPL_SSE2)
#error "SSE Vector class needs at least SSE2"
#endif

#include "const.h"
#include "macros.h"
#include <cstdlib>

#ifdef __3dNOW__
#include <mm3dnow.h>
#endif

namespace Vc
{
namespace SSE
{
    enum { VectorAlignment = 16 };

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6 && !defined(VC_DONT_FIX_SSE_SHIFT)
    static inline __m128i CONST _mm_sll_epi16(__m128i a, __m128i count) { __asm__("psllw %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m128i CONST _mm_sll_epi32(__m128i a, __m128i count) { __asm__("pslld %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m128i CONST _mm_sll_epi64(__m128i a, __m128i count) { __asm__("psllq %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m128i CONST _mm_srl_epi16(__m128i a, __m128i count) { __asm__("psrlw %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m128i CONST _mm_srl_epi32(__m128i a, __m128i count) { __asm__("psrld %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m128i CONST _mm_srl_epi64(__m128i a, __m128i count) { __asm__("psrlq %1,%0" : "+x"(a) : "x"(count)); return a; }
#endif

#if defined(__GNUC__) && !defined(NVALGRIND)
    static inline __m128i CONST _mm_setallone() { __m128i r; __asm__("pcmpeqb %0,%0":"=x"(r)); return r; }
#else
    static inline __m128i CONST _mm_setallone() { __m128i r = _mm_setzero_si128(); return _mm_cmpeq_epi8(r, r); }
#endif
    static inline __m128i CONST _mm_setallone_si128() { return _mm_setallone(); }
    static inline __m128d CONST _mm_setallone_pd() { return _mm_castsi128_pd(_mm_setallone()); }
    static inline __m128  CONST _mm_setallone_ps() { return _mm_castsi128_ps(_mm_setallone()); }

    static inline __m128i CONST _mm_setone_epi8 ()  { return _mm_set1_epi8(1); }
    static inline __m128i CONST _mm_setone_epu8 ()  { return _mm_setone_epi8(); }
    static inline __m128i CONST _mm_setone_epi16()  { return _mm_load_si128(reinterpret_cast<const __m128i *>(c_general::one16)); }
    static inline __m128i CONST _mm_setone_epu16()  { return _mm_setone_epi16(); }
    static inline __m128i CONST _mm_setone_epi32()  { return _mm_load_si128(reinterpret_cast<const __m128i *>(c_general::one32)); }
    static inline __m128i CONST _mm_setone_epu32()  { return _mm_setone_epi32(); }

    static inline __m128  CONST _mm_setone_ps()     { return _mm_load_ps(c_general::oneFloat); }
    static inline __m128d CONST _mm_setone_pd()     { return _mm_load_pd(c_general::oneDouble); }

    static inline __m128d CONST _mm_setabsmask_pd() { return _mm_load_pd(reinterpret_cast<const double *>(c_general::absMaskDouble)); }
    static inline __m128  CONST _mm_setabsmask_ps() { return _mm_load_ps(reinterpret_cast<const float *>(c_general::absMaskFloat)); }
    static inline __m128d CONST _mm_setsignmask_pd(){ return _mm_load_pd(reinterpret_cast<const double *>(c_general::signMaskDouble)); }
    static inline __m128  CONST _mm_setsignmask_ps(){ return _mm_load_ps(reinterpret_cast<const float *>(c_general::signMaskFloat)); }

    //X         static inline __m128i CONST _mm_setmin_epi8 () { return _mm_slli_epi8 (_mm_setallone_si128(),  7); }
    static inline __m128i CONST _mm_setmin_epi16() { return _mm_load_si128(reinterpret_cast<const __m128i *>(c_general::minShort)); }
    static inline __m128i CONST _mm_setmin_epi32() { return _mm_load_si128(reinterpret_cast<const __m128i *>(c_general::signMaskFloat)); }

    //X         static inline __m128i CONST _mm_cmplt_epu8 (__m128i a, __m128i b) { return _mm_cmplt_epi8 (
    //X                 _mm_xor_si128(a, _mm_setmin_epi8 ()), _mm_xor_si128(b, _mm_setmin_epi8 ())); }
    //X         static inline __m128i CONST _mm_cmpgt_epu8 (__m128i a, __m128i b) { return _mm_cmpgt_epi8 (
    //X                 _mm_xor_si128(a, _mm_setmin_epi8 ()), _mm_xor_si128(b, _mm_setmin_epi8 ())); }
    static inline __m128i CONST _mm_cmplt_epu16(__m128i a, __m128i b) { return _mm_cmplt_epi16(
            _mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16())); }
    static inline __m128i CONST _mm_cmpgt_epu16(__m128i a, __m128i b) { return _mm_cmpgt_epi16(
            _mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16())); }
    static inline __m128i CONST _mm_cmplt_epu32(__m128i a, __m128i b) { return _mm_cmplt_epi32(
            _mm_xor_si128(a, _mm_setmin_epi32()), _mm_xor_si128(b, _mm_setmin_epi32())); }
    static inline __m128i CONST _mm_cmpgt_epu32(__m128i a, __m128i b) { return _mm_cmpgt_epi32(
            _mm_xor_si128(a, _mm_setmin_epi32()), _mm_xor_si128(b, _mm_setmin_epi32())); }
} // namespace SSE
} // namespace Vc

// SSE3
#ifdef VC_IMPL_SSE3
#include <pmmintrin.h>
#elif defined _PMMINTRIN_H_INCLUDED
#error "SSE3 was disabled but something includes <pmmintrin.h>. Please fix your code."
#endif
// SSSE3
#ifdef VC_IMPL_SSSE3
#include <tmmintrin.h>
namespace Vc
{
namespace SSE
{

    // not overriding _mm_set1_epi8 because this one should only be used for non-constants
    static inline __m128i CONST set1_epi8(int a) {
#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 5
        return _mm_shuffle_epi8(_mm_cvtsi32_si128(a), _mm_setzero_si128());
#else
        // GCC 4.5 nows about the pshufb improvement
        return _mm_set1_epi8(a);
#endif
    }

} // namespace SSE
} // namespace Vc
#elif defined _TMMINTRIN_H_INCLUDED
#error "SSSE3 was disabled but something includes <tmmintrin.h>. Please fix your code."
#else
namespace Vc
{
namespace SSE
{
    static inline __m128i CONST _mm_abs_epi8 (__m128i a) {
        __m128i negative = _mm_cmplt_epi8 (a, _mm_setzero_si128());
        return _mm_add_epi8 (_mm_xor_si128(a, negative), _mm_and_si128(negative,  _mm_setone_epi8()));
    }
    // positive value:
    //   negative == 0
    //   a unchanged after xor
    //   0 >> 31 -> 0
    //   a + 0 -> a
    // negative value:
    //   negative == -1
    //   a xor -1 -> -a - 1
    //   -1 >> 31 -> 1
    //   -a - 1 + 1 -> -a
    static inline __m128i CONST _mm_abs_epi16(__m128i a) {
        __m128i negative = _mm_cmplt_epi16(a, _mm_setzero_si128());
        return _mm_add_epi16(_mm_xor_si128(a, negative), _mm_srli_epi16(negative, 15));
    }
    static inline __m128i CONST _mm_abs_epi32(__m128i a) {
        __m128i negative = _mm_cmplt_epi32(a, _mm_setzero_si128());
        return _mm_add_epi32(_mm_xor_si128(a, negative), _mm_srli_epi32(negative, 31));
    }
    static inline __m128i CONST set1_epi8(int a) {
        return _mm_set1_epi8(a);
    }
    static inline __m128i CONST _mm_alignr_epi8(__m128i a, __m128i b, const int s) {
        switch (s) {
            case  0: return b;
            case  1: return _mm_or_si128(_mm_slli_si128(a, 15), _mm_srli_si128(b,  1));
            case  2: return _mm_or_si128(_mm_slli_si128(a, 14), _mm_srli_si128(b,  2));
            case  3: return _mm_or_si128(_mm_slli_si128(a, 13), _mm_srli_si128(b,  3));
            case  4: return _mm_or_si128(_mm_slli_si128(a, 12), _mm_srli_si128(b,  4));
            case  5: return _mm_or_si128(_mm_slli_si128(a, 11), _mm_srli_si128(b,  5));
            case  6: return _mm_or_si128(_mm_slli_si128(a, 10), _mm_srli_si128(b,  6));
            case  7: return _mm_or_si128(_mm_slli_si128(a,  9), _mm_srli_si128(b,  7));
            case  8: return _mm_or_si128(_mm_slli_si128(a,  8), _mm_srli_si128(b,  8));
            case  9: return _mm_or_si128(_mm_slli_si128(a,  7), _mm_srli_si128(b,  9));
            case 10: return _mm_or_si128(_mm_slli_si128(a,  6), _mm_srli_si128(b, 10));
            case 11: return _mm_or_si128(_mm_slli_si128(a,  5), _mm_srli_si128(b, 11));
            case 12: return _mm_or_si128(_mm_slli_si128(a,  4), _mm_srli_si128(b, 12));
            case 13: return _mm_or_si128(_mm_slli_si128(a,  3), _mm_srli_si128(b, 13));
            case 14: return _mm_or_si128(_mm_slli_si128(a,  2), _mm_srli_si128(b, 14));
            case 15: return _mm_or_si128(_mm_slli_si128(a,  1), _mm_srli_si128(b, 15));
            case 16: return a;
            case 17: return _mm_srli_si128(a,  1);
            case 18: return _mm_srli_si128(a,  2);
            case 19: return _mm_srli_si128(a,  3);
            case 20: return _mm_srli_si128(a,  4);
            case 21: return _mm_srli_si128(a,  5);
            case 22: return _mm_srli_si128(a,  6);
            case 23: return _mm_srli_si128(a,  7);
            case 24: return _mm_srli_si128(a,  8);
            case 25: return _mm_srli_si128(a,  9);
            case 26: return _mm_srli_si128(a, 10);
            case 27: return _mm_srli_si128(a, 11);
            case 28: return _mm_srli_si128(a, 12);
            case 29: return _mm_srli_si128(a, 13);
            case 30: return _mm_srli_si128(a, 14);
            case 31: return _mm_srli_si128(a, 15);
        }
        return _mm_setzero_si128();
    }

} // namespace SSE
} // namespace Vc

#endif

// SSE4.1
#ifdef VC_IMPL_SSE4_1
#include <smmintrin.h>
#else
#ifdef _SMMINTRIN_H_INCLUDED
#error "SSE4.1 was disabled but something includes <smmintrin.h>. Please fix your code."
#endif
namespace Vc
{
namespace SSE
{
    static inline __m128d INTRINSIC _mm_blendv_pd(__m128d a, __m128d b, __m128d c) {
        return _mm_or_pd(_mm_andnot_pd(c, a), _mm_and_pd(c, b));
    }
    static inline __m128  INTRINSIC _mm_blendv_ps(__m128  a, __m128  b, __m128  c) {
        return _mm_or_ps(_mm_andnot_ps(c, a), _mm_and_ps(c, b));
    }
    static inline __m128i INTRINSIC _mm_blendv_epi8(__m128i a, __m128i b, __m128i c) {
        return _mm_or_si128(_mm_andnot_si128(c, a), _mm_and_si128(c, b));
    }

    // only use the following blend functions with immediates as mask and, of course, compiling
    // with optimization
    static inline __m128d INTRINSIC _mm_blend_pd(__m128d a, __m128d b, const int mask) {
        switch (mask) {
        case 0x0:
            return a;
        case 0x1:
            return _mm_shuffle_pd(b, a, 2);
        case 0x2:
            return _mm_shuffle_pd(a, b, 2);
        case 0x3:
            return b;
        default:
            abort();
        }
    }
    static inline __m128  INTRINSIC _mm_blend_ps(__m128  a, __m128  b, const int mask) {
        __m128i c;
        switch (mask) {
        case 0x0:
            return a;
        case 0x1:
            c = _mm_srli_si128(_mm_setallone_si128(), 12);
            break;
        case 0x2:
            c = _mm_slli_si128(_mm_srli_si128(_mm_setallone_si128(), 12), 4);
            break;
        case 0x3:
            c = _mm_srli_si128(_mm_setallone_si128(), 8);
            break;
        case 0x4:
            c = _mm_slli_si128(_mm_srli_si128(_mm_setallone_si128(), 12), 8);
            break;
        case 0x5:
            c = _mm_set_epi32(0, -1, 0, -1);
            break;
        case 0x6:
            c = _mm_slli_si128(_mm_srli_si128(_mm_setallone_si128(), 8), 4);
            break;
        case 0x7:
            c = _mm_srli_si128(_mm_setallone_si128(), 4);
            break;
        case 0x8:
            c = _mm_slli_si128(_mm_setallone_si128(), 12);
            break;
        case 0x9:
            c = _mm_set_epi32(-1, 0, 0, -1);
            break;
        case 0xa:
            c = _mm_set_epi32(-1, 0, -1, 0);
            break;
        case 0xb:
            c = _mm_set_epi32(-1, 0, -1, -1);
            break;
        case 0xc:
            c = _mm_slli_si128(_mm_setallone_si128(), 8);
            break;
        case 0xd:
            c = _mm_set_epi32(-1, -1, 0, -1);
            break;
        case 0xe:
            c = _mm_slli_si128(_mm_setallone_si128(), 4);
            break;
        case 0xf:
            return b;
        default: // may not happen
            abort();
            c = _mm_setzero_si128();
            break;
        }
        __m128 _c = _mm_castsi128_ps(c);
        return _mm_or_ps(_mm_andnot_ps(_c, a), _mm_and_ps(_c, b));
    }
    static inline __m128i INTRINSIC _mm_blend_epi16(__m128i a, __m128i b, const int mask) {
        __m128i c;
        switch (mask) {
        case 0x00:
            return a;
        case 0x01:
            c = _mm_srli_si128(_mm_setallone_si128(), 14);
            break;
        case 0x03:
            c = _mm_srli_si128(_mm_setallone_si128(), 12);
            break;
        case 0x07:
            c = _mm_srli_si128(_mm_setallone_si128(), 10);
            break;
        case 0x0f:
            return _mm_unpackhi_epi64(_mm_slli_si128(b, 8), a);
        case 0x1f:
            c = _mm_srli_si128(_mm_setallone_si128(), 6);
            break;
        case 0x3f:
            c = _mm_srli_si128(_mm_setallone_si128(), 4);
            break;
        case 0x7f:
            c = _mm_srli_si128(_mm_setallone_si128(), 2);
            break;
        case 0x80:
            c = _mm_slli_si128(_mm_setallone_si128(), 14);
            break;
        case 0xc0:
            c = _mm_slli_si128(_mm_setallone_si128(), 12);
            break;
        case 0xe0:
            c = _mm_slli_si128(_mm_setallone_si128(), 10);
            break;
        case 0xf0:
            c = _mm_slli_si128(_mm_setallone_si128(), 8);
            break;
        case 0xf8:
            c = _mm_slli_si128(_mm_setallone_si128(), 6);
            break;
        case 0xfc:
            c = _mm_slli_si128(_mm_setallone_si128(), 4);
            break;
        case 0xfe:
            c = _mm_slli_si128(_mm_setallone_si128(), 2);
            break;
        case 0xff:
            return b;
        case 0xcc:
            return _mm_unpacklo_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_epi32(b, _MM_SHUFFLE(3, 1, 3, 1)));
        case 0x33:
            return _mm_unpacklo_epi32(_mm_shuffle_epi32(b, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 3, 1)));
        default:
            const __m128i shift = _mm_set_epi16(0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000, -0x7fff);
            c = _mm_srai_epi16(_mm_mullo_epi16(_mm_set1_epi16(mask), shift), 15);
            break;
        }
        return _mm_or_si128(_mm_andnot_si128(c, a), _mm_and_si128(c, b));
    }

    static inline __m128i CONST _mm_max_epi8 (__m128i a, __m128i b) {
        return _mm_blendv_epi8(b, a, _mm_cmpgt_epi8 (a, b));
    }
    static inline __m128i CONST _mm_max_epi32(__m128i a, __m128i b) {
        return _mm_blendv_epi8(b, a, _mm_cmpgt_epi32(a, b));
    }
//X         static inline __m128i CONST _mm_max_epu8 (__m128i a, __m128i b) {
//X             return _mm_blendv_epi8(b, a, _mm_cmpgt_epu8 (a, b));
//X         }
    static inline __m128i CONST _mm_max_epu16(__m128i a, __m128i b) {
        return _mm_blendv_epi8(b, a, _mm_cmpgt_epu16(a, b));
    }
    static inline __m128i CONST _mm_max_epu32(__m128i a, __m128i b) {
        return _mm_blendv_epi8(b, a, _mm_cmpgt_epu32(a, b));
    }
//X         static inline __m128i CONST _mm_min_epu8 (__m128i a, __m128i b) {
//X             return _mm_blendv_epi8(a, b, _mm_cmpgt_epu8 (a, b));
//X         }
    static inline __m128i CONST _mm_min_epu16(__m128i a, __m128i b) {
        return _mm_blendv_epi8(a, b, _mm_cmpgt_epu16(a, b));
    }
    static inline __m128i CONST _mm_min_epu32(__m128i a, __m128i b) {
        return _mm_blendv_epi8(a, b, _mm_cmpgt_epu32(a, b));
    }
    static inline __m128i CONST _mm_min_epi8 (__m128i a, __m128i b) {
        return _mm_blendv_epi8(a, b, _mm_cmpgt_epi8 (a, b));
    }
    static inline __m128i CONST _mm_min_epi32(__m128i a, __m128i b) {
        return _mm_blendv_epi8(a, b, _mm_cmpgt_epi32(a, b));
    }
    static inline __m128i INTRINSIC _mm_cvtepu8_epi16(__m128i epu8) {
        return _mm_unpacklo_epi8(epu8, _mm_setzero_si128());
    }
    static inline __m128i INTRINSIC _mm_cvtepi8_epi16(__m128i epi8) {
        return _mm_unpacklo_epi8(epi8, _mm_cmplt_epi8(epi8, _mm_setzero_si128()));
    }
    static inline __m128i INTRINSIC _mm_cvtepu16_epi32(__m128i epu16) {
        return _mm_unpacklo_epi16(epu16, _mm_setzero_si128());
    }
    static inline __m128i INTRINSIC _mm_cvtepi16_epi32(__m128i epu16) {
        return _mm_unpacklo_epi16(epu16, _mm_cmplt_epi16(epu16, _mm_setzero_si128()));
    }
    static inline __m128i INTRINSIC _mm_cvtepu8_epi32(__m128i epu8) {
        return _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(epu8));
    }
    static inline __m128i INTRINSIC _mm_cvtepi8_epi32(__m128i epi8) {
        const __m128i neg = _mm_cmplt_epi8(epi8, _mm_setzero_si128());
        const __m128i epi16 = _mm_unpacklo_epi8(epi8, neg);
        return _mm_unpacklo_epi16(epi16, _mm_unpacklo_epi8(neg, neg));
    }
    static inline __m128i INTRINSIC _mm_stream_load_si128(__m128i *mem) {
        return _mm_load_si128(mem);
    }

} // namespace SSE
} // namespace Vc
#endif

// SSE4.2
#ifdef VC_IMPL_SSE4_2
#include <nmmintrin.h>
#elif defined _NMMINTRIN_H_INCLUDED
#error "SSE4.2 was disabled but something includes <nmmintrin.h>. Please fix your code."
#endif

namespace Vc
{
namespace SSE
{
    static inline float INTRINSIC extract_float_imm(const __m128 v, const size_t i) {
        float f;
        switch (i) {
        case 0:
            f = _mm_cvtss_f32(v);
            break;
#if defined VC_IMPL_SSE4_1 && !defined VC_MSVC
        default:
#ifdef VC_GCC
            f = __builtin_ia32_vec_ext_v4sf(static_cast<__v4sf>(v), (i));
#else
            // MSVC fails to compile this because it can't optimize i to an immediate
            _MM_EXTRACT_FLOAT(f, v, i);
#endif
            break;
#else
        case 1:
            f = _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), 4)));
            break;
        case 2:
            f = _mm_cvtss_f32(_mm_movehl_ps(v, v));
            break;
        case 3:
            f = _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), 12)));
            break;
#endif
        }
        return f;
    }
    static inline double INTRINSIC extract_double_imm(const __m128d v, const size_t i) {
        if (i == 0) {
            return _mm_cvtsd_f64(v);
        }
        return _mm_cvtsd_f64(_mm_castps_pd(_mm_movehl_ps(_mm_castpd_ps(v), _mm_castpd_ps(v))));
    }
    static inline float INTRINSIC extract_float(const __m128 v, const size_t i) {
#ifdef VC_GCC
        if (__builtin_constant_p(i)) {
            return extract_float_imm(v, i);
//X         if (index <= 1) {
//X             unsigned long long tmp = _mm_cvtsi128_si64(_mm_castps_si128(v));
//X             if (index == 0) tmp &= 0xFFFFFFFFull;
//X             if (index == 1) tmp >>= 32;
//X             return Common::AliasingEntryHelper<EntryType>(tmp);
//X         }
        } else {
            typedef float float4[4] MAY_ALIAS;
            const float4 &data = reinterpret_cast<const float4 &>(v);
            return data[i];
        }
#else
        union { __m128 v; float m[4]; } u;
        u.v = v;
        return u.m[i];
#endif
    }

    static inline __m128  INTRINSIC _mm_stream_load(const float *mem) {
#ifdef VC_IMPL_SSE4_1
        return _mm_castsi128_ps(_mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<float *>(mem))));
#else
        return _mm_load_ps(mem);
#endif
    }
    static inline __m128d INTRINSIC _mm_stream_load(const double *mem) {
#ifdef VC_IMPL_SSE4_1
        return _mm_castsi128_pd(_mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<double *>(mem))));
#else
        return _mm_load_pd(mem);
#endif
    }
    static inline __m128i INTRINSIC _mm_stream_load(const int *mem) {
#ifdef VC_IMPL_SSE4_1
        return _mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<int *>(mem)));
#else
        return _mm_load_si128(reinterpret_cast<const __m128i *>(mem));
#endif
    }
    static inline __m128i INTRINSIC _mm_stream_load(const unsigned int *mem) {
        return _mm_stream_load(reinterpret_cast<const int *>(mem));
    }
    static inline __m128i INTRINSIC _mm_stream_load(const short *mem) {
        return _mm_stream_load(reinterpret_cast<const int *>(mem));
    }
    static inline __m128i INTRINSIC _mm_stream_load(const unsigned short *mem) {
        return _mm_stream_load(reinterpret_cast<const int *>(mem));
    }
    static inline __m128i INTRINSIC _mm_stream_load(const signed char *mem) {
        return _mm_stream_load(reinterpret_cast<const int *>(mem));
    }
    static inline __m128i INTRINSIC _mm_stream_load(const unsigned char *mem) {
        return _mm_stream_load(reinterpret_cast<const int *>(mem));
    }
} // namespace SSE
} // namespace Vc

#include "shuffle.h"

#endif // SSE_INTRINSICS_H
