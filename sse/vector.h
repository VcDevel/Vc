/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of
    the License, or (at your option) version 3.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301, USA.

*/

/* The log implementations are based on code from Julien Pommier which carries the following
   copyright information:
 */
/*
   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/
/* Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#ifndef SSE_VECTOR_H
#define SSE_VECTOR_H

#include "intrinsics.h"
#include <algorithm>
#include <limits>

#ifndef _M128
# define _M128 __m128
#endif

#ifndef _M128I
# define _M128I __m128i
#endif

#ifndef _M128D
# define _M128D __m128d
#endif

#ifndef ALIGN
# ifdef __GNUC__
#  define ALIGN(n) __attribute__((aligned(n)))
# else
#  define ALIGN(n) __declspec(align(n))
# endif
#endif

#ifdef __GNUC__
#define CONST __attribute__((const))
#else
#define CONST
#endif

#define CAT_HELPER(a, b) a##b
#define CAT(a, b) CAT_HELPER(a, b)

#define unrolled_loop16(_it_, _start_, _end_, _code_) \
if (_start_ +  0 < _end_) { enum { _it_ = (_start_ +  0) < _end_ ? (_start_ +  0) : _start_ }; _code_ } \
if (_start_ +  1 < _end_) { enum { _it_ = (_start_ +  1) < _end_ ? (_start_ +  1) : _start_ }; _code_ } \
if (_start_ +  2 < _end_) { enum { _it_ = (_start_ +  2) < _end_ ? (_start_ +  2) : _start_ }; _code_ } \
if (_start_ +  3 < _end_) { enum { _it_ = (_start_ +  3) < _end_ ? (_start_ +  3) : _start_ }; _code_ } \
if (_start_ +  4 < _end_) { enum { _it_ = (_start_ +  4) < _end_ ? (_start_ +  4) : _start_ }; _code_ } \
if (_start_ +  5 < _end_) { enum { _it_ = (_start_ +  5) < _end_ ? (_start_ +  5) : _start_ }; _code_ } \
if (_start_ +  6 < _end_) { enum { _it_ = (_start_ +  6) < _end_ ? (_start_ +  6) : _start_ }; _code_ } \
if (_start_ +  7 < _end_) { enum { _it_ = (_start_ +  7) < _end_ ? (_start_ +  7) : _start_ }; _code_ } \
if (_start_ +  8 < _end_) { enum { _it_ = (_start_ +  8) < _end_ ? (_start_ +  8) : _start_ }; _code_ } \
if (_start_ +  9 < _end_) { enum { _it_ = (_start_ +  9) < _end_ ? (_start_ +  9) : _start_ }; _code_ } \
if (_start_ + 10 < _end_) { enum { _it_ = (_start_ + 10) < _end_ ? (_start_ + 10) : _start_ }; _code_ } \
if (_start_ + 11 < _end_) { enum { _it_ = (_start_ + 11) < _end_ ? (_start_ + 11) : _start_ }; _code_ } \
if (_start_ + 12 < _end_) { enum { _it_ = (_start_ + 12) < _end_ ? (_start_ + 12) : _start_ }; _code_ } \
if (_start_ + 13 < _end_) { enum { _it_ = (_start_ + 13) < _end_ ? (_start_ + 13) : _start_ }; _code_ } \
if (_start_ + 14 < _end_) { enum { _it_ = (_start_ + 14) < _end_ ? (_start_ + 14) : _start_ }; _code_ } \
if (_start_ + 15 < _end_) { enum { _it_ = (_start_ + 15) < _end_ ? (_start_ + 15) : _start_ }; _code_ } \
do {} while ( false )

#define for_all_vector_entries(_it_, _code_) \
  unrolled_loop16(_it_, 0, Size, _code_)

namespace SSE
{
    namespace Internal
    {
        ALIGN(16) extern const unsigned int   _IndexesFromZero4[4];
        ALIGN(16) extern const unsigned short _IndexesFromZero8[8];
    } // namespace Internal

    enum { VectorAlignment = 16 };
    template<typename T> class Vector;
    class Float8Mask;
    template<unsigned int VectorSize> class Mask;

    /*
     * Hack to create a vector object with 8 floats
     */
    class float8 {};

    class M256 {
        public:
            inline M256() {}
            inline M256(_M128 a, _M128 b) { d[0] = a; d[1] = b; }
            inline _M128 &operator[](int i) { return d[i]; }
            inline const _M128 &operator[](int i) const { return d[i]; }
        private:
            _M128 d[2];
    };

#define STORE_VECTOR(type, name, vec) \
    union { __m128i p; type v[16 / sizeof(type)]; } CAT(u, __LINE__); \
    _mm_store_si128(&CAT(u, __LINE__).p, vec); \
    const type *const name = &CAT(u, __LINE__).v[0]

    template<typename To, typename From> static inline To mm128_reinterpret_cast(From v) CONST;
    template<typename To, typename From> static inline To mm128_reinterpret_cast(From v) { return v; }
    template<> inline _M128I mm128_reinterpret_cast<_M128I, _M128 >(_M128  v) CONST;
    template<> inline _M128I mm128_reinterpret_cast<_M128I, _M128D>(_M128D v) CONST;
    template<> inline _M128  mm128_reinterpret_cast<_M128 , _M128D>(_M128D v) CONST;
    template<> inline _M128  mm128_reinterpret_cast<_M128 , _M128I>(_M128I v) CONST;
    template<> inline _M128D mm128_reinterpret_cast<_M128D, _M128I>(_M128I v) CONST;
    template<> inline _M128D mm128_reinterpret_cast<_M128D, _M128 >(_M128  v) CONST;
    template<> inline _M128I mm128_reinterpret_cast<_M128I, _M128 >(_M128  v) { return _mm_castps_si128(v); }
    template<> inline _M128I mm128_reinterpret_cast<_M128I, _M128D>(_M128D v) { return _mm_castpd_si128(v); }
    template<> inline _M128  mm128_reinterpret_cast<_M128 , _M128D>(_M128D v) { return _mm_castpd_ps(v);    }
    template<> inline _M128  mm128_reinterpret_cast<_M128 , _M128I>(_M128I v) { return _mm_castsi128_ps(v); }
    template<> inline _M128D mm128_reinterpret_cast<_M128D, _M128I>(_M128I v) { return _mm_castsi128_pd(v); }
    template<> inline _M128D mm128_reinterpret_cast<_M128D, _M128 >(_M128  v) { return _mm_castps_pd(v);    }


    template<unsigned int Size> struct IndexTypeHelper;
    template<> struct IndexTypeHelper<2u> { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<4u> { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<8u> { typedef unsigned short Type; };
    template<> struct IndexTypeHelper<16u>{ typedef unsigned char  Type; };

    template<typename T> struct VectorHelper {};

    template<typename VectorType, typename EntryType> class VectorMemoryUnion
    {
        public:
            typedef EntryType AliasingEntryType __attribute__((__may_alias__));
            inline VectorMemoryUnion() {}
            inline VectorMemoryUnion(const VectorType &x) : data(x) {}

            VectorType &v() { return data; }
            const VectorType &v() const { return data; }

            AliasingEntryType &m(int index) {
                return reinterpret_cast<AliasingEntryType *>(&data)[index];
            }

            EntryType m(int index) const {
                return reinterpret_cast<const AliasingEntryType *>(&data)[index];
            }

        private:
            VectorType data;
    };

    template<typename T> struct VectorHelperSize;

    template<typename T> class VectorBase {
        friend struct VectorHelperSize<float>;
        friend struct VectorHelperSize<double>;
        friend struct VectorHelperSize<int>;
        friend struct VectorHelperSize<unsigned int>;
        friend struct VectorHelperSize<short>;
        friend struct VectorHelperSize<unsigned short>;
        friend struct VectorHelperSize<float8>;
        public:
            enum { Size = 16 / sizeof(T) };
            typedef _M128I VectorType;
            typedef T EntryType;
            typedef VectorBase<typename IndexTypeHelper<Size>::Type> IndexType;
            typedef Mask<Size> MaskType;

            inline Vector<EntryType> &operator|= (const Vector<EntryType> &x);
            inline Vector<EntryType> &operator&= (const Vector<EntryType> &x);
            inline Vector<EntryType> &operator^= (const Vector<EntryType> &x);
            inline Vector<EntryType> &operator>>=(const Vector<EntryType> &x);
            inline Vector<EntryType> &operator<<=(const Vector<EntryType> &x);

            inline Vector<EntryType> operator| (const Vector<EntryType> &x) const;
            inline Vector<EntryType> operator& (const Vector<EntryType> &x) const;
            inline Vector<EntryType> operator^ (const Vector<EntryType> &x) const;
            inline Vector<EntryType> operator>>(const Vector<EntryType> &x) const;
            inline Vector<EntryType> operator<<(const Vector<EntryType> &x) const;

        protected:
            inline VectorBase() {}
            inline VectorBase(VectorType x) : d(x) {}

            VectorMemoryUnion<VectorType, EntryType> d;

            static const T *_IndexesFromZero() {
                if (Size == 4) {
                    return reinterpret_cast<const T *>(Internal::_IndexesFromZero4);
                } else if (Size == 8) {
                    return reinterpret_cast<const T *>(Internal::_IndexesFromZero8);
                }
                return 0;
            }
    };

    template<> class VectorBase<float8> {
        friend struct VectorHelperSize<float8>;
        public:
            enum { Size = 8 };
            typedef M256 VectorType;
            typedef float EntryType;
            typedef VectorBase<IndexTypeHelper<Size>::Type> IndexType;
            typedef Float8Mask MaskType;

        protected:
            inline VectorBase() {}
            inline VectorBase(const VectorType &x) : d(x) {}

            VectorMemoryUnion<VectorType, EntryType> d;
    };

    template<> class VectorBase<float> {
        friend struct VectorHelperSize<float>;
        public:
            enum { Size = 16 / sizeof(float) };
            typedef _M128 VectorType;
            typedef float EntryType;
            typedef VectorBase<IndexTypeHelper<Size>::Type> IndexType;
            typedef Mask<Size> MaskType;

        protected:
            inline VectorBase() {}
            inline VectorBase(VectorType x) : d(x) {}

            VectorMemoryUnion<VectorType, EntryType> d;
    };

    template<> class VectorBase<double> {
        friend struct VectorHelperSize<double>;
        public:
            enum { Size = 16 / sizeof(double) };
            typedef _M128D VectorType;
            typedef double EntryType;
            typedef VectorBase<IndexTypeHelper<Size>::Type> IndexType;
            typedef Mask<Size> MaskType;

        protected:
            inline VectorBase() {}
            inline VectorBase(VectorType x) : d(x) {}

            VectorMemoryUnion<VectorType, EntryType> d;
    };

    template<typename T> struct VectorHelperSize
    {
        typedef VectorBase<T> Base;
        typedef typename Base::VectorType VectorType;
        typedef typename Base::EntryType  EntryType;
        typedef typename Base::IndexType  IndexType;
        typedef VectorMemoryUnion<VectorType, EntryType> UnionType;

        enum { Size = Base::Size, Shift = sizeof(EntryType) };

        static inline VectorType my_set4(const EntryType *m, const unsigned long long int a,
                const unsigned long long int b, const unsigned long long int c,
                const unsigned long long int d) CONST {
            VectorType v;
            __m128 t1, t2, t3;
            __asm__("movd 0(%4,%5,4), %3\n\t"
                    "movd 0(%4,%6,4), %2\n\t"
                    "movd 0(%4,%7,4), %1\n\t"
                    "movd 0(%4,%8,4), %0\n\t"
                    "unpcklps %3, %2\n\t"
                    "unpcklps %1, %0\n\t"
                    "movlhps %2, %0\n\t"
                    : "=x"(v), "=x"(t1), "=x"(t2), "=x"(t3)
                    : "r"(m), "r"(a), "r"(b), "r"(c), "r"(d)
                    :
                   );
            return v;
        }
        static inline void gather(Base &v, const IndexType &indexes, const EntryType *baseAddr) {
            if (Size == 2) {
                v.d.v() = mm128_reinterpret_cast<VectorType>(_mm_set_pd(
                            baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]));
            } else if (Size == 4) {
                v.d.v() = my_set4(baseAddr, indexes.d.m(3), indexes.d.m(2), indexes.d.m(1), indexes.d.m(0));
            } else if (Size == 8) {
                v.d.v() = mm128_reinterpret_cast<VectorType>(_mm_set_epi16(
                            baseAddr[indexes.d.m(7)], baseAddr[indexes.d.m(6)],
                            baseAddr[indexes.d.m(5)], baseAddr[indexes.d.m(4)],
                            baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                            baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]));
            } else {
                for_all_vector_entries(i,
                        v.d.m(i) = baseAddr[indexes.d.m(i)];
                        );
            }
        }
        template<typename AliasingT>
        static inline void maskedGatherHelper(AliasingT &vEntry, const int mask, const EntryType &value, const int bitMask) {
#ifdef __GNUC__
            register EntryType t;
            asm(
                    "test %4,%2\n\t"
                    "mov %5,%1\n\t"
                    "cmovne %3,%1\n\t"
                    "mov %1,%0"
                    : "=m"(vEntry), "=&r"(t)
                    : "r"(mask), "m"(value),
#ifdef NO_OPTIMIZATION
                    "m"
#else
                    "n"
#endif
                    (bitMask), "m"(vEntry)
                    :
               );
#else
            if (mask & bitMask) {
                vEntry = value;
            }
#endif
        }
        template<typename AliasingT>
        static inline void maskedScatterHelper(const AliasingT &vEntry, const int mask, EntryType &value, const int bitMask) {
#ifdef __GNUC__
            register EntryType t;
            asm(
                    "test %4,%2\n\t"
                    "mov %3,%1\n\t"
                    "cmovne %5,%1\n\t"
                    "mov %1,%0"
                    : "=m"(value), "=&r"(t)
                    : "r"(mask), "m"(value),
#ifdef NO_OPTIMIZATION
                    "m"
#else
                    "n"
#endif
                    (bitMask), "m"(vEntry)
                    :
               );
#else
            if (mask & bitMask) {
                value = vEntry;
            }
#endif
        }
        static inline void gather(Base &v, const IndexType &indexes, int mask, const EntryType *baseAddr) {
            for_all_vector_entries(i,
                    maskedGatherHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)], 1 << i * Shift);
                );
        }
        template<typename S1> static inline void gather(Base &v, const IndexType &indexes,
                const S1 *baseAddr, const EntryType S1::* member1) {
//X             if (Size == 8) {
//X                 for_all_vector_entries(i,
//X                         v.d.v() = _mm_insert_epi16(v.d.v(), baseAddr[indexes.d.m(i)].*(member1), i);
//X                         );
//X             } else {
                for_all_vector_entries(i,
                        v.d.m(i) = baseAddr[indexes.d.m(i)].*(member1);
                        );
//X             }
        }
        template<typename S1> static inline void gather(Base &v, const IndexType &indexes, int mask,
                const S1 *baseAddr, const EntryType S1::* member1) {
            for_all_vector_entries(i,
                    maskedGatherHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)].*(member1), 1 << i * Shift);
                );
        }
        template<typename S1, typename S2> static inline void gather(Base &v, const IndexType &indexes,
                const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2) {
            for_all_vector_entries(i,
                    v.d.m(i) = baseAddr[indexes.d.m(i)].*(member1).*(member2);
                );
        }
        template<typename S1, typename S2> static inline void gather(Base &v, const IndexType &indexes, int mask,
                const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2) {
            for_all_vector_entries(i,
                    maskedGatherHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)].*(member1).*(member2), 1 << i * Shift);
                );
        }
        static inline void scatter(const Base &v, const IndexType &indexes, EntryType *baseAddr) {
            for_all_vector_entries(i,
                    baseAddr[indexes.d.m(i)] = v.d.m(i);
                    );
        }
        static inline void scatter(const Base &v, const IndexType &indexes, int mask, EntryType *baseAddr) {
            for_all_vector_entries(i,
                    maskedScatterHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)], 1 << i * Shift);
                    );
        }
        template<typename S1> static inline void scatter(const Base &v, const IndexType &indexes,
                S1 *baseAddr, EntryType S1::* member1) {
            for_all_vector_entries(i,
                    baseAddr[indexes.d.m(i)].*(member1) = v.d.m(i);
                    );
        }
        template<typename S1> static inline void scatter(const Base &v, const IndexType &indexes, int mask,
                S1 *baseAddr, EntryType S1::* member1) {
            for_all_vector_entries(i,
                    maskedScatterHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)].*(member1), 1 << i * Shift);
                    );
        }
        template<typename S1, typename S2> static inline void scatter(const Base &v, const IndexType &indexes,
                S1 *baseAddr, S2 S1::* member1, EntryType S2::* member2) {
            for_all_vector_entries(i,
                    baseAddr[indexes.d.m(i)].*(member1).*(member2) = v.d.m(i);
                    );
        }
        template<typename S1, typename S2> static inline void scatter(const Base &v, const IndexType &indexes, int mask,
                S1 *baseAddr, S2 S1::* member1, EntryType S2::* member2) {
            for_all_vector_entries(i,
                    maskedScatterHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)].*(member1).*(member2), 1 << i * Shift);
                    );
        }
    };

    template<> struct VectorHelperSize<float8>
    {
        typedef VectorBase<float8> Base;
        typedef Base::VectorType VectorType;
        typedef Base::EntryType  EntryType;
        typedef Base::IndexType  IndexType;
        typedef VectorMemoryUnion<VectorType, EntryType> UnionType;

        enum { Size = Base::Size, Shift = 1 };

        static inline void gather(Base &v, const IndexType &indexes, const EntryType *baseAddr) {
            v.d.v()[0] = VectorHelperSize<float>::my_set4(baseAddr,
                    indexes.d.m(3), indexes.d.m(2), indexes.d.m(1), indexes.d.m(0));
            v.d.v()[1] = VectorHelperSize<float>::my_set4(baseAddr,
                    indexes.d.m(7), indexes.d.m(6), indexes.d.m(5), indexes.d.m(4));
        }

        static inline void gather(Base &v, const IndexType &indexes, int mask, const EntryType *baseAddr) {
            for_all_vector_entries(i,
                    VectorHelperSize<float>::maskedGatherHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)], 1 << i * Shift);
                );
        }
        template<typename S1> static inline void gather(Base &v, const IndexType &indexes,
                const S1 *baseAddr, const EntryType S1::* member1) {
            for_all_vector_entries(i,
                    v.d.m(i) = baseAddr[indexes.d.m(i)].*(member1);
                    );
        }
        template<typename S1> static inline void gather(Base &v, const IndexType &indexes, int mask,
                const S1 *baseAddr, const EntryType S1::* member1) {
            for_all_vector_entries(i,
                    VectorHelperSize<float>::maskedGatherHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)].*(member1), 1 << i * Shift);
                );
        }
        template<typename S1, typename S2> static inline void gather(Base &v, const IndexType &indexes,
                const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2) {
            for_all_vector_entries(i,
                    v.d.m(i) = baseAddr[indexes.d.m(i)].*(member1).*(member2);
                );
        }
        template<typename S1, typename S2> static inline void gather(Base &v, const IndexType &indexes, int mask,
                const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2) {
            for_all_vector_entries(i,
                    VectorHelperSize<float>::maskedGatherHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)].*(member1).*(member2), 1 << i * Shift);
                );
        }
        static inline void scatter(const Base &v, const IndexType &indexes, EntryType *baseAddr) {
            for_all_vector_entries(i,
                    baseAddr[indexes.d.m(i)] = v.d.m(i);
                    );
        }
        static inline void scatter(const Base &v, const IndexType &indexes, int mask, EntryType *baseAddr) {
            for_all_vector_entries(i,
                    VectorHelperSize<float>::maskedScatterHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)], 1 << i * Shift);
                    );
        }
        template<typename S1> static inline void scatter(const Base &v, const IndexType &indexes,
                S1 *baseAddr, EntryType S1::* member1) {
            for_all_vector_entries(i,
                    baseAddr[indexes.d.m(i)].*(member1) = v.d.m(i);
                    );
        }
        template<typename S1> static inline void scatter(const Base &v, const IndexType &indexes, int mask,
                S1 *baseAddr, EntryType S1::* member1) {
            for_all_vector_entries(i,
                    VectorHelperSize<float>::maskedScatterHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)].*(member1), 1 << i * Shift);
                    );
        }
        template<typename S1, typename S2> static inline void scatter(const Base &v, const IndexType &indexes,
                S1 *baseAddr, S2 S1::* member1, EntryType S2::* member2) {
            for_all_vector_entries(i,
                    baseAddr[indexes.d.m(i)].*(member1).*(member2) = v.d.m(i);
                    );
        }
        template<typename S1, typename S2> static inline void scatter(const Base &v, const IndexType &indexes, int mask,
                S1 *baseAddr, S2 S1::* member1, EntryType S2::* member2) {
            for_all_vector_entries(i,
                    VectorHelperSize<float>::maskedScatterHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)].*(member1).*(member2), 1 << i * Shift);
                    );
        }
    };

#undef OP_DECL
#undef PARENT_DATA
#undef PARENT_DATA_CONST

        template<typename From, typename To> struct StaticCastHelper {};
        template<> struct StaticCastHelper<float       , int         > { static _M128I cast(const _M128  &v) { return _mm_cvttps_epi32(v); } };
        template<> struct StaticCastHelper<double      , int         > { static _M128I cast(const _M128D &v) { return _mm_cvttpd_epi32(v); } };
        template<> struct StaticCastHelper<int         , int         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<unsigned int, int         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<float       , unsigned int> { static _M128I cast(const _M128  &v) { return _mm_cvttps_epi32(v); } };
        template<> struct StaticCastHelper<double      , unsigned int> { static _M128I cast(const _M128D &v) { return _mm_cvttpd_epi32(v); } };
        template<> struct StaticCastHelper<int         , unsigned int> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<unsigned int, unsigned int> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<float       , float       > { static _M128  cast(const _M128  &v) { return v; } };
        template<> struct StaticCastHelper<double      , float       > { static _M128  cast(const _M128D &v) { return _mm_cvtpd_ps(v); } };
        template<> struct StaticCastHelper<int         , float       > { static _M128  cast(const _M128I &v) { return _mm_cvtepi32_ps(v); } };
        template<> struct StaticCastHelper<unsigned int, float       > { static _M128  cast(const _M128I &v) { return _mm_cvtepi32_ps(v); } };
        template<> struct StaticCastHelper<float       , double      > { static _M128D cast(const _M128  &v) { return _mm_cvtps_pd(v); } };
        template<> struct StaticCastHelper<double      , double      > { static _M128D cast(const _M128D &v) { return v; } };
        template<> struct StaticCastHelper<int         , double      > { static _M128D cast(const _M128I &v) { return _mm_cvtepi32_pd(v); } };
        template<> struct StaticCastHelper<unsigned int, double      > { static _M128D cast(const _M128I &v) { return _mm_cvtepi32_pd(v); } };

        template<> struct StaticCastHelper<unsigned short, float8        > { static  M256  cast(const _M128I &v) {
            return M256(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v, _mm_setzero_si128())),
                        _mm_cvtepi32_ps(_mm_unpackhi_epi16(v, _mm_setzero_si128())));
        } };
//X         template<> struct StaticCastHelper<short         , float8        > { static  M256  cast(const _M128I &v) {
//X             const _M128I neg = _mm_cmplt_epi16(v, _mm_setzero_si128());
//X             return M256(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v, neg)),
//X                         _mm_cvtepi32_ps(_mm_unpackhi_epi16(v, neg)));
//X         } };
        template<> struct StaticCastHelper<float8        , short         > { static _M128I cast(const  M256  &v) { return _mm_packs_epi32(_mm_cvttps_epi32(v[0]), _mm_cvttps_epi32(v[1])); } };
        template<> struct StaticCastHelper<float8        , unsigned short> { static _M128I cast(const  M256  &v) { return _mm_packs_epi32(_mm_cvttps_epi32(v[0]), _mm_cvttps_epi32(v[1])); } };

        template<> struct StaticCastHelper<float         , short         > { static _M128I cast(const _M128  &v) { return _mm_packs_epi32(_mm_cvttps_epi32(v), _mm_setzero_si128()); } };
        template<> struct StaticCastHelper<short         , short         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<unsigned short, short         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<float         , unsigned short> { static _M128I cast(const _M128  &v) { return _mm_packs_epi32(_mm_cvttps_epi32(v), _mm_setzero_si128()); } };
        template<> struct StaticCastHelper<short         , unsigned short> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<unsigned short, unsigned short> { static _M128I cast(const _M128I &v) { return v; } };

        template<typename From, typename To> struct ReinterpretCastHelper {};
        template<> struct ReinterpretCastHelper<float       , int         > { static _M128I cast(const _M128  &v) { return _mm_castps_si128(v); } };
        template<> struct ReinterpretCastHelper<double      , int         > { static _M128I cast(const _M128D &v) { return _mm_castpd_si128(v); } };
        template<> struct ReinterpretCastHelper<int         , int         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<unsigned int, int         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<float       , unsigned int> { static _M128I cast(const _M128  &v) { return _mm_castps_si128(v); } };
        template<> struct ReinterpretCastHelper<double      , unsigned int> { static _M128I cast(const _M128D &v) { return _mm_castpd_si128(v); } };
        template<> struct ReinterpretCastHelper<int         , unsigned int> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<unsigned int, unsigned int> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<float       , float       > { static _M128  cast(const _M128  &v) { return v; } };
        template<> struct ReinterpretCastHelper<double      , float       > { static _M128  cast(const _M128D &v) { return _mm_castpd_ps(v); } };
        template<> struct ReinterpretCastHelper<int         , float       > { static _M128  cast(const _M128I &v) { return _mm_castsi128_ps(v); } };
        template<> struct ReinterpretCastHelper<unsigned int, float       > { static _M128  cast(const _M128I &v) { return _mm_castsi128_ps(v); } };
        template<> struct ReinterpretCastHelper<float       , double      > { static _M128D cast(const _M128  &v) { return _mm_castps_pd(v); } };
        template<> struct ReinterpretCastHelper<double      , double      > { static _M128D cast(const _M128D &v) { return v; } };
        template<> struct ReinterpretCastHelper<int         , double      > { static _M128D cast(const _M128I &v) { return _mm_castsi128_pd(v); } };
        template<> struct ReinterpretCastHelper<unsigned int, double      > { static _M128D cast(const _M128I &v) { return _mm_castsi128_pd(v); } };

        template<> struct ReinterpretCastHelper<unsigned short, short         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<unsigned short, unsigned short> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<short         , unsigned short> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<short         , short         > { static _M128I cast(const _M128I &v) { return v; } };

        template<typename T> struct CtorTypeHelper { typedef T Type; };
        template<> struct CtorTypeHelper<short> { typedef int Type; };
        template<> struct CtorTypeHelper<unsigned short> { typedef unsigned int Type; };
        template<> struct CtorTypeHelper<float> { typedef double Type; };

        template<typename T> struct ExpandTypeHelper { typedef T Type; };
        template<> struct ExpandTypeHelper<short> { typedef int Type; };
        template<> struct ExpandTypeHelper<unsigned short> { typedef unsigned int Type; };
        template<> struct ExpandTypeHelper<float> { typedef double Type; };

#define OP0(name, code) static inline VectorType name() { return code; }
#define OP1(name, code) static inline VectorType name(const VectorType &a) { return code; }
#define OP2(name, code) static inline VectorType name(const VectorType &a, const VectorType &b) { return code; }
#define OP3(name, code) static inline VectorType name(const VectorType &a, const VectorType &b, const VectorType &c) { return code; }

        template<> struct VectorHelper<_M128>
        {
            typedef _M128 VectorType;
            static inline VectorType load(const float *x) { return _mm_load_ps(x); }
            static inline VectorType loadUnaligned(const float *x) { return _mm_loadu_ps(x); }
            static inline void store(float *mem, const VectorType &x) { _mm_store_ps(mem, x); }
            static inline void storeStreaming(float *mem, const VectorType &x) { _mm_stream_ps(mem, x); }
            OP0(zero, _mm_setzero_ps())
            OP2(or_, _mm_or_ps(a, b))
            OP2(xor_, _mm_xor_ps(a, b))
            OP2(and_, _mm_and_ps(a, b))
            OP2(andnot_, _mm_andnot_ps(a, b))
            OP3(blend, _mm_blendv_ps(a, b, c))
        };

        template<> struct VectorHelper<M256>
        {
            typedef M256 VectorType;
            static inline VectorType load(const float *x) {
                return VectorType(_mm_load_ps(x), _mm_load_ps(x + 4));
            }
            static inline VectorType loadUnaligned(const float *x) {
                return VectorType(_mm_loadu_ps(x), _mm_loadu_ps(x + 4));
            }
            static inline void store(float *mem, const VectorType &x) {
                _mm_store_ps(mem, x[0]);
                _mm_store_ps(mem + 4, x[1]);
            }
            static inline void storeStreaming(float *mem, const VectorType &x) {
                _mm_stream_ps(mem, x[0]);
                _mm_stream_ps(mem + 4, x[1]);
            }
            OP0(zero, VectorType(_mm_setzero_ps(), _mm_setzero_ps()))
            OP2(or_, VectorType(_mm_or_ps(a[0], b[0]), _mm_or_ps(a[1], b[1])))
            OP2(xor_, VectorType(_mm_xor_ps(a[0], b[0]), _mm_xor_ps(a[1], b[1])))
            OP2(and_, VectorType(_mm_and_ps(a[0], b[0]), _mm_and_ps(a[1], b[1])))
            OP2(andnot_, VectorType(_mm_andnot_ps(a[0], b[0]), _mm_andnot_ps(a[1], b[1])))
            OP3(blend, VectorType(_mm_blendv_ps(a[0], b[0], c[0]), _mm_blendv_ps(a[1], b[1], c[1])))
        };

        template<> struct VectorHelper<_M128D>
        {
            typedef _M128D VectorType;
            static inline VectorType load(const double *x) { return _mm_load_pd(x); }
            static inline VectorType loadUnaligned(const double *x) { return _mm_loadu_pd(x); }
            static inline void store(double *mem, const VectorType &x) { _mm_store_pd(mem, x); }
            static inline void storeStreaming(double *mem, const VectorType &x) { _mm_stream_pd(mem, x); }
            OP0(zero, _mm_setzero_pd())
            OP2(or_, _mm_or_pd(a, b))
            OP2(xor_, _mm_xor_pd(a, b))
            OP2(and_, _mm_and_pd(a, b))
            OP2(andnot_, _mm_andnot_pd(a, b))
            OP3(blend, _mm_blendv_pd(a, b, c))
        };

        template<> struct VectorHelper<_M128I>
        {
            typedef _M128I VectorType;
            template<typename T> static inline VectorType load(const T *x) { return _mm_load_si128(reinterpret_cast<const VectorType *>(x)); }
            template<typename T> static inline VectorType loadUnaligned(const T *x) { return _mm_loadu_si128(reinterpret_cast<const VectorType *>(x)); }
            template<typename T> static inline void store(T *mem, const VectorType &x) { _mm_store_si128(reinterpret_cast<VectorType *>(mem), x); }
            template<typename T> static inline void storeStreaming(T *mem, const VectorType &x) { _mm_stream_si128(reinterpret_cast<VectorType *>(mem), x); }
            OP0(zero, _mm_setzero_si128())
            OP2(or_, _mm_or_si128(a, b))
            OP2(xor_, _mm_xor_si128(a, b))
            OP2(and_, _mm_and_si128(a, b))
            OP2(andnot_, _mm_andnot_si128(a, b))
            OP3(blend, _mm_blendv_epi8(a, b, c))
        };
#undef OP1
#undef OP2
#undef OP3

#define OP1(op) \
        static inline VectorType op(const VectorType &a) { return CAT(_mm_##op##_, SUFFIX)(a); }
#define OP(op) \
        static inline VectorType op(const VectorType &a, const VectorType &b) { return CAT(_mm_##op##_ , SUFFIX)(a, b); }
#define OP_(op) \
        static inline VectorType op(const VectorType &a, const VectorType &b) { return CAT(_mm_##op    , SUFFIX)(a, b); }
#define OPx(op, op2) \
        static inline VectorType op(const VectorType &a, const VectorType &b) { return CAT(_mm_##op2##_, SUFFIX)(a, b); }
#define OPcmp(op) \
        static inline VectorType cmp##op(const VectorType &a, const VectorType &b) { return CAT(_mm_cmp##op##_, SUFFIX)(a, b); }
#define OP_CAST_(op) \
        static inline VectorType op(const VectorType &a, const VectorType &b) { return CAT(_mm_castps_, SUFFIX)( \
            _mm_##op##ps(CAT(CAT(_mm_cast, SUFFIX), _ps)(a), \
              CAT(CAT(_mm_cast, SUFFIX), _ps)(b))); \
        }
#define MINMAX \
        static inline VectorType min(VectorType a, VectorType b) { return CAT(_mm_min_, SUFFIX)(a, b); } \
        static inline VectorType max(VectorType a, VectorType b) { return CAT(_mm_max_, SUFFIX)(a, b); }

        // _mm_sll_* does not take a count parameter with different counts per vector element. So we
        // have to do it manually
#define SHIFT4 \
            static inline VectorType sll(VectorType v, __m128i count) { \
                STORE_VECTOR(unsigned int, shifts, count); \
                union { __m128i v; unsigned int i[4]; } data; \
                _mm_store_si128(&data.v, v); \
                data.i[0] <<= shifts[0]; \
                data.i[1] <<= shifts[1]; \
                data.i[2] <<= shifts[2]; \
                data.i[3] <<= shifts[3]; \
                return data.v; } \
            static inline VectorType slli(VectorType v, int count) { return CAT(_mm_slli_, SUFFIX)(v, count); } \
            static inline VectorType srl(VectorType v, __m128i count) { \
                STORE_VECTOR(unsigned int, shifts, count); \
                union { __m128i v; unsigned int i[4]; } data; \
                _mm_store_si128(&data.v, v); \
                data.i[0] >>= shifts[0]; \
                data.i[1] >>= shifts[1]; \
                data.i[2] >>= shifts[2]; \
                data.i[3] >>= shifts[3]; \
                return data.v; } \
            static inline VectorType srli(VectorType v, int count) { return CAT(_mm_srli_, SUFFIX)(v, count); }
#define SHIFT8 \
            static inline VectorType sll(VectorType v, __m128i count) { \
                STORE_VECTOR(unsigned short, shifts, count); \
                union { __m128i v; unsigned short i[8]; } data; \
                _mm_store_si128(&data.v, v); \
                data.i[0] <<= shifts[0]; \
                data.i[1] <<= shifts[1]; \
                data.i[2] <<= shifts[2]; \
                data.i[3] <<= shifts[3]; \
                data.i[4] <<= shifts[4]; \
                data.i[5] <<= shifts[5]; \
                data.i[6] <<= shifts[6]; \
                data.i[7] <<= shifts[7]; \
                return data.v; } \
            static inline VectorType slli(VectorType v, int count) { return CAT(_mm_slli_, SUFFIX)(v, count); } \
            static inline VectorType srl(VectorType v, __m128i count) { \
                STORE_VECTOR(unsigned short, shifts, count); \
                union { __m128i v; unsigned short i[8]; } data; \
                _mm_store_si128(&data.v, v); \
                data.i[0] >>= shifts[0]; \
                data.i[1] >>= shifts[1]; \
                data.i[2] >>= shifts[2]; \
                data.i[3] >>= shifts[3]; \
                data.i[4] >>= shifts[4]; \
                data.i[5] >>= shifts[5]; \
                data.i[6] >>= shifts[6]; \
                data.i[7] >>= shifts[7]; \
                return data.v; } \
            static inline VectorType srli(VectorType v, int count) { return CAT(_mm_srli_, SUFFIX)(v, count); }

        template<> struct VectorHelper<double> {
            typedef _M128D VectorType;
#define SUFFIX pd

            static inline VectorType notMaskedToZero(VectorType a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_pd(mask), a); }
            static inline VectorType set(const double a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline VectorType set(const double a, const double b) { return CAT(_mm_set_, SUFFIX)(a, b); }
            static inline VectorType zero() { return CAT(_mm_setzero_, SUFFIX)(); }
            static inline VectorType one()  { return CAT(_mm_setone_, SUFFIX)(); }// set(1.); }

            static inline void multiplyAndAdd(VectorType &v1, VectorType v2, VectorType v3) { v1 = add(mul(v1, v2), v3); }
            static inline VectorType mul(VectorType a, VectorType b, _M128 _mask) {
                _M128D mask = _mm_castps_pd(_mask);
                return _mm_or_pd(
                    _mm_and_pd(mask, _mm_mul_pd(a, b)),
                    _mm_andnot_pd(mask, a)
                    );
            }
            static inline VectorType div(VectorType a, VectorType b, _M128 _mask) {
                _M128D mask = _mm_castps_pd(_mask);
                return _mm_or_pd(
                    _mm_and_pd(mask, _mm_div_pd(a, b)),
                    _mm_andnot_pd(mask, a)
                    );
            }

            OP(add) OP(sub) OP(mul) OP(div)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt)
            static inline VectorType rsqrt(VectorType x) {
                return _mm_div_pd(one(), sqrt(x));
            }
            static inline VectorType isNaN(VectorType x) {
                return _mm_cmpunord_pd(x, x);
            }
            static inline VectorType isFinite(VectorType x) {
                return _mm_cmpord_pd(x, _mm_mul_pd(zero(), x));
            }
            static VectorType log(VectorType x) {
                const _M128D one = set(1.);
                const _M128D invalid_mask = cmplt(x, zero());
                const _M128D infinity_mask = cmpeq(x, zero());

                x = max(x, set(std::numeric_limits<double>::min()));  // lazy: cut off denormalized numbers

                _M128I emm0 = _mm_srli_epi64(_mm_castpd_si128(x), 52);
                emm0 = _mm_sub_epi32(emm0, _mm_set1_epi32(1023));
                _M128D e = _mm_cvtepi32_pd(_mm_shuffle_epi32(emm0, _MM_SHUFFLE(2, 0, 2, 0)));
                e = add(e, one);

                // keep only the fractional part
                const union {
                    unsigned long long int i;
                    double d;
                } mantissa_mask = { 0x800fffffffffffffull };
                x = _mm_and_pd(x, set(mantissa_mask.d));
                x = _mm_or_pd(x, set(0.5));

                const _M128D mask = cmplt(x, set(0.70710678118654757273731092936941422522068023681640625));

                const _M128D tmp = _mm_and_pd(x, mask);
                x = sub(x, one);
                x = add(x, tmp);

                e = sub(e, _mm_and_pd(one, mask));

                const _M128D z = mul(x, x);

                static const union {
                    unsigned short s[6 * 4];
                    double d[6];
                } P = { {
                    0x1bb0,0x93c3,0xb4c2,0x3f1a,
                    0x52f2,0x3f56,0xd6f5,0x3fdf,
                    0x6911,0xed92,0xd2ba,0x4012,
                    0xeb2e,0xc63e,0xff72,0x402c,
                    0xc84d,0x924b,0xefd6,0x4031,
                    0xdcf8,0x7d7e,0xd563,0x401e
                } };
                static const union {
                    unsigned short s[5 * 4];
                    double d[5];
                } Q = { {
                    0xef8e,0xae97,0x9320,0x4026,
                    0xc033,0x4e19,0x9d2c,0x4046,
                    0xbdbd,0xa326,0xbf33,0x4054,
                    0xae21,0xeb5e,0xc9e2,0x4051,
                    0x25b2,0x9e1f,0x200a,0x4037
                } };

                _M128D y = set(P.d[0]);
                for (int i = 1; i < 6; ++i) {
                    y = add(mul(y, x), set(P.d[i]));
                }
                _M128D y2 = add(set(Q.d[0]), x);
                for (int i = 1; i < 5; ++i) {
                    y2 = add(mul(y2, x), set(Q.d[i]));
                }
                y = mul(y, x);
                y = div(y, y2);

                y = mul(y, z);
                y = sub(y, mul(e, set(2.121944400546905827679e-4)));
                y = sub(y, mul(z, set(0.5)));

                x = add(x, y);
                x = add(x, mul(e, set(0.693359375)));
                x = _mm_or_pd(x, invalid_mask); // negative arg will be NAN
                x = _mm_or_pd(_mm_andnot_pd(infinity_mask, x), _mm_and_pd(infinity_mask, set(-std::numeric_limits<double>::infinity())));
                return x;
            }
            static inline VectorType abs(const VectorType a) {
                return CAT(_mm_and_, SUFFIX)(a, _mm_setabsmask_pd());
            }

            MINMAX
#undef SUFFIX
        };

        template<> struct VectorHelper<float> {
            typedef float EntryType;
            typedef _M128 VectorType;
#define SUFFIX ps

            static inline VectorType notMaskedToZero(VectorType a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(mask, a); }
            static inline VectorType set(const float a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline VectorType set(const float a, const float b, const float c, const float d) { return CAT(_mm_set_, SUFFIX)(a, b, c, d); }
            static inline VectorType zero() { return CAT(_mm_setzero_, SUFFIX)(); }
            static inline VectorType one()  { return CAT(_mm_setone_, SUFFIX)(); }// set(1.f); }
            static inline _M128 concat(_M128D a, _M128D b) { return _mm_movelh_ps(_mm_cvtpd_ps(a), _mm_cvtpd_ps(b)); }

            static bool pack(VectorType &v1, _M128I &_m1, VectorType &v2, _M128I &_m2) {
                {
                    VectorType &m1 = reinterpret_cast<VectorType &>(_m1);
                    VectorType &m2 = reinterpret_cast<VectorType &>(_m2);
                    const int m1Mask = _mm_movemask_ps(m1);
                    const int m2Mask = _mm_movemask_ps(m2);
                    if (0 == (m1Mask & 8)) {
                        if (0 == (m1Mask & 4)) {
                            if (0 == (m1Mask & 2)) {
                                if (0 == m1Mask) {
                                    m1 = m2;
                                    v1 = v2;
                                    m2 = zero();
                                    return _mm_movemask_ps(m1) == 15;
                                }
                                if (0 == (m2Mask & 8)) {
#ifdef __SSSE3__
                                    v1 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(v1), _mm_castps_si128(v2), sizeof(float)));
#else
                                    v1 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(3, 3, 1, 2));
                                    v1 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(0, 2, 2, 3));
#endif
                                }
                            }
                        }
                    }
                }



                // there are 256 different m1.m2 combinations
                VectorType &m1 = reinterpret_cast<VectorType &>(_m1);
                VectorType &m2 = reinterpret_cast<VectorType &>(_m2);
                const int m1Mask = _mm_movemask_ps(m1);
                switch (m1Mask) {
                case 15: // v1 is full, nothing to do
                    return true;
                // 240 left
                case 0:  // v1 is empty, take v2
                    m1 = m2;
                    v1 = v2;
                    m2 = zero();
                    return _mm_movemask_ps(m1) == 15;
                // 224 left
                default:
                    {
                        VectorType tmp;
                        const int m2Mask = _mm_movemask_ps(m2);
                        switch (m2Mask) {
                        case 15: // v2 is full, just swap
                            tmp = v1;
                            v1 = v2;
                            v2 = tmp;
                            tmp = m1;
                            m1 = m2;
                            m2 = tmp;
                            return true;
                // 210 left
                        case 0: // v2 is empty, nothing to be gained from packing
                            return false;
                // 196 left
                        }
                        // m1 and m2 are neither full nor empty
                        tmp = _mm_or_ps(m1, m2);
                        const int m3Mask = _mm_movemask_ps(tmp);
                        // m3Mask tells use where both vectors have no entries
                        const int m4Mask = _mm_movemask_ps(_mm_and_ps(m1, m2));
                        // m3Mask tells use where both vectors have entries
                        if (m4Mask == 0 || m3Mask == 15) {
                            // m4Mask == 0: No overlap, simply move all we can from v2 into v1.
                            //              Empties v2.
                            // m3Mask == 15: Simply merge the parts from v2 into v1 where v1 is
                            //               empty.
                            const VectorType m2Move = _mm_andnot_ps(m1, m2); // the part to be moved into v1
                            v1 = _mm_add_ps(
                                    _mm_and_ps(v1, m1),
                                    _mm_and_ps(v2, m2Move)
                                    );
                            m1 = tmp;
                            m2 = _mm_andnot_ps(m2Move, m2);
                            return m3Mask == 15;
                // 
                        }
                        if ((m4Mask & 3) == 3) {
                            // the high values are available
                            tmp = _mm_unpackhi_ps(v1, v2);
                            v2  = _mm_unpacklo_ps(v1, v2);
                            v1  = tmp;
                            tmp = _mm_unpackhi_ps(m1, m2);
                            m2  = _mm_unpacklo_ps(m1, m2);
                            m1  = tmp;
                            return true;
                        }
                        if ((m4Mask & 12) == 12) {
                            // the low values are available
                            tmp = _mm_unpacklo_ps(v1, v2);
                            v2  = _mm_unpackhi_ps(v1, v2);
                            v1  = tmp;
                            tmp = _mm_unpacklo_ps(m1, m2);
                            m2  = _mm_unpackhi_ps(m1, m2);
                            m1  = tmp;
                            return true;
                        }
                        if ((m4Mask & 5) == 5) {
                            tmp = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(0, 2, 0, 2));
                            v2  = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(1, 3, 1, 3));
                            v1  = tmp;
                            tmp = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(0, 2, 0, 2));
                            m2  = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(1, 3, 1, 3));
                            m1  = tmp;
                            return true;
                        }
                        if ((m4Mask & 6) == 6) {
                            tmp = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(1, 2, 1, 2));
                            v2  = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(0, 3, 0, 3));
                            v1  = tmp;
                            tmp = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(1, 2, 1, 2));
                            m2  = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(0, 3, 0, 3));
                            m1  = tmp;
                            return true;
                        }
                        if ((m4Mask & 9) == 9) {
                            tmp = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(0, 3, 0, 3));
                            v2  = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(1, 2, 1, 2));
                            v1  = tmp;
                            tmp = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(0, 3, 0, 3));
                            m2  = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(1, 2, 1, 2));
                            m1  = tmp;
                            return true;
                        }
                        if ((m4Mask & 10) == 10) {
                            tmp = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(1, 3, 1, 3));
                            v2  = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(0, 2, 0, 2));
                            v1  = tmp;
                            tmp = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(1, 3, 1, 3));
                            m2  = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(0, 2, 0, 2));
                            m1  = tmp;
                            return true;
                        }
                        float *const vv1 = reinterpret_cast<float *>(&v1);
                        float *const vv2 = reinterpret_cast<float *>(&v2);
                        unsigned int *const mm1 = reinterpret_cast<unsigned int *>(&_m1);
                        unsigned int *const mm2 = reinterpret_cast<unsigned int *>(&_m2);
                        int j = 0;
                        for (int i = 0; i < 4; ++i) {
                            if (!(m1Mask & (1 << i))) { // v1 entry not set, take the first from v2
                                while (j < 4 && !(m2Mask & (1 << j))) {
                                    ++j;
                                }
                                if (j < 4) {
                                    vv1[i] = vv2[j];
                                    mm1[i] = 0xffffffff;
                                    mm2[j] = 0;
                                    ++j;
                                }
                            }
                        }
                        return _mm_movemask_ps(m1) == 15;
//X                         // m4Mask has exactly one bit set
//X                         switch (m4Mask) {
//X                         case 1:
//X                             // x___    xx__    xx__    xx__    xxx_    x_x_    x_x_    x_xx    x__x  + mirrored horizontally
//X                             // x___    x___    x_x_    x__x    x___    x___    x__x    x___    x___
//X                             break;
//X                         case 2:
//X                             break;
//X                         case 4:
//X                             break;
//X                         case 8:
//X                             break;
//X                         }
                    }
                }
            }

            static inline void multiplyAndAdd(VectorType &v1, VectorType v2, VectorType v3) { v1 = add(mul(v1, v2), v3); }
            static inline VectorType mul(VectorType a, VectorType b, _M128 mask) {
                return _mm_or_ps(
                    _mm_and_ps(mask, _mm_mul_ps(a, b)),
                    _mm_andnot_ps(mask, a)
                    );
            }
            static inline VectorType div(VectorType a, VectorType b, _M128 mask) {
                return _mm_or_ps(
                    _mm_and_ps(mask, _mm_div_ps(a, b)),
                    _mm_andnot_ps(mask, a)
                    );
            }

            OP(add) OP(sub) OP(mul) OP(div)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt) OP1(rsqrt)
            static inline VectorType isNaN(VectorType x) {
                return _mm_cmpunord_ps(x, x);
            }
            static inline VectorType isFinite(VectorType x) {
                return _mm_cmpord_ps(x, _mm_mul_ps(zero(), x));
            }
            static VectorType log(VectorType x) {
                const _M128 one = set(1.);
                const _M128 invalid_mask = cmplt(x, zero());
                const _M128 infinity_mask = cmpeq(x, zero());

                x = max(x, set(std::numeric_limits<float>::min()));  // cut off denormalized stuff

                const _M128I emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
                _M128 e = _mm_cvtepi32_ps(_mm_sub_epi32(emm0, _mm_set1_epi32(127)));
                e = add(e, one);

                // keep only the fractional part
                const union {
                    unsigned int i;
                    float f;
                } mantissa_mask = { 0x807fffff };
                x = _mm_and_ps(x, set(mantissa_mask.f));
                x = _mm_or_ps(x, set(0.5));

                const _M128 mask = cmplt(x, set(0.707106781186547524f));

                const _M128 tmp = _mm_and_ps(x, mask);
                x = sub(x, one);
                x = add(x, tmp);

                e = sub(e, _mm_and_ps(one, mask));

                const _M128 z = mul(x, x);

                _M128 y = set( 7.0376836292e-2f);
                y = mul(y, x);
                y = add(y, set(-1.1514610310e-1f));
                y = mul(y, x);
                y = add(y, set( 1.1676998740e-1f));
                y = mul(y, x);
                y = add(y, set(-1.2420140846e-1f));
                y = mul(y, x);
                y = add(y, set( 1.4249322787e-1f));
                y = mul(y, x);
                y = add(y, set(-1.6668057665e-1f));
                y = mul(y, x);
                y = add(y, set( 2.0000714765e-1f));
                y = mul(y, x);
                y = add(y, set(-2.4999993993e-1f));
                y = mul(y, x);
                y = add(y, set( 3.3333331174e-1f));
                y = mul(y, x);

                y = mul(y, z);
                y = add(y, mul(e, set(-2.12194440e-4f)));
                y = sub(y, mul(z, set(0.5)));

                x = add(x, y);
                x = add(x, mul(e, set(0.693359375f)));
                x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
                x = _mm_or_ps(_mm_andnot_ps(infinity_mask, x), _mm_and_ps(infinity_mask, set(-std::numeric_limits<float>::infinity())));
                return x;
            }
            static inline VectorType abs(const VectorType a) {
                return CAT(_mm_and_, SUFFIX)(a, _mm_setabsmask_ps());
            }

            MINMAX
#undef SUFFIX
        };

        template<> struct VectorHelper<float8> {
            typedef float EntryType;
            typedef M256 VectorType;

            static inline VectorType set(const float a) {
                const _M128 x = _mm_set1_ps(a);
                return VectorType(x, x);
            }
            static inline VectorType set(const float a, const float b, const float c, const float d) {
                const _M128 x = _mm_set_ps(a, b, c, d);
                return VectorType(x, x);
            }
            static inline VectorType set(const float a, const float b, const float c, const float d,
                    const float e, const float f, const float g, const float h) {
                return VectorType(_mm_set_ps(a, b, c, d), _mm_set_ps(e, f, g, h));
            }
            static inline VectorType zero() { return VectorType(_mm_setzero_ps(), _mm_setzero_ps()); }
            static inline VectorType one()  { return set(1.f); }

#define REUSE_FLOAT_IMPL1(fun) \
            static inline VectorType fun(const VectorType &x) { \
                return VectorType(VectorHelper<float>::fun(x[0]), VectorHelper<float>::fun(x[1])); \
            }
#define REUSE_FLOAT_IMPL2(fun) \
            static inline VectorType fun(const VectorType &x, const VectorType &y) { \
                return VectorType(VectorHelper<float>::fun(x[0], y[0]), VectorHelper<float>::fun(x[1], y[1])); \
            }
#define REUSE_FLOAT_IMPL3(fun) \
            static inline VectorType fun(const VectorType &x, const VectorType &y, const VectorType &z) { \
                return VectorType(VectorHelper<float>::fun(x[0], y[0], z[0]), VectorHelper<float>::fun(x[1], y[1], z[1])); \
            }
            REUSE_FLOAT_IMPL1(sqrt)
            REUSE_FLOAT_IMPL1(rsqrt)
            REUSE_FLOAT_IMPL1(isNaN)
            REUSE_FLOAT_IMPL1(isFinite)
            REUSE_FLOAT_IMPL1(log)
            REUSE_FLOAT_IMPL1(abs)

            REUSE_FLOAT_IMPL2(notMaskedToZero)
            REUSE_FLOAT_IMPL2(add)
            REUSE_FLOAT_IMPL2(sub)
            REUSE_FLOAT_IMPL2(mul)
            REUSE_FLOAT_IMPL2(div)
            REUSE_FLOAT_IMPL2(cmple)
            REUSE_FLOAT_IMPL2(cmpnle)
            REUSE_FLOAT_IMPL2(cmplt)
            REUSE_FLOAT_IMPL2(cmpnlt)
            REUSE_FLOAT_IMPL2(cmpeq)
            REUSE_FLOAT_IMPL2(cmpneq)
            REUSE_FLOAT_IMPL2(min)
            REUSE_FLOAT_IMPL2(max)

            static inline void multiplyAndAdd(VectorType &a, const VectorType &b, const VectorType &c) {
                VectorHelper<float>::multiplyAndAdd(a[0], b[0], c[0]);
                VectorHelper<float>::multiplyAndAdd(a[1], b[1], c[1]);
            }
            REUSE_FLOAT_IMPL3(mul)
            REUSE_FLOAT_IMPL3(div)
#undef REUSE_FLOAT_IMPL3
#undef REUSE_FLOAT_IMPL2
#undef REUSE_FLOAT_IMPL1
        };

        template<> struct VectorHelper<int> {
            typedef int EntryType;
            typedef _M128I VectorType;
#define SUFFIX si128

            OP_(or_) OP_(and_) OP_(xor_)
            static inline VectorType zero() { return CAT(_mm_setzero_, SUFFIX)(); }
            static inline VectorType notMaskedToZero(VectorType a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }
#undef SUFFIX
#define SUFFIX epi32
            static inline VectorType one() { return CAT(_mm_setone_, SUFFIX)(); }

            static inline VectorType set(const int a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline VectorType set(const int a, const int b, const int c, const int d) { return CAT(_mm_set_, SUFFIX)(a, b, c, d); }

            static inline void multiplyAndAdd(VectorType &v1, VectorType v2, VectorType v3) { v1 = add(mul(v1, v2), v3); }

            SHIFT4


            OP1(abs)

            MINMAX
#ifdef __SSE4_1__
            static inline VectorType mul(VectorType a, VectorType b) { return _mm_mullo_epi32(a, b); }
            static inline VectorType mul(VectorType a, VectorType b, _M128 _mask) {
                _M128I mask = _mm_castps_si128(_mask);
                return _mm_or_si128(
                    _mm_and_si128(mask, _mm_mullo_epi32(a, b)),
                    _mm_andnot_si128(mask, a)
                    );
            }
#else
            static inline VectorType mul(const VectorType a, const VectorType b, _M128 _mask) {
                const int mask = _mm_movemask_ps(_mask);
                STORE_VECTOR(int, _a, a);
                STORE_VECTOR(int, _b, b);
                union {
                    int i[4];
                    VectorType v;
                } x = { {
                    (mask & 1 ? _a[0] * _b[0] : _a[0]),
                    (mask & 2 ? _a[1] * _b[1] : _a[1]),
                    (mask & 4 ? _a[2] * _b[2] : _a[2]),
                    (mask & 8 ? _a[3] * _b[3] : _a[3])
                } };
                return x.v;
            }
            static inline VectorType mul(const VectorType a, const VectorType b) {
                STORE_VECTOR(int, _a, a);
                STORE_VECTOR(int, _b, b);
                union {
                    int i[4];
                    VectorType v;
                } x = { {
                    _a[0] * _b[0],
                    _a[1] * _b[1],
                    _a[2] * _b[2],
                    _a[3] * _b[3]
                } };
                return x.v;
//X                 VectorType hi = _mm_mulhi_epi16(a, b);
//X                 hi = _mm_slli_epi32(hi, 16);
//X                 VectorType lo = _mm_mullo_epi16(a, b);
//X                 return or_(hi, lo);
            }
#endif

            static inline VectorType div(const VectorType a, const VectorType b, _M128 _mask) {
                const int mask = _mm_movemask_ps(_mask);
                STORE_VECTOR(int, _a, a);
                STORE_VECTOR(int, _b, b);
                union {
                    int i[4];
                    VectorType v;
                } x = { {
                    (mask & 1 ? _a[0] / _b[0] : _a[0]),
                    (mask & 2 ? _a[1] / _b[1] : _a[1]),
                    (mask & 4 ? _a[2] / _b[2] : _a[2]),
                    (mask & 8 ? _a[3] / _b[3] : _a[3])
                } };
                return x.v;
            }
            static inline VectorType div(const VectorType a, const VectorType b) {
                STORE_VECTOR(int, _a, a);
                STORE_VECTOR(int, _b, b);
                union {
                    int i[4];
                    VectorType v;
                } x = { {
                    _a[0] / _b[0],
                    _a[1] / _b[1],
                    _a[2] / _b[2],
                    _a[3] / _b[3]
                } };
                return x.v;
            }

            OP(add) OP(sub)
            OPcmp(eq)
            OPcmp(lt)
            OPcmp(gt)
            static inline VectorType cmpneq(const VectorType &a, const VectorType &b) { _M128I x = cmpeq(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static inline VectorType cmpnlt(const VectorType &a, const VectorType &b) { _M128I x = cmplt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static inline VectorType cmple (const VectorType &a, const VectorType &b) { _M128I x = cmpgt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static inline VectorType cmpnle(const VectorType &a, const VectorType &b) { return cmpgt(a, b); }
#undef SUFFIX
        };

        template<> struct VectorHelper<unsigned int> {
            typedef unsigned int EntryType;
            typedef _M128I VectorType;
#define SUFFIX si128
            OP_CAST_(or_) OP_CAST_(and_) OP_CAST_(xor_)
            static inline VectorType zero() { return CAT(_mm_setzero_, SUFFIX)(); }
            static inline VectorType notMaskedToZero(VectorType a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }

#undef SUFFIX
#define SUFFIX epu32
            static inline VectorType one() { return CAT(_mm_setone_, SUFFIX)(); }

            MINMAX

            static inline VectorType mul(VectorType a, VectorType b, _M128 _mask) {
                _M128I mask = _mm_castps_si128(_mask);
                return _mm_or_si128(
                    _mm_and_si128(mask, mul(a, b)),
                    _mm_andnot_si128(mask, a)
                    );
            }
            static inline VectorType mul(const VectorType a, const VectorType b) {
                VectorType hi = _mm_mulhi_epu16(a, b);
                hi = _mm_slli_epi32(hi, 16);
                VectorType lo = _mm_mullo_epi16(a, b);
                return or_(hi, lo);
            }
            static inline VectorType div(const VectorType a, const VectorType b, _M128 _mask) {
                const int mask = _mm_movemask_ps(_mask);
                STORE_VECTOR(unsigned int, _a, a);
                STORE_VECTOR(unsigned int, _b, b);
                union {
                    unsigned int i[4];
                    VectorType v;
                } x = { {
                    (mask & 1 ? _a[0] / _b[0] : _a[0]),
                    (mask & 2 ? _a[1] / _b[1] : _a[1]),
                    (mask & 4 ? _a[2] / _b[2] : _a[2]),
                    (mask & 8 ? _a[3] / _b[3] : _a[3])
                } };
                return x.v;
            }
            static inline VectorType div(const VectorType a, const VectorType b) {
                STORE_VECTOR(unsigned int, _a, a);
                STORE_VECTOR(unsigned int, _b, b);
                union {
                    unsigned int i[4];
                    VectorType v;
                } x = { {
                    _a[0] / _b[0],
                    _a[1] / _b[1],
                    _a[2] / _b[2],
                    _a[3] / _b[3]
                } };
                return x.v;
            }

#undef SUFFIX
#define SUFFIX epi32
            static inline VectorType set(const unsigned int a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline VectorType set(const unsigned int a, const unsigned int b, const unsigned int c, const unsigned int d) { return CAT(_mm_set_, SUFFIX)(a, b, c, d); }

            SHIFT4
            OP(add) OP(sub)
            OPcmp(eq)
            static inline VectorType cmpneq(const VectorType &a, const VectorType &b) { return _mm_andnot_si128(cmpeq(a, b), _mm_setallone_si128()); }

#ifdef USE_CORRECT_UNSIGNED_COMPARE
            static inline VectorType cmplt(const VectorType &a, const VectorType &b) {
                return _mm_cmplt_epi32(_mm_xor_si128(a, _mm_setmin_epi32()), _mm_xor_si128(b, _mm_setmin_epi32()));
            }
            static inline VectorType cmpgt(const VectorType &a, const VectorType &b) {
                return _mm_cmpgt_epi32(_mm_xor_si128(a, _mm_setmin_epi32()), _mm_xor_si128(b, _mm_setmin_epi32()));
            }
#else
            OPcmp(lt)
            OPcmp(gt)
#endif
            static inline VectorType cmpnlt(const VectorType &a, const VectorType &b) { return _mm_andnot_si128(cmplt(a, b), _mm_setallone_si128()); }
            static inline VectorType cmple (const VectorType &a, const VectorType &b) { return _mm_andnot_si128(cmpgt(a, b), _mm_setallone_si128()); }
            static inline VectorType cmpnle(const VectorType &a, const VectorType &b) { return cmpgt(a, b); }

#undef SUFFIX
        };

        template<> struct VectorHelper<signed short> {
            typedef _M128I VectorType;
            typedef signed short EntryType;
#define SUFFIX si128

            OP_(or_) OP_(and_) OP_(xor_)
            static inline VectorType zero() { return CAT(_mm_setzero_, SUFFIX)(); }
            static inline VectorType notMaskedToZero(VectorType a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }
            static inline _M128I concat(_M128I a, _M128I b) { return _mm_packs_epi32(a, b); }
            static inline _M128I expand0(_M128I x) { return _mm_srai_epi32(_mm_unpacklo_epi16(x, x), 16); }
            static inline _M128I expand1(_M128I x) { return _mm_srai_epi32(_mm_unpackhi_epi16(x, x), 16); }

#undef SUFFIX
#define SUFFIX epi16
            static inline VectorType one() { return CAT(_mm_setone_, SUFFIX)(); }
            SHIFT8

            static inline VectorType set(const EntryType a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline VectorType set(const EntryType a, const EntryType b, const EntryType c, const EntryType d,
                    const EntryType e, const EntryType f, const EntryType g, const EntryType h) {
                return CAT(_mm_set_, SUFFIX)(a, b, c, d, e, f, g, h);
            }

            static inline void multiplyAndAdd(VectorType &v1, VectorType v2, VectorType v3) {
                v1 = add(mul(v1, v2), v3); }

            OP1(abs)

            static inline VectorType mul(VectorType a, VectorType b, _M128 _mask) {
                _M128I mask = _mm_castps_si128(_mask);
                return _mm_or_si128(
                    _mm_and_si128(mask, mul(a, b)),
                    _mm_andnot_si128(mask, a)
                    );
            }
            OPx(mul, mullo)
            OP(min) OP(max)

            static inline VectorType div(const VectorType a, const VectorType b, _M128 _mask) {
                const int mask = _mm_movemask_epi8(_mm_castps_si128(_mask));
                STORE_VECTOR(EntryType, _a, a);
                STORE_VECTOR(EntryType, _b, b);
                union {
                    EntryType i[8];
                    VectorType v;
                } x = { {
                    (mask & 0x0001 ? _a[0] / _b[0] : _a[0]),
                    (mask & 0x0004 ? _a[1] / _b[1] : _a[1]),
                    (mask & 0x0010 ? _a[2] / _b[2] : _a[2]),
                    (mask & 0x0040 ? _a[3] / _b[3] : _a[3]),
                    (mask & 0x0100 ? _a[4] / _b[4] : _a[4]),
                    (mask & 0x0400 ? _a[5] / _b[5] : _a[5]),
                    (mask & 0x1000 ? _a[6] / _b[6] : _a[6]),
                    (mask & 0x4000 ? _a[7] / _b[7] : _a[7])
                } };
                return x.v;
            }
            static inline VectorType div(const VectorType a, const VectorType b) {
                STORE_VECTOR(EntryType, _a, a);
                STORE_VECTOR(EntryType, _b, b);
                union {
                    EntryType i[8];
                    VectorType v;
                } x = { {
                    _a[0] / _b[0],
                    _a[1] / _b[1],
                    _a[2] / _b[2],
                    _a[3] / _b[3],
                    _a[4] / _b[4],
                    _a[5] / _b[5],
                    _a[6] / _b[6],
                    _a[7] / _b[7]
                } };
                return x.v;
            }

            OP(add) OP(sub)
            OPcmp(eq)
            OPcmp(lt)
            OPcmp(gt)
            static inline VectorType cmpneq(const VectorType &a, const VectorType &b) { _M128I x = cmpeq(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static inline VectorType cmpnlt(const VectorType &a, const VectorType &b) { _M128I x = cmplt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static inline VectorType cmple (const VectorType &a, const VectorType &b) { _M128I x = cmpgt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static inline VectorType cmpnle(const VectorType &a, const VectorType &b) { return cmpgt(a, b); }
#undef SUFFIX
        };

        template<> struct VectorHelper<unsigned short> {
            typedef _M128I VectorType;
            typedef unsigned short EntryType;
#define SUFFIX si128
            OP_CAST_(or_) OP_CAST_(and_) OP_CAST_(xor_)
            static inline VectorType zero() { return CAT(_mm_setzero_, SUFFIX)(); }
            static inline VectorType notMaskedToZero(VectorType a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }
#ifdef __SSE4_1__
            static inline _M128I concat(_M128I a, _M128I b) { return _mm_packus_epi32(a, b); }
#else
            // XXX too bad, but this is broken without SSE 4.1
            static inline _M128I concat(_M128I a, _M128I b) { return _mm_packs_epi32(a, b); }
#endif
            static inline _M128I expand0(_M128I x) { return _mm_srli_epi32(_mm_unpacklo_epi16(x, x), 16); }
            static inline _M128I expand1(_M128I x) { return _mm_srli_epi32(_mm_unpackhi_epi16(x, x), 16); }

#undef SUFFIX
#define SUFFIX epu16
            static inline VectorType one() { return CAT(_mm_setone_, SUFFIX)(); }
            static inline VectorType div(const VectorType a, const VectorType b, _M128 _mask) {
                const int mask = _mm_movemask_epi8(_mm_castps_si128(_mask));
                STORE_VECTOR(EntryType, _a, a);
                STORE_VECTOR(EntryType, _b, b);
                union {
                    EntryType i[8];
                    VectorType v;
                } x = { {
                    (mask & 0x0001 ? _a[0] / _b[0] : _a[0]),
                    (mask & 0x0004 ? _a[1] / _b[1] : _a[1]),
                    (mask & 0x0010 ? _a[2] / _b[2] : _a[2]),
                    (mask & 0x0040 ? _a[3] / _b[3] : _a[3]),
                    (mask & 0x0100 ? _a[4] / _b[4] : _a[4]),
                    (mask & 0x0400 ? _a[5] / _b[5] : _a[5]),
                    (mask & 0x1000 ? _a[6] / _b[6] : _a[6]),
                    (mask & 0x4000 ? _a[7] / _b[7] : _a[7])
                } };
                return x.v;
            }
            static inline VectorType div(const VectorType a, const VectorType b) {
                STORE_VECTOR(EntryType, _a, a);
                STORE_VECTOR(EntryType, _b, b);
                union {
                    EntryType i[8];
                    VectorType v;
                } x = { {
                    _a[0] / _b[0],
                    _a[1] / _b[1],
                    _a[2] / _b[2],
                    _a[3] / _b[3],
                    _a[4] / _b[4],
                    _a[5] / _b[5],
                    _a[6] / _b[6],
                    _a[7] / _b[7]
                } };
                return x.v;
            }

            static inline VectorType mul(VectorType a, VectorType b, _M128 _mask) {
                _M128I mask = _mm_castps_si128(_mask);
                return _mm_or_si128(
                    _mm_and_si128(mask, mul(a, b)),
                    _mm_andnot_si128(mask, a)
                    );
            }
#undef SUFFIX
#define SUFFIX epi16
            SHIFT8
            OPx(mul, mullo) // should work correctly for all values
            OP(min) OP(max) // XXX breaks for values with MSB set
            static inline VectorType set(const EntryType a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline VectorType set(const EntryType a, const EntryType b, const EntryType c,
                    const EntryType d, const EntryType e, const EntryType f,
                    const EntryType g, const EntryType h) {
                return CAT(_mm_set_, SUFFIX)(a, b, c, d, e, f, g, h);
            }

            OP(add) OP(sub)
            OPcmp(eq)
            static inline VectorType cmpneq(const VectorType &a, const VectorType &b) { return _mm_andnot_si128(cmpeq(a, b), _mm_setallone_si128()); }

#ifdef USE_CORRECT_UNSIGNED_COMPARE
            static inline VectorType cmplt(const VectorType &a, const VectorType &b) {
                return _mm_cmplt_epi16(_mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16()));
            }
            static inline VectorType cmpgt(const VectorType &a, const VectorType &b) {
                return _mm_cmpgt_epi16(_mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16()));
            }
#else
            OPcmp(lt)
            OPcmp(gt)
#endif
            static inline VectorType cmpnlt(const VectorType &a, const VectorType &b) { return _mm_andnot_si128(cmplt(a, b), _mm_setallone_si128()); }
            static inline VectorType cmple (const VectorType &a, const VectorType &b) { return _mm_andnot_si128(cmpgt(a, b), _mm_setallone_si128()); }
            static inline VectorType cmpnle(const VectorType &a, const VectorType &b) { return cmpgt(a, b); }
#undef SUFFIX
        };
#undef SHIFT4
#undef SHIFT8
#undef OP1
#undef OP
#undef OP_
#undef OPx
#undef OPcmp
#undef CAT
#undef CAT_HELPER

namespace VectorSpecialInitializerZero { enum ZEnum { Zero }; }
namespace VectorSpecialInitializerOne { enum OEnum { One }; }
namespace VectorSpecialInitializerRandom { enum REnum { Random }; }
namespace VectorSpecialInitializerIndexesFromZero { enum IEnum { IndexesFromZero }; }

template<unsigned int Size1> struct MaskHelper;
template<> struct MaskHelper<2> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_pd(_mm_castps_pd(k1)) == _mm_movemask_pd(_mm_castps_pd(k2)); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_pd(_mm_castps_pd(k1)) != _mm_movemask_pd(_mm_castps_pd(k2)); }
};
template<> struct MaskHelper<4> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_ps(k1) == _mm_movemask_ps(k2); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_ps(k1) != _mm_movemask_ps(k2); }
};
template<> struct MaskHelper<8> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_epi8(_mm_castps_si128(k1)) == _mm_movemask_epi8(_mm_castps_si128(k2)); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_epi8(_mm_castps_si128(k1)) != _mm_movemask_epi8(_mm_castps_si128(k2)); }
};

template<unsigned int VectorSize> class Mask
{
    friend class Mask<2u>;
    friend class Mask<4u>;
    friend class Mask<8u>;
    friend class Float8Mask;
    public:
        inline Mask() {}
        inline Mask(const __m128  &x) : k(x) {}
        inline Mask(const __m128d &x) : k(_mm_castpd_ps(x)) {}
        inline Mask(const __m128i &x) : k(_mm_castsi128_ps(x)) {}
        inline explicit Mask(VectorSpecialInitializerZero::ZEnum) : k(_mm_setzero_ps()) {}
        inline Mask(const Mask &rhs) : k(rhs.k) {}
        inline Mask(const Mask<VectorSize / 2> *a)
          : k(_mm_castsi128_ps(_mm_packs_epi16(a[0].dataI(), a[1].dataI()))) {}

        template<unsigned int OtherSize> explicit inline Mask(const Mask<OtherSize> &x)
        {
            _M128I tmp = x.dataI();
            if (OtherSize < VectorSize) {
                tmp = _mm_packs_epi16(tmp, _mm_setzero_si128());
                if (VectorSize / OtherSize >= 4u) { tmp = _mm_packs_epi16(tmp, _mm_setzero_si128()); }
                if (VectorSize / OtherSize >= 8u) { tmp = _mm_packs_epi16(tmp, _mm_setzero_si128()); }
            } else if (OtherSize > VectorSize) {
                tmp = _mm_unpacklo_epi8(tmp, tmp);
                if (OtherSize / VectorSize >= 4u) { tmp = _mm_unpacklo_epi8(tmp, tmp); }
                if (OtherSize / VectorSize >= 8u) { tmp = _mm_unpacklo_epi8(tmp, tmp); }
            }
            k = _mm_castsi128_ps(tmp);
        }

        inline void expand(Mask<VectorSize / 2> *x) const
        {
            enum { Shuf = _MM_SHUFFLE(1, 1, 0, 0) };
            if (VectorSize == 16u) {
                x[0].k = _mm_castsi128_ps(_mm_unpacklo_epi8 (dataI(), dataI()));
                x[1].k = _mm_castsi128_ps(_mm_unpackhi_epi8 (dataI(), dataI()));
            } else if (VectorSize == 8u) {
                x[0].k = _mm_castsi128_ps(_mm_unpacklo_epi16(dataI(), dataI()));
                x[1].k = _mm_castsi128_ps(_mm_unpackhi_epi16(dataI(), dataI()));
            } else if (VectorSize == 4u) {
                x[0].k = _mm_castsi128_ps(_mm_unpacklo_epi32(dataI(), dataI()));
                x[1].k = _mm_castsi128_ps(_mm_unpackhi_epi32(dataI(), dataI()));
            } else if (VectorSize == 2u) {
                x[0].k = _mm_castsi128_ps(_mm_unpacklo_epi64(dataI(), dataI()));
                x[1].k = _mm_castsi128_ps(_mm_unpackhi_epi64(dataI(), dataI()));
            }
        }

        inline bool operator==(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpeq (k, rhs.k); }
        inline bool operator!=(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpneq(k, rhs.k); }

        inline Mask operator&&(const Mask &rhs) const { return _mm_and_ps(k, rhs.k); }
        inline Mask operator& (const Mask &rhs) const { return _mm_and_ps(k, rhs.k); }
        inline Mask operator||(const Mask &rhs) const { return _mm_or_ps (k, rhs.k); }
        inline Mask operator| (const Mask &rhs) const { return _mm_or_ps (k, rhs.k); }
        inline Mask operator!() const { return _mm_andnot_si128(dataI(), _mm_setallone_si128()); }

        inline Mask &operator&=(const Mask &rhs) { k = _mm_and_ps(k, rhs.k); return *this; }
        inline Mask &operator|=(const Mask &rhs) { k = _mm_or_ps (k, rhs.k); return *this; }

        inline bool isFull () const { return
#ifdef __SSE4_1__
            _mm_testc_si128(_mm_setzero_si128(), dataI()); // return 1 if (0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff) == (~0 & k)
#else
            _mm_movemask_epi8(dataI()) == 0xffff;
#endif
        }
        inline bool isEmpty() const { return
#ifdef __SSE4_1__
            _mm_testz_si128(dataI(), dataI()); // return 1 if (0, 0, 0, 0) == (k & k)
#else
            _mm_movemask_epi8(dataI()) == 0x0000;
#endif
        }

        inline operator bool() const { return isFull(); }

        inline int toInt() const { return _mm_movemask_epi8(dataI()); }

        inline _M128  data () const { return k; }
        inline _M128I dataI() const { return _mm_castps_si128(k); }
        inline _M128D dataD() const { return _mm_castps_pd(k); }

        template<unsigned int OtherSize> inline Mask<OtherSize> cast() const { return Mask<OtherSize>(k); }

        inline bool operator[](int index) const {
            if (VectorSize == 2) {
                return _mm_movemask_pd(dataD()) & (1 << index);
            } else if (VectorSize == 4) {
                return _mm_movemask_ps(k) & (1 << index);
            } else if (VectorSize == 8) {
                return _mm_movemask_epi8(dataI()) & (1 << 2 * index);
            } else if (VectorSize == 16) {
                return _mm_movemask_epi8(dataI()) & (1 << index);
            }
            return false;
        }

    private:
        _M128 k;
};

class Float8Mask
{
    enum {
        PartialSize = 4,
        VectorSize = 8
    };
    public:
        inline Float8Mask() {}
        inline Float8Mask(const M256 &x) : k(x) {}
        inline explicit Float8Mask(VectorSpecialInitializerZero::ZEnum) {
            k[0] = _mm_setzero_ps();
            k[1] = _mm_setzero_ps();
        }
        inline Float8Mask(const Mask<VectorSize> &a) {
            k[0] = _mm_castsi128_ps(_mm_unpacklo_epi16(a.dataI(), a.dataI()));
            k[1] = _mm_castsi128_ps(_mm_unpackhi_epi16(a.dataI(), a.dataI()));
        }
        inline operator Mask<VectorSize>() const {
            return _mm_packs_epi32(_mm_castps_si128(k[0]), _mm_castps_si128(k[1]));
        }

        inline bool operator==(const Float8Mask &rhs) const {
            return MaskHelper<PartialSize>::cmpeq (k[0], rhs.k[0])
                && MaskHelper<PartialSize>::cmpeq (k[1], rhs.k[1]);
        }
        inline bool operator!=(const Float8Mask &rhs) const {
            return MaskHelper<PartialSize>::cmpneq(k[0], rhs.k[0])
                && MaskHelper<PartialSize>::cmpneq(k[1], rhs.k[1]);
        }

        inline Float8Mask operator&&(const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm_and_ps(k[0], rhs.k[0]);
            r.k[1] = _mm_and_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator& (const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm_and_ps(k[0], rhs.k[0]);
            r.k[1] = _mm_and_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator||(const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm_or_ps(k[0], rhs.k[0]);
            r.k[1] = _mm_or_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator| (const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm_or_ps(k[0], rhs.k[0]);
            r.k[1] = _mm_or_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator!() const {
            Float8Mask r;
            r.k[0] = _mm_andnot_ps(k[0], _mm_setallone_ps());
            r.k[1] = _mm_andnot_ps(k[1], _mm_setallone_ps());
            return r;
        }
        inline Float8Mask &operator&=(const Float8Mask &rhs) {
            k[0] = _mm_and_ps(k[0], rhs.k[0]);
            k[1] = _mm_and_ps(k[1], rhs.k[1]);
            return *this;
        }
        inline Float8Mask &operator|=(const Float8Mask &rhs) {
            k[0] = _mm_or_ps (k[0], rhs.k[0]);
            k[1] = _mm_or_ps (k[1], rhs.k[1]);
            return *this;
        }

        inline bool isFull () const { return
#ifdef __SSE4_1__
            _mm_testc_si128(_mm_setzero_si128(), _mm_castps_si128(k[0])) &&
            _mm_testc_si128(_mm_setzero_si128(), _mm_castps_si128(k[1]));
#else
            _mm_movemask_ps(k[0]) == 0xf &&
            _mm_movemask_ps(k[1]) == 0xf;
#endif
        }
        inline bool isEmpty() const { return
#ifdef __SSE4_1__
            _mm_testz_si128(_mm_castps_si128(k[0]), _mm_castps_si128(k[0])) &&
            _mm_testz_si128(_mm_castps_si128(k[1]), _mm_castps_si128(k[1]));
#else
            _mm_movemask_ps(k[0]) == 0x0 &&
            _mm_movemask_ps(k[1]) == 0x0;
#endif
        }

        inline operator bool() const { return isFull(); }

        inline int toInt() const { return (_mm_movemask_ps(k[1]) << 4) + _mm_movemask_ps(k[0]); }

        inline const M256 &data () const { return k; }

        inline bool operator[](int index) const {
            return toInt() & (1 << index);
        }

    private:
        M256 k;
};

template<typename T>
class WriteMaskedVector
{
    friend class Vector<T>;
    typedef typename VectorBase<T>::MaskType Mask;
    public:
        //prefix
        inline Vector<T> &operator++() {
            vec->data() = VectorHelper<T>::add(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return *vec;
        }
        inline Vector<T> &operator--() {
            vec->data() = VectorHelper<T>::sub(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return *vec;
        }
        //postfix
        inline Vector<T> operator++(int) {
            Vector<T> ret(*vec);
            vec->data() = VectorHelper<T>::add(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return ret;
        }
        inline Vector<T> operator--(int) {
            Vector<T> ret(*vec);
            vec->data() = VectorHelper<T>::sub(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return ret;
        }

        inline Vector<T> &operator+=(Vector<T> x) {
            vec->data() = VectorHelper<T>::add(vec->data(), VectorHelper<T>::notMaskedToZero(x.data(), mask.data()));
            return *vec;
        }
        inline Vector<T> &operator-=(Vector<T> x) {
            vec->data() = VectorHelper<T>::sub(vec->data(), VectorHelper<T>::notMaskedToZero(x.data(), mask.data()));
            return *vec;
        }
        inline Vector<T> &operator*=(Vector<T> x) {
            vec->data() = VectorHelper<T>::mul(vec->data(), x.data(), mask.data());
            return *vec;
        }
        inline Vector<T> &operator/=(Vector<T> x) {
            vec->data() = VectorHelper<T>::div(vec->data(), x.data(), mask.data());
            return *vec;
        }

        inline Vector<T> &operator=(Vector<T> x) {
            vec->assign(x, mask);
            return *vec;
        }
    private:
        WriteMaskedVector(Vector<T> *v, Mask k) : vec(v), mask(k) {}
        Vector<T> *vec;
        Mask mask;
};

template<typename T>
class Vector : public VectorBase<T>
{
    public:
        typedef VectorBase<T> Base;
        enum { Size = Base::Size };
        typedef typename Base::VectorType VectorType;
        typedef typename Base::EntryType  EntryType;
        typedef Vector<typename IndexTypeHelper<Size>::Type> IndexType;
        typedef typename Base::MaskType Mask;

        /**
         * uninitialized
         */
        inline Vector() {}

        /**
         * initialized to 0 in all 128 bits
         */
        inline explicit Vector(VectorSpecialInitializerZero::ZEnum) : Base(VectorHelper<VectorType>::zero()) {}

        /**
         * initialized to 1 for all entries in the vector
         */
        inline explicit Vector(VectorSpecialInitializerOne::OEnum) : Base(VectorHelper<VectorType>::one()) {}

        /**
         * initialized to 0, 1 (, 2, 3 (, 4, 5, 6, 7))
         */
        inline explicit Vector(VectorSpecialInitializerIndexesFromZero::IEnum) : Base(VectorHelper<VectorType>::load(Base::_IndexesFromZero())) {}

        /**
         * initialize with given _M128 vector
         */
        inline Vector(const VectorType &x) : Base(x) {}

        template<typename OtherT>
        explicit inline Vector(const Vector<OtherT> &x) : Base(StaticCastHelper<OtherT, T>::cast(x.data())) {}

        /**
         * initialize all values with the given value
         */
        inline Vector(EntryType a)
        {
            data() = VectorHelper<T>::set(a);
        }

        /**
         * Initialize the vector with the given data. \param x must point to 64 byte aligned 512
         * byte data.
         */
        inline explicit Vector(const EntryType *x) : Base(VectorHelper<VectorType>::load(x)) {}

        inline Vector(const Vector<typename CtorTypeHelper<T>::Type> *a)
            : Base(VectorHelper<T>::concat(a[0].data(), a[1].data()))
        {}

        inline void expand(Vector<typename ExpandTypeHelper<T>::Type> *x) const
        {
            if (Size == 8u) {
                x[0].data() = VectorHelper<T>::expand0(data());
                x[1].data() = VectorHelper<T>::expand1(data());
            }
        }

        static inline Vector broadcast4(const EntryType *x) { return Vector<T>(x); }

        inline void load(const EntryType *mem) { data() = VectorHelper<VectorType>::load(mem); }

        static inline Vector loadUnaligned(const EntryType *mem) { return VectorHelper<VectorType>::loadUnaligned(mem); }

        inline void makeZero() { data() = VectorHelper<VectorType>::zero(); }

        /**
         * Set all entries to zero where the mask is set. I.e. a 4-vector with a mask of 0111 would
         * set the last three entries to 0.
         */
        inline void makeZero(const Mask &k) { data() = VectorHelper<VectorType>::andnot_(mm128_reinterpret_cast<VectorType>(k.data()), data()); }

        /**
         * Store the vector data to the given memory. The memory must be 64 byte aligned and of 512
         * bytes size.
         */
        inline void store(EntryType *mem) const { VectorHelper<VectorType>::store(mem, data()); }

        /**
         * Non-temporal store variant. Writes to the memory without polluting the cache.
         */
        inline void storeStreaming(EntryType *mem) const { VectorHelper<VectorType>::storeStreaming(mem, data()); }

        inline const Vector<T> &dcba() const { return *this; }
        inline const Vector<T> cdab() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(2, 3, 0, 1))); }
        inline const Vector<T> badc() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(1, 0, 3, 2))); }
        inline const Vector<T> aaaa() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(0, 0, 0, 0))); }
        inline const Vector<T> bbbb() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(1, 1, 1, 1))); }
        inline const Vector<T> cccc() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(2, 2, 2, 2))); }
        inline const Vector<T> dddd() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(3, 3, 3, 3))); }
        inline const Vector<T> dacb() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(3, 0, 2, 1))); }

        inline Vector(const EntryType *array, const IndexType &indexes) { VectorHelperSize<T>::gather(*this, indexes, array); }
        inline Vector(const EntryType *array, const IndexType &indexes, const Mask &mask) {
            VectorHelperSize<T>::gather(*this, indexes, mask.toInt(), array);
        }

        inline void gather(const EntryType *array, const IndexType &indexes) { VectorHelperSize<T>::gather(*this, indexes, array); }
        inline void gather(const EntryType *array, const IndexType &indexes, const Mask &mask) {
            VectorHelperSize<T>::gather(*this, indexes, mask.toInt(), array);
        }

        inline void scatter(EntryType *array, const IndexType &indexes) const { VectorHelperSize<T>::scatter(*this, indexes, array); }
        inline void scatter(EntryType *array, const IndexType &indexes, const Mask &mask) const {
            VectorHelperSize<T>::scatter(*this, indexes, mask.toInt(), array);
        }

        /**
         * \param array An array of objects where one member should be gathered
         * \param member1 A member pointer to the member of the class/struct that should be gathered
         * \param indexes The indexes in the array. The correct offsets are calculated
         *                automatically.
         * \param mask Optional mask to select only parts of the vector that should be gathered
         */
        template<typename S1> inline Vector(const S1 *array, const EntryType S1::* member1, const IndexType &indexes) {
            VectorHelperSize<T>::gather(*this, indexes, array, member1);
        }
        template<typename S1> inline Vector(const S1 *array, const EntryType S1::* member1,
                const IndexType &indexes, const Mask &mask) {
            VectorHelperSize<T>::gather(*this, indexes, mask.toInt(), array, member1);
        }
        template<typename S1, typename S2> inline Vector(const S1 *array, const S2 S1::* member1,
                const EntryType S2::* member2, const IndexType &indexes) {
            VectorHelperSize<T>::gather(*this, indexes, array, member1, member2);
        }
        template<typename S1, typename S2> inline Vector(const S1 *array, const S2 S1::* member1,
                const EntryType S2::* member2, const IndexType &indexes, const Mask &mask) {
            VectorHelperSize<T>::gather(*this, indexes, mask.toInt(), array, member1, member2);
        }

        template<typename S1> inline void gather(const S1 *array, const EntryType S1::* member1,
                const IndexType &indexes) {
            VectorHelperSize<T>::gather(*this, indexes, array, member1);
        }
        template<typename S1> inline void gather(const S1 *array, const EntryType S1::* member1,
                const IndexType &indexes, const Mask &mask) {
            VectorHelperSize<T>::gather(*this, indexes, mask.toInt(), array, member1);
        }
        template<typename S1, typename S2> inline void gather(const S1 *array, const S2 S1::* member1,
                const EntryType S2::* member2, const IndexType &indexes) {
            VectorHelperSize<T>::gather(*this, indexes, array, member1, member2);
        }
        template<typename S1, typename S2> inline void gather(const S1 *array, const S2 S1::* member1,
                const EntryType S2::* member2, const IndexType &indexes, const Mask &mask) {
            VectorHelperSize<T>::gather(*this, indexes, mask.toInt(), array, member1, member2);
        }

        template<typename S1> inline void scatter(S1 *array, EntryType S1::* member1,
                const IndexType &indexes) const {
            VectorHelperSize<T>::scatter(*this, indexes, array, member1);
        }
        template<typename S1> inline void scatter(S1 *array, EntryType S1::* member1,
                const IndexType &indexes, const Mask &mask) const {
            VectorHelperSize<T>::scatter(*this, indexes, mask.toInt(), array, member1);
        }
        template<typename S1, typename S2> inline void scatter(S1 *array, S2 S1::* member1,
                EntryType S2::* member2, const IndexType &indexes) const {
            VectorHelperSize<T>::scatter(*this, indexes, array, member1, member2);
        }
        template<typename S1, typename S2> inline void scatter(S1 *array, S2 S1::* member1,
                EntryType S2::* member2, const IndexType &indexes, const Mask &mask) const {
            VectorHelperSize<T>::scatter(*this, indexes, mask.toInt(), array, member1, member2);
        }

        //prefix
        inline Vector &operator++() { data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return *this; }
        //postfix
        inline Vector operator++(int) { const Vector<T> r = *this; data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return r; }

        inline EntryType operator[](int index) const {
            return Base::d.m(index);
        }

#define OP1(fun) \
        inline Vector fun() const { return Vector<T>(VectorHelper<T>::fun(data())); } \
        inline Vector &fun##_eq() { data() = VectorHelper<T>::fun(data()); return *this; }
        OP1(sqrt)
        OP1(abs)
#undef OP1

#define OP(symbol, fun) \
        inline Vector &operator symbol##=(const Vector<T> &x) { data() = VectorHelper<T>::fun(data(), x.data()); return *this; } \
        inline Vector operator symbol(const Vector<T> &x) const { return Vector<T>(VectorHelper<T>::fun(data(), x.data())); }

        OP(+, add)
        OP(-, sub)
        OP(*, mul)
        OP(/, div)
        OP(|, or_)
        OP(&, and_)
        OP(^, xor_)
#undef OP
#define OPcmp(symbol, fun) \
        inline Mask operator symbol(const Vector<T> &x) const { return VectorHelper<T>::fun(data(), x.data()); }

        OPcmp(==, cmpeq)
        OPcmp(!=, cmpneq)
        OPcmp(>=, cmpnlt)
        OPcmp(>, cmpnle)
        OPcmp(<, cmplt)
        OPcmp(<=, cmple)
#undef OPcmp

        inline void multiplyAndAdd(const Vector<T> &factor, const Vector<T> &summand) {
            VectorHelper<T>::multiplyAndAdd(data(), factor, summand);
        }

        inline void assign( const Vector<T> &v, const Mask &mask ) {
            const VectorType k = mm128_reinterpret_cast<VectorType>(mask.data());
            data() = VectorHelper<VectorType>::blend(data(), v.data(), k);
        }

        template<typename T2> inline Vector<T2> staticCast() const { return StaticCastHelper<T, T2>::cast(data()); }
        template<typename T2> inline Vector<T2> reinterpretCast() const { return ReinterpretCastHelper<T, T2>::cast(data()); }

        inline WriteMaskedVector<T> operator()(Mask k) { return WriteMaskedVector<T>(this, k); }

        /**
         * \return \p true  This vector was completely filled. m2 might be 0 or != 0. You still have
         *                  to test this.
         *         \p false This vector was not completely filled. m2 is all 0.
         */
        //inline bool pack(Mask &m1, Vector<T> &v2, Mask &m2) {
            //return VectorHelper<T>::pack(data(), m1.data, v2.data(), m2.data);
        //}

        VectorType &data() { return Base::d.v(); }
        const VectorType &data() const { return Base::d.v(); }
};

template<typename T> class SwizzledVector : public Vector<T> {};

template<typename T> inline Vector<T> operator+(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return v.operator+(x); }
template<typename T> inline Vector<T> operator*(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return v.operator*(x); }
template<typename T> inline Vector<T> operator-(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) - v; }
template<typename T> inline Vector<T> operator/(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) / v; }
template<typename T> inline typename Vector<T>::Mask  operator< (const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) <  v; }
template<typename T> inline typename Vector<T>::Mask  operator<=(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) <= v; }
template<typename T> inline typename Vector<T>::Mask  operator> (const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) >  v; }
template<typename T> inline typename Vector<T>::Mask  operator>=(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) >= v; }
template<typename T> inline typename Vector<T>::Mask  operator==(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) == v; }
template<typename T> inline typename Vector<T>::Mask  operator!=(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) != v; }

#define OP_IMPL(T, symbol, fun) \
  template<> inline Vector<T> &VectorBase<T>::operator symbol##=(const Vector<T> &x) { d.v() = VectorHelper<T>::fun(d.v(), x.d.v()); return *static_cast<Vector<T> *>(this); } \
  template<> inline Vector<T>  VectorBase<T>::operator symbol(const Vector<T> &x) const { return Vector<T>(VectorHelper<T>::fun(d.v(), x.d.v())); }
  OP_IMPL(int, &, and_)
  OP_IMPL(int, |, or_)
  OP_IMPL(int, ^, xor_)
  OP_IMPL(int, <<, sll)
  OP_IMPL(int, >>, srl)
  OP_IMPL(unsigned int, &, and_)
  OP_IMPL(unsigned int, |, or_)
  OP_IMPL(unsigned int, ^, xor_)
  OP_IMPL(unsigned int, <<, sll)
  OP_IMPL(unsigned int, >>, srl)
  OP_IMPL(short, &, and_)
  OP_IMPL(short, |, or_)
  OP_IMPL(short, ^, xor_)
  OP_IMPL(short, <<, sll)
  OP_IMPL(short, >>, srl)
  OP_IMPL(unsigned short, &, and_)
  OP_IMPL(unsigned short, |, or_)
  OP_IMPL(unsigned short, ^, xor_)
  OP_IMPL(unsigned short, <<, sll)
  OP_IMPL(unsigned short, >>, srl)
#undef OP_IMPL
#undef OP_IMPL2

  template<typename T> static inline Vector<T> min  (const Vector<T> &x, const Vector<T> &y) { return VectorHelper<T>::min(x.data(), y.data()); }
  template<typename T> static inline Vector<T> max  (const Vector<T> &x, const Vector<T> &y) { return VectorHelper<T>::max(x.data(), y.data()); }
  template<typename T> static inline Vector<T> min  (const Vector<T> &x, const typename Vector<T>::EntryType &y) { return min(x.data(), Vector<T>(y).data()); }
  template<typename T> static inline Vector<T> max  (const Vector<T> &x, const typename Vector<T>::EntryType &y) { return max(x.data(), Vector<T>(y).data()); }
  template<typename T> static inline Vector<T> min  (const typename Vector<T>::EntryType &x, const Vector<T> &y) { return min(Vector<T>(x).data(), y.data()); }
  template<typename T> static inline Vector<T> max  (const typename Vector<T>::EntryType &x, const Vector<T> &y) { return max(Vector<T>(x).data(), y.data()); }
  template<typename T> static inline Vector<T> sqrt (const Vector<T> &x) { return VectorHelper<T>::sqrt(x.data()); }
  template<typename T> static inline Vector<T> rsqrt(const Vector<T> &x) { return VectorHelper<T>::rsqrt(x.data()); }
  template<typename T> static inline Vector<T> abs  (const Vector<T> &x) { return VectorHelper<T>::abs(x.data()); }
  template<typename T> static inline Vector<T> sin  (const Vector<T> &x) { return VectorHelper<T>::sin(x.data()); }
  template<typename T> static inline Vector<T> cos  (const Vector<T> &x) { return VectorHelper<T>::cos(x.data()); }
  template<typename T> static inline Vector<T> log  (const Vector<T> &x) { return VectorHelper<T>::log(x.data()); }
  template<typename T> static inline Vector<T> log10(const Vector<T> &x) { return VectorHelper<T>::log10(x.data()); }

  template<typename T> static inline typename Vector<T>::Mask isfinite(const Vector<T> &x) { return VectorHelper<T>::isFinite(x.data()); }
  template<typename T> static inline typename Vector<T>::Mask isnan(const Vector<T> &x) { return VectorHelper<T>::isNaN(x.data()); }
#undef ALIGN
#undef STORE_VECTOR
} // namespace SSE

#undef CONST

#endif // SSE_VECTOR_H
